import pandas as pd
import numpy as np
import pytz
from Constant_Factors import (
    NG_Tune, BL_Tune, Solar_Installed, Wind_Installed, Import_Limit, Hydro_Tune,
    Solar_Tune, Wind_Tune, total_def_limit, Factor_NSC,
    Full_EV_Fleet, NG_Installed, Hydro_Installed,
    energy_sources, calculate_lcoe
)
from Battery_Functions2 import calculate_battery_wear, interpolate_soc
import os
import math
import json
from pathlib import Path

NG_Fact = 0
Grid_Store_Max = 1000

Baseline_years_to_replace = 12.787
EV_Battery_Cost = 10000

# Main simulation loop
installed_solar = [Solar_Installed]
installed_wind = [Wind_Installed * 3200]
output_yearly = []
output_hourly = []
failed_constraints = []

SOCIAL_COST_CARBON_PER_TONNE = 190.0  # $/metric tonne CO2

CARBON_INTENSITY = {
    "solar": {"fixed": 33.0, "variable": 10.0},
    "hydro": {"fixed": 6.204, "variable": 1.9},
    "wind": {"fixed": 12.34, "variable": 0.74},
    "battery": {"fixed": 35.4, "variable": 0.0},
    "V2G": {"fixed": 0, "variable": 0.0},  # Set to 0 as V2G carbon costs calculated by accelerated battery wear
    "nuclear": {"fixed": 2.7, "variable": 12.0},
    "natural_gas": {"fixed": 0.82, "variable": 460.0},
}

EV_LCA_GCO2_PER_MILE = {
    "BOL_EOL": {2025: 4.0, 2035: 5.3, 2050: 5.2},
    "battery": {2025: 36.0, 2035: 26.0, 2050: 21.0},
    "vehicle": {2025: 31.0, 2035: 26.0, 2050: 25.0},
}

ICE_LCA_GCO2_PER_MILE = {
    "BOL_EOL": {2025: 6.0, 2035: 4.3, 2050: 4.2},
    "vehicle_ops": {2025: 311.0, 2035: 258.0, 2050: 223.0},
    "production_fuel": {2025: 72.0, 2035: 54.0, 2050: 46.0},
    "vehicle": {2025: 38.0, 2035: 33.0, 2050: 33.0},
}

def annual_extra_battery_carbon_cost(
    extra_battery_cost_per_vehicle_year: float,
    battery_lca_gco2_per_mile_2050: float = EV_LCA_GCO2_PER_MILE["battery"][2050],
    ev_battery_cost: float = EV_Battery_Cost,
    annual_miles_per_vehicle: float = 12000.0,
    social_cost_per_tonne: float = SOCIAL_COST_CARBON_PER_TONNE,
) -> float:
    """
    Convert incremental annual battery wear cost into an equivalent additional
    battery-manufacturing carbon cost.

    2050 battery LCA gCO2/mile is sclaed by the fraction of one battery
    consumed per year due to accelerated replacement.
    """
    if ev_battery_cost <= 0:
        return 0.0

    extra_battery_fraction_per_year = max(extra_battery_cost_per_vehicle_year, 0.0) / ev_battery_cost

    extra_gco2_per_vehicle_year = (
        battery_lca_gco2_per_mile_2050
        * annual_miles_per_vehicle
        * extra_battery_fraction_per_year
    )

    extra_tonnes_per_vehicle_year = extra_gco2_per_vehicle_year / 1_000_000.0

    return extra_tonnes_per_vehicle_year * social_cost_per_tonne

def get_lca_value(values_by_year: dict[int, float], year: int = 2050) -> float:
    """
    For this optimizer run, use fixed 2050 LCA values only.
    """
    return values_by_year[2050]

def carbon_cost_from_intensity(
    generated_mwh: float,
    used_mwh: float,
    fixed_gco2_per_kwh: float,
    variable_gco2_per_kwh: float,
    social_cost_per_tonne: float = SOCIAL_COST_CARBON_PER_TONNE,
) -> float:
    fixed_tonnes = generated_mwh * fixed_gco2_per_kwh / 1000.0
    variable_tonnes = used_mwh * variable_gco2_per_kwh / 1000.0
    return (fixed_tonnes + variable_tonnes) * social_cost_per_tonne


def calculate_carbon_costs(dynamic_values: dict) -> tuple[dict, float]:
    carbon_costs = {}

    for source, values in dynamic_values.items():
        if source not in CARBON_INTENSITY:
            continue

        intensity = CARBON_INTENSITY[source]

        generated_mwh = values.get(
            "annual_energy_generated_mwh",
            values.get("annual_energy_used_mwh", 0.0),
        )
        used_mwh = values.get("annual_energy_used_mwh", 0.0)

        carbon_costs[source] = carbon_cost_from_intensity(
            generated_mwh=generated_mwh,
            used_mwh=used_mwh,
            fixed_gco2_per_kwh=intensity["fixed"],
            variable_gco2_per_kwh=intensity["variable"],
        )

    return carbon_costs, sum(carbon_costs.values())

# Load data
csv_file_path = '/Users/g0g/PycharmProjects/Grid_Optimizer/Grid_Builder/ChargeTimeFix.csv'
df = pd.read_csv(csv_file_path)
df['Date'] = pd.date_range(start='2021-01-01 00:00', end='2021-12-30 23:00', freq='h')

# Step 1: Localize the Date column to UTC and convert to Pacific Time
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
pacific_zone = pytz.timezone('US/Pacific')
df['Date_Pacific'] = df['Date'].dt.tz_convert(pacific_zone)

# Add constant columns
df['Nuke_AVG'] = 0.9
df['Nuke_STDV'] = 0.03


def make_json_safe(obj):
    """
    Recursively convert NumPy/Pandas objects into normal Python types
    so they can be written to JSON.
    """
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(value) for value in obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if pd.isna(obj):
        return None

    return obj
def save_audit_report(audit_report: dict, label: str, iteration: int | None = None):
    """
    Save detailed cost audit report for hand checking.
    """
    if audit_report is None:
        return

    audit_report = make_json_safe(audit_report)

    output_dir = Path("audit_reports")
    output_dir.mkdir(exist_ok=True)

    if iteration is None:
        base_name = label
    else:
        base_name = f"{label}_iteration_{iteration}"

    json_path = output_dir / f"{base_name}_audit_report.json"
    lcoe_csv_path = output_dir / f"{base_name}_lcoe_breakdown.csv"
    summary_csv_path = output_dir / f"{base_name}_summary.csv"

    safe_audit_report = make_json_safe(audit_report)

    with open(json_path, "w") as f:
        json.dump(safe_audit_report, f, indent=4)

    rows = []
    for source, values in audit_report["lcoe_breakdown"].items():
        row = {"source": source}
        row.update(values)
        rows.append(row)

    pd.DataFrame(rows).to_csv(lcoe_csv_path, index=False)

    summary_rows = []
    for section, values in audit_report.items():
        if isinstance(values, dict) and section != "lcoe_breakdown":
            for key, value in values.items():
                if not isinstance(value, dict):
                    summary_rows.append({
                        "section": section,
                        "item": key,
                        "value": value,
                    })

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    print(f"Saved audit report: {json_path}")

def do_parallel_monte_carlo(Share_EV_init, evcharge, n_simulations=10,
                            SC_CR=0.0475, V2G_CR=0.0475, V2G_DR=0.0475,
                            SC_Floor=30, V2G_Floor=30, Cap=0.8, V2G_Connect_Cost=None):
    """
    Runs the simulation multiple times.
    Note: SC_Floor and V2G_Floor are now expressed in percentage points (e.g., 30 means 30%).
    """

    SC_Cap = Cap
    V2G_Cap = Cap
    simulation_results = []
    battery_wear_calculated = False
    final_q_V2G = None
    final_q_SC = None
    Factor_Smart_Charge = 0.8

    print("\nMONTE CARLO RECEIVED VALUES")
    print(f"BL_installed      = {BL_installed}")
    print(f"Factor_V2G        = {Factor_V2G}")
    print(f"Grid_Store_Max    = {Grid_Store_Max}")
    print(f"Wind_Installed    = {Wind_Installed}")
    print(f"Solar_Installed   = {Solar_Installed}")
    print(f"SC_CR             = {SC_CR}")
    print(f"V2G_CR            = {V2G_CR}")
    print(f"V2G_DR            = {V2G_DR}")
    print(f"Cap               = {Cap}")

    for sim_index in range(n_simulations):
        #print(f"Running Simulation {sim_index + 1}/{n_simulations}")
        failures = []
        max_def_max = 0
        all_deficits = []
        curtailments = []

        total_charge_SC_Fleet = 40 * (Factor_Smart_Charge - Factor_V2G) * Full_EV_Fleet * Share_EV_init
        total_charge_V2G_Fleet = 40 * Factor_V2G * Full_EV_Fleet * Share_EV_init
        SC_Battery_bank = SC_Cap * total_charge_SC_Fleet
        V2G_battery_bank = V2G_Cap * total_charge_V2G_Fleet

        SC_hourly_battery_bank = []
        V2G_hourly_battery_bank = []
        V2G_ChargingEnergy = 0
        SC_ChargingEnergy = 0
        hourly_deficits = []
        total_deficit = 0

        # Sample base demand
        demand = np.random.normal(
            df['Demand_AVG'],
            df['Demand_STDV']
        ).astype('float32')

        base_demand_sum = np.sum(demand)

        # Sample total EV demand once
        total_ev_demand = (
                np.random.normal(df['Charge_KWh_perCar'], df['Charge_STDV_perCar'])
                .astype('float32')
                * Share_EV_init
                * Full_EV_Fleet
        )

        # Split EV demand into NSC and SC/V2G portions
        NSC_EV_Demand = total_ev_demand * Factor_NSC
        SCV2G_EV_Demand = total_ev_demand * (Factor_Smart_Charge)

        NSC_EV_Demand_sum = np.sum(NSC_EV_Demand)
        SCV2G_EV_Demand_sum = np.sum(SCV2G_EV_Demand)

        # Only NSC is added directly to hourly grid demand here
        demand += NSC_EV_Demand

        # Power generation calculations
        solar_gen = sum(
            np.maximum(np.random.normal(df[f"{solar}_mean"], df[f"{solar}_std"]), 0)
            * Solar_Installed / 5
            for solar in ['Kern', 'Topaz', 'Deser_Sunlight', 'CA_Valley_Solar_Ranch', 'Mount_Signal']
        )
        wind_gen = sum(
            np.maximum(np.random.normal(df[f"Wind{i}_Mean"], df[f"Wind{i}_STDV"]), 0)
            * Wind_Installed / 2
            for i in range(1, 3)
        )
        solar_gen = (solar_gen / 1000) * Solar_Tune
        wind_gen = wind_gen * Wind_Tune
        hydro_gen = np.maximum(
            np.random.normal(df["Hydro_mean"], df["Hydro_STDV"]),
            0,
        ) * Hydro_Tune
        natural_gas_gen = np.random.normal(df['NG_mean'], df['NG_STDV']) * NG_Tune * NG_Fact
        nuclear_gen = np.random.normal(df['Nuke_AVG'], df['Nuke_STDV']) * BL_installed * BL_Tune

        solar_gen_sum = np.sum(solar_gen)
        wind_gen_sum = np.sum(wind_gen)
        hydro_gen_sum = np.sum(hydro_gen)
        natural_gas_gen_sum = np.sum(natural_gas_gen)
        nuclear_gen_sum = np.sum(nuclear_gen)

        power_deficit = demand - (solar_gen + wind_gen + nuclear_gen)

        # Align natural gas and hydro generation with the deficit order
        sorted_deficit_indices = np.argsort(-power_deficit)
        sorted_natural_gas_gen = np.sort(natural_gas_gen)[::-1]
        sorted_hydro_gen = np.sort(hydro_gen)[::-1]
        realigned_natural_gas_gen = np.zeros_like(natural_gas_gen)
        realigned_hydro_gen = np.zeros_like(hydro_gen)
        for i, idx in enumerate(sorted_deficit_indices):
            realigned_natural_gas_gen[idx] = sorted_natural_gas_gen[i]
            realigned_hydro_gen[idx] = sorted_hydro_gen[i]
        power_deficit_sort = power_deficit - realigned_natural_gas_gen - realigned_hydro_gen

        total_curtailment = 0
        total_v2g_tendered = 0
        Grid_Store = Grid_Store_Max * 0.6
        total_GS_used = 1
        EV_power_modeled_total = 0
        SC_DISCH_Total = 0
        V2G_DISCH_Total = 0

        top_deficit_indices = set(np.argsort(power_deficit_sort)[-10:])
        top_deficit_audit = []

        v2g_dispatch_limit_audit = []
        max_v2g_dispatch_violation = 0.0
        hours_v2g_dispatch_over_limit = 0

        # Hourly loop for managing deficits and battery operations
        for hour, deficit in enumerate(power_deficit_sort):

            deficit_start = float(deficit)
            grid_tendered_this_hour = 0.0
            SC_discharge = (df['1_Car_KWh'][hour]) * (Factor_Smart_Charge - Factor_V2G) * Full_EV_Fleet * Share_EV_init
            SC_DISCH_Total += SC_discharge
            EV_power_modeled_total += SC_discharge
            V2G_discharge = (df['1_Car_KWh'][hour]) * Factor_V2G * Full_EV_Fleet * Share_EV_init
            V2G_DISCH_Total += V2G_discharge

            EV_power_modeled_total += V2G_discharge
            V2G_battery_bank = max(V2G_battery_bank - V2G_discharge, 0)
            SC_Battery_bank = max(SC_Battery_bank - SC_discharge, 0)

            SC_hourly_battery_bank.append(SC_Battery_bank / total_charge_SC_Fleet * 100)
            hourly_deficits.append(deficit)

            v2g_tendered_this_hour = 0.0
            v2g_max_dispatch_this_hour = float(total_charge_V2G_Fleet * V2G_DR)

            if deficit > 12800 and V2G_battery_bank > 0.45 * total_charge_V2G_Fleet:
                V2G_Charge = np.min([
                    total_charge_V2G_Fleet * V2G_DR,
                    deficit - 12800,
                    V2G_battery_bank - (0.35 * total_charge_V2G_Fleet),
                ])
                V2G_Charge = max(V2G_Charge, 0)
                V2G_battery_bank -= V2G_Charge
                deficit -= V2G_Charge
                total_v2g_tendered += V2G_Charge
                v2g_tendered_this_hour = float(V2G_Charge)

            V2G_hourly_battery_bank.append(V2G_battery_bank / total_charge_V2G_Fleet * 100)

            v2g_dispatch_violation = max(
                v2g_tendered_this_hour - v2g_max_dispatch_this_hour,
                0.0,
            )

            if v2g_dispatch_violation > 1e-6:
                hours_v2g_dispatch_over_limit += 1
                max_v2g_dispatch_violation = max(
                    max_v2g_dispatch_violation,
                    v2g_dispatch_violation,
                )

            v2g_dispatch_limit_audit.append({
                "hour_index": int(hour),
                "v2g_tendered_this_hour_mwh": float(v2g_tendered_this_hour),
                "v2g_max_dispatch_this_hour_mwh": float(v2g_max_dispatch_this_hour),
                "v2g_dispatch_violation_mwh": float(v2g_dispatch_violation),
                "v2g_soc_percent_after_dispatch": float(
                    V2G_battery_bank / total_charge_V2G_Fleet * 100
                ),
            })

            if deficit > 12800 and Grid_Store > 0:
                grid_charge = np.min([Grid_Store_Max * 0.25, deficit - 12800, Grid_Store])
                grid_charge = max(grid_charge, 0)
                Grid_Store -= grid_charge
                deficit -= grid_charge
                total_GS_used += grid_charge
                grid_tendered_this_hour = float(grid_charge)

            if V2G_battery_bank / total_charge_V2G_Fleet < SC_Battery_bank / total_charge_SC_Fleet:
                if deficit < 12800:
                    charge_amount = min(12800 - deficit,
                                        (V2G_Cap * total_charge_V2G_Fleet) - V2G_battery_bank,
                                        V2G_CR * total_charge_V2G_Fleet)
                    V2G_battery_bank = min(V2G_battery_bank + charge_amount, V2G_Cap * total_charge_V2G_Fleet)
                    V2G_ChargingEnergy += charge_amount
                    deficit += charge_amount

                if deficit < 12800:
                    charge_amount = min(12800 - deficit,
                                        (SC_Cap * total_charge_SC_Fleet) - SC_Battery_bank,
                                        SC_CR * total_charge_SC_Fleet)
                    SC_Battery_bank = min(SC_Battery_bank + charge_amount, SC_Cap * total_charge_SC_Fleet)
                    deficit += charge_amount
                    SC_ChargingEnergy += charge_amount

            if SC_Battery_bank / total_charge_SC_Fleet < V2G_battery_bank / total_charge_V2G_Fleet:
                if deficit < 12800:
                    charge_amount = min(12800 - deficit,
                                        (SC_Cap * total_charge_SC_Fleet) - SC_Battery_bank,
                                        SC_CR * total_charge_SC_Fleet)
                    SC_Battery_bank = min(SC_Battery_bank + charge_amount, SC_Cap * total_charge_SC_Fleet)
                    deficit += charge_amount
                    SC_ChargingEnergy += charge_amount

                if deficit < 12800:
                    charge_amount = min(12800 - deficit,
                                        (V2G_Cap * total_charge_V2G_Fleet) - V2G_battery_bank,
                                        V2G_CR * total_charge_V2G_Fleet)
                    V2G_battery_bank = min(V2G_battery_bank + charge_amount, V2G_Cap * total_charge_V2G_Fleet)
                    V2G_ChargingEnergy += charge_amount
                    deficit += charge_amount

            # Append post-charge battery percentages
            SC_hourly_battery_bank.append(SC_Battery_bank / total_charge_SC_Fleet * 100)
            V2G_hourly_battery_bank.append(V2G_battery_bank / total_charge_V2G_Fleet * 100)

            if deficit < 12800:
                grid_charge = min(12800 - deficit, Grid_Store_Max - Grid_Store, 0.25 * Grid_Store_Max)
                Grid_Store += grid_charge
                deficit += grid_charge

            if deficit < 0:
                total_curtailment += -deficit

            if hour in top_deficit_indices:
                top_deficit_audit.append({
                    "hour_index": int(hour),
                    "pre_dispatch_deficit_mw": deficit_start,
                    "excess_above_import_limit_mw": max(deficit_start - Import_Limit, 0),
                    "v2g_tendered_mwh": v2g_tendered_this_hour,
                    "grid_storage_tendered_mwh": grid_tendered_this_hour,
                    "post_dispatch_deficit_mw": float(deficit),
                    "v2g_soc_percent_after": float(
                        V2G_battery_bank / total_charge_V2G_Fleet * 100
                    ),
                    "grid_store_after_mwh": float(Grid_Store),
                })

            curtailments.append(total_curtailment)
            total_deficit += max(deficit, 0)
            max_def = max(deficit, 0)
            if max_def > max_def_max:
                max_def_max = max_def

        all_deficits.append(total_deficit)

        # Constraint checks
        if total_deficit > total_def_limit:
            failures.append(f"Excess Deficit ({total_deficit:.2f})")
        if max_def_max > Import_Limit:
            failures.append(f"Excess Single Hour Deficit ({max_def_max:.2f})")
        if np.min(SC_hourly_battery_bank) < SC_Floor:
            failures.append("Low SC Battery Bank")
        if np.min(V2G_hourly_battery_bank) < V2G_Floor:
            failures.append("Low V2G Battery Bank")

        simulation_results.append({
            "failed": bool(failures),
            "failure_reason": ", ".join(failures),

            "total_energy_deficit": total_deficit,
            "max_hourly_deficit": max_def_max,
            "import_limit": Import_Limit,
            "annual_import_limit": total_def_limit,

            "passed_hourly_import_limit": max_def_max <= Import_Limit,
            "passed_annual_import_limit": total_deficit <= total_def_limit,

            "total_curtailment": total_curtailment,
            "total_GS_used": total_GS_used,
            "total_V2G_used": total_v2g_tendered,
            "final_grid_store": Grid_Store,
            "grid_store_start": Grid_Store_Max * 0.6,
            "grid_store_capacity": Grid_Store_Max,
        })

    pass_count = sum(1 for result in simulation_results if not result['failed'])
    total_runs = len(simulation_results)

    for idx, sim in enumerate(simulation_results, start=1):
        print(
            f"Run {idx}:",
            "FAILED" if sim["failed"] else "PASSED",
            sim["failure_reason"],
            "max_def", sim["max_hourly_deficit"],
            "annual_import", sim["total_energy_deficit"],
            "GS_used", sim["total_GS_used"],
            "V2G_used", sim["total_V2G_used"],
            "final_GS", sim["final_grid_store"],
        )

    if not battery_wear_calculated:
        temp_data = pd.read_csv("/Users/g0g/PycharmProjects/Grid_Optimizer/Grid_Builder/CAL_Temps_Minutely.csv")['Temperature'].values
        V2G_minutely_bank = interpolate_soc(np.array(V2G_hourly_battery_bank))
        SC_minutely_bank = interpolate_soc(np.array(SC_hourly_battery_bank))
        final_q_V2G = calculate_battery_wear(V2G_minutely_bank, temp_data)
        final_q_SC = calculate_battery_wear(SC_minutely_bank, temp_data)
        SC_years_to_failure = np.log(0.8) / np.log(final_q_SC)
        SC_extra_cost_per_vehicle_year = (
                EV_Battery_Cost / SC_years_to_failure
                - EV_Battery_Cost / Baseline_years_to_replace
        )

        SC_vehicle_count = (
                (Factor_Smart_Charge - Factor_V2G)
                * Full_EV_Fleet
                * 1000
        )

        SC_Fleet_Annual_Cost = SC_extra_cost_per_vehicle_year * SC_vehicle_count

        V2G_years_to_replace = np.log(0.8) / np.log(final_q_V2G)
        V2G_extra_cost_per_vehicle_year = (
                EV_Battery_Cost / V2G_years_to_replace
                - EV_Battery_Cost / Baseline_years_to_replace
        )

        V2G_vehicle_count = Factor_V2G * Full_EV_Fleet * Share_EV_init * 1000
        V2G_Fleet_Cost = V2G_extra_cost_per_vehicle_year * V2G_vehicle_count

        SC_extra_battery_carbon_cost = (
                annual_extra_battery_carbon_cost(SC_extra_cost_per_vehicle_year)
                * SC_vehicle_count
        )

        V2G_extra_battery_carbon_cost = (
                annual_extra_battery_carbon_cost(V2G_extra_cost_per_vehicle_year)
                * V2G_vehicle_count
        )

        total_extra_battery_wear_carbon_cost = (
                SC_extra_battery_carbon_cost
                + V2G_extra_battery_carbon_cost
        )

        # Needed for V2G LCOE Calculation
        Number_V2G_Connections = Factor_V2G * Share_EV_init * Full_EV_Fleet

        demand_total = base_demand_sum + NSC_EV_Demand_sum + SCV2G_EV_Demand_sum
        total_system_power = demand_total - total_deficit

        dynamic_values = {
            "solar": {
                "installed_capacity_mw": Solar_Installed,
                "annual_energy_generated_mwh": solar_gen_sum,
                "annual_energy_used_mwh": solar_gen_sum,
            },
            "wind": {
                "installed_capacity_mw": Wind_Installed * 3200,
                "annual_energy_generated_mwh": wind_gen_sum,
                "annual_energy_used_mwh": wind_gen_sum,
            },
            "hydro": {
                "installed_capacity_mw": Hydro_Installed,
                "annual_energy_generated_mwh": hydro_gen_sum,
                "annual_energy_used_mwh": hydro_gen_sum,
            },
            "nuclear": {
                "installed_capacity_mw": BL_installed,
                "annual_energy_generated_mwh": nuclear_gen_sum,
                "annual_energy_used_mwh": nuclear_gen_sum,
            },
            "natural_gas": {
                "installed_capacity_mw": NG_Installed,
                "annual_energy_generated_mwh": natural_gas_gen_sum,
                "annual_energy_used_mwh": natural_gas_gen_sum,
            },
            "battery": {
                "installed_capacity_mw": Grid_Store_Max / 4,
                "annual_energy_generated_mwh": total_GS_used,
                "annual_energy_used_mwh": total_GS_used,
            },
            "V2G": {
                "installed_capacity_mw": Number_V2G_Connections * 40 * V2G_CR,
                "annual_energy_generated_mwh": total_v2g_tendered,
                "annual_energy_used_mwh": total_v2g_tendered,
            },
        }

        print(f"Factor V2G (INSIDE LCOE) {Factor_V2G}")
        lcoe_values = {
            source: (
                calculate_lcoe(
                    params,
                    dynamic_values[source]['installed_capacity_mw'],
                    dynamic_values[source]['annual_energy_used_mwh'],
                    source_name=source,
                    V2G_CR=V2G_CR,
                    num_connections= Number_V2G_Connections * 1000,
                    V2G_Connect_Cost=V2G_Connect_Cost,

                )
                if source == "V2G"
                else calculate_lcoe(
                    params,
                    dynamic_values[source]['installed_capacity_mw'],
                    dynamic_values[source]['annual_energy_used_mwh'],

                )
            )
            for source, params in energy_sources.items()
            if source in dynamic_values
        }


    def calculate_combined_lcoe(lcoe_values, dynamic_values, total_system_power):
        weighted_costs = sum(
            (lcoe_values[source] if not math.isinf(lcoe_values[source]) else 0) *
            dynamic_values[source]['annual_energy_used_mwh']
            for source in lcoe_values
        )

        combined_lcoe = (weighted_costs + SC_Fleet_Annual_Cost + V2G_Fleet_Cost) / total_system_power
        return combined_lcoe


    print(f"MASTER DEBUG \n"
          f"Factor V2G {Factor_V2G} "
          f"SC_CR {SC_CR} "
          f"V2G_CR {V2G_CR} "
          f"V2G_DR {V2G_DR} "
          f"GS_Store {Grid_Store_Max}")

    combined_lcoe = calculate_combined_lcoe(lcoe_values, dynamic_values, total_system_power)

    carbon_costs, total_carbon_cost = calculate_carbon_costs(dynamic_values)

    total_carbon_cost_with_battery_wear = (
            total_carbon_cost
            + total_extra_battery_wear_carbon_cost
    )

    carbon_cost_per_mwh = total_carbon_cost_with_battery_wear / total_system_power

    extra_battery_wear_carbon_cost_per_mwh = (
            total_extra_battery_wear_carbon_cost / total_system_power
    )

    combined_lcoe_carbon_adjusted = combined_lcoe + carbon_cost_per_mwh

    status = "success" if pass_count >= 8 else "failure"

    lcoe_audit = {}

    for source in lcoe_values:
        source_energy_mwh = dynamic_values[source]["annual_energy_used_mwh"]
        source_lcoe = lcoe_values[source]

        source_weighted_cost = (
            0 if math.isinf(source_lcoe)
            else source_lcoe * source_energy_mwh
        )

        lcoe_audit[source] = {
            "installed_capacity_mw": dynamic_values[source]["installed_capacity_mw"],
            "annual_energy_generated_mwh": dynamic_values[source]["annual_energy_generated_mwh"],
            "annual_energy_used_mwh": source_energy_mwh,
            "source_lcoe_per_mwh": source_lcoe,
            "source_weighted_annual_cost": source_weighted_cost,
            "source_contribution_to_system_lcoe_per_mwh": source_weighted_cost / total_system_power,
            "carbon_cost_total": carbon_costs.get(source, 0.0),
            "carbon_cost_per_system_mwh": carbon_costs.get(source, 0.0) / total_system_power,
        }

    max_hourly_deficit_across_runs = max(
        r["max_hourly_deficit"] for r in simulation_results
    )

    max_annual_import_across_runs = max(
        r["total_energy_deficit"] for r in simulation_results
    )

    hourly_import_pass_all_runs = all(
        r["max_hourly_deficit"] <= Import_Limit for r in simulation_results
    )

    annual_import_pass_all_runs = all(
        r["total_energy_deficit"] <= total_def_limit for r in simulation_results
    )

    failure_reasons = [
        r["failure_reason"] for r in simulation_results
    ]

    top_v2g_dispatch_hours = sorted(
        v2g_dispatch_limit_audit,
        key=lambda x: x["v2g_tendered_this_hour_mwh"],
        reverse=True,
    )[:10]

    audit_report = {
        "inputs": {
            "BL_installed": BL_installed,
            "Factor_V2G": Factor_V2G,
            "Factor_Smart_Charge": Factor_Smart_Charge,
            "Grid_Store_Max": Grid_Store_Max,
            "Wind_Installed_model_units": Wind_Installed,
            "Wind_Installed_mw": Wind_Installed * 3200,
            "Solar_Installed_mw": Solar_Installed,
            "SC_CR": SC_CR,
            "V2G_CR": V2G_CR,
            "Cap": Cap,
            "V2G_Connect_Cost": V2G_Connect_Cost,
            "EV_Battery_Cost": EV_Battery_Cost,
            "Baseline_years_to_replace": Baseline_years_to_replace,
            "SOCIAL_COST_CARBON_PER_TONNE": SOCIAL_COST_CARBON_PER_TONNE,
        },
        "system_totals": {
            "demand_total_mwh": demand_total,
            "total_system_power_mwh": total_system_power,
            "total_energy_deficit_mwh": total_deficit,
            "total_curtailment_mwh": total_curtailment,
            "total_grid_storage_used_mwh": total_GS_used,
            "total_v2g_tendered_mwh": total_v2g_tendered,
        },
        "vehicle_counts": {
            "SC_vehicle_count": SC_vehicle_count,
            "V2G_vehicle_count": V2G_vehicle_count,
            "Number_V2G_Connections_millions": Number_V2G_Connections,
            "Number_V2G_Connections_actual": Number_V2G_Connections * 1000,
        },
        "battery_wear": {
            "final_q_SC": final_q_SC,
            "final_q_V2G": final_q_V2G,
            "SC_years_to_failure": SC_years_to_failure,
            "V2G_years_to_replace": V2G_years_to_replace,
            "Baseline_years_to_replace": Baseline_years_to_replace,
            "SC_extra_cost_per_vehicle_year": SC_extra_cost_per_vehicle_year,
            "V2G_extra_cost_per_vehicle_year": V2G_extra_cost_per_vehicle_year,
            "SC_Fleet_Annual_Cost": SC_Fleet_Annual_Cost,
            "V2G_Fleet_Cost": V2G_Fleet_Cost,
            "SC_Fleet_Cost_per_system_mwh": SC_Fleet_Annual_Cost / total_system_power,
            "V2G_Fleet_Cost_per_system_mwh": V2G_Fleet_Cost / total_system_power,
            "SC_extra_battery_carbon_cost": SC_extra_battery_carbon_cost,
            "V2G_extra_battery_carbon_cost": V2G_extra_battery_carbon_cost,
            "total_extra_battery_wear_carbon_cost": total_extra_battery_wear_carbon_cost,
            "extra_battery_wear_carbon_cost_per_mwh": extra_battery_wear_carbon_cost_per_mwh,
        },
        "lcoe_breakdown": lcoe_audit,
        "totals": {
            "combined_LCOE_raw": combined_lcoe,
            "total_carbon_cost": total_carbon_cost,
            "total_carbon_cost_with_battery_wear": total_carbon_cost_with_battery_wear,
            "carbon_cost_per_mwh": carbon_cost_per_mwh,
            "combined_LCOE_carbon_adjusted": combined_lcoe_carbon_adjusted,
        },
        "reliability_summary": {
            "pass_count": pass_count,
            "total_runs": total_runs,

            "max_hourly_deficit_across_runs": max_hourly_deficit_across_runs,
            "import_limit": Import_Limit,
            "hourly_import_pass_all_runs": hourly_import_pass_all_runs,

            "max_annual_import_across_runs": max_annual_import_across_runs,
            "annual_import_limit": total_def_limit,
            "annual_import_pass_all_runs": annual_import_pass_all_runs,

            "failure_reasons": failure_reasons,
        },
        "top_deficit_dispatch_audit": sorted(
            top_deficit_audit,
            key=lambda x: x["pre_dispatch_deficit_mw"],
            reverse=True,
        ),
        "v2g_dispatch_limit_summary": {
            "total_charge_V2G_Fleet_mwh": total_charge_V2G_Fleet,
            "V2G_DR": V2G_DR,
            "expected_max_v2g_dispatch_per_hour_mwh": total_charge_V2G_Fleet * V2G_DR,
            "max_actual_v2g_dispatch_hour_mwh": max(
                x["v2g_tendered_this_hour_mwh"] for x in v2g_dispatch_limit_audit
            ),
            "hours_v2g_dispatch_over_limit": hours_v2g_dispatch_over_limit,
            "max_v2g_dispatch_violation_mwh": max_v2g_dispatch_violation,
            "top_10_v2g_dispatch_hours": top_v2g_dispatch_hours,
        },
        "grid_storage_audit": {
            "Grid_Store_Max": Grid_Store_Max,
            "grid_store_start_mwh": Grid_Store_Max * 0.6,
            "grid_store_final_mwh": Grid_Store,
            "total_GS_used_mwh": total_GS_used,
            "grid_storage_unused": total_GS_used <= 1e-6,
            "battery_installed_capacity_mw": Grid_Store_Max / 4,
            "battery_lcoe": lcoe_values.get("battery"),
            "battery_weighted_cost": (
                lcoe_values.get("battery", 0) * total_GS_used
                if not math.isinf(lcoe_values.get("battery", float("inf")))
                else 0
            ),
        },
    }

    return {
        "status": status,
        "details": simulation_results,
        "pass_count": pass_count,
        "total_runs": total_runs,
        "battery_wear": {"final_q_V2G": final_q_V2G, "final_q_SC": final_q_SC},
        "lcoe_values": lcoe_values,
        'power_deficit': power_deficit,
        'total_energy_deficit': np.mean(all_deficits),
        'final_battery_bank': V2G_battery_bank,
        'total_charge': total_charge_SC_Fleet,
        'SOC_V2G': V2G_hourly_battery_bank,
        'SOC_SC': SC_hourly_battery_bank,
        'total_curtailment': total_curtailment,
        'total_V2G': total_v2g_tendered,
        'V2G_per_veh': total_v2g_tendered / (Full_EV_Fleet * Factor_Smart_Charge),
        'wind_power_sum': wind_gen_sum,
        'solar_power_sum': solar_gen_sum,
        'combined_LCOE': combined_lcoe,
        "carbon_costs": carbon_costs,
        "total_carbon_cost": total_carbon_cost,
        "carbon_cost_per_mwh": carbon_cost_per_mwh,
        "combined_LCOE_carbon_adjusted": combined_lcoe_carbon_adjusted,
        'total_GS_used': total_GS_used,
        "total_carbon_cost_with_battery_wear": total_carbon_cost_with_battery_wear,
        "extra_battery_wear_carbon_cost": total_extra_battery_wear_carbon_cost,
        "extra_battery_wear_carbon_cost_per_mwh": extra_battery_wear_carbon_cost_per_mwh,
        "SC_extra_battery_carbon_cost": SC_extra_battery_carbon_cost,
        "V2G_extra_battery_carbon_cost": V2G_extra_battery_carbon_cost,
        "audit_report": audit_report,
    }

# Load the data
file_path = "Small_Sample_6.csv"
data = pd.read_csv(file_path)

failed_constraints = []
unique_scenarios = set()
tested_combinations = []

passed_10_of_10 = []

# Global variables for simulation
BL_installed = None
Factor_V2G = None
Grid_Store_Max = None
Wind_Installed = None
Solar_Installed = None
CR = None
Cap = None


def generate_random_combination(data: pd.DataFrame):
    """
    Randomly sample one row and drop it so it isn't re-selected.
    """
    sample = data.sample(n=1)
    combination = sample.iloc[0].tolist()
    data.drop(sample.index, inplace=True)
    return combination


def count_passes(results: dict) -> tuple[int, int]:
    """
    Count how many Monte Carlo runs passed.

    Assumes results["details"] is a list of run dictionaries and each run has
    a boolean key 'failed'. This matches your earlier failure-reason logic.

    Returns:
        (pass_count, total_runs)
    """
    details = results.get("details", [])

    if not details:
        # Fallback:
        # If no details exist, interpret status='success' as 10/10 and failure as 0/10.
        if results.get("status") == "success":
            return 10, 10
        return 0, 10

    total_runs = len(details)
    pass_count = sum(1 for sim in details if not sim.get("failed", False))
    return pass_count, total_runs


def evaluate_combination(combination):
    global data, unique_scenarios
    global BL_installed, Factor_V2G, Grid_Store_Max, Wind_Installed
    global Solar_Installed, CR, Cap, V2G_Connect_Cost

    # If we've already tested this exact tuple, skip it
    tup = tuple(combination[:8])
    if tup in unique_scenarios:
        return None
    unique_scenarios.add(tup)

    # Unpack the first 8 values into the globals that do_parallel_monte_carlo expects
    (
        BL_installed,
        Factor_V2G,
        Grid_Store_Max,
        Wind_Installed,
        Solar_Installed,
        CR,
        Cap,
        V2G_Connect_Cost,
    ) = combination[:8]


    # Call your existing Monte Carlo function
    results = do_parallel_monte_carlo(
        Share_EV_init=1,
        evcharge=0,
        SC_CR=CR,
        V2G_CR=CR,
        V2G_DR=CR,
        Cap=Cap,
        V2G_Connect_Cost=V2G_Connect_Cost,
    )

    print(f"CR is {CR}")

    # Count passes from the individual simulations
    details = results.get("details", [])
    if details:
        total_runs = len(details)
        pass_count = sum(1 for sim in details if not sim.get("failed", False))
    else:
        # fallback if details missing
        total_runs = 10
        pass_count = 10 if results.get("status") == "success" else 0

    print(f"Pass count: {pass_count}/{total_runs}")

    decision_cols = data.columns[:8]

    # Only prune on scenarios that fail the new reliability threshold
    if pass_count < 8:
        print(f"Combination failed reliability threshold (<8/{total_runs}): {combination}")

        data_before = data.shape[0]
        to_drop = data.index[(data[decision_cols] <= combination[:8]).all(axis=1)]
        data.drop(to_drop, inplace=True)
        rows_removed = data_before - data.shape[0]

        if rows_removed > 0:
            print(f"  → Pruned {rows_removed} rows after reliability failure.")
        else:
            print("  → No rows pruned on reliability failure.")

        failed_constraints.append(combination + [pass_count, total_runs])
        print(f"Remaining rows in pruned sample space: {len(data)}")
        return None

    # Preserve your original variable name
    combined_lcoe = results["combined_LCOE"]
    carbon_adjusted_lcoe = results["combined_LCOE_carbon_adjusted"]
    carbon_cost_per_mwh = results["carbon_cost_per_mwh"]
    total_carbon_cost = results["total_carbon_cost"]
    carbon_costs_json = json.dumps(results["carbon_costs"])

    record = combination + [combined_lcoe, carbon_adjusted_lcoe, carbon_cost_per_mwh, total_carbon_cost, carbon_costs_json, pass_count, total_runs]

    if pass_count == total_runs:
        passed_10_of_10.append(record)
        tested_combinations.append(record)
        print(f"Combination passed at {pass_count}/{total_runs} with LCOE: {combined_lcoe:.4f}")
    else:
        print(
            f"Combination met pruning threshold but was not kept: "
            f"{pass_count}/{total_runs}"
        )
    print(f"CARBON ADJUSTED lCOE IS {carbon_adjusted_lcoe:.4f}")

    # Pruning of dominated expensive scenarios
    if carbon_adjusted_lcoe > 110:
        data_before = data.shape[0]
        to_drop = data.index[(data[decision_cols] >= combination[:8]).all(axis=1)]
        data.drop(to_drop, inplace=True)
        rows_removed = data_before - data.shape[0]

        if rows_removed > 0:
            print(f"  → Pruned {rows_removed} dominated rows due to high carbon-adjusted LCOE (>110).")
        else:
            print("  → No dominated rows pruned for this high-LCOE combination.")

    print(f"Remaining rows in pruned sample space: {len(data)}")
    audit_report = results.get("audit_report")
    return {
        "raw_lcoe": combined_lcoe,
        "carbon_adjusted_lcoe": carbon_adjusted_lcoe,
        "pass_count": pass_count,
        "audit_report": audit_report,
    }

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_EVERY = 5000

def atomic_csv_write(df: pd.DataFrame, path: Path):
    """
    Write CSV safely: write temp file first, then replace.
    Prevents corrupted checkpoint files if Python crashes during write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def save_checkpoint(
    data: pd.DataFrame,
    iteration: int,
    best_raw_combination=None,
    best_raw_lcoe: float | None = None,
    best_raw_pass_count: int | None = None,
    best_carbon_combination=None,
    best_carbon_adjusted_lcoe: float | None = None,
    best_carbon_pass_count: int | None = None,
):
    """
    Save remaining sample space and all accumulated outputs.
    """
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print(f"\n--- Saving checkpoint at iteration {iteration} ---")

    # Remaining unexplored/pruned sample space
    data.reset_index(drop=True, inplace=True)
    atomic_csv_write(
        data,
        CHECKPOINT_DIR / f"remaining_sample_space_iter_{iteration}.csv"
    )

    # Also keep an overwrite-style latest checkpoint
    atomic_csv_write(
        data,
        CHECKPOINT_DIR / "remaining_sample_space_latest.csv"
    )

    result_cols = data.columns.to_list() + [
        "Combined_LCOE",
        "Carbon_Adjusted_LCOE",
        "Carbon_Cost_Per_MWh",
        "Total_Carbon_Cost",
        "Carbon_Costs_JSON",
        "Pass_Count",
        "Total_Runs",
    ]

    failed_cols = data.columns.to_list() + ["Pass_Count", "Total_Runs"]

    if tested_combinations:
        atomic_csv_write(
            pd.DataFrame(tested_combinations, columns=result_cols),
            CHECKPOINT_DIR / "all_passing_combos_8plus_latest.csv"
        )

    if passed_10_of_10:
        atomic_csv_write(
            pd.DataFrame(passed_10_of_10, columns=result_cols),
            CHECKPOINT_DIR / "passing_combos_10_of_10_latest.csv"
        )

    if failed_constraints:
        atomic_csv_write(
            pd.DataFrame(failed_constraints, columns=failed_cols),
            CHECKPOINT_DIR / "failed_reliability_combos_latest.csv"
        )

    metadata = {
        "iteration": iteration,
        "remaining_rows": int(data.shape[0]),
        "tested_passing_count": len(tested_combinations),
        "failed_reliability_count": len(failed_constraints),
        "passed_10_of_10_count": len(passed_10_of_10),

        "best_raw_combination": best_raw_combination,
        "best_raw_lcoe": best_raw_lcoe,
        "best_raw_pass_count": best_raw_pass_count,

        "best_carbon_combination": best_carbon_combination,
        "best_carbon_adjusted_lcoe": best_carbon_adjusted_lcoe,
        "best_carbon_pass_count": best_carbon_pass_count,
    }

    with open(CHECKPOINT_DIR / "checkpoint_metadata_latest.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"--- Checkpoint saved. Remaining rows: {data.shape[0]} ---\n")


def save_results(data: pd.DataFrame):
    """
    Save 10/10 passing scenarios, failed reliability cases, and remaining search space.
    """

    result_cols = data.columns.to_list() + [
        "Combined_LCOE",
        "Carbon_Adjusted_LCOE",
        "Carbon_Cost_Per_MWh",
        "Total_Carbon_Cost",
        "Carbon_Costs_JSON",
        "Pass_Count",
        "Total_Runs",
    ]

    if passed_10_of_10:
        pd.DataFrame(passed_10_of_10, columns=result_cols).to_csv(
            "passing_combos_10_of_10.csv",
            index=False,
        )
        print("Saved 10/10 scenarios to 'passing_combos_10_of_10.csv'")
    else:
        print("No 10/10 scenarios found.")

    if failed_constraints:
        failed_cols = data.columns.to_list() + ["Pass_Count", "Total_Runs"]
        pd.DataFrame(failed_constraints, columns=failed_cols).to_csv(
            "failed_reliability_combos.csv",
            index=False,
        )
        print("Saved failed reliability scenarios to 'failed_reliability_combos.csv'")

    data.reset_index(drop=True, inplace=True)
    data.to_csv("EvenLower_Leftovers.csv", index=False)
    print("Saved remaining search space to 'EvenLower_Leftovers.csv'")


def random_search_from_csv(data: pd.DataFrame, max_iterations: int = 30000):
    """
    Sequential random search:
      - Sample one combination
      - Evaluate it
      - Prune failed reliability cases
      - Keep only 10/10 passing scenarios
      - Track best raw LCOE and best carbon-adjusted LCOE separately
      - Checkpoint every CHECKPOINT_EVERY iterations
    """
    print("Starting sequential random search with dual objectives...")

    first_audit_saved = False
    last_audit_report = None
    last_audit_iteration = None

    best_raw_combination = None
    best_raw_lcoe = float("inf")
    best_raw_pass_count = -1

    best_carbon_combination = None
    best_carbon_adjusted_lcoe = float("inf")
    best_carbon_pass_count = -1

    iteration = 0

    for iteration in range(1, max_iterations + 1):
        if data.empty:
            print("Sample space fully explored. Stopping search.")
            break

        print(f"\nRows remaining before iteration {iteration}: {len(data)}")

        combination = generate_random_combination(data)
        print(f"[Iter {iteration}] Evaluating combination: {combination}")

        result = evaluate_combination(combination)

        if result is not None and result.get("audit_report") is not None:
            last_audit_report = result["audit_report"]
            last_audit_iteration = iteration

            if not first_audit_saved:
                save_audit_report(
                    audit_report=result["audit_report"],
                    label="first_valid",
                    iteration=iteration,
                )
                first_audit_saved = True

        if result is not None:
            raw_lcoe = result["raw_lcoe"]
            carbon_adjusted_lcoe = result["carbon_adjusted_lcoe"]
            pass_count = result["pass_count"]

            # Only optimize among fully reliable 10/10 cases.
            if pass_count == 10:
                if raw_lcoe < best_raw_lcoe:
                    best_raw_lcoe = raw_lcoe
                    best_raw_pass_count = pass_count
                    best_raw_combination = combination.copy()

                    save_audit_report(
                        audit_report=result.get("audit_report"),
                        label="best_raw",
                        iteration=None,
                    )

                    print(
                        f"  → New best RAW solution: "
                        f"reliability={best_raw_pass_count}/10, "
                        f"Raw LCOE={best_raw_lcoe:.4f}, "
                        f"Carbon-adjusted LCOE={carbon_adjusted_lcoe:.4f}, "
                        f"combination={best_raw_combination}"
                    )

                if carbon_adjusted_lcoe < best_carbon_adjusted_lcoe:
                    best_carbon_adjusted_lcoe = carbon_adjusted_lcoe
                    best_carbon_pass_count = pass_count
                    best_carbon_combination = combination.copy()

                    save_audit_report(
                        audit_report=result.get("audit_report"),
                        label="best_carbon_adjusted",
                        iteration=None,
                    )

                    print(
                        f"  → New best CARBON-ADJUSTED solution: "
                        f"reliability={best_carbon_pass_count}/10, "
                        f"Carbon-adjusted LCOE={best_carbon_adjusted_lcoe:.4f}, "
                        f"Raw LCOE={raw_lcoe:.4f}, "
                        f"combination={best_carbon_combination}"
                    )

        if iteration % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                data=data,
                iteration=iteration,
                best_raw_combination=best_raw_combination,
                best_raw_lcoe=best_raw_lcoe if best_raw_lcoe != float("inf") else None,
                best_raw_pass_count=best_raw_pass_count,
                best_carbon_combination=best_carbon_combination,
                best_carbon_adjusted_lcoe=(
                    best_carbon_adjusted_lcoe
                    if best_carbon_adjusted_lcoe != float("inf")
                    else None
                ),
                best_carbon_pass_count=best_carbon_pass_count,
            )

    save_results(data)

    save_checkpoint(
        data=data,
        iteration=iteration,
        best_raw_combination=best_raw_combination,
        best_raw_lcoe=best_raw_lcoe if best_raw_lcoe != float("inf") else None,
        best_raw_pass_count=best_raw_pass_count,
        best_carbon_combination=best_carbon_combination,
        best_carbon_adjusted_lcoe=(
            best_carbon_adjusted_lcoe
            if best_carbon_adjusted_lcoe != float("inf")
            else None
        ),
        best_carbon_pass_count=best_carbon_pass_count,
    )

    print("\nFinal best solutions:")

    if best_raw_combination:
        print(
            f"Best RAW LCOE solution: {best_raw_combination} | "
            f"Reliability={best_raw_pass_count}/10 | "
            f"Raw LCOE={best_raw_lcoe:.4f}"
        )
    else:
        print("No valid 10/10 raw-LCOE solution found.")

    if best_carbon_combination:
        print(
            f"Best CARBON-ADJUSTED solution: {best_carbon_combination} | "
            f"Reliability={best_carbon_pass_count}/10 | "
            f"Carbon-adjusted LCOE={best_carbon_adjusted_lcoe:.4f}"
        )
    else:
        print("No valid 10/10 carbon-adjusted solution found.")

    if last_audit_report is not None:
        save_audit_report(
            audit_report=last_audit_report,
            label="last_valid",
            iteration=last_audit_iteration,
        )

    return {
        "best_raw_combination": best_raw_combination,
        "best_raw_lcoe": best_raw_lcoe,
        "best_raw_pass_count": best_raw_pass_count,
        "best_carbon_combination": best_carbon_combination,
        "best_carbon_adjusted_lcoe": best_carbon_adjusted_lcoe,
        "best_carbon_pass_count": best_carbon_pass_count,
    }


# Run the search
best_results = random_search_from_csv(
    data,
    max_iterations=100000
)
