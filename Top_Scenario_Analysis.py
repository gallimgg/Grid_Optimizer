import pandas as pd
import numpy as np
import pytz
from Constant_Factors import (
    NG_Tune, BL_Tune, Solar_Installed, Wind_Installed, Import_Limit, Hydro_Tune,
    Solar_Tune, Wind_Tune, total_def_limit, Factor_NSC,
    Factor_Smart_Charge, Full_EV_Fleet, NG_Installed, Hydro_Installed,
    energy_sources, calculate_lcoe
)
from Battery_Functions import calculate_battery_wear, interpolate_soc
import os
import math
from pathlib import Path
import json

NG_Decrementor = True
Nuke_Builder = True
Grid_Store_Builder = True
Factor_V2G = 0.35

Factor_NSC = 0.2
Factor_Smart_Charge = 1 - Factor_NSC
NG_Fact = 0.0
NG_Installed = 0.0

Grid_Store_Max = 5000

Baseline_years_to_replace = 12.787
Annual_Miles = 11130
EV_Battery_Cost = 10000
Baseload_Installed = 5149
V2G_Connect_Cost = 2000

# Main simulation loop
installed_solar = [Solar_Installed]
installed_wind = [Wind_Installed * 3200]
output_yearly = []
output_hourly = []
failed_constraints = []

NG_Fact = 0

# Define the folder name
folder_name = f"Output_Files_Merge/V2G_{Factor_V2G}_SC_{Factor_Smart_Charge}_NatGasDec_{NG_Decrementor}_NukeBuild_{Nuke_Builder}"
os.makedirs(folder_name, exist_ok=True)

# Load data
csv_file_path = 'Grid_Input_Data.csv'
df = pd.read_csv(csv_file_path)
df['Date'] = pd.date_range(start='2021-01-01 00:00', end='2021-12-30 23:00', freq='h')

# Step 1: Localize the Date column to UTC and convert to Pacific Time
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
pacific_zone = pytz.timezone('US/Pacific')
df['Date_Pacific'] = df['Date'].dt.tz_convert(pacific_zone)

# Add constant columns
df['Nuke_AVG'] = 0.9
df['Nuke_STDV'] = 0.03

SOCIAL_COST_CARBON_CASES = {
    "SCC_51": 51.0,
    "SCC_120": 120.0,
    "SCC_190": 190.0,
    "SCC_340": 340.0,
}

CARBON_INTENSITY = {
    "solar": {"fixed": 33.0, "variable": 10.0},
    "hydro": {"fixed": 6.204, "variable": 1.9},
    "wind": {"fixed": 12.34, "variable": 0.74},
    "battery": {"fixed": 0, "variable": 0.0},
    "V2G": {"fixed": 35.4, "variable": 0.0},
    "nuclear": {"fixed": 2.7, "variable": 12.0},
    "natural_gas": {"fixed": 0.82, "variable": 460.0},
}

Annual_Miles = 11130
VEHICLE_COUNT_MULTIPLIER = 1000  # keep if Full_EV_Fleet is in thousands

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

def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def annual_extra_battery_carbon_cost(
    extra_battery_cost_per_vehicle_year: float,
    battery_lca_gco2_per_mile_2050: float = EV_LCA_GCO2_PER_MILE["battery"][2050],
    ev_battery_cost: float = EV_Battery_Cost,
    annual_miles_per_vehicle: float = 12000.0,
    social_cost_per_tonne: float = None,
) -> float:
    """
    Convert incremental annual battery wear cost into an equivalent additional
    battery-manufacturing carbon cost.

    We scale the 2050 battery LCA gCO2/mile by the fraction of one battery
    consumed per year due to accelerated replacement.
    """

    if social_cost_per_tonne is None:
        social_cost_per_tonne = 190.0

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
def interpolate_by_year(year: int, values_by_year: dict[int, float]) -> float:
    years = sorted(values_by_year)

    if year <= years[0]:
        return values_by_year[years[0]]

    if year >= years[-1]:
        return values_by_year[years[-1]]

    for y0, y1 in zip(years[:-1], years[1:]):
        if y0 <= year <= y1:
            v0 = values_by_year[y0]
            v1 = values_by_year[y1]
            return v0 + (v1 - v0) * (year - y0) / (y1 - y0)

    raise ValueError(f"Could not interpolate value for year {year}")


def get_lca_value(
    values_by_year: dict[int, float],
    year: int,
    method: str,
) -> float:
    if method == "constant_2025":
        return values_by_year[2025]

    if method == "interpolated":
        return interpolate_by_year(year, values_by_year)

    raise ValueError(f"Unknown LCA method: {method}")


def annual_vehicle_lifecycle_co2_tonnes(
    vehicle_count: float,
    annual_miles: float,
    gco2_per_mile: float,
) -> float:
    return vehicle_count * annual_miles * gco2_per_mile / 1_000_000.0

def carbon_cost_from_intensity(
    generated_mwh: float,
    used_mwh: float,
    fixed_gco2_per_kwh: float,
    variable_gco2_per_kwh: float,
    social_cost_per_tonne: float,
) -> float:
    fixed_tonnes = generated_mwh * fixed_gco2_per_kwh / 1000.0
    variable_tonnes = used_mwh * variable_gco2_per_kwh / 1000.0
    return (fixed_tonnes + variable_tonnes) * social_cost_per_tonne

def calculate_carbon_costs(
    dynamic_values: dict,
    social_cost_per_tonne: float,
) -> tuple[dict, float]:
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
            social_cost_per_tonne=social_cost_per_tonne,
        )

    return carbon_costs, sum(carbon_costs.values())

def calculate_carbon_emissions_tonnes(dynamic_values: dict) -> tuple[dict, float]:
    carbon_tonnes = {}

    for source, values in dynamic_values.items():
        if source not in CARBON_INTENSITY:
            continue

        intensity = CARBON_INTENSITY[source]

        generated_mwh = values.get(
            "annual_energy_generated_mwh",
            values.get("annual_energy_used_mwh", 0.0),
        )
        used_mwh = values.get("annual_energy_used_mwh", 0.0)

        fixed_tonnes = generated_mwh * intensity["fixed"] / 1000.0
        variable_tonnes = used_mwh * intensity["variable"] / 1000.0

        carbon_tonnes[source] = fixed_tonnes + variable_tonnes

    return carbon_tonnes, sum(carbon_tonnes.values())

def do_parallel_monte_carlo(
    Share_EV_init,
    evcharge,
    year,
    n_simulations=1,
    min_pass_count=1,
    SC_CR=0.0475,
    V2G_CR=0.0475,
    V2G_DR=0.0475,
    SC_Floor=30,
    V2G_Floor=30,
    Cap=0.85,
    V2G_Connect_Cost=V2G_Connect_Cost,
):
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

        total_ev_demand_sum = np.sum(total_ev_demand)

        # Split EV demand into NSC and SC/V2G portions
        NSC_EV_Demand = total_ev_demand * Factor_NSC
        SCV2G_EV_Demand = total_ev_demand * (Factor_Smart_Charge)

        NSC_EV_Demand_sum = np.sum(NSC_EV_Demand)
        SCV2G_EV_Demand_sum = np.sum(SCV2G_EV_Demand)

        # Only NSC is added directly to hourly grid demand here
        demand += NSC_EV_Demand

        demand_sum = base_demand_sum + NSC_EV_Demand_sum

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
        nuclear_gen = np.random.normal(df['Nuke_AVG'], df['Nuke_STDV']) * Baseload_Installed * BL_Tune

        solar_gen_sum = np.sum(solar_gen)
        wind_gen_sum = np.sum(wind_gen)
        hydro_gen_sum = np.sum(hydro_gen)
        natural_gas_gen_sum = np.sum(natural_gas_gen)
        print(f"XXXXX natural gas gen sum is {natural_gas_gen_sum} NG fact is {NG_Fact}")
        nuclear_gen_sum = np.sum(nuclear_gen)
        #print(f"BL installed is {BL_installed}, Nuclear sum is {nuclear_gen_sum}")
        total_power_sum = solar_gen_sum + wind_gen_sum + hydro_gen_sum + nuclear_gen_sum + natural_gas_gen_sum

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
        dj = 0
        total_v2g_tendered = 0
        Grid_Store = Grid_Store_Max * 0.6
        total_GS_used = 1
        EV_power_modeled_total = 0
        SC_DISCH_Total = 0
        V2G_DISCH_Total = 0

        # Hourly loop for managing deficits and battery operations
        for hour, deficit in enumerate(power_deficit_sort):
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

            if hour == 0:
                v2g_debug = {
                    "hours_deficit_over_cap": 0,
                    "hours_soc_above_trigger": 0,
                    "hours_v2g_dispatched": 0,
                    "v2g_potential_mwh": 0.0,
                    "v2g_blocked_by_soc": 0,
                    "v2g_blocked_by_no_deficit": 0,
                }

            if deficit > 12800:
                v2g_debug["hours_deficit_over_cap"] += 1

                if V2G_battery_bank > 0.45 * total_charge_V2G_Fleet:
                    v2g_debug["hours_soc_above_trigger"] += 1
                    v2g_debug["v2g_potential_mwh"] += min(
                        total_charge_V2G_Fleet * V2G_DR,
                        deficit - 12800,
                        V2G_battery_bank - 0.35 * total_charge_V2G_Fleet,
                    )
                else:
                    v2g_debug["v2g_blocked_by_soc"] += 1
            else:
                v2g_debug["v2g_blocked_by_no_deficit"] += 1

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

                if V2G_Charge > 0:
                    v2g_debug["hours_v2g_dispatched"] += 1

            V2G_hourly_battery_bank.append(V2G_battery_bank / total_charge_V2G_Fleet * 100)

            if deficit > 12800 and Grid_Store > 0:
                grid_charge = np.min([Grid_Store_Max * 0.25, deficit - 12800, Grid_Store])
                grid_charge = max(grid_charge, 0)
                Grid_Store -= grid_charge
                deficit -= grid_charge
                total_GS_used += grid_charge

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

            curtailments.append(total_curtailment)
            total_deficit += max(deficit, 0)
            max_def = max(deficit, 0)
            if max_def > max_def_max:
                max_def_max = max_def

        print("V2G DEBUG:", v2g_debug)
        print("total_v2g_tendered:", total_v2g_tendered)
        print("final V2G SOC %:", V2G_battery_bank / total_charge_V2G_Fleet * 100)
        print("min V2G SOC %:", np.min(V2G_hourly_battery_bank))
        print("max deficit entering dispatch:", np.max(power_deficit_sort))
        print("hours entering dispatch above 12800:", np.sum(power_deficit_sort > 12800))
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
            'failed': bool(failures),
            'failure_reason': ", ".join(failures),
            'total_energy_deficit': total_deficit,
            'total_curtailment': total_curtailment,
            'SOC_V2G': V2G_hourly_battery_bank,
            'SOC_SC': SC_hourly_battery_bank,
            'power_deficit': power_deficit,
        })

    pass_count = sum(not result['failed'] for result in simulation_results)

    if pass_count < min_pass_count:
        return {
            "status": "failure",
            "details": simulation_results,
            "pass_count": pass_count,
            "min_pass_count": min_pass_count,
        }

    if not battery_wear_calculated:
        temp_data = pd.read_csv("CAL_Temps_Minutely.csv")[
            'Temperature'].values
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
                * Share_EV_init
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

        # Needed for V2G LCOE Calculation
        Number_V2G_Connections = Factor_V2G * Share_EV_init * Full_EV_Fleet

        demand_total = base_demand_sum + NSC_EV_Demand_sum + SCV2G_EV_Demand_sum
        print(f"DEMAND TOTAL {demand_total} NSC_EV_Demand_sum {NSC_EV_Demand_sum}, SCV2G {SCV2G_EV_Demand_sum}")
        total_system_power = demand_total - total_deficit
        print(f"XXXX TSP2 = {total_system_power}, total deficit is {total_deficit}")
        gross_generation_by_source = {
            "solar": solar_gen_sum,
            "wind": wind_gen_sum,
            "hydro": hydro_gen_sum,
            "nuclear": nuclear_gen_sum,
            "natural_gas": natural_gas_gen_sum,
        }

        gross_generation_total = sum(gross_generation_by_source.values())

        ev_demand_breakdown = {
            "total_ev_demand_mwh": total_ev_demand_sum,
            "nsc_ev_demand_mwh": float(np.sum(NSC_EV_Demand)),
            "sc_discharge_total_mwh": SC_DISCH_Total,
            "v2g_discharge_total_mwh": V2G_DISCH_Total,
            "sc_charging_energy_mwh": SC_ChargingEnergy,
            "v2g_charging_energy_mwh": V2G_ChargingEnergy,
        }

        storage_summary = {
            "grid_store_capacity_mwh": Grid_Store_Max,
            "grid_store_power_limit_mw": Grid_Store_Max * 0.25,
            "grid_store_start_mwh": Grid_Store_Max * 0.6,
            "grid_store_final_mwh": Grid_Store,
            "grid_store_used_mwh": total_GS_used,
            "grid_store_cycles_equivalent": (
                total_GS_used / Grid_Store_Max if Grid_Store_Max > 0 else 0.0
            ),
            "v2g_tendered_mwh": total_v2g_tendered,
            "curtailment_mwh": total_curtailment,
        }

        system_balance = {
            "base_demand_mwh": demand_sum,
            "total_ev_demand_mwh": total_ev_demand_sum,
            "demand_total_mwh": demand_total,
            "gross_generation_total_mwh": gross_generation_total,
            "gross_generation_by_source_mwh": gross_generation_by_source,
            "total_imports_or_unserved_proxy_mwh": total_deficit,
            "curtailment_mwh": total_curtailment,
            "gross_generation_minus_curtailment_mwh": gross_generation_total - total_curtailment,
            "served_energy_check_mwh": demand_total - total_deficit,
            "generation_to_demand_ratio": (
                gross_generation_total / demand_total if demand_total > 0 else None
            ),
            "curtailment_fraction_of_generation": (
                total_curtailment / gross_generation_total if gross_generation_total > 0 else None
            ),
            "import_fraction_of_demand": (
                total_deficit / demand_total if demand_total > 0 else None
            ),
        }

        NG_LCOE_FLOOR_THRESHOLD = 65_511_758.83  # MWh/year threshold
        NG_LCOE_FLOOR_THRESHOLD_0 = 50000
        NG_LCOE_PEG = 78  # $/MWh
        NG_LCOE_PEG_0 = 0

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
                "installed_capacity_mw": Baseload_Installed,
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

        lcoe_values = {
            source: (
                calculate_lcoe(
                    params,
                    dynamic_values[source]['installed_capacity_mw'],
                    dynamic_values[source]['annual_energy_used_mwh'],
                    source_name=source,
                    V2G_CR=V2G_CR,
                    num_connections=Number_V2G_Connections * 1000,
                    V2G_Connect_Cost=V2G_Connect_Cost
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

        # ---- PEG NG LCOE WHEN NG ENERGY IS TINY ----
        if "natural_gas" in lcoe_values and natural_gas_gen_sum < NG_LCOE_FLOOR_THRESHOLD:
            print(
                f"[NG PEG] natural_gas_gen_sum={natural_gas_gen_sum:.2f} < {NG_LCOE_FLOOR_THRESHOLD:.2f}; "
                f"pegging NG LCOE to {NG_LCOE_PEG:.2f} $/MWh (was {lcoe_values['natural_gas']})"
            )
            lcoe_values["natural_gas"] = NG_LCOE_PEG

        if "natural_gas" in lcoe_values and natural_gas_gen_sum < NG_LCOE_FLOOR_THRESHOLD_0:
            print(
                f"[NG PEG] natural_gas_gen_sum={natural_gas_gen_sum:.2f} < {NG_LCOE_FLOOR_THRESHOLD_0:.2f}; "
                f"pegging NG LCOE to {NG_LCOE_PEG_0:.2f} $/MWh (was {lcoe_values['natural_gas']})"
            )
            lcoe_values["natural_gas"] = NG_LCOE_PEG_0

    def calculate_transport_lifecycle_co2(
            year: int,
            Share_EV_init: float,
            SC_years_to_failure: float,
            V2G_years_to_replace: float,
            method: str,
    ) -> dict:
        total_vehicle_count = Full_EV_Fleet * VEHICLE_COUNT_MULTIPLIER
        ev_vehicle_count = total_vehicle_count * Share_EV_init

        gas_vehicle_count = total_vehicle_count * (1.0 - Share_EV_init)
        nsc_vehicle_count = ev_vehicle_count * Factor_NSC
        sc_vehicle_count = ev_vehicle_count * (Factor_Smart_Charge - Factor_V2G)
        v2g_vehicle_count = ev_vehicle_count * Factor_V2G

        sc_battery_wear_adjustment = Baseline_years_to_replace / SC_years_to_failure
        v2g_battery_wear_adjustment = Baseline_years_to_replace / V2G_years_to_replace

        ev_bol_eol = get_lca_value(EV_LCA_GCO2_PER_MILE["BOL_EOL"], year, method)
        ev_battery = get_lca_value(EV_LCA_GCO2_PER_MILE["battery"], year, method)
        ev_vehicle = get_lca_value(EV_LCA_GCO2_PER_MILE["vehicle"], year, method)

        ice_bol_eol = get_lca_value(ICE_LCA_GCO2_PER_MILE["BOL_EOL"], year, method)
        ice_vehicle_ops = get_lca_value(ICE_LCA_GCO2_PER_MILE["vehicle_ops"], year, method)
        ice_production_fuel = get_lca_value(ICE_LCA_GCO2_PER_MILE["production_fuel"], year, method)
        ice_vehicle = get_lca_value(ICE_LCA_GCO2_PER_MILE["vehicle"], year, method)

        ice_total_gco2_per_mile = (
                ice_bol_eol
                + ice_vehicle_ops
                + ice_production_fuel
                + ice_vehicle
        )

        baseline_ice_transport_co2_tonnes = annual_vehicle_lifecycle_co2_tonnes(
            vehicle_count=total_vehicle_count,
            annual_miles=Annual_Miles,
            gco2_per_mile=ice_total_gco2_per_mile,
        )

        modeled_gas_transport_co2_tonnes = annual_vehicle_lifecycle_co2_tonnes(
            vehicle_count=gas_vehicle_count,
            annual_miles=Annual_Miles,
            gco2_per_mile=ice_total_gco2_per_mile,
        )

        nsc_ev_gco2_per_mile = ev_bol_eol + ev_battery + ev_vehicle

        sc_ev_gco2_per_mile = (
                ev_bol_eol
                + ev_battery * sc_battery_wear_adjustment
                + ev_vehicle
        )

        v2g_ev_gco2_per_mile = (
                ev_bol_eol
                + ev_battery * v2g_battery_wear_adjustment
                + ev_vehicle
        )

        nsc_ev_transport_co2_tonnes = annual_vehicle_lifecycle_co2_tonnes(
            vehicle_count=nsc_vehicle_count,
            annual_miles=Annual_Miles,
            gco2_per_mile=nsc_ev_gco2_per_mile,
        )

        sc_ev_transport_co2_tonnes = annual_vehicle_lifecycle_co2_tonnes(
            vehicle_count=sc_vehicle_count,
            annual_miles=Annual_Miles,
            gco2_per_mile=sc_ev_gco2_per_mile,
        )

        v2g_ev_transport_co2_tonnes = annual_vehicle_lifecycle_co2_tonnes(
            vehicle_count=v2g_vehicle_count,
            annual_miles=Annual_Miles,
            gco2_per_mile=v2g_ev_gco2_per_mile,
        )

        modeled_transport_co2_tonnes = (
                modeled_gas_transport_co2_tonnes
                + nsc_ev_transport_co2_tonnes
                + sc_ev_transport_co2_tonnes
                + v2g_ev_transport_co2_tonnes
        )

        avoided_transport_co2_tonnes = (
                baseline_ice_transport_co2_tonnes
                - modeled_transport_co2_tonnes
        )

        return {
            "baseline_ice_transport_co2_tonnes": baseline_ice_transport_co2_tonnes,
            "modeled_transport_co2_tonnes": modeled_transport_co2_tonnes,
            "avoided_transport_co2_tonnes": avoided_transport_co2_tonnes,
            "sc_battery_wear_adjustment": sc_battery_wear_adjustment,
            "v2g_battery_wear_adjustment": v2g_battery_wear_adjustment,
            "ice_total_gco2_per_mile": ice_total_gco2_per_mile,
            "nsc_ev_gco2_per_mile": nsc_ev_gco2_per_mile,
            "sc_ev_gco2_per_mile": sc_ev_gco2_per_mile,
            "v2g_ev_gco2_per_mile": v2g_ev_gco2_per_mile,
        }

    def calculate_combined_lcoe(lcoe_values, dynamic_values, total_system_power):
        weighted_costs = sum(
            (lcoe_values[source] if not math.isinf(lcoe_values[source]) else 0) *
            dynamic_values[source]['annual_energy_used_mwh']
            for source in lcoe_values
        )
        print(f"WHAT ACTUALLY GOES IN LCOE TSP {total_system_power}")
        combined_lcoe = (weighted_costs + SC_Fleet_Annual_Cost + V2G_Fleet_Cost) / total_system_power
        return combined_lcoe

    print("\n--- RAW LCOE DEBUG ---")
    #print("BL variable:", BL_installed if "BL_installed" in globals() else None)
    print("Baseload variable:", Baseload_Installed if "Baseload_Installed" in globals() else None)
    print("demand_sum:", demand_sum)
    print("total_ev_demand_sum:", total_ev_demand_sum)
    print("demand_total:", demand_total)
    print("total_deficit:", total_deficit)


    for source in lcoe_values:
        e = dynamic_values[source]["annual_energy_used_mwh"]
        l = lcoe_values[source]
        c = 0 if math.isinf(l) else l * e
        print(source, "capacity:", dynamic_values[source]["installed_capacity_mw"],
              "energy:", e, "lcoe:", l, "weighted_cost:", c,
              "system_contribution:", c / total_system_power)

    print("V2G_CR:", V2G_CR)
    print("V2G_DR:", V2G_DR)
    print("SC_CR:", SC_CR)
    print("Factor_V2G", Factor_V2G)


    combined_lcoe = calculate_combined_lcoe(lcoe_values, dynamic_values, total_system_power)

    grid_carbon_tonnes_by_source, total_grid_carbon_tonnes = (
        calculate_carbon_emissions_tonnes(dynamic_values)
    )

    carbon_cost_by_case = {}
    carbon_adjusted_lcoe_by_case = {}
    vehicle_carbon_cost_by_case = {}
    total_carbon_cost_by_case = {}
    total_co2_tonnes_by_lca_case = {}

    for scc_case, social_cost_per_tonne in SOCIAL_COST_CARBON_CASES.items():
        _, grid_carbon_cost = calculate_carbon_costs(
            dynamic_values,
            social_cost_per_tonne=social_cost_per_tonne,
        )

        SC_extra_battery_carbon_cost = (
                annual_extra_battery_carbon_cost(
                    SC_extra_cost_per_vehicle_year,
                    annual_miles_per_vehicle=Annual_Miles,
                    social_cost_per_tonne=social_cost_per_tonne,
                )
                * SC_vehicle_count
        )

        V2G_extra_battery_carbon_cost = (
                annual_extra_battery_carbon_cost(
                    V2G_extra_cost_per_vehicle_year,
                    annual_miles_per_vehicle=Annual_Miles,
                    social_cost_per_tonne=social_cost_per_tonne,
                )
                * V2G_vehicle_count
        )

        total_extra_battery_wear_carbon_cost = (
                SC_extra_battery_carbon_cost
                + V2G_extra_battery_carbon_cost
        )

        grid_plus_battery_carbon_cost = (
                grid_carbon_cost
                + total_extra_battery_wear_carbon_cost
        )

        carbon_cost_by_case[scc_case] = grid_plus_battery_carbon_cost

        carbon_adjusted_lcoe_by_case[scc_case] = (
                combined_lcoe
                + grid_plus_battery_carbon_cost / total_system_power
        )

        for lca_method in ["constant_2025", "interpolated"]:
            vehicle_lca = calculate_transport_lifecycle_co2(
                year=year,
                Share_EV_init=Share_EV_init,
                SC_years_to_failure=SC_years_to_failure,
                V2G_years_to_replace=V2G_years_to_replace,
                method=lca_method,
            )

            vehicle_carbon_cost = (
                    vehicle_lca["modeled_transport_co2_tonnes"]
                    * social_cost_per_tonne
            )

            total_carbon_cost = (
                    grid_plus_battery_carbon_cost
                    + vehicle_carbon_cost
            )

            full_case_key = f"{scc_case}_{lca_method}"

            vehicle_carbon_cost_by_case[full_case_key] = vehicle_carbon_cost
            total_carbon_cost_by_case[full_case_key] = total_carbon_cost

            carbon_adjusted_lcoe_by_case[full_case_key] = (
                    combined_lcoe
                    + total_carbon_cost / total_system_power
            )

            total_co2_tonnes_by_lca_case[lca_method] = (
                    total_grid_carbon_tonnes
                    + vehicle_lca["modeled_transport_co2_tonnes"]
            )



    return {
        "status": "success",
        "details": simulation_results,
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
        'combined_lcoe': combined_lcoe,
        'total_GS_used': total_GS_used,
        'Baseload_Installed': Baseload_Installed,
        'Grid_Store_Max': Grid_Store_Max,
        'carbon_cost_by_case': carbon_cost_by_case,
        'carbon_adjusted_lcoe_by_case': carbon_adjusted_lcoe_by_case,
        'vehicle_carbon_cost_by_case': vehicle_carbon_cost_by_case,
        'total_carbon_cost_by_case': total_carbon_cost_by_case,
        'total_grid_carbon_tonnes': total_grid_carbon_tonnes,
        'total_co2_tonnes_by_lca_case': total_co2_tonnes_by_lca_case,
        "system_balance": system_balance,
        "gross_generation_by_source_mwh": gross_generation_by_source,
        "ev_demand_breakdown": ev_demand_breakdown,
        "storage_summary": storage_summary,
        "pass_count": pass_count,
        "min_pass_count": min_pass_count,
    }



all_years_data = pd.DataFrame()

reliability_cases = {
    "6_of_10": 6,
    "8_of_10": 8,
    "10_of_10": 10,
}

initial_state = {
    "Solar_Installed": Solar_Installed,
    "Wind_Installed": Wind_Installed,
    "Baseload_Installed": Baseload_Installed,
    "Grid_Store_Max": Grid_Store_Max,
    "NG_Fact": NG_Fact,
    "NG_Installed": NG_Installed,
}

all_reliability_outputs = {}

for reliability_label, min_pass_count in reliability_cases.items():
    print(f"\n==============================")
    print(f"Running reliability case: {reliability_label}")
    print(f"Minimum passes required: {min_pass_count}/10")
    print(f"==============================\n")

    # Independent state for this reliability path
    Solar_Installed = initial_state["Solar_Installed"]
    Wind_Installed = initial_state["Wind_Installed"]
    Baseload_Installed = initial_state["Baseload_Installed"]
    Grid_Store_Max = initial_state["Grid_Store_Max"]
    NG_Fact = initial_state["NG_Fact"]
    NG_Installed = initial_state["NG_Installed"]

    output_yearly = []

    reliability_folder_name = (
        f"{folder_name}/{reliability_label}_pass_path"
    )
    os.makedirs(reliability_folder_name, exist_ok=True)

    # ============================================================
    # SINGLE 2050 NO-NATURAL-GAS AUDIT RUN
    # ============================================================

    RUN_YEAR = 2050
    RUN_I = RUN_YEAR - 2022 + 12  # matches original loop logic

    # ---- Put your optimized parameters here ----
    Solar_Installed = 30000  # MW
    Wind_Installed = 40000 / 3200  # normalized, because code multiplies by 3200
    Baseload_Installed = 25000  # MW nuclear
    Grid_Store_Max = 15000  # MWh
    Factor_V2G = 0.5
    Factor_NSC = 0.15
    Factor_Smart_Charge = 1 - Factor_NSC
    V2G_Connect_Cost = 2000

    SC_CR = 0.0475
    V2G_CR = 0.0475
    V2G_DR = 0.0475
    SC_Floor = 30
    V2G_Floor = 30
    Cap = 0.8

    # ---- Force no natural gas ----
    NG_Fact = 0.0
    NG_Installed = 0.0

    folder_name = (
        "Output_Files_Merge/"
        f"single_2050_no_ng_"
        f"V2G_{Factor_V2G}_SC_{Factor_Smart_Charge}"
    )
    os.makedirs(folder_name, exist_ok=True)

    #Share_EV_init = (
     #                       np.exp(c1 + c2 * RUN_I) / (1 + np.exp(c1 + RUN_I * c2))
     #               ) - 0.01

    Share_EV_init = 1

    print("\n" + "=" * 80)
    print("RUNNING SINGLE 2050 NO-NATURAL-GAS AUDIT CASE")
    print("=" * 80)
    print(f"Year: {RUN_YEAR}")
    print(f"Share_EV_init: {Share_EV_init:.6f}")
    print(f"Solar_Installed MW: {Solar_Installed:,.2f}")
    print(f"Wind_Installed MW: {Wind_Installed * 3200:,.2f}")
    print(f"Baseload_Installed MW: {Baseload_Installed:,.2f}")
    print(f"Grid_Store_Max MWh: {Grid_Store_Max:,.2f}")
    print(f"Factor_V2G: {Factor_V2G}")
    print(f"Factor_Smart_Charge: {Factor_Smart_Charge}")
    print(f"NG_Fact: {NG_Fact}")
    print(f"NG_Installed: {NG_Installed}")
    print("=" * 80 + "\n")

    results = do_parallel_monte_carlo(
        Share_EV_init=Share_EV_init,
        evcharge=0,
        year=RUN_YEAR,
        n_simulations=1,
        min_pass_count=1,
        SC_CR=SC_CR,
        V2G_CR=V2G_CR,
        V2G_DR=V2G_DR,
        SC_Floor=SC_Floor,
        V2G_Floor=V2G_Floor,
        Cap=Cap,
        V2G_Connect_Cost=V2G_Connect_Cost,
    )

    print("\n" + "=" * 80)
    print("SINGLE 2050 RUN RESULT")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Pass Count: {results.get('pass_count')}/10")

    if results["status"] == "failure":
        for idx, sim in enumerate(results["details"], start=1):
            print(
                f"Run {idx}: "
                f"failed={sim['failed']}, "
                f"reason={sim['failure_reason']}, "
                f"total_deficit={sim['total_energy_deficit']:.2f}, "
                f"curtailment={sim['total_curtailment']:.2f}"
            )
    else:
        print(f"Combined LCOE: {results['combined_lcoe']:.4f}")
        print(f"Total Deficit: {results['total_energy_deficit']:.2f}")
        print(f"Curtailment: {results['total_curtailment']:.2f}")
        print(f"Total V2G: {results['total_V2G']:.2f}")
        print(f"Total Grid Battery Used: {results['total_GS_used']:.2f}")
        print(f"Grid CO2 Tonnes: {results['total_grid_carbon_tonnes']:.2f}")

        print("\nLCOE values:")
        for source, value in results["lcoe_values"].items():
            print(f"  {source}: {value}")

        print("\nCarbon-adjusted LCOE:")
        for case, value in results["carbon_adjusted_lcoe_by_case"].items():
            print(f"  {case}: {value}")

        print("\nSystem balance:")
        for key, value in results["system_balance"].items():
            print(f"  {key}: {value}")

        print("\nStorage summary:")
        for key, value in results["storage_summary"].items():
            print(f"  {key}: {value}")

        print("\nEV demand breakdown:")
        for key, value in results["ev_demand_breakdown"].items():
            print(f"  {key}: {value}")

        audit_report = {
            "run_type": "single_2050_no_natural_gas",
            "input_parameters": {
                "Year": RUN_YEAR,
                "Share_EV_init": Share_EV_init,
                "Solar_Installed_MW": Solar_Installed,
                "Wind_Installed_MW": Wind_Installed * 3200,
                "Baseload_Installed_MW": Baseload_Installed,
                "Grid_Store_Max_MWh": Grid_Store_Max,
                "Factor_V2G": Factor_V2G,
                "Factor_NSC": Factor_NSC,
                "Factor_Smart_Charge": Factor_Smart_Charge,
                "SC_CR": SC_CR,
                "V2G_CR": V2G_CR,
                "V2G_DR": V2G_DR,
                "SC_Floor": SC_Floor,
                "V2G_Floor": V2G_Floor,
                "Cap": Cap,
                "V2G_Connect_Cost": V2G_Connect_Cost,
                "NG_Fact": NG_Fact,
                "NG_Installed": NG_Installed,
            },
            "results": results,
        }

        audit_path = Path(folder_name) / "single_2050_no_ng_audit_report.json"
        with open(audit_path, "w") as f:
            json.dump(to_builtin(audit_report), f, indent=4)

        summary_df = pd.DataFrame([{
            "Year": RUN_YEAR,
            "EV_Fleet_Size": Share_EV_init,
            "Installed_Solar": Solar_Installed,
            "Installed_Wind": Wind_Installed * 3200,
            "Installed_Nuclear": Baseload_Installed,
            "Installed_Grid_Batteries": Grid_Store_Max,
            "NG_Fact": NG_Fact,
            "Combined_LCOE": results["combined_lcoe"],
            "Total_Deficit": results["total_energy_deficit"],
            "Curtailment": results["total_curtailment"],
            "V2G_Total": results["total_V2G"],
            "Total_GS_Used": results["total_GS_used"],
            "Total_Grid_CO2_Tonnes": results["total_grid_carbon_tonnes"],
            "Carbon_Adjusted_LCOE_SCC_51": results["carbon_adjusted_lcoe_by_case"]["SCC_51"],
            "Carbon_Adjusted_LCOE_SCC_120": results["carbon_adjusted_lcoe_by_case"]["SCC_120"],
            "Carbon_Adjusted_LCOE_SCC_190": results["carbon_adjusted_lcoe_by_case"]["SCC_190"],
            "Carbon_Adjusted_LCOE_SCC_340": results["carbon_adjusted_lcoe_by_case"]["SCC_340"],
        }])

        summary_path = Path(folder_name) / "single_2050_no_ng_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved audit report to: {audit_path}")
        print(f"Saved summary CSV to: {summary_path}")

        output_yearly_df_case = pd.DataFrame(output_yearly)

        output_yearly_df_case.to_csv(
            f"{reliability_folder_name}/output_yearly_with_battery.csv",
            index=False,
        )

        #all_reliability_outputs[reliability_label] = output_yearly_df_case

    audit_lcoes = []

    for run_idx in range(50):
        results = do_parallel_monte_carlo(
            Share_EV_init=0.98,
            evcharge=0,
            year=2050,
            n_simulations=1,
            min_pass_count=1,
            SC_CR=SC_CR,
            V2G_CR=V2G_CR,
            V2G_DR=V2G_DR,
            SC_Floor=SC_Floor,
            V2G_Floor=V2G_Floor,
            Cap=Cap,
            V2G_Connect_Cost=V2G_Connect_Cost,
        )

        if results["status"] == "success":
            audit_lcoes.append(results["combined_lcoe"])

    print("\n===== 2050 STOCHASTIC LCOE CHECK =====")
    print(f"Successful runs: {len(audit_lcoes)}")
    print(f"Mean LCOE: {np.mean(audit_lcoes):.3f}")
    print(f"Std LCOE: {np.std(audit_lcoes):.3f}")
    print(f"Min LCOE: {np.min(audit_lcoes):.3f}")
    print(f"Max LCOE: {np.max(audit_lcoes):.3f}")







def save_final_audit_report(output_yearly_df, results, folder_name):
    last_row = output_yearly_df.iloc[-1].to_dict()

    audit_report = {
        "final_year_summary": last_row,
        "final_model_state": {
            "Solar_Installed_MW": Solar_Installed,
            "Wind_Installed_MW": Wind_Installed * 3200,
            "Baseload_Installed_MW": Baseload_Installed,
            "Grid_Store_Max_MWh": Grid_Store_Max,
            "NG_Fact": NG_Fact,
            "Factor_V2G": Factor_V2G,
            "Factor_NSC": Factor_NSC,
            "Factor_Smart_Charge": Factor_Smart_Charge,
            "system_balance": results.get("system_balance"),
            "gross_generation_by_source_mwh": results.get("gross_generation_by_source_mwh"),
            "ev_demand_breakdown": results.get("ev_demand_breakdown"),
            "storage_summary": results.get("storage_summary"),

        },
        "final_results_detail": {
            "combined_lcoe": results.get("combined_lcoe"),
            "lcoe_values": results.get("lcoe_values"),
            "battery_wear": results.get("battery_wear"),
            "carbon_cost_by_case": results.get("carbon_cost_by_case"),
            "carbon_adjusted_lcoe_by_case": results.get("carbon_adjusted_lcoe_by_case"),
            "vehicle_carbon_cost_by_case": results.get("vehicle_carbon_cost_by_case"),
            "total_carbon_cost_by_case": results.get("total_carbon_cost_by_case"),
            "total_grid_carbon_tonnes": results.get("total_grid_carbon_tonnes"),
            "total_co2_tonnes_by_lca_case": results.get("total_co2_tonnes_by_lca_case"),
            "total_curtailment": results.get("total_curtailment"),
            "total_V2G": results.get("total_V2G"),
            "total_GS_used": results.get("total_GS_used"),
            "total_energy_deficit": results.get("total_energy_deficit"),
        },
    }

    audit_path = Path(folder_name) / "final_iteration_audit_report.json"
    with open(audit_path, "w") as f:
        json.dump(to_builtin(audit_report), f, indent=4)

    pd.DataFrame([last_row]).to_csv(
        Path(folder_name) / "final_iteration_summaryCO2.csv",
        index=False,
    )

    print(f"Saved final audit report to: {audit_path}")


# Print the variable selection in an easy-to-read format
print("Variable Selection:")
print(f"  BL_installed: {Baseload_Installed}")
print(f"  Factor_V2G: {Factor_V2G}")
print(f"  Grid_Store_Max: {Grid_Store_Max}")
print(f"  Wind_Installed: {Wind_Installed * 3200:.2f} MW")  # Convert normalized wind back to MW
print(f"  Solar_Installed: {Solar_Installed:.2f} MW")








