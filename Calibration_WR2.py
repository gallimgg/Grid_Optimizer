import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Constant_Factors import (
    NG_Tune, BL_Tune, Solar_Installed, Wind_Installed,
    Baseload_Installed, Import_Limit, Hydro_Tune, Solar_Tune, Wind_Tune
)

# EV Data
Share_EV_init = 0
Factor_NSC = 0.2
Factor_Smart_Charge = 1

# Load and preprocess CSV data
csv_file_path = 'CAL_Check.csv'
df = pd.read_csv(csv_file_path)

# ensure date index is the same length as df
df['Date'] = pd.date_range(
    start='2021-01-01 00:00',
    end='2021-12-30 23:00',
    freq='h'
)

# Add constant columns to the DataFrame
df['Nuke_AVG'] = 0.9
df['Nuke_STDV'] = 0.03

# actual deficit from data
df['Power_Difference'] = df['Demand_AVG'] - df['ActualGen_AVG']


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute coefficient of determination R^2.
    y_true and y_pred must be 1-D arrays of the same length.
    """
    # make sure we're working with float arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # guard against division by zero (e.g. flat series)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def do_monte_carlo(Share_EV_init, evcharge):
    power_deficit_df = pd.DataFrame()
    max_def_max = 0
    all_iterations_deficits = []

    # Initialize lists to track sums across iterations
    solar_sums = []
    wind_sums = []
    BL_sums = []
    power_deficit_sums = []
    power_deficit_avgs = []
    natural_gas_sums = []
    hydro_sums = []
    max_deficits = []
    curtailments = []

    for iteration in range(20):  # Number of Monte Carlo iterations
        # Simulate demand
        demand = np.random.normal(df['Demand_AVG'], df['Demand_STDV']).astype('float32')
        demand_sum = np.sum(demand)

        # Simulate solar and wind generation
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
            np.random.normal(df['Hydro_mean'], df['Hydro_STDV']),
            0
        ) * Hydro_Tune

        # Simulate other generation
        natural_gas_gen = np.random.normal(df['NG_mean'], df['NG_STDV']) * NG_Tune
        nuclear_gen = (
            np.random.normal(df['Nuke_AVG'], df['Nuke_STDV']) * Baseload_Installed
        ) * BL_Tune

        solar_gen_sum = np.sum(solar_gen)
        wind_gen_sum = np.sum(wind_gen)
        hydro_gen_sum = np.sum(hydro_gen)
        natural_gas_gen_sum = np.sum(natural_gas_gen)
        nuclear_gen_sum = np.sum(nuclear_gen)
        total_power_sum = solar_gen_sum + wind_gen_sum + hydro_gen_sum + nuclear_gen_sum + natural_gas_gen_sum

        # Calculate power deficit after nuclear and renewables
        power_deficit = demand - nuclear_gen - (solar_gen + wind_gen)

        # Sort power deficit and align natural_gas_gen and hydro_gen
        sorted_deficit_indices = np.argsort(-power_deficit)  # Indices of highest deficits
        sorted_natural_gas_gen = np.sort(natural_gas_gen)[::-1]  # Descending order
        sorted_hydro_gen = np.sort(hydro_gen)[::-1]  # Descending order

        # Create new arrays for aligned generation
        realigned_natural_gas_gen = np.zeros_like(natural_gas_gen)
        realigned_hydro_gen = np.zeros_like(hydro_gen)

        # Assign sorted generation values to indices of highest deficits
        for i, idx in enumerate(sorted_deficit_indices):
            realigned_natural_gas_gen[idx] = sorted_natural_gas_gen[i]
            realigned_hydro_gen[idx] = sorted_hydro_gen[i]

        power_deficit_sort = power_deficit - realigned_natural_gas_gen - realigned_hydro_gen
        power_deficit_sort_sum = np.sum(power_deficit_sort)
        print(f'the power deficit sort sum is {power_deficit_sort_sum}')
        print(f'the total curtailments are {total_power_sum + power_deficit_sort_sum - demand_sum}')

        # Calculate curtailment for this iteration
        curtailment = np.sum(np.maximum(8200 - power_deficit_sort, 0))
        curtailments.append(curtailment)

        # Track sums for this iteration
        solar_sums.append(np.nansum(solar_gen))
        wind_sums.append(np.nansum(wind_gen))
        BL_sums.append(np.nansum(nuclear_gen))
        power_deficit_sums.append(np.nansum(power_deficit_sort))
        power_deficit_avgs.append(np.nanmean(power_deficit_sort))
        natural_gas_sums.append(np.nansum(natural_gas_gen))
        hydro_sums.append(np.nansum(hydro_gen))

        # Add power_deficit_sort as a new column in the DataFrame
        power_deficit_df[f'Iteration_{iteration + 1}'] = power_deficit_sort

        # Record results for this iteration
        all_iterations_deficits.append(power_deficit_sort)

        # Record results
        max_def = np.max(np.maximum(power_deficit_sort, 0))
        max_deficits.append(max_def)
        if max_def > max_def_max:
            max_def_max = max_def
            if max_def_max > Import_Limit:
                print("Simulation Failed")

    # Compute averages across iterations
    avg_solar = np.mean(solar_sums)
    avg_wind = np.mean(wind_sums)
    avg_BL = np.mean(BL_sums)
    avg_power_deficit = np.mean(power_deficit_sums)
    actual_power_deficit = np.nansum(df['Power_Difference'])
    avg_deficit_mean = np.mean(power_deficit_avgs)
    avg_natural_gas = np.mean(natural_gas_sums)
    avg_hydro = np.mean(hydro_sums)
    actual_demand = np.nansum(df['Demand_AVG'])
    actual_max_def_avg = np.max(df['Power_Difference'])
    avg_curtailment = np.mean(curtailments)

    deficit_20th_percentile = np.percentile(power_deficit_avgs, 90)
    actual_20th_percentile = np.percentile(df['Power_Difference'], 90)

    avg_max_deficit = np.mean(max_deficits)

    print("\n=== Average Results Across All Iterations ===")
    print(f"Average Solar Generation: {avg_solar:.2f}")
    print(f"Average Wind Generation: {avg_wind:.2f}")
    print(f"Average Nuclear Generation: {avg_BL:.2f}")
    print(f"Average Power Deficit (Sum): {avg_power_deficit:.2f}")
    print(f"Average Power Deficit actual data (Sum): {actual_power_deficit:.2f}")
    print(f"Average Power Deficit (Mean): {avg_deficit_mean:.2f}")
    print(f"20th Percentile of Average Power Deficit: {deficit_20th_percentile:.2f}")
    print(f"20th Percentile of the actual values: {actual_20th_percentile:.2f}")
    print(f"Average Natural Gas Generation: {avg_natural_gas:.2f}")
    print(f"Average Hydro Generation: {avg_hydro:.2f}")
    print(f"Actual demand total: {actual_demand:.2f}")
    print(f"Average Maximum Deficit Across Iterations: {avg_max_deficit:.2f}")
    print(f"Actual Maximum Deficit Across Iterations: {actual_max_def_avg:.2f}")
    print(f"Average Annual Curtailment: {avg_curtailment:.2f}")
    print(f"The Max def max is: {max_def_max:.2f}")

    # Calculate the average deficit across all iterations (row-wise mean)
    power_deficit_df['Average_Deficit'] = power_deficit_df.mean(axis=1)

    # now compute R^2 vs actual
    actual_deficit = df['Power_Difference'].to_numpy()
    modeled_deficit = power_deficit_df['Average_Deficit'].to_numpy()
    r2 = r2_score(actual_deficit, modeled_deficit)
    print(f"R-squared (modeled vs. actual deficit): {r2:.4f}")

    return {
        'power_deficit_df': power_deficit_df,
        'max_def': max_def_max,
        'r2': r2,
    }


evcharge = 0
results = do_monte_carlo(Share_EV_init, evcharge)

power_deficit_df = results['power_deficit_df']
r2 = results['r2']

# Save the results to a CSV file
power_deficit_df.to_csv("output_df.csv", index=False)

# --- Styling parameters (easy to tweak later) ---
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 14
FONT_WEIGHT = 'bold'

# Plot the average deficit
plt.figure(figsize=(12, 7))

plt.scatter(
    df.index,
    power_deficit_df['Average_Deficit'],
    label='Modeled Deficit',
    s=5  # slightly larger for visibility
)

plt.scatter(
    df.index,
    df['Power_Difference'],
    label='California Grid Actual Deficit',
    s=5
)

# Axis labels
plt.xlabel('Hour of Year', fontsize=LABEL_SIZE, fontweight=FONT_WEIGHT)
plt.ylabel('Deficit (MW)', fontsize=LABEL_SIZE, fontweight=FONT_WEIGHT)

# Title
plt.title(
    f'Average Power Deficit Across Monte Carlo Iterations (R² = {r2:.3f})',
    fontsize=TITLE_SIZE,
    fontweight=FONT_WEIGHT
)

# Tick labels
plt.xticks(fontsize=TICK_SIZE, fontweight=FONT_WEIGHT)
plt.yticks(fontsize=TICK_SIZE, fontweight=FONT_WEIGHT)

# Legend
plt.legend(fontsize=LEGEND_SIZE)

# Optional: make axes lines thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Layout and save
plt.tight_layout()
plot_path = "installed_capacity_growth_calibration.png"
plt.savefig(plot_path, format='png', dpi=300)  # higher DPI for clarity
plt.show()
