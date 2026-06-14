import numpy as np
from itertools import product
import pandas as pd
from Constant_Factors import Solar_Tune, Wind_Tune, BL_Tune, Hydro_Tune

# Define variable bounds and step sizes
VARIABLE_BOUNDS = {
    'BL_installed': (10000, 40000),
    'Factor_V2G': (0.10, 0.7),
    'Grid_Store_Max': (10000, 60000),
    'Wind_Installed': (20000 / 3200, 70000 / 3200),
    'Solar_Installed': (20000, 70000),
    'CR': (0.0187, 0.0475),
    'Cap': (0.8, 1),
    "V2G_Connect_Cost": 2000,
}

STEP_SIZES = {
    'BL_installed': 5000,
    'Factor_V2G': 0.2,
    'Grid_Store_Max': 5000,
    'Wind_Installed': 10000 / 3200,
    'Solar_Installed': 10000,
    'CR': 0.0072,
    'Cap': 0.05,
}


total_power = 392526404.9

LCOE_MAX = 100
LCOE_MIN = 50


# Generate all combinations within bounds and step sizes
def generate_combinations(bounds, steps):
    combinations = []

    for key, value in bounds.items():
        if isinstance(value, tuple):
            low, high = value
            step = steps[key]
            vals = np.arange(low, high + step / 2, step)
        else:
            # fixed scalar value
            vals = np.array([value])

        combinations.append(vals)

    return pd.DataFrame(
        list(product(*combinations)),
        columns=bounds.keys()
    )

# Generate combinations
combinations = generate_combinations(VARIABLE_BOUNDS, STEP_SIZES)

# Convert to DataFrame
columns = list(VARIABLE_BOUNDS.keys())
df_combinations = pd.DataFrame(combinations, columns=columns)

data = df_combinations

# Constants for the calculation
Solar_CF = 0.302675
Wind_CF = 0.505289
Hydro = 22116617.6 * Hydro_Tune
HOURS_IN_YEAR = 8760

# Calculate the new column
data['Power_Capacity'] = (
    data['Solar_Installed'] * Solar_CF *Solar_Tune * HOURS_IN_YEAR +
    data['Wind_Installed'] * Wind_CF * Wind_Tune * HOURS_IN_YEAR * 3200 +
    data['BL_installed'] * HOURS_IN_YEAR * 0.9 * BL_Tune +
    Hydro
)


# Save the updated DataFrame to a new CSV file
output_file_path = "NO_CR_Wpw.csv"
data.to_csv(output_file_path, index=False)

print(f"File saved to: {output_file_path}")

def crf(rate, n_years):
    return (rate * (1 + rate)**n_years) / ((1 + rate)**n_years - 1)

# Load your CSV
df = data
print(len(df))
# Constants
CAPEX_PER_KW = {
    "Battery": 1316,    # $/kW, example
    "Wind": 1718,
    "Solar": 1327,
    "Nuclear": 7030,
    "Hydro": 0 # won't be installing new hydropower
}
V2G_CAPEX_PER_CAR = 2000  # example, change if needed
N_CARS = 29000000

CRF = {
    "Battery": crf(0.05, 20),
    "Wind": crf(0.05, 30),
    "Solar": crf(0.05, 30),
    "Nuclear": crf(0.05, 60),
    "Hydro": crf(0.05, 60),
    "V2G": crf(0.05, 30)
}

# Annualized CAPEX
df['Annual_CAPEX_Battery'] = df['Grid_Store_Max'] * 1000 * CAPEX_PER_KW["Battery"] * CRF["Battery"]
df['Annual_CAPEX_Wind'] = df['Wind_Installed'] * 1000 * CAPEX_PER_KW["Wind"] * CRF["Wind"] * 3200
df['Annual_CAPEX_Solar'] = df['Solar_Installed'] * 1000 * CAPEX_PER_KW["Solar"] * CRF["Solar"]
df['Annual_CAPEX_Nuclear'] = df['BL_installed'] * 1000 * CAPEX_PER_KW["Nuclear"] * CRF["Nuclear"]
df['Annual_CAPEX_Hydro'] = 0  # Add if you model hydro separately
df['Annual_CAPEX_V2G'] = N_CARS * df['Factor_V2G'] * V2G_CAPEX_PER_CAR * CRF["V2G"]

# Total annual CAPEX
df['Annual_CAPEX_Total'] = (
    df['Annual_CAPEX_Battery'] +
    df['Annual_CAPEX_Wind'] +
    df['Annual_CAPEX_Solar'] +
    df['Annual_CAPEX_Nuclear'] +
    df['Annual_CAPEX_Hydro'] +
    df['Annual_CAPEX_V2G']
)

df['LCOE_CAPEX'] = df['Annual_CAPEX_Total']/total_power

df_filtered = df[(df['LCOE_CAPEX'] <= LCOE_MAX) & (df['LCOE_CAPEX'] > LCOE_MIN)]

# Transition to Column adjust (Removed legacy column)
df = df_filtered

# Columns to remove
cols_to_remove = [
    "Annual_CAPEX_Battery", "Annual_CAPEX_Wind", "Annual_CAPEX_Solar",
    "Annual_CAPEX_Nuclear", "Annual_CAPEX_Hydro", "Annual_CAPEX_V2G",
    "Annual_CAPEX_Total", "LCOE_CAPEX"
]

# Drop the columns
df = df.drop(columns=cols_to_remove, errors='ignore')  # errors='ignore' prevents crashing if any column is missing

# (Optional) Save the result to a new file
df.to_csv("Small_Sample_6.csv", index=False)

