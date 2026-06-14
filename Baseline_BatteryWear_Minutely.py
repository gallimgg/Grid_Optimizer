import pandas as pd
import numpy as np
from blast.models import Lfp_Gr_250AhPrismatic


def convert_hourly_to_minutely(temp_data):
    """
    Convert hourly temperature data to minutely data by repeating each value 60 times.
    """
    return np.repeat(temp_data, 60)

def calculate_battery_wear(
    soc_data,
    temp_data,
    battery_model=Lfp_Gr_250AhPrismatic,
):
    """
    Calculate battery wear using minute-by-minute SOC and temperature data.
    """
    if len(soc_data) != len(temp_data):
        raise ValueError(
            f"SOC and temperature data lengths must match. "
            f"SOC length: {len(soc_data)}, temperature length: {len(temp_data)}"
        )

    time_s = np.arange(len(soc_data)) * 60

    blast_data = {
        "Time_s": time_s,
        "SOC": soc_data,
        "Temperature_C": temp_data,
    }

    battery = battery_model()

    battery.simulate_battery_life(
        input_timeseries=blast_data,
        is_constant_input=False,
    )

    results_df = pd.DataFrame(battery.outputs)
    final_q = results_df["q"].iloc[-1]

    return final_q

# File paths
soc_filepath = "Baseline_Battery_Data.csv"
temp_filepath = "CAL_Temps.csv"

# Load SOC data
soc_data = pd.read_csv(soc_filepath)["SOC"].values

# Load and process temperature data
hourly_temp_data = pd.read_csv(temp_filepath)["Temperature"].values
minutely_temp_data = convert_hourly_to_minutely(hourly_temp_data)

# Validate lengths
if len(soc_data) != len(minutely_temp_data):
    raise ValueError("SOC data and minutely temperature data lengths must match.")

# Calculate battery wear
final_q = calculate_battery_wear(soc_data, minutely_temp_data)
print(f"Final battery capacity (q): {final_q * 100:.2f}%")
print(f"Baseline years to replace: {np.log(0.8)/np.log(final_q)}") #Note Hand calculated result from Final battery q will be different due to rounding




