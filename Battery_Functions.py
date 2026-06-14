import numpy as np
import pandas as pd
from blast.models import Lfp_Gr_250AhPrismatic  # Replace with your preferred battery model


def interpolate_soc(hourly_soc):
    """
    Interpolate hourly SOC data to minutely resolution with separate
    discharge and charge phases.

    Args:
        hourly_soc (np.ndarray): Array of hourly SOC data (0-100%).

    Returns:
        np.ndarray: Array of minutely SOC data (0-1 normalized).
    """
    if len(hourly_soc) % 2 != 0:
        raise ValueError("SOC data length must be even, with alternating lowest and highest SOC.")

    # Split into lowest and highest SOC values
    lowest_soc = hourly_soc[::2]   # Every other point starting from index 0
    highest_soc = hourly_soc[1::2]  # Every other point starting from index 1

    # Initialize the minutely SOC array
    minutely_soc = []

    # Simulate each hour
    for low, high in zip(lowest_soc, highest_soc):
        # Discharge phase (first 30 minutes)
        total_discharge_soc = high - low
        discharge_per_minute = total_discharge_soc / 30
        current_soc = high
        for _ in range(30):
            minutely_soc.append(current_soc / 100.0)  # Normalize to 0–1
            current_soc -= discharge_per_minute

        # Charge phase (last 30 minutes)
        charge_per_minute = total_discharge_soc / 30
        for _ in range(30):
            minutely_soc.append(current_soc / 100.0)  # Normalize to 0–1
            current_soc += charge_per_minute

    return np.array(minutely_soc)


def calculate_battery_wear(
    minute_soc_data,
    minute_temp_data,
    battery_model=Lfp_Gr_250AhPrismatic,
    simulation_years=1,
):
    """
    Calculate battery wear using minute-by-minute SOC and temperature data.

    Arguments:
        minute_soc_data (np.ndarray): Minutely SOC data (normalized to 0–1 Share of Battery Capacity),
                                      representing ONE typical year of operation.
        minute_temp_data (np.ndarray): Minutely temperature data (degrees C),
        battery_model (class): BLAST battery model class.
        simulation_years (int): Number of years to simulate by repeating the
                                input profile.

    Returns:
        float: Final battery capacity (q) after the simulation.
    """
    # Validate input data
    if len(minute_soc_data) != len(minute_temp_data):
        raise ValueError("SOC and temperature data lengths must match.")

    if simulation_years <= 0:
        raise ValueError("simulation_years must be a positive integer.")

    # Repeat the 1-year profile simulation_years times
    soc_tiled = np.tile(minute_soc_data, simulation_years)
    temp_tiled = np.tile(minute_temp_data, simulation_years)

    # Prepare time data for BLAST (one point per minute, in seconds)
    n_points = len(soc_tiled)
    time_s = np.arange(n_points, dtype=float) * 60.0

    # Prepare BLAST input data
    blast_data = {
        "Time_s": time_s,
        "SOC": soc_tiled,              # SOC normalized (0–1)
        "Temperature_C": temp_tiled,   # Temperature in Celsius
    }

    # Initialize the battery model
    battery = battery_model()

    # Simulate battery life
    battery.simulate_battery_life(blast_data)

    # Extract the final battery capacity (q)
    results_df = pd.DataFrame(battery.outputs)
    if "q" not in results_df.columns:
        raise KeyError(
            f"'q' (capacity) not found in BLAST outputs. "
            f"Available columns: {list(results_df.columns)}"
        )

    final_q = results_df["q"].iloc[-1]  # Last capacity value

    return final_q
