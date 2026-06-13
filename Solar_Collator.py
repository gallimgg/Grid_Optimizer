import os
import pandas as pd
import pvlib

# Base directory containing the location folders
base_dir = "CA_SOL2"

# Define the locations
locations = [
    {"lat": 34.926, "lon": -118.333, "name": "Kern"},
    {"lat": 35.388, "lon": -120.067, "name": "Topaz"},
    {"lat": 33.98, "lon": -117.37, "name": "Deser_Sunlight"},
    {"lat": 35.28, "lon": -120.66, "name": "CA_Valley_Solar_Ranch"},
    {"lat": 32.68, "lon": -115.64, "name": "Mount_Signal"}
]

def load_location_data(location):
    """Load and concatenate all CSVs for a location."""
    folder_name = f"{location['lat']}_{location['lon']}"
    location_dir = os.path.join(base_dir, folder_name)

    all_data = []
    for file in os.listdir(location_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(location_dir, file)
            df = pd.read_csv(file_path, parse_dates=['time'])

            # Remove February 29th if it exists
            df = df[~((df['time'].dt.month == 2) & (df['time'].dt.day == 29))]

            all_data.append(df)

    # Concatenate and ensure the index is datetime
    combined_data = pd.concat(all_data).sort_values('time')
    combined_data.set_index('time', inplace=True)
    return combined_data

def calculate_poa(data, lat, lon):
    """Calculate Plane-of-Array (POA) irradiance."""
    system = {
        'axis_tilt': 0,
        'axis_azimuth': 180,
        'backtrack': True,
        'gcr': 0.4
    }

    # Calculate solar position
    solar_position = pvlib.solarposition.get_solarposition(
        time=data.index, latitude=lat, longitude=lon
    )

    # Align solar position with irradiance data
    solar_position = solar_position.reindex(data.index, method='nearest')

    # Create a tracker for single-axis tracking
    tracker = pvlib.tracking.singleaxis(
        apparent_zenith=solar_position['apparent_zenith'],
        apparent_azimuth=solar_position['azimuth'],
        **system
    )

    # Calculate total irradiance (POA)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracker['surface_tilt'],
        surface_azimuth=tracker['surface_azimuth'],
        dni=data['DNI'],
        dhi=data['DHI'],
        ghi=data['GHI'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )

    # Add 'hour_of_year' for grouping (1 to 8760)
    poa['hour_of_year'] = (
        (poa.index.day_of_year - 1) * 24 + poa.index.hour
    ).astype(int)

    return poa[['poa_global', 'hour_of_year']]

# Prepare the output DataFrame with 8760 rows
output_df = pd.DataFrame(index=range(8760))

for location in locations:
    print(f"Processing {location['name']}...")

    # Load and clean data for the location
    data = load_location_data(location)

    # Calculate POA irradiance
    poa_irradiance = calculate_poa(data, location['lat'], location['lon'])

    # Drop any rows with missing data
    poa_irradiance = poa_irradiance.dropna()

    # Group by 'hour_of_year' and compute mean & std
    grouped = poa_irradiance.groupby('hour_of_year')['poa_global'].agg(['mean', 'std'])

    # Ensure exactly 8760 rows by reindexing if needed
    grouped = grouped.reindex(range(8760), fill_value=0)

    # Store results in the output DataFrame
    output_df[f"{location['name']}_mean"] = grouped['mean'].values
    output_df[f"{location['name']}_std"] = grouped['std'].values

# Save the results to CSV
output_df.to_csv("SOL_COLLATED.csv", index=False)
print("Saved SOL_COLLATED.csv successfully.")






