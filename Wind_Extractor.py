import xarray as xr
import numpy as np


# Load the GRIB file
file_path = 'data.grib'
data = xr.open_dataset(file_path, engine='cfgrib')

# Define the time range
start_time = "2010-01-01"
end_time = "2023-12-31"

# Wind farm coordinates (lat/lon pairs)
wind_farm_coords = [
    (38.75, -123.50),  # Altamont Pass Wind Farm
    (38, -123),  # Tehachapi Pass Wind Farm
]

# Extract the u100 and v100 data for the wind farm locations
wind_speed_data = []

for lat, lon in wind_farm_coords:
    try:
        # Select the nearest lat/lon using method='nearest'
        u100_subset = data['u100'].sel(latitude=lat, longitude=lon, method="nearest")
        v100_subset = data['v100'].sel(latitude=lat, longitude=lon, method="nearest")

        print(f"Processing location: lat={lat}, lon={lon}")

        # Apply the time range with slice
        u100_time_filtered = u100_subset.sel(time=slice(start_time, end_time))
        v100_time_filtered = v100_subset.sel(time=slice(start_time, end_time))

        # Calculate wind speed
        wind_speed_subset = np.sqrt(u100_time_filtered ** 2 + v100_time_filtered ** 2)

        # Add latitude, longitude, and wind speed to the data
        wind_speed_subset = wind_speed_subset.assign_coords(lat=lat, lon=lon)
        wind_speed_data.append(wind_speed_subset)

        print(f"Successfully processed data for lat={lat}, lon={lon}")
    except KeyError as e:
        print(f"Data not found for lat={lat}, lon={lon}. Error: {e}")
    except Exception as e:
        print(f"An error occurred for lat={lat}, lon={lon}: {e}")

# Check if there's valid data to concatenate
if wind_speed_data:
    # Concatenate the wind speed data for the different lat/lon pairs
    california_wind_speed = xr.concat(wind_speed_data, dim="lat_lon")

    # Name the DataArray and convert to a DataFrame
    california_wind_speed.name = 'wind_speed_100m'
    df = california_wind_speed.to_dataframe().reset_index()

    # Save the DataFrame to a CSV file
    df.to_csv('Land_2_Site.csv', index=False)

    # Display the first few rows of the data
    print(df.head())

    # Calculate min and max wind speed
    min_wind_speed = df['wind_speed_100m'].min()
    max_wind_speed = df['wind_speed_100m'].max()

    print(f"Minimum wind speed: {min_wind_speed} m/s")
    print(f"Maximum wind speed: {max_wind_speed} m/s")
else:
    print("No valid wind speed data was found for the given coordinates.")
