import pandas as pd

# Load the CSV file
file_path = 'Land_2_Site.csv'
df = pd.read_csv(file_path)

# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Filter out leap day data explicitly (Feb 29)
df = df[~((df['time'].dt.month == 2) & (df['time'].dt.day == 29))]

# Create a new column for 'hour_of_year' (from 0 to 8759)
df['hour_of_year'] = df['time'].dt.dayofyear * 24 + df['time'].dt.hour - 24

# Function to convert wind speed to power using GE 3.2 MW Turbine
def wind_speed_to_power(speed):
    if speed < 3:
        return 0
    elif 3 < speed <= 3.5:
        return 54
    elif 3.5 < speed <= 4.5:
        return 234
    elif 4.5 < speed <= 5.5:
        return 497
    elif 5.5 < speed <= 6.5:
        return 880
    elif 6.5 < speed <= 7.5:
        return 1407
    elif 7.5 < speed <= 8.5:
        return 2060
    elif 8.5 < speed <= 9.5:
        return 2709
    elif 9.5 < speed <= 10.5:
        return 3156
    elif 10.5 < speed <= 26:
        return 3421
    else:
        return 0

# Apply the wind speed to power conversion function
df['power_output'] = df['wind_speed_100m'].apply(wind_speed_to_power)

# Compute the average power output for each hour of the year for each location (lat, lon)
average_power_output = df.groupby(['lat', 'lon', 'hour_of_year'])['power_output'].mean().reset_index()

# Pivot the DataFrame to make each lat/lon pair a separate column
pivot_df = average_power_output.pivot(index='hour_of_year', columns=['lat', 'lon'], values='power_output')

# Rename columns for clarity 
pivot_df.columns = [f"({lat}, {lon})" for lat, lon in pivot_df.columns]

# Ensure we now have 8760 rows
pivot_df = pivot_df.loc[:8759]

# Save the reshaped DataFrame to a new CSV file
pivot_df.to_csv('Land_2site_out.csv')

# Display the first few rows of the pivoted data
#print(pivot_df)
#pivot_df.to_csv("CA_Wind_Power_Avg.csv")
average_power_per_location = pivot_df.mean()
print(average_power_per_location)
