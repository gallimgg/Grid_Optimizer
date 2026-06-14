import pandas as pd

# Load the data
file_path = "CAL.csv"
data = pd.read_csv(file_path, low_memory=False)

# Convert timestamp columns to datetime, handling invalid entries as NaT
for col in data.columns:
    if "timestamp" in col:
        data[col] = pd.to_datetime(data[col], errors="coerce")

# Initialize an empty list to collect yearly data
yearly_data_frames = []

# Iterate over years to align their data one by one
for year in range(2019, 2024):
    year_columns = [f"{year}_COL", f"{year}_NG", f"{year}_NUC", f"{year}_OIL", f"{year}_OTH",
                    f"{year}_SUN", f"{year}_WAT", f"{year}_WND", f"{year}_timestamp"]
    if all(col in data.columns for col in year_columns):  # Ensure all required columns exist
        year_data = data[year_columns].copy()
        year_data.rename(columns={f"{year}_timestamp": "timestamp"}, inplace=True)

        # Drop rows with invalid timestamps
        year_data = year_data.dropna(subset=["timestamp"])

        # Ensure unique timestamps by prefixing the year
        year_data['timestamp'] = year_data['timestamp'].dt.strftime(f"{year}-%m-%d %H:%M:%S")

        # Reset index for easier concatenation
        year_data.reset_index(drop=True, inplace=True)

        # Rename columns to include the year
        year_data.columns = [f"{col.split('_')[1]}_{year}" if col != "timestamp" else "timestamp" for col in
                             year_data.columns]

        # Append to the list of yearly data frames
        yearly_data_frames.append(year_data)

# Concatenate all years' data side by side
aligned_data = pd.concat(yearly_data_frames, axis=1)

# Save the aligned data to a CSV file
output_file = "CAL_Aligned.csv"
aligned_data.to_csv(output_file, index=False)

# Print confirmation
print(f"Aligned data saved to '{output_file}'")



