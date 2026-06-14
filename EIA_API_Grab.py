import requests
import pandas as pd
import calendar  # Use to check leap years

# API endpoint and parameters
url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
api_key = "YOUR_API_KEY"  # Replace with your actual API key

# Years to fetch
years = list(range(2018, 2024))

# Initialize an empty DataFrame to store all years side by side
all_years_data = pd.DataFrame()

# Query the API for each year
for year in years:
    print(f"Fetching data for {year}...")
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "start": f"{year}-01-01T00",
        "end": f"{year}-12-31T23",
        "facets[respondent][]": "CAL",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }

    # Temporary DataFrame for the year
    year_data = pd.DataFrame()

    # Pagination loop
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("response", {}).get("data", [])
            if not results:
                break

            # Append results to the year's DataFrame
            df = pd.DataFrame(results)
            year_data = pd.concat([year_data, df], ignore_index=True)

            # Update offset for pagination
            params["offset"] += params["length"]
        else:
            print(f"Failed to fetch data for {year}. Status code: {response.status_code}, Error: {response.text}")
            break

    # Process data for the year
    if not year_data.empty:
        # Convert 'period' to datetime and remove leap day for non-leap years
        year_data["period"] = pd.to_datetime(year_data["period"])
        if not calendar.isleap(year):
            year_data = year_data[~((year_data["period"].dt.month == 2) & (year_data["period"].dt.day == 29))]

        # Pivot the data to format by fuel type
        pivoted = year_data.pivot_table(
            index="period",
            columns="fueltype",
            values="value",
            aggfunc="sum"
        )
        pivoted.columns = [f"{year}_{col}" for col in pivoted.columns]  # Add year prefix to columns
        pivoted[f"{year}_timestamp"] = pivoted.index  # Add timestamp for comparison

        # Align all years side by side
        if all_years_data.empty:
            all_years_data = pivoted
        else:
            all_years_data = pd.merge(all_years_data, pivoted, how="outer", left_index=True, right_index=True)

# Ensure final formatting
all_years_data.reset_index(drop=True, inplace=True)
all_years_data.fillna(0, inplace=True)

# Save the final file
output_file = "CAL.csv"
all_years_data.to_csv(output_file, index=False)

print(f"Data saved to '{output_file}'")

