import os, io, time, requests
import pandas as pd

# Your NSRDB API key
API_KEY = 'YOUR_API_KEY'

# Define the locations to fetch data for
locations = [
    {"lat": 34.926, "lon": -118.333, "name": "Kern"},
    {"lat": 35.388, "lon": -120.067, "name": "Topaz"},
    {"lat": 33.98,  "lon": -117.37,  "name": "Deser_Sunlight"},
    {"lat": 35.28,  "lon": -120.66,  "name": "CA_Valley_Solar_Ranch"},
    {"lat": 32.68,  "lon": -115.64,  "name": "Mount_Signal"}
]

years = range(1998, 2025)

base_dir = "CA_SOL2"
os.makedirs(base_dir, exist_ok=True)

V4_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
ATTRS = "dni,dhi,ghi,solar_zenith_angle,air_temperature,wind_speed"

def fetch_nsrdb_v4(lat: float, lon: float, year: int) -> pd.DataFrame | None:
    """Fetch hourly (60-min) solar data from NSRDB GOES Aggregated v4."""
    params = {
        "api_key": API_KEY,
        "wkt": f"POINT({lon} {lat})",
        "names": year,
        "leap_day": "false",
        "interval": "60",      # 30 or 60 allowed
        "utc": "true",
        "full_name": "Test User",
        "email": "grant.gallimore@und.edu",
        "affiliation": "Test Organization",
        "reason": "research",
        "attributes": ATTRS
    }

    r = requests.get(V4_URL, params=params, timeout=120)
    if r.status_code != 200:
        print(f"Failed to fetch data: {r.status_code} - {r.text[:200]}")
        return None

    # v4 CSV has two header rows before the time-series header row
    content = io.StringIO(r.content.decode("utf-8", errors="ignore"))
    df = pd.read_csv(content, skiprows=2)

    # Ensure expected time columns exist; v4 adds 'Minute'
    # for interval=60 minute will be 0 in all rows.
    if "Minute" not in df.columns:
        df["Minute"] = 0

    # Build a timestamp index
    ts_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    missing = [c for c in ts_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected time columns in CSV: {missing}")

    df["time"] = pd.to_datetime(df[ts_cols].rename(columns={"Minute": "Min"}).assign(Second=0)
                                .apply(lambda s: f"{int(s['Year']):04d}-{int(s['Month']):02d}-{int(s['Day']):02d} "
                                                 f"{int(s['Hour']):02d}:{int(s['Min']):02d}:00", axis=1))
    df.set_index("time", inplace=True)
    return df

def save_data(location: dict, year: int, data: pd.DataFrame) -> None:
    folder_name = f"{location['name']}_{location['lat']}_{location['lon']}"
    location_dir = os.path.join(base_dir, folder_name)
    os.makedirs(location_dir, exist_ok=True)
    path = os.path.join(location_dir, f"{year}.csv")
    data.to_csv(path)
    print(f"Saved {path}")

for location in locations:
    for year in years:
        df = fetch_nsrdb_v4(location["lat"], location["lon"], year)
        if df is not None:
            save_data(location, year, df)
        time.sleep(1)  # CSV rate limit = 1 req/sec; keep at least 1s spacing
print("All data fetched and saved successfully.")
