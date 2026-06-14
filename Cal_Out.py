#!/usr/bin/env python3
"""
Compute 8760 hourly statistics across all years for NG and WAT.

Input format: wide CSV with repeating year blocks such as:
    NG_2019, WAT_2019, ..., timestamp,
    NG_2020, WAT_2020, ..., timestamp.1,
    NG_2021, WAT_2021, ..., timestamp.2,
    ...

Output: CSV with columns
    hour_of_year (0..8759), WAT_mean, WAT_std, NG_mean, NG_std

Just set INPUT_FILE and OUTPUT_FILE and run in PyCharm.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_FILE  = "CAL_Aligned.csv"
OUTPUT_FILE = "hourly_WAT_NG_8760_stats.csv"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def find_timestamp_columns(df: pd.DataFrame):
    """Return all timestamp-like columns in left-to-right order."""
    ts_cols = [c for c in df.columns if (c == "timestamp") or c.startswith("timestamp")]
    return [c for c in df.columns if c in ts_cols]


def detect_years(df: pd.DataFrame):
    """Return list of years that have both NG_<year> and WAT_<year> columns."""
    ng_years  = set(int(m.group(1)) for col in df.columns
                    if (m := re.match(r"^NG_(\d{4})$", col)))
    wat_years = set(int(m.group(1)) for col in df.columns
                    if (m := re.match(r"^WAT_(\d{4})$", col)))
    years = sorted(ng_years & wat_years)
    if not years:
        raise ValueError("No usable NG_YYYY and WAT_YYYY year columns found.")
    return years


def pick_timestamp_for_year(df: pd.DataFrame, year: int, ts_cols: list):
    """
    Select the timestamp column corresponding to a given year.
    Prefers timestamp_YYYY, otherwise picks the timestamp* whose values contain that year.
    Fallback: the first timestamp column.
    """
    # Exact match?
    exact = f"timestamp_{year}"
    if exact in df.columns:
        return pd.to_datetime(df[exact], errors="coerce")

    # Try to infer from values
    parsed_cache = {}
    for c in ts_cols:
        s = pd.to_datetime(df[c], errors="coerce")
        parsed_cache[c] = s
        if not s.isna().all():
            years_present = s.dt.year.dropna().unique()
            if len(years_present) and (year in years_present):
                return s

    # Fallback
    return pd.to_datetime(df[ts_cols[0]], errors="coerce")


def build_year_frame(df: pd.DataFrame, year: int, ts_cols: list) -> pd.DataFrame:
    """Return a per-year DataFrame with datetime, NG, WAT, year."""
    ng_col  = f"NG_{year}"
    wat_col = f"WAT_{year}"

    dt = pick_timestamp_for_year(df, year, ts_cols)

    out = pd.DataFrame({
        "datetime": dt,
        "NG": pd.to_numeric(df[ng_col], errors="coerce"),
        "WAT": pd.to_numeric(df[wat_col], errors="coerce"),
    }).dropna(subset=["datetime"])

    return out


def drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Feb 29 for leap year alignment to 8760 hours."""
    d = df.copy()
    mono = d["datetime"]
    mask = (mono.dt.month == 2) & (mono.dt.day == 29)
    return d.loc[~mask].reset_index(drop=True)


def add_hour_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_year (0..8759)."""
    base_year = df["datetime"].dt.year.min()
    base = pd.Timestamp(year=base_year, month=1, day=1)

    dt = df["datetime"].dt.tz_localize(None)
    base_n = pd.Timestamp(year=base_year, month=1, day=1)

    hrs = (dt - base_n).dt.days * 24 + dt.dt.hour

    df2 = df.copy()
    df2["hour_of_year"] = hrs.astype(int)

    # Keep range 0..8759 only
    df2 = df2[(df2["hour_of_year"] >= 0) & (df2["hour_of_year"] < 8760)]
    return df2


# ---------------------------------------------------------
# Main computation
# ---------------------------------------------------------

def compute_stats(input_file: str, output_file: str):
    df = pd.read_csv(input_file)

    years = detect_years(df)
    ts_cols = find_timestamp_columns(df)

    frames = []

    for y in years:
        yf = build_year_frame(df, y, ts_cols)
        yf = drop_feb29(yf)
        yf = add_hour_of_year(yf)
        frames.append(yf)

    if not frames:
        raise RuntimeError("No per-year frames could be built.")

    all_years = pd.concat(frames, ignore_index=True)

    stats = (all_years
             .groupby("hour_of_year", as_index=False)
             .agg(WAT_mean=("WAT", "mean"),
                  WAT_std=("WAT", "std"),
                  NG_mean=("NG", "mean"),
                  NG_std=("NG", "std")))

    # Ensure a full 8760 row index
    full = pd.DataFrame({"hour_of_year": np.arange(8760)})
    merged = full.merge(stats, on="hour_of_year", how="left")

    merged.to_csv(output_file, index=False)
    print(f"Saved {output_file} with shape {merged.shape}")


# ---------------------------------------------------------
# Run directly in PyCharm
# ---------------------------------------------------------

if __name__ == "__main__":
    compute_stats(INPUT_FILE, OUTPUT_FILE)
