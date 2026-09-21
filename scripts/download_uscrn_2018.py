#!/usr/bin/env python3
"""
Download & Parse NOAA USCRN Hourly Ground Station Observations (2018)
======================================================================
Retrieves independent, un-assimilated station observations from the
U.S. Climate Reference Network (USCRN) for the target winter 2018 case.

Prerequisites:
  - Run on LOGIN node (gpulogin1) with outbound internet access.
  - No credentials or API keys required (public NOAA NCEI archive).

Usage:
  python scripts/download_uscrn_2018.py
  python scripts/download_uscrn_2018.py --date 2018-01-15
"""

import argparse
import concurrent.futures
import io
import os
import re
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Download & parse USCRN ground observations")
parser.add_argument("--date", default="2018-01-15", help="Target launch date YYYY-MM-DD (default: 2018-01-15)")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/data/obs)")
parser.add_argument("--workers", type=int, default=16, help="Concurrent download workers (default: 16)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "data", "obs")
os.makedirs(OUTDIR, exist_ok=True)

YEAR = args.date.split("-")[0]
BASE_URL = f"https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{YEAR}/"
csv_out = os.path.join(OUTDIR, f"uscrn_stations_{args.date}.csv")

# Target dates: launch day and rollout day (e.g. 2018-01-15 and 2018-01-16)
dt_t0 = pd.to_datetime(args.date)
target_dates = {
    dt_t0.strftime("%Y%m%d"),
    (dt_t0 + pd.Timedelta(days=1)).strftime("%Y%m%d"),
}

print("=" * 68)
print(f"NOAA USCRN Ground Station Downloader ({args.date})")
print("=" * 68)
print(f"Source URL:       {BASE_URL}")
print(f"Target Dates:     {sorted(list(target_dates))}")
print(f"Output CSV:       {csv_out}")
print(f"Workers:          {args.workers}")

# 1. Fetch Station File Directory
print("\n[1/3] Scanning NOAA NCEI station directory ...")
t0 = time.time()
try:
    req = urllib.request.Request(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
        files = re.findall(r'href=[\"\'](CRNH0203-[^\"\']+\.txt)[\"\']', html)
    files = sorted(list(set(files)))
    print(f"      Found {len(files)} station files in directory ({time.time() - t0:.2f} s)")
except Exception as e:
    print(f"ERROR: Failed to access NOAA directory: {e}")
    sys.exit(1)

# 2. Parallel Stream & Extract Matching Records
print(f"\n[2/3] Streaming and filtering observations across {len(files)} stations ...")
COL_NAMES = [
    "WBANNO", "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "CRX_VN",
    "LONGITUDE", "LATITUDE", "AIR_TEMPERATURE", "PRECIPITATION", "SOLAR_RADIATION",
    "SR_FLAG", "SURFACE_TEMPERATURE", "ST_TYPE", "ST_FLAG", "RELATIVE_HUMIDITY",
    "RH_FLAG", "SOIL_MOISTURE_5", "SOIL_TEMPERATURE_5", "WETNESS", "WET_FLAG",
    "WIND_1_5", "WIND_FLAG",
]

def fetch_and_filter_station(fname):
    url = BASE_URL + fname
    matched_lines = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            for line in resp:
                parts = line.decode("utf-8", errors="ignore").split()
                if len(parts) > 2 and parts[1] in target_dates:
                    matched_lines.append(line.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    return matched_lines

t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    results = list(executor.map(fetch_and_filter_station, files))

all_lines = [line for sublist in results for line in sublist]
print(f"      Retrieved {len(all_lines)} raw hourly lines in {time.time() - t_start:.2f} s")

# 3. Clean and Structure DataFrame
print("\n[3/3] Parsing station coordinates and temperatures ...")
if not all_lines:
    print("ERROR: No observation records matched target dates.")
    sys.exit(1)

df = pd.read_csv(
    io.StringIO("".join(all_lines)),
    sep=r"\s+",
    names=COL_NAMES,
    usecols=["WBANNO", "UTC_DATE", "UTC_TIME", "LONGITUDE", "LATITUDE", "AIR_TEMPERATURE"],
    dtype={"WBANNO": str, "UTC_DATE": str, "UTC_TIME": str},
)

# Filter missing values (-9999.0)
df = df[df["AIR_TEMPERATURE"] > -90.0].copy()

# Filter CONUS bounding box (24-50 deg N, -125 to -66 deg E)
df = df[
    (df["LATITUDE"] >= 24.0)
    & (df["LATITUDE"] <= 50.0)
    & (df["LONGITUDE"] >= -125.0)
    & (df["LONGITUDE"] <= -66.0)
].copy()

# Convert Celsius to Kelvin
df["T2M_K"] = (df["AIR_TEMPERATURE"] + 273.15).round(3)

# Longitude in [0, 360) format matching GraphCast
df["LON_360"] = ((df["LONGITUDE"] + 360.0) % 360.0).round(4)

# Format Datetime
df["DATETIME_UTC"] = pd.to_datetime(
    df["UTC_DATE"] + " " + df["UTC_TIME"].str.zfill(4), format="%Y%m%d %H%M"
)

# Save to CSV
df.to_csv(csv_out, index=False)

unique_stations = df["WBANNO"].nunique()
print("=" * 68)
print("SUCCESS: Real CONUS Ground Observations Downloaded & Processed!")
print("=" * 68)
print(f"Output File:      {csv_out}")
print(f"File Size:        {os.path.getsize(csv_out) / 1e3:.1f} KB")
print(f"Total Records:    {len(df)}")
print(f"Unique Stations:  {unique_stations} pristine reference sites across CONUS")
print(f"Time Coverage:    {df['DATETIME_UTC'].min()} to {df['DATETIME_UTC'].max()}")
print("\nSample stations:")
print(df[["WBANNO", "DATETIME_UTC", "LATITUDE", "LON_360", "AIR_TEMPERATURE", "T2M_K"]].head(5).to_string(index=False))
print("=" * 68)
