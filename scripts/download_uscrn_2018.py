#!/usr/bin/env python3
"""
Download & Parse NOAA USCRN Hourly Ground Truth Observations (2018)
====================================================================
Retrieves independent, un-assimilated station observations from the
U.S. Climate Reference Network (USCRN) for independent verification of:
  - 2m Air Temperature at +6h, +12h, +18h, +24h
  - Stations: ~140 pristine reference sites across CONUS

Prerequisites:
  - Run on LOGIN node (gpulogin1) with outbound internet.
  - No credentials or API keys required (public NOAA NCEI archive).

Usage:
  python scripts/download_uscrn_2018.py
  python scripts/download_uscrn_2018.py --date 2018-01-15
"""

import argparse
import io
import os
import sys
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser(description="Download and parse USCRN station observations")
parser.add_argument("--date", default="2018-01-15", help="Target date YYYY-MM-DD (default: 2018-01-15)")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/data/obs)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "data", "obs")
os.makedirs(OUTDIR, exist_ok=True)

YEAR = args.date.split("-")[0]
TAR_URL = f"https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{YEAR}/CRNH0203-{YEAR}.tar.gz"

print("=" * 65)
print(f"NOAA USCRN Station Observation Downloader ({args.date})")
print("=" * 65)
print(f"Target Year:      {YEAR}")
print(f"Target Date:      {args.date}")
print(f"Archive URL:      {TAR_URL}")
print(f"Output directory: {OUTDIR}")

tar_path = os.path.join(OUTDIR, f"CRNH0203-{YEAR}.tar.gz")
if not os.path.exists(tar_path):
    print("\n[1/3] Downloading USCRN CONUS annual archive (~60 MB) ...")
    try:
        urllib.request.urlretrieve(TAR_URL, tar_path)
        print(f"      Downloaded to: {tar_path}")
    except Exception as e:
        print(f"ERROR: Failed to download USCRN archive: {e}")
        sys.exit(1)
else:
    print(f"\n[1/3] USCRN archive already exists: {tar_path}")

# Target timestamps (UTC): 2018-01-15 12:00 through 2018-01-16 12:00
target_dates = [args.date.replace("-", ""), (pd.to_datetime(args.date) + pd.Timedelta(days=1)).strftime("%Y%m%d")]
print(f"\n[2/3] Extracting and filtering station observations for {target_dates} ...")

# Column specification according to USCRN hourly02 format
COL_NAMES = [
    "WBANNO", "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "CRX_VN",
    "LONGITUDE", "LATITUDE", "AIR_TEMPERATURE", "PRECIPITATION", "SOLAR_RADIATION",
    "SR_FLAG", "SURFACE_TEMPERATURE", "ST_TYPE", "ST_FLAG", "RELATIVE_HUMIDITY",
    "RH_FLAG", "SOIL_MOISTURE_5", "SOIL_TEMPERATURE_5", "WETNESS", "WET_FLAG",
    "WIND_1_5", "WIND_FLAG"
]

all_records = []

with tarfile.open(tar_path, "r:gz") as tar:
    members = [m for m in tar.getmembers() if m.name.endswith(".txt")]
    print(f"      Found {len(members)} station files in archive.")
    
    for m in members:
        f = tar.extractfile(m)
        if f is None:
            continue
        try:
            df = pd.read_csv(
                f,
                sep=r"\s+",
                names=COL_NAMES,
                usecols=["WBANNO", "UTC_DATE", "UTC_TIME", "LONGITUDE", "LATITUDE", "AIR_TEMPERATURE"],
                dtype={"WBANNO": str, "UTC_DATE": str, "UTC_TIME": str},
            )
            # Filter to target dates
            df_filtered = df[df["UTC_DATE"].isin(target_dates)].copy()
            if not df_filtered.empty:
                all_records.append(df_filtered)
        except Exception:
            continue

if not all_records:
    print("ERROR: No observations extracted for target dates.")
    sys.exit(1)

obs_df = pd.concat(all_records, ignore_index=True)
# Clean invalid values (-9999.0)
obs_df = obs_df[obs_df["AIR_TEMPERATURE"] > -90.0].copy()

# Convert Celsius to Kelvin
obs_df["T2M_K"] = obs_df["AIR_TEMPERATURE"] + 273.15

# Format datetime
obs_df["DATETIME_UTC"] = pd.to_datetime(obs_df["UTC_DATE"] + " " + obs_df["UTC_TIME"].str.zfill(4), format="%Y%m%d %H%M")

# Save clean tabular CSV
csv_out = os.path.join(OUTDIR, f"uscrn_stations_{args.date}.csv")
obs_df.to_csv(csv_out, index=False)

unique_stations = obs_df["WBANNO"].nunique()
print(f"\n[3/3] Processed {len(obs_df)} valid temperature observations across {unique_stations} stations.")
print(f"      Saved station table: {csv_out}")

print("\n" + "=" * 65)
print(f"SUCCESS: Independent Ground Verification Observations Ready!")
print(f"  USCRN CSV: {csv_out}")
print("=" * 65)
