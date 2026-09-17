#!/usr/bin/env python3
"""
Download & Format ERA5 for 2018 Case Date (GraphCast Format)
============================================================
Retrieves ERA5 pressure levels and single-level fields for the winter 2018 case:
  - Date: 2018-01-15 12:00 UTC (launch time t0)
  - Input frames (t-6h, t0): 2018-01-15 06:00 and 12:00 UTC
  - Forecast rollout targets (+6h, +12h, +18h, +24h): 18:00, 00:00, 06:00, 12:00 UTC
  - Grid: 1.0° × 1.0° global
  - Pressure levels (13 levels): [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

Static fields (land_sea_mask, geopotential_at_surface) are automatically copied
from the existing reference sample file to guarantee exact topography match.

Prerequisites:
  - Run on LOGIN node (gpulogin1) with outbound internet.
  - CDS API configured in ~/.cdsapirc.

Usage:
  python scripts/download_era5_2018.py
  python scripts/download_era5_2018.py --date 2018-01-15 --time 12:00
"""

import argparse
import datetime
import os
import sys
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Download and prepare ERA5 case dataset for GraphCast")
parser.add_argument("--date", default="2018-01-15", help="Target launch date (YYYY-MM-DD). Default: 2018-01-15")
parser.add_argument("--time", default="12:00", help="Launch UTC time (HH:MM). Default: 12:00")
parser.add_argument("--steps", type=int, default=4, help="Number of 6h forecast steps (default: 4 = 24h)")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/data/era5)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "data", "era5")
os.makedirs(OUTDIR, exist_ok=True)

SAMPLE_REF = os.path.join(PROJ, "data", "sample", "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc")

# ---------------------------------------------------------------------------
# 1. Setup Time Coordinates
# ---------------------------------------------------------------------------
t0 = datetime.datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M")
t_minus_1 = t0 - datetime.timedelta(hours=6)

# All timestamps needed: [t-6h, t0, t+6h, t+12h, ...]
all_datetimes = [t_minus_1 + datetime.timedelta(hours=6 * i) for i in range(2 + args.steps)]

print("=" * 65)
print("ERA5 Download & Preparation for GraphCast")
print("=" * 65)
print(f"Launch Date (t0): {t0.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Input times:      {t_minus_1.strftime('%Y-%m-%d %H:%M')}, {t0.strftime('%Y-%m-%d %H:%M')}")
print(f"Rollout horizon:  {args.steps} steps ({args.steps * 6} hours) -> {all_datetimes[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"Output directory: {OUTDIR}")

# Unique days and hours to request
req_years = sorted(list(set(dt.strftime("%Y") for dt in all_datetimes)))
req_months = sorted(list(set(dt.strftime("%m") for dt in all_datetimes)))
req_days = sorted(list(set(dt.strftime("%d") for dt in all_datetimes)))
req_times = sorted(list(set(dt.strftime("%H:00") for dt in all_datetimes)))

print(f"Requesting Days:  {req_days}, Months: {req_months}, Hours: {req_times}")

# ---------------------------------------------------------------------------
# 2. Check CDS API
# ---------------------------------------------------------------------------
try:
    import cdsapi
except ImportError:
    print("\nERROR: 'cdsapi' is not installed in the current environment.")
    print("Please run: pip install cdsapi")
    sys.exit(1)

cdsapirc = os.path.expanduser("~/.cdsapirc")
if not os.path.exists(cdsapirc):
    print(f"\nERROR: CDS credentials not found at {cdsapirc}.")
    print("Please create ~/.cdsapirc with your ECMWF Copernicus Climate Data Store UID and API key:")
    print("  url: https://cds.climate.copernicus.eu/api")
    print("  key: <YOUR-CDS-API-KEY>")
    sys.exit(1)

client = cdsapi.Client()

# ---------------------------------------------------------------------------
# 3. Download Pressure Levels Dataset
# ---------------------------------------------------------------------------
pl_file = os.path.join(OUTDIR, f"raw_pl_{args.date}.nc")
if not os.path.exists(pl_file):
    print("\n[1/3] Downloading 3D pressure levels from CDS ...")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "temperature",
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
                "specific_humidity",
            ],
            "pressure_level": [
                "50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"
            ],
            "year": req_years,
            "month": req_months,
            "day": req_days,
            "time": req_times,
            "grid": ["1.0", "1.0"],
        },
        pl_file,
    )
    print(f"      Saved: {pl_file}")
else:
    print(f"\n[1/3] Pressure levels file already exists: {pl_file}")

# ---------------------------------------------------------------------------
# 4. Download Single Levels Dataset
# ---------------------------------------------------------------------------
sfc_file = os.path.join(OUTDIR, f"raw_sfc_{args.date}.nc")
if not os.path.exists(sfc_file):
    print("\n[2/3] Downloading surface / single-level fields from CDS ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "total_precipitation",
                "toa_incident_solar_radiation",
            ],
            "year": req_years,
            "month": req_months,
            "day": req_days,
            "time": req_times,
            "grid": ["1.0", "1.0"],
        },
        sfc_file,
    )
    print(f"      Saved: {sfc_file}")
else:
    print(f"\n[2/3] Surface fields file already exists: {sfc_file}")

# ---------------------------------------------------------------------------
# 5. Format and Merge into GraphCast Dataset
# ---------------------------------------------------------------------------
print("\n[3/3] Formatting and packaging into GraphCast schema ...")

ds_pl = xr.open_dataset(pl_file)
ds_sfc = xr.open_dataset(sfc_file)

# Check and standardize coordinate names
coord_map = {
    "latitude": "lat",
    "longitude": "lon",
    "isobaricInhPa": "level",
    "valid_time": "time",
}
for old, new in coord_map.items():
    if old in ds_pl:
        ds_pl = ds_pl.rename({old: new})
    if old in ds_sfc:
        ds_sfc = ds_sfc.rename({old: new})

# Standardize variable names to match GraphCast
var_map_pl = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}
var_map_sfc = {
    "t2m": "2m_temperature",
    "2t": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "10u": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation_6hr",
    "tisr": "toa_incident_solar_radiation",
}

for old, new in var_map_pl.items():
    if old in ds_pl.data_vars and new not in ds_pl.data_vars:
        ds_pl = ds_pl.rename({old: new})
for old, new in var_map_sfc.items():
    if old in ds_sfc.data_vars and new not in ds_sfc.data_vars:
        ds_sfc = ds_sfc.rename({old: new})

# Merge 3D and 2D
ds_merged = xr.merge([ds_pl, ds_sfc], compat="override")

# Select exact required timestamps
time_coords = [np.datetime64(dt) for dt in all_datetimes]
ds_selected = ds_merged.sel(time=time_coords)

# Ensure batch dimension of size 1 exists
if "batch" not in ds_selected.dims:
    ds_selected = ds_selected.expand_dims("batch", axis=0)

# Attach static fields from reference sample
if os.path.exists(SAMPLE_REF):
    print(f"      Attaching static fields (land_sea_mask, geopotential_at_surface) from sample ...")
    with open(SAMPLE_REF, "rb") as f:
        ref_ds = xr.open_dataset(f)
        for static_var in ["land_sea_mask", "geopotential_at_surface"]:
            if static_var in ref_ds:
                ds_selected[static_var] = ref_ds[static_var]
else:
    print(f"      Warning: Reference sample {SAMPLE_REF} not found. Static fields will need to be added.")

# Save final NetCDF
out_nc = os.path.join(OUTDIR, f"source-era5_date-{args.date}_res-1.0_levels-13_steps-{args.steps:02d}.nc")
ds_selected.to_netcdf(out_nc)
print(f"\n{'=' * 65}")
print(f"SUCCESS: Real 2018 Case Dataset Prepared!")
print(f"  Saved to: {out_nc}")
print(f"  File size: {os.path.getsize(out_nc) / 1e6:.1f} MB")
print(f"You can now run forecasts on this real 2018 date:")
print(f"  python scripts/test_forecast.py --sample {out_nc}")
print(f"  python scripts/test_balanced_insertion.py --sample {out_nc}")
print("=" * 65)
