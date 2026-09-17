#!/usr/bin/env python3
"""
Download & Format ERA5 Case Dataset from Cloud (No CDS API Key Required)
========================================================================
Downloads ERA5 pressure levels and single-level fields for any historical date
directly from Google Cloud's public WeatherBench 2 archive:
  gs://weatherbench2/datasets/era5/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr

Features:
  - 100% Anonymous access: No ECMWF / CDS API account or credentials needed.
  - Direct 1.0° resolution (181 lat x 360 lon), matching GraphCast_small natively.
  - Slices exactly the 13 required pressure levels:
      [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa.
  - Computes exact 6-hour accumulated precipitation (total_precipitation_6hr).
  - Formats all dimensions, coordinates, and static fields to match the official
    DeepMind GraphCast NetCDF schema.

Prerequisites:
  - Run on LOGIN node (gpulogin1) with outbound internet.
  - conda environment with xarray, zarr, and gcsfs installed:
      pip install zarr gcsfs

Usage:
  python scripts/download_era5_cloud.py
  python scripts/download_era5_cloud.py --date 2018-01-15 --time 12:00 --steps 4
"""

import argparse
import datetime
import os
import sys
import time
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Download ERA5 case dataset from Cloud (WeatherBench 2)")
parser.add_argument("--date", default="2018-01-15", help="Launch date (YYYY-MM-DD). Default: 2018-01-15")
parser.add_argument("--time", default="12:00", help="Launch UTC time (HH:MM). Default: 12:00")
parser.add_argument("--steps", type=int, default=4, help="Forecast steps (6h each, default: 4 = 24h)")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--out", default=None, help="Explicit output NetCDF path")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = os.path.join(PROJ, "data", "era5")
os.makedirs(OUTDIR, exist_ok=True)

out_file = args.out or os.path.join(
    OUTDIR, f"source-era5_date-{args.date}_res-1.0_levels-13_steps-{args.steps:02d}.nc"
)
SAMPLE_REF = os.path.join(PROJ, "data", "sample", "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc")

# ---------------------------------------------------------------------------
# 1. Setup Time Targets
# ---------------------------------------------------------------------------
t0 = datetime.datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M")
t_minus_6 = t0 - datetime.timedelta(hours=6)

# GraphCast needs 2 input frames (t-6h, t0) followed by `steps` rollout frames (t+6h, t+12h, ...)
all_dts = [t_minus_6 + datetime.timedelta(hours=6 * i) for i in range(2 + args.steps)]
dt_strings = [dt.strftime("%Y-%m-%dT%H:%M:%S") for dt in all_dts]

print("=" * 68)
print("ERA5 Cloud Downloader for GraphCast (Anonymous Google Cloud Stream)")
print("=" * 68)
print(f"Target Launch Time (t0):  {t0.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Input frames (t-6h, t0):  {all_dts[0].strftime('%Y-%m-%d %H:%M')}, {all_dts[1].strftime('%Y-%m-%d %H:%M')}")
print(f"Forecast horizon:         {args.steps} steps ({args.steps * 6} hours) -> {all_dts[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"Total timestamps needed:  {len(all_dts)}")
print(f"Output NetCDF file:       {out_file}")

# ---------------------------------------------------------------------------
# 2. Connect to Cloud Archive (WeatherBench 2 Anonymous Zarr)
# ---------------------------------------------------------------------------
print("\n[1/5] Connecting to Google Cloud WeatherBench 2 archive ...")
cloud_zarr_url = "gs://weatherbench2/datasets/era5/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"

try:
    ds_cloud = xr.open_zarr(cloud_zarr_url, storage_options=dict(token="anon"))
except Exception as e:
    print(f"\nERROR: Failed to open cloud Zarr store: {e}")
    print("Ensure 'zarr' and 'gcsfs' are installed: pip install zarr gcsfs")
    sys.exit(1)

# Target 13 pressure levels for GraphCast_small
TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
target_times = [np.datetime64(s) for s in dt_strings]

# ---------------------------------------------------------------------------
# 3. Stream 3D Atmospheric Variables
# ---------------------------------------------------------------------------
print("\n[2/5] Streaming 3D pressure level fields (13 levels) ...")
t_start = time.time()
vars_3d = [
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
]

ds_3d_raw = ds_cloud[vars_3d].sel(time=target_times, level=TARGET_LEVELS).compute()
print(f"      Downloaded 3D variables in {time.time() - t_start:.1f} s ({ds_3d_raw.nbytes / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# 4. Stream 2D Surface Variables
# ---------------------------------------------------------------------------
print("\n[3/5] Streaming 2D surface / single-level fields ...")
t_start = time.time()
vars_2d = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "toa_incident_solar_radiation",
]

ds_2d_raw = ds_cloud[vars_2d].sel(time=target_times).compute()
print(f"      Downloaded 2D variables in {time.time() - t_start:.1f} s ({ds_2d_raw.nbytes / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# 5. Compute 6-Hour Accumulated Precipitation (total_precipitation_6hr)
# ---------------------------------------------------------------------------
print("\n[4/5] Computing 6-hour accumulated precipitation (total_precipitation_6hr) ...")
t_start = time.time()
tp_list = []
for dt_target in all_dts:
    dt_start = dt_target - datetime.timedelta(hours=5)
    # ERA5 1h precipitation slice for preceding 6 hours
    tp_window = ds_cloud["total_precipitation"].sel(
        time=slice(dt_start.strftime("%Y-%m-%dT%H:%M:%S"), dt_target.strftime("%Y-%m-%dT%H:%M:%S"))
    ).compute()
    tp_6h = tp_window.sum(dim="time")
    tp_list.append(tp_6h)

# Concatenate along time dimension
tp_6h_all = xr.concat(tp_list, dim=xr.DataArray(target_times, dims=["time"], name="time"))
print(f"      Precipitation computed in {time.time() - t_start:.1f} s")

# ---------------------------------------------------------------------------
# 6. Harmonize Coordinates & Build GraphCast Schema
# ---------------------------------------------------------------------------
print("\n[5/5] Formatting into GraphCast official schema ...")

# Ensure coordinate ordering and naming matches GraphCast:
# lat: [-90 to 90], lon: [0 to 359], level: int32 [50 to 1000]
lat_vals = ds_cloud["latitude"].values.astype(np.float32)
lon_vals = ds_cloud["longitude"].values.astype(np.float32)
level_vals = np.array(TARGET_LEVELS, dtype=np.int32)
time_coords = np.array([np.timedelta64(6 * i, "h") for i in range(len(all_dts))], dtype="timedelta64[ns]")
datetime_coords = np.array([[np.datetime64(dt) for dt in all_dts]], dtype="datetime64[ns]")

ds_out = xr.Dataset()

# Coordinates
ds_out.coords["lon"] = xr.DataArray(lon_vals, dims=["lon"])
ds_out.coords["lat"] = xr.DataArray(lat_vals, dims=["lat"])
ds_out.coords["level"] = xr.DataArray(level_vals, dims=["level"])
ds_out.coords["time"] = xr.DataArray(time_coords, dims=["time"])
ds_out.coords["datetime"] = xr.DataArray(datetime_coords, dims=["batch", "time"])

# Static variables
if os.path.exists(SAMPLE_REF):
    print("      Loading static topography & land mask from reference sample ...")
    with open(SAMPLE_REF, "rb") as f:
        ref_ds = xr.open_dataset(f)
        ds_out["geopotential_at_surface"] = xr.DataArray(
            ref_ds["geopotential_at_surface"].values.astype(np.float32),
            dims=["lat", "lon"],
        )
        ds_out["land_sea_mask"] = xr.DataArray(
            ref_ds["land_sea_mask"].values.astype(np.float32),
            dims=["lat", "lon"],
        )
else:
    print("      Extracting static topography & land mask from WeatherBench 2 ...")
    z_sfc = ds_cloud["geopotential_at_surface"].transpose("latitude", "longitude").values.astype(np.float32)
    lsm = ds_cloud["land_sea_mask"].transpose("latitude", "longitude").values.astype(np.float32)
    ds_out["geopotential_at_surface"] = xr.DataArray(z_sfc, dims=["lat", "lon"])
    ds_out["land_sea_mask"] = xr.DataArray(lsm, dims=["lat", "lon"])

# Add 2D variables: shape (batch=1, time, lat, lon)
for v in vars_2d:
    # WB2 ordering is (time, longitude, latitude) -> transpose to (time, latitude, longitude)
    arr = ds_2d_raw[v].transpose("time", "latitude", "longitude").values.astype(np.float32)
    ds_out[v] = xr.DataArray(arr[np.newaxis, ...], dims=["batch", "time", "lat", "lon"])

# Add total_precipitation_6hr
tp_arr = tp_6h_all.transpose("time", "latitude", "longitude").values.astype(np.float32)
ds_out["total_precipitation_6hr"] = xr.DataArray(tp_arr[np.newaxis, ...], dims=["batch", "time", "lat", "lon"])

# Add 3D variables: shape (batch=1, time, level, lat, lon)
for v in vars_3d:
    # WB2 ordering is (time, level, longitude, latitude) -> transpose to (time, level, latitude, longitude)
    arr = ds_3d_raw[v].transpose("time", "level", "latitude", "longitude").values.astype(np.float32)
    ds_out[v] = xr.DataArray(arr[np.newaxis, ...], dims=["batch", "time", "level", "lat", "lon"])

# Save NetCDF
print(f"\nWriting dataset to NetCDF: {out_file} ...")
ds_out.to_netcdf(out_file)
file_size_mb = os.path.getsize(out_file) / 1e6

print("=" * 68)
print("SUCCESS: Real Case ERA5 Dataset Downloaded & Formatted!")
print("=" * 68)
print(f"File path: {out_file}")
print(f"File size: {file_size_mb:.1f} MB")
print(f"Variables: {list(ds_out.data_vars.keys())}")
print(f"Dimensions: {dict(ds_out.dims)}")
print("\nYou can now run GraphCast directly on this real 2018 case on GPU:")
print(f"  python scripts/test_forecast.py --sample {out_file}")
print(f"  python scripts/test_balanced_insertion.py --sample {out_file}")
print("=" * 68)
