#!/usr/bin/env python3
"""
Download ERA5 at 0.25 deg / 37 levels for full GraphCast (0.25 deg checkpoint)
==============================================================================
Source: Google ARCO-ERA5 (public, anonymous):
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
Writes a GraphCast-schema Zarr store frame by frame (so memory stays ~1-2 GB):
  data/era5/source-era5_date-<launch>_res-0.25_levels-37_steps-<N>.zarr
Frames: launch-6h, launch, launch+6h, ..., launch+6h*N   (N+2 frames)

Size: ~1 GB per frame uncompressed (37 levels x 6 variables x 721 x 1440 + surface);
the 72 h-cycling case (--steps 24, 26 frames) needs ~15-25 GB on disk. Put data/era5
on nobackup (see SETUP_NCCS_PRISM.md) before running.

Run on the LOGIN node (internet):
  python scripts/download_era5_arco025.py --date 2018-01-12 --time 12:00 --steps 24
"""
import argparse
import datetime as dt
import os
import shutil
import time

import numpy as np
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--date", default="2018-01-12")
ap.add_argument("--time", default="12:00")
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
ap.add_argument("--store", default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
ap.add_argument("--out", default=None)
args = ap.parse_args()

launch = dt.datetime.fromisoformat(f"{args.date}T{args.time}")
frames = [launch - dt.timedelta(hours=6) + dt.timedelta(hours=6 * i) for i in range(args.steps + 2)]
out = args.out or os.path.join(args.proj, "data", "era5",
                               f"source-era5_date-{args.date}_res-0.25_levels-37_steps-{args.steps:02d}.zarr")
os.makedirs(os.path.dirname(out), exist_ok=True)
print(f"frames {frames[0]} .. {frames[-1]} ({len(frames)})\n-> {out}")

LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
          500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
V3 = ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
      "vertical_velocity", "specific_humidity"]
V2 = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
      "mean_sea_level_pressure", "toa_incident_solar_radiation"]

src = xr.open_zarr(args.store, chunks=None,
                   storage_options=dict(token="anon") if args.store.startswith("gs://") else None)
src = src.rename({"latitude": "lat", "longitude": "lon"})
lat_asc = src.lat.values[0] > src.lat.values[-1]          # ARCO latitude runs 90 -> -90


def fix(da):
    da = da.sortby("lat") if lat_asc else da
    return da.astype(np.float32)


if os.path.exists(out):
    shutil.rmtree(out)

# static fields (first write)
t_first = np.datetime64(frames[0])
static = xr.Dataset()
for v in ("geopotential_at_surface", "land_sea_mask"):
    da = src[v]
    if "time" in da.dims:
        da = da.sel(time=t_first)
    static[v] = fix(da).drop_vars([c for c in fix(da).coords if c not in ("lat", "lon")])
static = static.load()

for i, t in enumerate(frames):
    t0 = time.time()
    tt = np.datetime64(t)
    fr = xr.Dataset()
    for v in V3:
        fr[v] = fix(src[v].sel(time=tt, level=LEVELS)).load()
    for v in V2:
        fr[v] = fix(src[v].sel(time=tt)).load()
    tp = src["total_precipitation"].sel(time=slice(np.datetime64(t - dt.timedelta(hours=5)), tt)).sum("time")
    fr["total_precipitation_6hr"] = fix(tp).load()
    fr = fr.drop_vars([c for c in fr.coords if c not in ("lat", "lon", "level")])
    fr = fr.expand_dims(time=[np.timedelta64(6 * i, "h").astype("timedelta64[ns]")]).expand_dims(batch=1)
    fr = fr.transpose("batch", "time", "level", "lat", "lon", missing_dims="ignore")
    for v in V2 + ["total_precipitation_6hr"]:
        fr[v] = fr[v].transpose("batch", "time", "lat", "lon")
    fr = fr.assign_coords(datetime=(("batch", "time"), np.array([[tt]], dtype="datetime64[ns]")))
    fr["level"] = fr["level"].astype(np.int32)
    if i == 0:
        enc = {"time": {"units": "hours", "dtype": "int64"},
               "datetime": {"units": "hours since 1970-01-01 00:00:00", "dtype": "int64"}}
        xr.merge([fr, static]).to_zarr(out, mode="w", encoding=enc)
    else:
        fr.to_zarr(out, append_dim="time")
    print(f"  frame {i + 1}/{len(frames)} {t:%Y-%m-%d %H} UTC  {time.time() - t0:.0f} s")

chk = xr.open_zarr(out, decode_timedelta=True)
print("done:", dict(chk.sizes), "\ndatetimes:", chk.datetime.values[0][[0, -1]])
