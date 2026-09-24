#!/usr/bin/env python3
"""
Hour-of-day climatologies of every GraphCast input variable, for anomaly initialization
=======================================================================================
Builds the mean and standard deviation of each GraphCast state variable (1°, 13 levels) for one
calendar month, per synoptic hour (00/06/12/18 UTC), over a range of years, from either

  --source merra2 : MERRA-2 daily files (local NCCS /css/merra2/MERRA2_all or data/merra2), converted
                    exactly as for the forecast inputs (scripts/prep_merra2.py: M2Converter)
  --source era5   : ERA5 1° from the public WeatherBench-2 archive (same source as the case file;
                    needs internet -> login node)

Both use the ERA5 case file as the grid/level template, so the two climatologies are directly
comparable. Output: data/clim/clim_<source>_m<MM>_<Y0>-<Y1>.nc with <var>_mean and <var>_std,
dims (hour, [level,] lat, lon).

Used by exp_main_real_obs.py --clim-era5 / --clim-provider for the anomaly-initialization arms
  M-MEAN: x = M - (clim_M - clim_E)                  (mean mapping)
  M-QM  : x = clim_E + (M - clim_M) * std_E / std_M  (mean + variance mapping)

Usage:
  python scripts/build_clim.py --source merra2 --years 2011-2017 --month 1 --workers 6   # compute node
  python scripts/build_clim.py --source era5   --years 2011-2017 --month 1 --workers 4   # login node
Keep the case year (2018) out of the years.
"""
import argparse
import calendar
import datetime as dt
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_merra2 import DEFAULT_ROOTS, M2Converter, STATE_2D, STATE_3D  # noqa: E402

ap = argparse.ArgumentParser(description="Hour-of-day monthly climatology of GraphCast inputs")
ap.add_argument("--source", required=True, choices=["merra2", "era5"])
ap.add_argument("--years", default="2011-2017", help="Y0-Y1 inclusive (exclude the case year)")
ap.add_argument("--month", type=int, default=1)
ap.add_argument("--hours", default="0,6,12,18")
ap.add_argument("--stride", type=int, default=1, help="Use every n-th day")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
ap.add_argument("--era5-template", default=None, help="ERA5 GraphCast input file (grid, levels, orography)")
ap.add_argument("--merra2-dir", default=None)
ap.add_argument("--precip", default="PRECTOT", choices=["PRECTOT", "PRECTOTCORR"])
ap.add_argument("--out", default=None)
args = ap.parse_args()

Y0, Y1 = [int(y) for y in args.years.split("-")]
HOURS = [int(h) for h in args.hours.split(",")]
TEMPLATE = args.era5_template or os.path.join(args.proj, "data", "era5",
                                              "source-era5_date-2018-01-12_res-1.0_levels-13_steps-24.nc")
M2DIR = args.merra2_dir or os.path.join(args.proj, "data", "merra2")
OUT = args.out or os.path.join(args.proj, "data", "clim", f"clim_{args.source}_m{args.month:02d}_{Y0}-{Y1}.nc")
VARS = STATE_3D + STATE_2D
DAYS = [dt.datetime(y, args.month, d) for y in range(Y0, Y1 + 1)
        for d in range(1, calendar.monthrange(y, args.month)[1] + 1)][::args.stride]
WB2 = "gs://weatherbench2/datasets/era5/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"

_T = None
_CONV = None
NOTES = []
_ZARR = None


def _template():
    global _T
    if _T is None:
        _T = xr.open_dataset(TEMPLATE, decode_timedelta=True)[["geopotential_at_surface", "land_sea_mask"]].load()
        full = xr.open_dataset(TEMPLATE, decode_timedelta=True)
        _T = _T.assign_coords(level=full["level"], lat=full["lat"], lon=full["lon"])
        full.close()
    return _T


def day_frames_merra2(day):
    global _CONV
    if _CONV is None:
        _CONV = M2Converter(_template(), [M2DIR] + DEFAULT_ROOTS, precip=args.precip, verbose=False,
                            grid_day=day)
    out = []
    for h in HOURS:
        fr = _CONV.frame(day + dt.timedelta(hours=h))
        fr.pop("_below_frac", None)
        ex = fr.pop("_nan_extra", None)
        if ex:
            NOTES.append(f"{day:%Y-%m-%d} {h:02d}Z {ex}")
        out.append(fr)
    _CONV.close()
    return out


def day_frames_era5(day):
    global _ZARR
    if _ZARR is None:
        _ZARR = xr.open_zarr(WB2, storage_options=dict(token="anon"))
    T = _template()
    lev = [int(x) for x in T["level"].values]
    times = [np.datetime64(day + dt.timedelta(hours=h)) for h in HOURS]
    d3 = _ZARR[STATE_3D].sel(time=times, level=lev).compute()
    d2 = _ZARR[[v for v in STATE_2D if v != "total_precipitation_6hr"]].sel(time=times).compute()
    tp = _ZARR["total_precipitation"].sel(time=slice(np.datetime64(day - dt.timedelta(hours=5)),
                                                      np.datetime64(day + dt.timedelta(hours=max(HOURS))))).compute()
    out = []
    for i, (h, t) in enumerate(zip(HOURS, times)):
        fr = {v: d3[v].isel(time=i).transpose("level", "latitude", "longitude").values.astype(np.float32)
              for v in STATE_3D}
        for v in STATE_2D:
            if v == "total_precipitation_6hr":
                w = tp.sel(time=slice(t - np.timedelta64(5, "h"), t))       # hours ending t-5h ... t
                fr[v] = w.sum("time").transpose("latitude", "longitude").values.astype(np.float32)
            else:
                fr[v] = d2[v].isel(time=i).transpose("latitude", "longitude").values.astype(np.float32)
        out.append(fr)
    return out


def work(day):
    try:
        frames = day_frames_merra2(day) if args.source == "merra2" else day_frames_era5(day)
    except SystemExit as e:
        return day, None, str(e)
    except Exception as e:                          # a missing/corrupt file skips the day, reported at the end
        return day, None, repr(e)
    s1 = [{v: fr[v].astype(np.float64) for v in VARS} for fr in frames]
    note = "; ".join(NOTES)
    NOTES.clear()
    return day, s1, note or None


def main():
    T = _template()
    lat, lon = T["lat"].values, T["lon"].values
    print("=" * 76)
    print(f"Climatology: {args.source}, month {args.month:02d}, {Y0}-{Y1}, hours {HOURS}, {len(DAYS)} days "
          f"(stride {args.stride}), {args.workers} workers")
    print(f"template {TEMPLATE}\noutput   {OUT}")
    print("=" * 76)
    S1 = [{} for _ in HOURS]
    S2 = [{} for _ in HOURS]
    n = 0
    bad = []
    t_ = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(work, d) for d in DAYS]
        for k, f in enumerate(as_completed(futs), 1):
            day, s1, err = f.result()
            if s1 is not None and err:
                print(f"   NOTE {err[:200]}", flush=True)
            if s1 is None:
                bad.append((day, err))
                print(f"   SKIP {day:%Y-%m-%d}: {err[:120]}", flush=True)
                continue
            for i in range(len(HOURS)):
                for v in VARS:
                    a = s1[i][v]
                    if v not in S1[i]:
                        S1[i][v] = np.zeros_like(a)
                        S2[i][v] = np.zeros_like(a)
                    S1[i][v] += a
                    S2[i][v] += a * a
            n += 1
            if k % 10 == 0 or k == len(DAYS):
                el = time.time() - t_
                print(f"   {k}/{len(DAYS)} days  ({el/60:.1f} min, ~{el/k*(len(DAYS)-k)/60:.0f} min left)", flush=True)
    if n == 0:
        raise SystemExit("no days processed")
    ds = xr.Dataset(coords=dict(hour=np.array(HOURS, np.int32), level=T["level"].values, lat=lat, lon=lon))
    for v in VARS:
        mean = np.stack([S1[i][v] / n for i in range(len(HOURS))])
        var = np.stack([S2[i][v] / n for i in range(len(HOURS))]) - mean ** 2
        dims = ("hour", "level", "lat", "lon") if mean.ndim == 4 else ("hour", "lat", "lon")
        ds[f"{v}_mean"] = (dims, mean.astype(np.float32))
        ds[f"{v}_std"] = (dims, np.sqrt(np.clip(var, 0, None)).astype(np.float32))
    ds.attrs.update(source=args.source, month=args.month, years=f"{Y0}-{Y1}", n_days=n, stride=args.stride,
                    hours=",".join(map(str, HOURS)), template=os.path.basename(TEMPLATE),
                    skipped=";".join(f"{d:%Y-%m-%d}" for d, _ in bad))
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB): {n} days, {len(bad)} skipped, "
          f"{(time.time()-t_)/60:.1f} min")
    g = ds["2m_temperature_mean"].mean(("lat", "lon")).values
    print("global-mean 2 m T by hour:", {h: round(float(x), 2) for h, x in zip(HOURS, g)})


if __name__ == "__main__":
    main()
