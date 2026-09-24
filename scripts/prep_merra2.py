#!/usr/bin/env python3
"""
Build a GraphCast input file from MERRA-2 (Step 7, OPERATIONAL_DA_PLAN.md §14)
===============================================================================
Writes a NetCDF with exactly the same schema, grid (1°, 181 x 360), 13 pressure levels and
datetimes as the ERA5 input file, with every prognostic/input variable taken from MERRA-2:

  GraphCast variable          MERRA-2 source (collection: variable)
  temperature                 inst3_3d_asm_Np: T
  geopotential                inst3_3d_asm_Np: H * g
  u/v_component_of_wind       inst3_3d_asm_Np: U, V
  vertical_velocity           inst3_3d_asm_Np: OMEGA          (Pa/s, same as ERA5)
  specific_humidity           inst3_3d_asm_Np: QV
  2m_temperature              inst1_2d_asm_Nx: T2M            (height-corrected to ERA5 orography)
  10m_u/v_component_of_wind   inst1_2d_asm_Nx: U10M, V10M
  mean_sea_level_pressure     inst1_2d_asm_Nx: SLP
  total_precipitation_6hr     tavg1_2d_flx_Nx: PRECTOT (or PRECTOTCORR), 6 h sum ending at t, in m

Kept from the ERA5 file (they are part of the model, not of the analysis):
  geopotential_at_surface, land_sea_mask (static) and toa_incident_solar_radiation (forcing).

Processing:
  1. Below-ground points (MERRA-2 is undefined below the surface) are filled level by level from
     the lowest valid level above: T with a 6.5 K/km lapse rate, geopotential hydrostatically
     consistent with that T, u/v/q/omega held constant (on all 42 MERRA-2 levels, before regridding).
     --below-ground era5 instead takes ERA5 wherever most of the 1° cell is below MERRA-2's surface.
  2. First-order conservative regridding 0.5° x 0.625° -> 1° (as WeatherBench-2 ERA5 1°).
  3. 2 m T is moved from MERRA-2's orography to ERA5's with 6.5 K/km (--t2m-height-correct 1).

Usage (after scripts/download_merra2.sh):
  python scripts/prep_merra2.py                                   # defaults: 2018-01-12 ERA5 file
  python scripts/prep_merra2.py --era5 data/era5/<file>.nc --precip PRECTOTCORR
Output: data/merra2/source-merra2_<rest of the ERA5 file name>
"""
import argparse
import datetime as dt
import glob
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

G, RD, GAMMA = 9.80665, 287.05, 0.0065
A_EXP = RD * GAMMA / G

ap = argparse.ArgumentParser(description="MERRA-2 -> GraphCast input file (same schema as the ERA5 file)")
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
ap.add_argument("--era5", default=None, help="ERA5 GraphCast input file that defines grid, levels and times")
ap.add_argument("--merra2-dir", default=None, help="Directory with MERRA2_*.nc4 files (default <proj>/data/merra2)")
ap.add_argument("--out", default=None)
ap.add_argument("--precip", default="PRECTOT", choices=["PRECTOT", "PRECTOTCORR"])
ap.add_argument("--below-ground", default="extrap", choices=["extrap", "era5"])
ap.add_argument("--t2m-height-correct", type=int, default=1)
args = ap.parse_args()

ERA5 = args.era5 or os.path.join(args.proj, "data", "era5",
                                 "source-era5_date-2018-01-12_res-1.0_levels-13_steps-24.nc")
M2DIR = args.merra2_dir or os.path.join(args.proj, "data", "merra2")
OUT = args.out or os.path.join(M2DIR, os.path.basename(ERA5).replace("source-era5", "source-merra2"))

print("=" * 76)
print("MERRA-2 -> GraphCast input")
print("=" * 76)
E = xr.load_dataset(ERA5, decode_timedelta=True)
DT = E["datetime"].values
DT = DT[0] if DT.ndim == 2 else DT
LAT, LON = E["lat"].values.astype(np.float64), E["lon"].values.astype(np.float64)
LEVELS = [int(x) for x in E["level"].values]
print(f"ERA5 template : {ERA5}")
print(f"frames        : {DT[0]} ... {DT[-1]} ({len(DT)})   levels: {LEVELS}")
print(f"MERRA-2 dir   : {M2DIR}")
print(f"output        : {OUT}")


def m2file(coll, day):
    pats = [f"MERRA2_*.{coll}.{day:%Y%m%d}.nc4", f"MERRA2.{coll}.{day:%Y%m%d}.nc4"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(M2DIR, "**", p), recursive=True))
        if hits:
            return hits[0]
    raise SystemExit(f"Missing MERRA-2 {coll} for {day:%Y-%m-%d} in {M2DIR} (run scripts/download_merra2.sh)")


def const_file():
    hits = sorted(glob.glob(os.path.join(M2DIR, "**", "MERRA2*.const_2d_asm_Nx.00000000.nc4"), recursive=True))
    return hits[0] if hits else None


_cache = {}


def open_day(coll, day, variables):
    key = (coll, day.date())
    if key not in _cache:
        if len(_cache) > 6:
            _cache.pop(next(iter(_cache))).close()
        _cache[key] = xr.open_dataset(m2file(coll, day))[variables]
    return _cache[key]


# ---------------------------------------------------------------------------------------------
# conservative regridding (separable for regular lat-lon grids)
# ---------------------------------------------------------------------------------------------
def _edges(c, lo=None, hi=None):
    e = np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[:-1] + c[1:]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
    if lo is not None:
        e = np.clip(e, lo, hi)
    return e


def lat_weights(src, tgt):
    se, te = np.sin(np.deg2rad(_edges(src, -90, 90))), np.sin(np.deg2rad(_edges(tgt, -90, 90)))
    ov = np.clip(np.minimum(te[1:, None], se[None, 1:]) - np.maximum(te[:-1, None], se[None, :-1]), 0, None)
    return ov / ov.sum(1, keepdims=True)


def lon_weights(src, tgt):
    se, te = _edges(src), _edges(tgt)
    ov = 0
    for k in (-360.0, 0.0, 360.0):
        ov = ov + np.clip(np.minimum(te[1:, None], se[None, 1:] + k) - np.maximum(te[:-1, None], se[None, :-1] + k),
                          0, None)
    return ov / ov.sum(1, keepdims=True)


_d0 = pd.Timestamp(DT[0]).to_pydatetime()
_g = xr.open_dataset(m2file("inst3_3d_asm_Np", _d0))
SLAT = _g["lat"].values.astype(np.float64)
SLON_RAW = _g["lon"].values.astype(np.float64)
PLEV = _g["lev"].values.astype(np.float64)                     # hPa, 1000 ... 0.1
_g.close()
LON_ORDER = np.argsort(np.mod(SLON_RAW, 360.0))
SLON = np.mod(SLON_RAW, 360.0)[LON_ORDER]
WLAT, WLON = lat_weights(SLAT, LAT), lon_weights(SLON, LON)
P_ASC = np.argsort(PLEV)                                        # low pressure (top) -> high pressure
missing = [p for p in LEVELS if p not in set(PLEV.astype(int))]
if missing:
    raise SystemExit(f"MERRA-2 Np levels lack {missing}")
print(f"MERRA-2 grid  : {len(SLAT)} x {len(SLON)}, {len(PLEV)} levels -> conservative to {len(LAT)} x {len(LON)}")


def regrid(f):
    """(..., nlat_src, nlon_src) in MERRA-2 order -> (..., 181, 360)."""
    f = np.asarray(f, dtype=np.float64)[..., LON_ORDER]
    return np.einsum("ij,...jk,lk->...il", WLAT, f, WLON, optimize=True).astype(np.float32)


def fill_below_ground(T, PHI, others):
    """In place, on arrays (nlev, ny, nx) ordered by ASCENDING pressure. NaN = below ground."""
    p = PLEV[P_ASC]
    for k in range(1, len(p)):
        m = ~np.isfinite(T[k])
        if not m.any():
            continue
        r = (p[k] / p[k - 1]) ** A_EXP
        Tk1 = T[k - 1][m]
        T[k][m] = Tk1 * r
        PHI[k][m] = PHI[k - 1][m] - G * Tk1 / GAMMA * (r - 1.0)
        for X in others:
            X[k][m] = X[k - 1][m]


PHIS_E = E["geopotential_at_surface"].values
PHIS_E = PHIS_E[0] if PHIS_E.ndim == 3 else PHIS_E
cf = const_file()
PHIS_M = None
if cf is not None:
    PHIS_M = regrid(xr.open_dataset(cf)["PHIS"].isel(time=0).values)
    d = (PHIS_M - PHIS_E) / G
    print(f"orography     : MERRA-2 - ERA5 at 1°: mean {d.mean():+.1f} m, rms {np.sqrt((d**2).mean()):.1f} m, "
          f"max |d| {np.abs(d).max():.0f} m")
elif args.t2m_height_correct:
    print("WARNING: no const_2d_asm_Nx file -> 2 m T height correction disabled")

LEV_IDX = [int(np.where(PLEV[P_ASC].astype(int) == p)[0][0]) for p in LEVELS]
OUTV = {v: np.full(E[v].shape, np.nan, np.float32) for v in
        ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
         "specific_humidity", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
         "mean_sea_level_pressure", "total_precipitation_6hr"] if v in E}
MAP3 = {"temperature": "T", "u_component_of_wind": "U", "v_component_of_wind": "V",
        "vertical_velocity": "OMEGA", "specific_humidity": "QV"}
MAP2 = {"2m_temperature": "T2M", "10m_u_component_of_wind": "U10M", "10m_v_component_of_wind": "V10M",
        "mean_sea_level_pressure": "SLP"}


def put(v, i, arr):
    """Store a (level,)lat,lon array at time index i, whatever the dim order of the ERA5 variable."""
    dims = list(E[v].dims)
    core = [d for d in dims if d not in ("batch", "time")]
    src = ["level", "lat", "lon"] if arr.ndim == 3 else ["lat", "lon"]
    a = np.transpose(arr, [src.index(d) for d in core])
    idx = tuple(0 if d == "batch" else i if d == "time" else slice(None) for d in dims)
    OUTV[v][idx] = a


t_all = time.time()
below_frac = []
for i, t64 in enumerate(DT):
    t = pd.Timestamp(t64).to_pydatetime()
    tt = np.datetime64(t)
    # --- 3-D ---
    d3 = open_day("inst3_3d_asm_Np", t, ["T", "U", "V", "QV", "H", "OMEGA"]).sel(time=tt)
    arr = {k: d3[k].values.astype(np.float64)[P_ASC] for k in ["T", "U", "V", "QV", "H", "OMEGA"]}
    PHI = arr["H"] * G
    bg = ~np.isfinite(arr["T"])
    fill_below_ground(arr["T"], PHI, [arr["U"], arr["V"], arr["QV"], arr["OMEGA"]])
    for X in [arr[k] for k in ("T", "U", "V", "QV", "OMEGA")] + [PHI]:   # anything still missing (should not happen)
        if not np.all(np.isfinite(X)):
            raise SystemExit(f"unfilled NaNs at {t}")
    frac = regrid(bg[LEV_IDX].astype(np.float32))
    below_frac.append(float((frac[LEVELS.index(1000)] > 0.5).mean()))
    R3 = {v: regrid(arr[k][LEV_IDX]) for v, k in MAP3.items() if v in OUTV}
    R3["geopotential"] = regrid(PHI[LEV_IDX])
    for v, a in R3.items():
        if args.below_ground == "era5":
            ev = E[v].isel(batch=0, time=i).transpose("level", "lat", "lon").values
            a = np.where(frac > 0.5, ev, a)
        put(v, i, a)
    # --- 2-D ---
    d2 = open_day("inst1_2d_asm_Nx", t, ["T2M", "U10M", "V10M", "SLP"]).sel(time=tt)
    for v, k in MAP2.items():
        if v in OUTV:
            f = regrid(d2[k].values)
            if v == "2m_temperature" and args.t2m_height_correct and PHIS_M is not None:
                f = f + GAMMA * (PHIS_M - PHIS_E) / G
            put(v, i, f)
    # --- 6 h precipitation ending at t: hourly means stamped HH:30 in (t-6h, t) ---
    if "total_precipitation_6hr" in OUTV:
        tot = 0.0
        n = 0
        for h in range(6):
            ts = t - dt.timedelta(hours=h, minutes=30)
            dp = open_day("tavg1_2d_flx_Nx", ts, [args.precip]).sel(time=np.datetime64(ts), method="nearest",
                                                                   tolerance=np.timedelta64(10, "m"))
            tot = tot + dp[args.precip].values.astype(np.float64)
            n += 1
        put("total_precipitation_6hr", i, np.clip(regrid(tot * 3600.0 / 1000.0), 0, None))   # kg m-2 s-1 -> m
    print(f"   {t:%Y-%m-%d %H:%M}  done ({time.time() - t_all:5.0f} s)", flush=True)

out = E.copy(deep=True)
for v, a in OUTV.items():
    if np.isnan(a).any():
        raise SystemExit(f"{v} has unfilled values")
    out[v] = E[v].copy(data=a)
out.attrs.update(source="MERRA-2 (GMAO) mapped to the GraphCast 1-deg / 13-level input schema by prep_merra2.py",
                 below_ground=args.below_ground, precip=args.precip,
                 t2m_height_correct=int(bool(args.t2m_height_correct and PHIS_M is not None)),
                 static_and_forcing="geopotential_at_surface, land_sea_mask, toa_incident_solar_radiation from ERA5",
                 era5_template=os.path.basename(ERA5))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_netcdf(OUT)
print(f"\nwrote {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB) in {time.time() - t_all:.0f} s")
print(f"1° cells mostly below MERRA-2's surface at 1000 hPa: {100 * np.mean(below_frac):.1f} %")

# --- sanity check: MERRA-2 minus ERA5 ---
lat2 = np.repeat(LAT[:, None], len(LON), 1)
lon2 = np.repeat(LON[None, :], len(LAT), 0)
lsm = E["land_sea_mask"].values
lsm = lsm[0] if lsm.ndim == 3 else lsm
conus = (lat2 >= 25) & (lat2 <= 50) & (lon2 >= 235) & (lon2 <= 295) & (lsm > 0.5)
w = np.cos(np.deg2rad(lat2))
print("\nMERRA-2 minus ERA5 (mean over frames): global area-weighted bias / rms | CONUS land bias / rms")


def stat(v, lev=None, scale=1.0):
    a = out[v].isel(batch=0)
    b = E[v].isel(batch=0)
    if lev is not None:
        a, b = a.sel(level=lev), b.sel(level=lev)
    d = (a - b).transpose("time", "lat", "lon").values / scale
    gb = (d * w).sum((1, 2)) / w.sum()
    gr = np.sqrt((d ** 2 * w).sum((1, 2)) / w.sum())
    cb = d[:, conus].mean(1)
    cr = np.sqrt((d[:, conus] ** 2).mean(1))
    name = v if lev is None else f"{v}@{lev}"
    print(f"   {name:32s} {gb.mean():+8.3f} / {gr.mean():7.3f}   | {cb.mean():+8.3f} / {cr.mean():7.3f}")


stat("2m_temperature")
stat("mean_sea_level_pressure", scale=100.0)
stat("temperature", 850)
stat("geopotential", 500, scale=G)
stat("u_component_of_wind", 250)
stat("specific_humidity", 850, scale=1e-3)
stat("total_precipitation_6hr", scale=1e-3)
print("   (units: K, hPa, K, m, m/s, g/kg, mm per 6 h)")
