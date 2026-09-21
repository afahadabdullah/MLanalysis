#!/usr/bin/env python3
"""
Main experiment — insertion of REAL observations into a frozen ERA5-trained GraphCast
=====================================================================================
QUESTION
  What is the best way to insert new 2 m temperature information into GraphCast_small
  so that the forecast improves?  (surface-only, column, balanced, statistical, nudged)

BASE STATE that receives the observations (--base)
  bg    GraphCast's own 24 h forecast valid at t0-6h / t0 (from ERA5 at t0-30h/t0-24h):
        the operational-style background, i.e. the "company" case          [default]
  era5  ERA5 itself: can real observations improve even the training analysis?

INSERTED DATA (--obs-source)
  isd           NOAA ISD-Lite hourly ASOS/AWOS/SYNOP 2 m T (real stations)     [default]
  uscrn         NOAA USCRN reference stations (then USCRN is split used/withheld)
  merra2        MERRA-2 T2M (inst1_2d_asm_Nx) regridded to 1 deg, used as dense
                pseudo-stations on CONUS land points and analysed by OI
  merra2-field  MERRA-2 T2M replaces the base 2 m T on CONUS land directly (no OI)
  era5-synth    ERA5 2 m T + noise at random CONUS land points (twin control)
  Real stations: bilinear H operator, lapse-rate height correction (6.5 K/km) to model
  orography, gross/background QC, optional 1-deg super-obbing, 70/30 station split.

VERIFICATION
  * ERA5 analyses (gridded): CONUS 2 m T, CONUS T850, Z500 CONUS+downstream
  * USCRN stations (independent; never inserted unless --obs-source uscrn)
  * withheld stations of the inserted network
  "Gap closed" = (E_BASE - E_arm)/(E_BASE - E_ERA5); >100 % vs stations means the
  arm beat the forecast started from ERA5.

ARMS  ERA5 | BASE | DIR-1F | DIR-2F | COL-2F | BAL-2F | REG-2F | NUD-<type>
  DIR = 2 m T only; COL = + column T from a regression of background error profiles
  on the 2 m error (training region outside CONUS); BAL = COL + hypsometric Z;
  REG = COL + regressed Z; 1F/2F = insert at t0 only / at t0-6h and t0;
  NUD = step GraphCast from ERA5 at t0-24h, add alpha*increment after each 6 h step
  (only with --base bg).

DATA (login node):
  python scripts/download_era5_cloud.py --date 2018-01-14 --time 12:00 --steps 16
  python scripts/download_isd_lite.py   --start 2018-01-14T00 --end 2018-01-19T00
  python scripts/download_uscrn_range.py --start 2018-01-14T00 --end 2018-01-19T00
  (MERRA-2: point --merra2-dir at local NCCS copies of MERRA2_*.inst1_2d_asm_Nx.*.nc4)
RUN (GPU node):
  source activate_env.sh
  python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd
  python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source merra2 --merra2-dir <dir>
  python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --base era5
Outputs: runs/exp_main/<t0>_<source>_<base>/
"""

import argparse
import dataclasses
import datetime as dt
import functools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# GraphCast's graph message passing uses scatter-adds that are non-deterministic on GPU
# unless forced; without this, reruns differ by ~0.3 K and arm differences are noise.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()

import jax
import haiku as hk

try:
    from graphcast import (autoregressive, casting, checkpoint, data_utils,
                           graphcast, normalization, rollout)
except ImportError:  # weathernext layout
    from weathernext.weathernext1_graph import graphcast
    from weathernext.utils import (autoregressive, casting, checkpoint,
                                   data_utils, normalization, rollout)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

# =============================================================================
# 0. Arguments
# =============================================================================
ap = argparse.ArgumentParser(description="Exp 0 v3: cycling twin with model background")
ap.add_argument("--t0", default="2018-01-15T12:00", help="Launch time (UTC), ISO format")
ap.add_argument("--steps", type=int, default=12, help="Forecast steps of 6 h (default 12 = 72 h)")
ap.add_argument("--data", default=None, help="ERA5 NetCDF covering t0-30h ... t0+steps*6h")
ap.add_argument("--base", default="bg", choices=["bg", "era5"])
ap.add_argument("--obs-source", default="isd",
                choices=["isd", "uscrn", "merra2", "merra2-field", "era5-synth"])
ap.add_argument("--obs-file", default=None, help="Station CSV (sid,lat,lon,elev,time,t2m_K)")
ap.add_argument("--verify-file", default=None, help="USCRN CSV for independent verification")
ap.add_argument("--merra2-dir", default=None, help="Directory with MERRA2_*.inst1_2d_asm_Nx.*.nc4")
ap.add_argument("--superob", type=int, default=1, help="Average stations within a 1-deg cell (1/0)")
ap.add_argument("--max-dz", type=float, default=600.0, help="Reject stations |elev - model orog| > this (m)")
ap.add_argument("--allow-no-elev", action="store_true", help="Keep stations with unknown elevation (no height correction)")
ap.add_argument("--lapse", type=float, default=6.5, help="Lapse rate for station height correction (K/km)")
ap.add_argument("--verify-max-dz", type=float, default=150.0,
                help="Verification stations only where |station elev - model orog| < this (m); limits lapse-rate error")
ap.add_argument("--regress-region", default="conus-past", choices=["conus-past", "outside"],
                help="conus-past: CONUS land background errors at t0-18h/t0-12h; outside: NH land outside CONUS at t0-6h/t0")
ap.add_argument("--fix-weights", default="1.0,0.6,0.2", help="Fixed column weights at 1000,925,850 hPa for COL-FIX/BAL-FIX")
ap.add_argument("--n-obs", type=int, default=300, help="Pseudo-stations for era5-synth")
ap.add_argument("--withheld-frac", type=float, default=0.3)
ap.add_argument("--obs-noise", type=float, default=None, help="Obs error std (K); default by source")
ap.add_argument("--oi-L", type=float, default=250.0, help="OI Gaussian length scale (km)")
ap.add_argument("--col-top", type=int, default=500, help="Top level (hPa) of column increments")
ap.add_argument("--nud-types", default="DIR,BAL,FIX", help="Increment types used by nudging arms (DIR, BAL, FIX, REG)")
ap.add_argument("--nud-windows", default="6,24", help="Nudging window(s) in hours, comma-separated (e.g. '6', '24', or '6,24')")
ap.add_argument("--nud-tau", type=float, default=6.0, help="Nudging relaxation time (h)")
ap.add_argument("--nud-obs", default="all", choices=["all", "last2"],
                help="Nudge with obs at all cycles or only last 2")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
ap.add_argument("--outdir", default=None)
ap.add_argument("--dpi", type=int, default=150)
ap.add_argument("--skip-checks", action="store_true", help="Skip determinism/consistency checks")
args = ap.parse_args()

T0 = np.datetime64(dt.datetime.fromisoformat(args.t0))
t_launch = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=24)
PROJ = args.proj
DATA = args.data or os.path.join(
    PROJ, "data", "era5",
    f"source-era5_date-{t_launch:%Y-%m-%d}_res-1.0_levels-13_steps-16.nc")
TAG = dt.datetime.fromisoformat(args.t0).strftime("%Y%m%dT%H")
OUT = args.outdir or os.path.join(PROJ, "runs", "exp_main", f"{TAG}_{args.obs_source}_{args.base}")
if args.base == "era5" and args.nud_types:
    print("NOTE: nudging arms need --base bg; disabled for --base era5")
    args.nud_types = ""
SIGMA_O = args.obs_noise if args.obs_noise is not None else {
    "isd": 1.2, "uscrn": 1.0, "merra2": 0.8, "merra2-field": 0.8, "era5-synth": 0.5}[args.obs_source]
T_START = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=30)
T_END = dt.datetime.fromisoformat(args.t0) + dt.timedelta(hours=6 * args.steps)
OBS_DIR = os.path.join(PROJ, "data", "obs")


def _default_csv(prefix):
    import glob
    exact = os.path.join(OBS_DIR, f"{prefix}_{(T_START.replace(hour=0)):%Y%m%dT%H}_*.csv")
    c = sorted(glob.glob(exact)) or sorted(glob.glob(os.path.join(OBS_DIR, f"{prefix}_*.csv")))
    return c[0] if c else None
os.makedirs(OUT, exist_ok=True)
PARAMS = os.path.join(PROJ, "data", "params",
                      "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                      "mesh 2to5 - precipitation input and output.npz")
STATS = os.path.join(PROJ, "data", "stats")
RNG = np.random.default_rng(args.seed)
G, RD = 9.80665, 287.05

print("=" * 76)
print("MAIN EXPERIMENT — REAL-OBSERVATION INSERTION INTO GRAPHCAST")
print("=" * 76)
print(f"t0 = {args.t0}   steps = {args.steps} ({args.steps*6} h)   data = {DATA}")
print(f"base = {args.base}   obs = {args.obs_source} (sigma_o = {SIGMA_O} K)   out = {OUT}")

# =============================================================================
# 1. Model
# =============================================================================
print("\n[1] Loading GraphCast_small checkpoint and normalization stats ...")
with open(PARAMS, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params, model_config, task_config = ckpt.params, ckpt.model_config, ckpt.task_config
stats = {n: xr.load_dataset(os.path.join(STATS, f"{n}.nc")).compute()
         for n in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level"]}


def _wrapped(m_cfg, t_cfg):
    p = graphcast.GraphCast(m_cfg, t_cfg)
    p = casting.Bfloat16Cast(p)
    p = normalization.InputsAndResiduals(p, **stats)
    return autoregressive.Predictor(p, gradient_checkpointing=True)


@hk.transform_with_state
def _fwd(m_cfg, t_cfg, inputs, targets_template, forcings):
    return _wrapped(m_cfg, t_cfg)(inputs, targets_template=targets_template, forcings=forcings)


_jit = jax.jit(functools.partial(
    functools.partial(_fwd.apply, m_cfg=model_config, t_cfg=task_config),
    params=params, state={}))


def _run(rng, inputs, targets_template, forcings):
    return _jit(rng=rng, inputs=inputs, targets_template=targets_template, forcings=forcings)[0]

# =============================================================================
# 2. Data, time indexing, masks
# =============================================================================
print("\n[2] Loading ERA5 window ...")
try:
    DS = xr.load_dataset(DATA, decode_timedelta=True).compute()
except Exception:
    DS = xr.load_dataset(DATA).compute()

DATETIMES = DS.coords["datetime"].values
DATETIMES = DATETIMES[0] if DATETIMES.ndim == 2 else DATETIMES
hits = np.where(DATETIMES == T0)[0]
if len(hits) != 1:
    raise SystemExit(f"t0 {T0} not found in data times {DATETIMES[0]} ... {DATETIMES[-1]}")
I0 = int(hits[0])                 # index of t0
if I0 < 5:
    raise SystemExit("Data must start at t0-30h (need 5 frames before t0).")
if I0 + args.steps >= len(DATETIMES):
    raise SystemExit(f"Data ends at {DATETIMES[-1]}; need t0+{args.steps*6}h.")
print(f"   frames: {DATETIMES[0]} ... {DATETIMES[-1]}  (t0 index {I0})")

LATS, LONS = DS["lat"].values, DS["lon"].values
LEVELS = [int(x) for x in DS["level"].values]
LIDX = {p: i for i, p in enumerate(LEVELS)}
LATG, LONG = np.meshgrid(LATS, LONS, indexing="ij")
COSW = np.cos(np.deg2rad(LATG))
LSM = DS["land_sea_mask"].values if "land_sea_mask" in DS else np.ones_like(LATG)
LSM = LSM[0] if LSM.ndim == 3 else LSM

CONUS = (LATG >= 25) & (LATG <= 50) & (LONG >= 235) & (LONG <= 295)
CONUS_LAND = CONUS & (LSM > 0.5)
OI_BOX = (LATG >= 18) & (LATG <= 57) & (LONG >= 225) & (LONG <= 305)
DOWNSTREAM = (LATG >= 25) & (LATG <= 60) & (LONG >= 230) & (LONG <= 340)
ZSFC = DS["geopotential_at_surface"].values if "geopotential_at_surface" in DS else np.zeros_like(LATG)
ZSFC = ZSFC[0] if ZSFC.ndim == 3 else ZSFC
# regression region: NH mid-latitude land outside the CONUS/OI box, surface below ~1000 m
# (avoids points where 1000/925 hPa are below ground and only extrapolated)
TRAIN = (LATG >= 25) & (LATG <= 60) & (LSM > 0.5) & ~OI_BOX & (ZSFC < 1000.0 * 9.80665)


def window(start, nsteps):
    """inputs/targets/forcings for inputs at frames start,start+1 and nsteps targets."""
    sub = DS.isel(time=slice(start, start + 2 + nsteps))
    return data_utils.extract_inputs_targets_forcings(
        sub, target_lead_times=slice("6h", f"{nsteps*6}h"), **dataclasses.asdict(task_config))


_tmpl_inputs, _, _ = window(I0 - 1, 1)
STATE_VARS = [v for v in task_config.target_variables if v in _tmpl_inputs.data_vars]
REST = {v: [d for d in _tmpl_inputs[v].dims if d not in ("batch", "time")] for v in STATE_VARS}
print(f"   state variables: {STATE_VARS}")


def era5_frame(idx):
    return {v: DS[v].isel(time=idx).isel(batch=0).transpose(*REST[v]).values.astype(np.float32)
            for v in STATE_VARS}


def pred_frame(preds, k):
    out = {}
    for v in STATE_VARS:
        da = preds[v].isel(time=k)
        if "batch" in da.dims:
            da = da.isel(batch=0)
        out[v] = np.asarray(da.transpose(*REST[v]).values, dtype=np.float32)
    return out


def copy_frame(fr):
    return {v: a.copy() for v, a in fr.items()}


def set_pair(inp_t, A, B):
    inp = inp_t.copy(deep=True)
    for v in STATE_VARS:
        da = inp[v]
        order = ["batch", "time"] + REST[v]
        arr = da.transpose(*order).values.copy()
        arr[0, 0], arr[0, 1] = A[v], B[v]
        inp[v] = da.copy(data=np.transpose(arr, [order.index(d) for d in da.dims]))
    return inp


def forecast(A, B, start, nsteps):
    inp_t, tgt_t, frc = window(start, nsteps)
    inp = set_pair(inp_t, A, B)
    return rollout.chunked_prediction(_run, rng=jax.random.PRNGKey(0), inputs=inp,
                                      targets_template=tgt_t * np.nan, forcings=frc)


def wrms(field, mask):
    w = COSW * mask
    return float(np.sqrt(np.sum(w * field ** 2) / np.sum(w)))

# =============================================================================
# 3. Background: GraphCast 24 h forecast valid at t0-18 ... t0
# =============================================================================
print("\n[3] Background = GraphCast 24 h forecast from ERA5 at t0-30h/t0-24h ...")
S0 = I0 - 5
t_ = time.time()
bg_preds = forecast(era5_frame(S0), era5_frame(S0 + 1), S0, 4)
print(f"   done in {time.time()-t_:.1f} s (includes compile)")
BG_CHAIN = {S0 + 2 + k: pred_frame(bg_preds, k) for k in range(4)}   # frames t0-18..t0
BG_A, BG_B = BG_CHAIN[I0 - 1], BG_CHAIN[I0]
TRUE_A, TRUE_B = era5_frame(I0 - 1), era5_frame(I0)
if args.base == "era5":
    BASE_A, BASE_B = TRUE_A, TRUE_B
else:
    BASE_A, BASE_B = BG_A, BG_B
print(f"   background 2t error at t0 (CONUS land): {wrms(BG_B['2m_temperature']-TRUE_B['2m_temperature'], CONUS_LAND):.3f} K")

# =============================================================================
# 4. Observations (real or synthetic), H operator, QC and OI
# =============================================================================
print("\n[4] Observations ...")
ZMOD = ZSFC / G                                   # model orography (m)
GAMMA = args.lapse / 1000.0                       # K/m
OBS_IDX = [I0 - 3, I0 - 2, I0 - 1, I0]            # t0-18, -12, -6, t0


def interp2(field, la, lo):
    """Bilinear interpolation on the regular model grid (either latitude order)."""
    lo = np.mod(lo, 360.0)
    dla, dlo = LATS[1] - LATS[0], LONS[1] - LONS[0]
    fi, fj = (la - LATS[0]) / dla, (lo - LONS[0]) / dlo
    i0 = np.clip(np.floor(fi).astype(int), 0, len(LATS) - 2)
    j0 = np.floor(fj).astype(int) % len(LONS); j1 = (j0 + 1) % len(LONS)
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * (1 - wj) * field[i0, j0] + (1 - wi) * wj * field[i0, j1]
            + wi * (1 - wj) * field[i0 + 1, j0] + wi * wj * field[i0 + 1, j1])


def load_station_csv(path):
    d = pd.read_csv(path, parse_dates=["time"])
    d["lon"] = np.mod(d["lon"], 360.0)
    d = d[d.lat.between(24, 50) & d.lon.between(235, 294)]
    if not args.allow_no_elev:
        n0 = d.sid.nunique()
        d = d[np.isfinite(d.elev)]
        if d.sid.nunique() < n0:
            print(f"   dropped {n0 - d.sid.nunique()} stations with unknown elevation ({os.path.basename(path)})")
    return d


def load_merra2_t2m(times):
    import glob
    if not args.merra2_dir:
        raise SystemExit("--merra2-dir is required for merra2 sources")
    out = {}
    for t in times:
        t = pd.Timestamp(t)
        hits = glob.glob(os.path.join(args.merra2_dir, "**", f"MERRA2_*.inst1_2d_asm_Nx.{t:%Y%m%d}.nc4"),
                         recursive=True)
        if not hits:
            raise SystemExit(f"No MERRA-2 inst1_2d_asm_Nx file for {t:%Y-%m-%d} under {args.merra2_dir}")
        m = xr.open_dataset(hits[0])["T2M"].sel(time=t, method="nearest")
        m = m.assign_coords(lon=np.mod(m.lon, 360.0)).sortby("lon")
        # close the periodic seam, then bilinear to the 1-deg model grid
        m = xr.concat([m, m.isel(lon=0).assign_coords(lon=m.lon[0] + 360.0)], dim="lon")
        out[t] = m.interp(lat=xr.DataArray(LATS, dims="lat"), lon=xr.DataArray(LONS, dims="lon")).values
    return out


OBS_TIMES = [pd.Timestamp(DATETIMES[i]) for i in OBS_IDX]
M2FIELD = None
if args.obs_source == "era5-synth":
    cand = np.argwhere(CONUS_LAND)
    pick = cand[RNG.choice(len(cand), size=min(args.n_obs, len(cand)), replace=False)]
    rows = []
    for i in range(len(DATETIMES)):
        t = pd.Timestamp(DATETIMES[i])
        v = era5_frame(i)["2m_temperature"][pick[:, 0], pick[:, 1]] + RNG.normal(0, SIGMA_O, len(pick))
        rows.append(pd.DataFrame(dict(sid=[f"g{a}_{b}" for a, b in pick], lat=LATS[pick[:, 0]],
                                      lon=LONS[pick[:, 1]], elev=ZMOD[pick[:, 0], pick[:, 1]], time=t, t2m_K=v)))
    OBSDF = pd.concat(rows, ignore_index=True)
elif args.obs_source in ("merra2", "merra2-field"):
    M2 = load_merra2_t2m(OBS_TIMES)
    pts = np.argwhere(CONUS_LAND)
    rows = [pd.DataFrame(dict(sid=[f"m{a}_{b}" for a, b in pts], lat=LATS[pts[:, 0]], lon=LONS[pts[:, 1]],
                              elev=ZMOD[pts[:, 0], pts[:, 1]], time=t, t2m_K=M2[t][pts[:, 0], pts[:, 1]]))
            for t in OBS_TIMES]
    OBSDF = pd.concat(rows, ignore_index=True)
    if args.obs_source == "merra2-field":
        M2FIELD = {i: M2[t] for i, t in zip(OBS_IDX, OBS_TIMES)}
else:
    path = args.obs_file or _default_csv("isd_lite" if args.obs_source == "isd" else "uscrn")
    if not path or not os.path.exists(path):
        raise SystemExit(f"Observation CSV not found ({path}); run the downloader on the login node.")
    OBSDF = load_station_csv(path)
    print(f"   {args.obs_source}: {OBSDF.sid.nunique()} stations from {os.path.basename(path)}")

# height correction to model orography, and elevation QC
zm = interp2(ZMOD, OBSDF.lat.values, OBSDF.lon.values)
dz = OBSDF.elev.values - zm
OBSDF["y"] = OBSDF.t2m_K.values + np.where(np.isfinite(dz), GAMMA * dz, 0.0)
OBSDF = OBSDF[~(np.abs(np.nan_to_num(dz)) > args.max_dz)]

ALLDF = OBSDF.copy()                                  # all times (withheld verification at leads)
OBSDF = OBSDF[OBSDF.time.isin(OBS_TIMES)]              # insertion times only
# station split (by station, not by time)
sids = np.array(sorted(OBSDF.sid.unique()))
RNG.shuffle(sids)
n_hold = int(round(args.withheld_frac * len(sids)))
HOLD_SIDS, USE_SIDS = set(sids[:n_hold]), set(sids[n_hold:])
print(f"   stations after QC: {len(sids)} (used {len(USE_SIDS)}, withheld {len(HOLD_SIDS)})")
_hv = ALLDF[ALLDF.sid.isin(HOLD_SIDS)]
_hv = _hv.drop_duplicates("sid")
_dz = _hv.elev.values - interp2(ZMOD, _hv.lat.values, _hv.lon.values)
print(f"   withheld stations usable for verification (|dz| < {args.verify_max_dz:.0f} m): {int((np.abs(np.nan_to_num(_dz, nan=1e9)) < args.verify_max_dz).sum())}")

# independent verification network (USCRN) — never inserted
VER = None
if args.obs_source != "uscrn":
    vpath = args.verify_file or _default_csv("uscrn")
    if vpath and os.path.exists(vpath):
        VER = load_station_csv(vpath)
        vz = VER.elev.values - interp2(ZMOD, VER.lat.values, VER.lon.values)
        VER["y"] = VER.t2m_K.values + np.where(np.isfinite(vz), GAMMA * vz, 0.0)
        VER = VER[~(np.abs(np.nan_to_num(vz)) > args.max_dz)]
        _v1 = VER.drop_duplicates("sid"); _vz = _v1.elev.values - interp2(ZMOD, _v1.lat.values, _v1.lon.values)
        print(f"   verification: USCRN {VER.sid.nunique()} stations, {int((np.abs(_vz) < args.verify_max_dz).sum())} with |dz| < {args.verify_max_dz:.0f} m ({os.path.basename(vpath)})")
    else:
        print("   verification: no USCRN file found (station verification skipped)")
else:
    VER = ALLDF[ALLDF.sid.isin(HOLD_SIDS)].copy()

R_EARTH = 6371.0


def gc_dist(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat, dlon = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


_box = np.argwhere(OI_BOX)
OI_LOG = []


def obs_at(idx, which="use"):
    t = pd.Timestamp(DATETIMES[idx])
    d = OBSDF[(OBSDF.time == t) & OBSDF.sid.isin(USE_SIDS if which == "use" else HOLD_SIDS)]
    return d


def oi_increment(bg2t, idx, tag=""):
    """2 m T increment from the observations valid at frame idx."""
    if M2FIELD is not None:                                  # direct field replacement
        inc = np.where(CONUS_LAND, M2FIELD[idx] - bg2t, 0.0)
        OI_LOG.append(dict(tag=tag, time=str(DATETIMES[idx]), mode="field",
                           mean_innov=float(inc[CONUS_LAND].mean()),
                           rms_innov=float(np.sqrt((inc[CONUS_LAND] ** 2).mean()))))
        return inc.astype(np.float32)
    d = obs_at(idx)
    if d.empty:
        return np.zeros_like(bg2t)
    la, lo, y = d.lat.values, d.lon.values, d.y.values
    innov = y - interp2(bg2t, la, lo)
    keep = np.abs(innov) < 10.0                              # gross / background check
    la, lo, innov = la[keep], lo[keep], innov[keep]
    if args.superob:                                         # average within 1-deg cells
        ci = np.round((la - LATS[0]) / (LATS[1] - LATS[0])).astype(int)
        cj = np.round(np.mod(lo - LONS[0], 360) / (LONS[1] - LONS[0])).astype(int)
        g = pd.DataFrame(dict(c=ci * 10000 + cj, la=la, lo=lo, d=innov)).groupby("c").mean()
        la, lo, innov = g.la.values, g.lo.values, g.d.values
    so2 = SIGMA_O ** 2
    sb2 = max(float(np.var(innov)) - so2, 0.05)
    Coo = np.exp(-0.5 * (gc_dist(la[:, None], lo[:, None], la[None], lo[None]) / args.oi_L) ** 2)
    Cgo = np.exp(-0.5 * (gc_dist(LATS[_box[:, 0]][:, None], LONS[_box[:, 1]][:, None],
                                 la[None], lo[None]) / args.oi_L) ** 2)
    w = np.linalg.solve(sb2 * Coo + so2 * np.eye(len(innov)), innov)
    inc = np.zeros_like(bg2t)
    inc[_box[:, 0], _box[:, 1]] = sb2 * (Cgo @ w)
    OI_LOG.append(dict(tag=tag, time=str(DATETIMES[idx]), n_obs=int(len(innov)),
                       mean_innov=float(innov.mean()), rms_innov=float(np.sqrt((innov ** 2).mean())),
                       sigma_b=float(np.sqrt(sb2)), sigma_o=SIGMA_O))
    return inc.astype(np.float32)


def _vfilter(df):
    if df is None or df.empty:
        return df
    dz = df.elev.values - interp2(ZMOD, df.lat.values, df.lon.values)
    return df[~(np.abs(np.nan_to_num(dz, nan=1e9)) > args.verify_max_dz)]


def station_rmse(field2t, df):
    df = _vfilter(df)
    if df is None or df.empty:
        return np.nan
    return float(np.sqrt(np.mean((interp2(field2t, df.lat.values, df.lon.values) - df.y.values) ** 2)))


def station_bias(field2t, df):
    df = _vfilter(df)
    if df is None or df.empty:
        return np.nan
    return float(np.mean(interp2(field2t, df.lat.values, df.lon.values) - df.y.values))


# =============================================================================
# 5. Vertical regression (training region outside CONUS)
# =============================================================================
print("\n[5] Estimating vertical spreading by regression of background errors ...")
x_list, yT, yZ = [], {p: [] for p in LEVELS}, {p: [] for p in LEVELS}
if args.regress_region == "conus-past":
    # background errors over CONUS land at t0-18h and t0-12h (earlier cycles; no t0 information)
    REG_IDX, REG_MASK = (I0 - 3, I0 - 2), CONUS_LAND & (ZSFC < 1500.0 * G)
else:
    REG_IDX, REG_MASK = (I0 - 1, I0), TRAIN
print(f"   region: {args.regress_region}  ({int(REG_MASK.sum())} points x {len(REG_IDX)} times)")
for idx in REG_IDX:
    e = {v: era5_frame(idx)[v] - BG_CHAIN[idx][v] for v in ("2m_temperature", "temperature", "geopotential")}
    x_list.append(e["2m_temperature"][REG_MASK])
    for p in LEVELS:
        yT[p].append(e["temperature"][LIDX[p]][REG_MASK])
        yZ[p].append(e["geopotential"][LIDX[p]][REG_MASK])
X = np.concatenate(x_list); X = X - X.mean()
BT, BZ, R2T, R2Z = {}, {}, {}, {}
for p in LEVELS:
    for yy, B, R2 in ((np.concatenate(yT[p]), BT, R2T), (np.concatenate(yZ[p]), BZ, R2Z)):
        yy = yy - yy.mean()
        b = float(np.sum(X * yy) / np.sum(X * X))
        B[p] = b if p >= args.col_top else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            R2[p] = float(np.nan_to_num(np.corrcoef(X, yy)[0, 1] ** 2))
print("   level  b_T(K/K)  R2_T   b_Z(m2s-2/K)  R2_Z")
for p in sorted(LEVELS, reverse=True):
    print(f"   {p:5d}  {BT[p]:+7.3f}  {R2T[p]:.2f}   {BZ[p]:+9.2f}    {R2Z[p]:.2f}")

# =============================================================================
# 6. Increment operators
# =============================================================================


def hypsometric_phi(dT_by_level):
    """Integrate dPhi = Rd dT dln p upward from 1000 hPa (anchor dPhi(1000)=0)."""
    if not dT_by_level:
        return {}
    asc = sorted(LEVELS, reverse=True)
    dphi = {asc[0]: np.zeros_like(next(iter(dT_by_level.values())))}
    for pb, pt in zip(asc[:-1], asc[1:]):
        tl = 0.5 * (dT_by_level.get(pb, 0.0) + dT_by_level.get(pt, 0.0))
        dphi[pt] = dphi[pb] + RD * tl * np.log(pb / pt)
    return dphi


FIXW = dict(zip((1000, 925, 850), [float(x) for x in args.fix_weights.split(",")]))


def apply_increment(frame, inc2t, kind, scale=1.0):
    f = copy_frame(frame)
    inc = scale * inc2t
    f["2m_temperature"] += inc
    if kind in ("COL-FIX", "BAL-FIX", "FIX"):
        dT = {p: w * inc for p, w in FIXW.items() if p in LIDX}
        for p, v in dT.items():
            f["temperature"][LIDX[p]] += v
        if kind == "BAL-FIX":
            for p, v in hypsometric_phi(dT).items():
                f["geopotential"][LIDX[p]] += v
    if kind in ("COL", "BAL", "REG"):
        dT = {p: BT[p] * inc for p in LEVELS if BT[p] != 0.0}
        for p, v in dT.items():
            f["temperature"][LIDX[p]] += v
        if kind == "BAL":
            for p, v in hypsometric_phi(dT).items():
                f["geopotential"][LIDX[p]] += v
        if kind == "REG":
            for p in LEVELS:
                if BZ[p] != 0.0:
                    f["geopotential"][LIDX[p]] += BZ[p] * inc
    return f

# =============================================================================
# 7. Build initial pairs for every arm
# =============================================================================
print("\n[6] Building arms ...")
INC_A = oi_increment(BASE_A["2m_temperature"], I0 - 1, "static")
INC_B = oi_increment(BASE_B["2m_temperature"], I0, "static")
ARMS = {"ERA5": (TRUE_A, TRUE_B)}
if args.base == "bg":
    ARMS["BASE"] = (BG_A, BG_B)
else:
    ARMS["BASE"] = ARMS["ERA5"]
ARMS["DIR-1F"] = (BASE_A, apply_increment(BASE_B, INC_B, "DIR"))
for k in ("DIR", "COL", "BAL", "REG", "COL-FIX", "BAL-FIX"):
    ARMS[f"{k}-2F" if "FIX" not in k else f"{k}"] = (apply_increment(BASE_A, INC_A, k), apply_increment(BASE_B, INC_B, k))

ALPHA = 1.0 - np.exp(-6.0 / args.nud_tau)
nud_times = OBS_IDX if args.nud_obs == "all" else [I0 - 1, I0]


def nudge_chain(kind, alpha, window_h=24):
    if window_h == 6:
        # 6 h window: 1 step (t0-6h -> t0) with model dynamic adjustment
        if args.base == "bg":
            A = copy_frame(BG_CHAIN[I0 - 2])
            B = copy_frame(BG_CHAIN[I0 - 1])
        else:
            A = copy_frame(era5_frame(I0 - 2))
            B = copy_frame(era5_frame(I0 - 1))
        if alpha > 0:
            B = apply_increment(B, oi_increment(B["2m_temperature"], I0 - 1, f"nud6-{kind}"), kind, scale=alpha)
        C = pred_frame(forecast(A, B, I0 - 2, 1), 0)
        if alpha > 0:
            C = apply_increment(C, oi_increment(C["2m_temperature"], I0, f"nud6-{kind}"), kind, scale=alpha)
        return B, C
    elif window_h == 12:
        # 12 h window: 2 steps (t0-12h -> t0-6h -> t0)
        if args.base == "bg":
            A, B = copy_frame(BG_CHAIN[I0 - 3]), copy_frame(BG_CHAIN[I0 - 2])
        else:
            A, B = copy_frame(era5_frame(I0 - 3)), copy_frame(era5_frame(I0 - 2))
        for idx in [I0 - 1, I0]:
            C = pred_frame(forecast(A, B, idx - 2, 1), 0)
            if alpha > 0:
                C = apply_increment(C, oi_increment(C["2m_temperature"], idx, f"nud12-{kind}"), kind, scale=alpha)
            A, B = B, C
        return A, B
    elif window_h == 24:
        # 24 h window: 4 cycles (t0-18h, t0-12h, t0-6h, t0) from t0-30h/t0-24h
        A, B = era5_frame(S0), era5_frame(S0 + 1)
        for idx in OBS_IDX:
            C = pred_frame(forecast(A, B, idx - 2, 1), 0)
            if alpha > 0 and idx in nud_times:
                C = apply_increment(C, oi_increment(C["2m_temperature"], idx, f"nud24-{kind}"), kind, scale=alpha)
            A, B = B, C
        return A, B
    else:
        raise ValueError(f"Unsupported nudging window: {window_h}h (supported: 6, 12, 24)")


WINDOWS = [int(w.strip()) for w in args.nud_windows.split(",") if w.strip()]
for w in WINDOWS:
    for k in [x.strip() for x in args.nud_types.split(",") if x.strip()]:
        t_ = time.time()
        arm_key = f"NUD{w}-{k}" if (len(WINDOWS) > 1 or w != 24) else f"NUD-{k}"
        ARMS[arm_key] = nudge_chain(k, ALPHA, window_h=w)
        print(f"   {arm_key}: window={w}h, alpha={ALPHA:.2f} (tau={args.nud_tau} h), obs={args.nud_obs}, {time.time()-t_:.1f} s")

CHECKS = {}
if not args.skip_checks:
    print("\n[7] Consistency checks ...")
    chk_w = WINDOWS[0] if WINDOWS else 24
    A0, B0 = nudge_chain("DIR", 0.0, window_h=chk_w)
    dd = B0["2m_temperature"] - BG_B["2m_temperature"]
    CHECKS["stepwise_vs_rollout_max_2t_K"] = float(np.abs(dd).max())
    CHECKS["stepwise_vs_rollout_conus_rms_2t_K"] = wrms(dd, CONUS_LAND)
    p1 = forecast(TRUE_A, TRUE_B, I0 - 1, 2)
    p2 = forecast(TRUE_A, TRUE_B, I0 - 1, 2)
    dd = pred_frame(p1, 1)["2m_temperature"] - pred_frame(p2, 1)["2m_temperature"]
    CHECKS["rerun_max_2t_K"] = float(np.abs(dd).max())
    CHECKS["rerun_conus_rms_2t_K"] = wrms(dd, CONUS_LAND)
    CHECKS["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    for k, v in CHECKS.items():
        print(f"   {k}: {v:.3e}" if isinstance(v, float) else f"   {k}: {v}")
    if CHECKS["rerun_max_2t_K"] > 1e-3:
        print("   WARNING: reruns are not bit-identical; the NOISE curves in the figures give the floor.")

# =============================================================================
# 8. Forecasts and scores
# =============================================================================
print(f"\n[8] Running {len(ARMS)} forecasts x {args.steps} steps ...")
FC, ROWS = {}, []
for name, (A, B) in ARMS.items():
    t_ = time.time()
    FC[name] = forecast(A, B, I0 - 1, args.steps)
    print(f"   {name:8s} {time.time()-t_:5.1f} s")

NOISE_FC = forecast(TRUE_A, TRUE_B, I0 - 1, args.steps)        # rerun of ERA5 arm = run-to-run noise
LEADS = [6 * (k + 1) for k in range(args.steps)]
L850, L500 = LIDX[850], LIDX[500]
for name in ARMS:
    for k, lead in enumerate(LEADS):
        f, tr = pred_frame(FC[name], k), era5_frame(I0 + 1 + k)
        ROWS.append(dict(arm=name, lead_h=lead,
                         rmse_t2m_conus=wrms(f["2m_temperature"] - tr["2m_temperature"], CONUS_LAND),
                         rmse_t850_conus=wrms(f["temperature"][L850] - tr["temperature"][L850], CONUS),
                         rmse_z500_down=wrms((f["geopotential"][L500] - tr["geopotential"][L500]) / G, DOWNSTREAM),
                         bias_t2m_uscrn=station_bias(f["2m_temperature"],
                                                     None if VER is None else VER[VER.time == pd.Timestamp(DATETIMES[I0 + 1 + k])]),
                         rmse_t2m_uscrn=station_rmse(f["2m_temperature"],
                                                     None if VER is None else VER[VER.time == pd.Timestamp(DATETIMES[I0 + 1 + k])]),
                         rmse_t2m_withheld=station_rmse(f["2m_temperature"], ALLDF[(ALLDF.time == pd.Timestamp(DATETIMES[I0 + 1 + k]))
                                                                                   & ALLDF.sid.isin(HOLD_SIDS)])))
SC = pd.DataFrame(ROWS)
NOISE = pd.DataFrame([dict(lead_h=lead,
                           conus_rms_2t=wrms(pred_frame(NOISE_FC, k)["2m_temperature"] - pred_frame(FC["ERA5"], k)["2m_temperature"], CONUS_LAND))
                      for k, lead in enumerate(LEADS)])
NOISE.to_csv(os.path.join(OUT, "noise_floor.csv"), index=False)
for m in ("rmse_t2m_conus", "rmse_t850_conus", "rmse_z500_down", "rmse_t2m_uscrn", "rmse_t2m_withheld"):
    piv = SC.pivot(index="lead_h", columns="arm", values=m)
    for a in piv.columns:                                  # % error reduction relative to BASE
        SC.loc[SC.arm == a, m.replace("rmse", "impr")] = SC.loc[SC.arm == a, "lead_h"].map(
            100 * (piv["BASE"] - piv[a]) / piv["BASE"]).values
    gap = (piv["BASE"] - piv["ERA5"]).replace(0, np.nan)
    for a in piv.columns:
        SC.loc[SC.arm == a, m.replace("rmse", "gap")] = SC.loc[SC.arm == a, "lead_h"].map(
            (piv["BASE"] - piv[a]) / gap).values
SC.to_csv(os.path.join(OUT, "scores.csv"), index=False)

# ---- t0 diagnostics ----------------------------------------------------------
T0ROWS, PROF = [], {}
res_true = None


def hyps_residual(fr):
    z = fr["geopotential"]; t = fr["temperature"]
    th = (z[LIDX[850]] - z[LIDX[1000]])
    hy = RD * 0.5 * (t[LIDX[1000]] + t[LIDX[925]]) * np.log(1000 / 925) + \
        RD * 0.5 * (t[LIDX[925]] + t[LIDX[850]]) * np.log(925 / 850)
    return (th - hy) / G


res_true = hyps_residual(TRUE_B)
ERA_JUMP = wrms(era5_frame(I0 + 1)["2m_temperature"] - TRUE_B["2m_temperature"], CONUS)
for name, (A, B) in ARMS.items():
    e2 = B["2m_temperature"] - TRUE_B["2m_temperature"]
    f6 = pred_frame(FC[name], 0)
    T0ROWS.append(dict(
        arm=name,
        t0_err_t2m_conus=wrms(e2, CONUS_LAND),
        t0_err_withheld_obs=station_rmse(B["2m_temperature"], obs_at(I0, "hold")),
        t0_err_uscrn=station_rmse(B["2m_temperature"],
                                  None if VER is None else VER[VER.time == pd.Timestamp(DATETIMES[I0])]),
        t0_err_t850_conus=wrms(B["temperature"][L850] - TRUE_B["temperature"][L850], CONUS),
        hyps_resid_vs_era5_m=wrms(hyps_residual(B) - res_true, CONUS),
        first_step_jump_t2m=wrms(f6["2m_temperature"] - B["2m_temperature"], CONUS),
    ))
    PROF[name] = {"T": [wrms(B["temperature"][LIDX[p]] - TRUE_B["temperature"][LIDX[p]], CONUS) for p in LEVELS],
                  "Z": [wrms((B["geopotential"][LIDX[p]] - TRUE_B["geopotential"][LIDX[p]]) / G, CONUS) for p in LEVELS]}
T0DF = pd.DataFrame(T0ROWS)
T0DF["era5_6h_change_t2m"] = ERA_JUMP
T0DF.to_csv(os.path.join(OUT, "t0_diagnostics.csv"), index=False)

# retention of the t0 2 m increment (relative to BG forecast)
RET = {}
for name in ARMS:
    if name == "BASE":
        continue
    d0 = (ARMS[name][1]["2m_temperature"] - ARMS["BASE"][1]["2m_temperature"]) * CONUS
    nrm = np.sum(COSW * d0 ** 2)
    if nrm <= 0:
        continue
    RET[name] = [float(np.sum(COSW * (pred_frame(FC[name], k)["2m_temperature"]
                                       - pred_frame(FC["BASE"], k)["2m_temperature"]) * d0) / nrm)
                 for k in range(args.steps)]

# =============================================================================
# 9. Plots
# =============================================================================
print("\n[9] Plotting ...")
ORDER = ["ERA5", "BASE", "DIR-1F", "DIR-2F", "COL-2F", "BAL-2F", "REG-2F", "COL-FIX", "BAL-FIX"] + \
        [a for a in ARMS if a.startswith("NUD")]
for a in ARMS:
    if a not in ORDER:
        ORDER.append(a)
ORDER = [a for a in ORDER if a in ARMS]
COL = {"ERA5": "#222222", "BASE": "#9a9a9a", "DIR-1F": "#f4a3a3", "DIR-2F": "#d62728",
       "COL-2F": "#ff7f0e", "BAL-2F": "#1f77b4", "REG-2F": "#17becf", "COL-FIX": "#ffbb78", "BAL-FIX": "#9467bd",
       "NUD-DIR": "#e377c2", "NUD-COL": "#bcbd22", "NUD-BAL": "#2ca02c", "NUD-REG": "#8c564b", "NUD-FIX": "#17becf",
       "NUD6-DIR": "#f781bf", "NUD6-BAL": "#4daf4a", "NUD6-FIX": "#377eb8",
       "NUD12-DIR": "#e41a1c", "NUD12-BAL": "#984ea3", "NUD12-FIX": "#ff7f00",
       "NUD24-DIR": "#e377c2", "NUD24-BAL": "#2ca02c", "NUD24-FIX": "#9467bd"}
STY = {"ERA5": "--", "BASE": "--"}
EXT = [230, 300, 20, 55]


def mapax(fig, pos):
    if HAS_CARTOPY:
        ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.5")
        return ax, dict(transform=ccrs.PlateCarree())
    ax = fig.add_subplot(*pos)
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:])
    return ax, {}


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ {name}")


# (1) t0 maps: true background error, OI increment + stations, residual errors
fig = plt.figure(figsize=(17, 9))
lim = max(1.0, np.percentile(np.abs((TRUE_B["2m_temperature"] - BASE_B["2m_temperature"])[CONUS]), 98))
panels = [(f"ERA5 − BASE ({args.base}), 2 m T", TRUE_B["2m_temperature"] - BASE_B["2m_temperature"]),
          (f"Increment at t0 from {args.obs_source} (● used, ✕ withheld)", INC_B),
          ("Residual after DIR-2F (ERA5 − arm)", TRUE_B["2m_temperature"] - ARMS["DIR-2F"][1]["2m_temperature"]),
          ("ERA5 − BASE, T850", TRUE_B["temperature"][L850] - BASE_B["temperature"][L850]),
          ("Increment at T850: BAL-2F − BASE", ARMS["BAL-2F"][1]["temperature"][L850] - BASE_B["temperature"][L850]),
          ("Residual T850 after BAL-2F", TRUE_B["temperature"][L850] - ARMS["BAL-2F"][1]["temperature"][L850])]
for i, (title, fld) in enumerate(panels):
    ax, kw = mapax(fig, (2, 3, i + 1))
    im = ax.pcolormesh(LONS, LATS, fld, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", **kw)
    if i == 1 and M2FIELD is None:
        u, h_ = obs_at(I0, "use"), obs_at(I0, "hold")
        ax.scatter(u.lon, u.lat, s=5, c="k", **kw)
        ax.scatter(h_.lon, h_.lat, s=12, c="m", marker="x", **kw)
        if VER is not None and args.obs_source != "uscrn":
            v = VER[VER.time == pd.Timestamp(DATETIMES[I0])]
            ax.scatter(v.lon, v.lat, s=16, facecolors="none", edgecolors="g", **kw)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7, label="K")
fig.suptitle(f"Initial state at t0 = {args.t0} UTC — base {args.base}, data {args.obs_source} (○ = USCRN verification)", fontsize=13, weight="bold")
savefig(fig, "fig1_t0_maps.png")

# (2) vertical: regression coefficients and t0 error profiles
fig, axs = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
lv = np.array(LEVELS)
axs[0].plot([BT[p] for p in LEVELS], lv, "o-", color="#ff7f0e", label="b_T (K per K of 2 m T)")
axs[0].plot([R2T[p] for p in LEVELS], lv, ":", color="#ff7f0e", label="R² (T)")
axs[0].plot([BZ[p] / G / 10 for p in LEVELS], lv, "s-", color="#17becf", label="b_Z (10 m per K)")
axs[0].axhline(args.col_top, color="0.6", lw=0.8); axs[0].axvline(0, color="0.6", lw=0.8)
axs[0].set_title("Estimated vertical spreading\n(training region outside CONUS)")
axs[0].legend(fontsize=8)
for a in ORDER:
    axs[1].plot(PROF[a]["T"], lv, STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
    axs[2].plot(PROF[a]["Z"], lv, STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
axs[1].set_title("t0 temperature error vs ERA5 (CONUS RMS, K)")
axs[2].set_title("t0 height error vs ERA5 (CONUS RMS, m)")
axs[2].legend(fontsize=7, loc="upper right")
for ax in axs:
    ax.set_ylim(1000, 100); ax.grid(alpha=0.3)
axs[0].set_ylabel("Pressure (hPa)")
savefig(fig, "fig2_vertical_profiles.png")

# (3) forecast RMSE vs lead (3 metrics)
fig, axs = plt.subplots(1, 5, figsize=(27, 5))
for ax, m, ttl in zip(axs, ["rmse_t2m_withheld", "rmse_t2m_uscrn", "rmse_t2m_conus", "rmse_t850_conus", "rmse_z500_down"],
                      ["2 m T RMSE vs WITHHELD stations (K)", "2 m T RMSE vs USCRN stations (K)",
                       "CONUS-land 2 m T RMSE vs ERA5 grid (K)", "CONUS T850 RMSE vs ERA5 (K)",
                       "Z500 RMSE vs ERA5, CONUS+downstream (m)"]):
    for a in ORDER:
        s = SC[SC.arm == a]
        ax.plot(s.lead_h, s[m], STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
    ax.set_title(ttl); ax.set_xlabel("Lead (h)"); ax.set_xticks(LEADS); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle("Forecast error — observations (left two panels, primary for real data) and ERA5 analyses", weight="bold")
savefig(fig, "fig3_rmse_vs_lead.png")

# (4) gap closed
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
nb = NOISE.set_index("lead_h")["conus_rms_2t"]
for ax, m, ttl in zip(axs, ["impr_t2m_withheld", "impr_t2m_uscrn", "impr_t2m_conus"],
                      ["withheld stations", "USCRN", "ERA5 grid"]):
    for a in ORDER:
        if a == "BASE":
            continue
        s = SC[SC.arm == a]
        ax.plot(s.lead_h, s[m], STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
    if m == "impr_t2m_conus":
        base_rmse = SC[SC.arm == "BASE"].set_index("lead_h")["rmse_t2m_conus"]
        band = 100 * nb / base_rmse
        ax.fill_between(band.index, -band.values, band.values, color="0.85", label="run-to-run noise")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"2 m T error reduction vs BASE (%) — {ttl}")
    ax.set_xlabel("Lead (h)"); ax.set_xticks(LEADS); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
savefig(fig, "fig4_improvement_vs_base.png")

# (5) retention of the t0 increment
fig, ax = plt.subplots(figsize=(8, 5))
for a in ORDER:
    if a in RET:
        ax.plot([0] + LEADS, [1] + RET[a], STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("Lead (h)"); ax.set_ylabel("R(t)")
ax.set_title("Retention of the t0 2 m T increment (vs BASE forecast)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
savefig(fig, "fig5_retention.png")

# (6) t0 consistency and accuracy bars
fig, axs = plt.subplots(1, 5, figsize=(22, 4.5))
metrics = [("t0_err_withheld_obs", "t0 error at WITHHELD stations (K)"),
           ("t0_err_uscrn", "t0 error at USCRN stations (K)"),
           ("t0_err_t850_conus", "t0 T850 error (K)"),
           ("hyps_resid_vs_era5_m", "Hypsometric residual vs ERA5 (m)"),
           ("first_step_jump_t2m", "First-step 2 m T jump (K)")]
for ax, (m, ttl) in zip(axs, metrics):
    vals = [float(T0DF.loc[T0DF.arm == a, m].iloc[0]) for a in ORDER]
    ax.bar(range(len(ORDER)), vals, color=[COL.get(a, "k") for a in ORDER])
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=60, fontsize=8)
    ax.set_title(ttl, fontsize=10); ax.grid(axis="y", alpha=0.3)
    if m == "first_step_jump_t2m":
        ax.axhline(ERA_JUMP, color="k", ls="--", lw=1, label="ERA5 6 h change")
        ax.legend(fontsize=8)
savefig(fig, "fig6_t0_consistency.png")

# (7) day-1 and day-3 error maps
show = [a for a in ["BASE", "DIR-2F", "COL-FIX", "NUD6-BAL", "NUD24-BAL", "NUD-BAL", "ERA5"] if a in ARMS]
leads_show = [k for k in (3, 11) if k < args.steps]
fig = plt.figure(figsize=(4.2 * len(show), 3.6 * len(leads_show)))
for r, k in enumerate(leads_show):
    tr = era5_frame(I0 + 1 + k)["2m_temperature"]
    for c, a in enumerate(show):
        ax, kw = mapax(fig, (len(leads_show), len(show), r * len(show) + c + 1))
        fld = pred_frame(FC[a], k)["2m_temperature"] - tr
        im = ax.pcolormesh(LONS, LATS, fld, cmap="RdBu_r", vmin=-4, vmax=4, shading="auto", **kw)
        ax.set_title(f"{a}  +{LEADS[k]} h  rms={wrms(fld, CONUS):.2f} K", fontsize=9)
fig.colorbar(im, ax=fig.axes, shrink=0.6, label="forecast − ERA5, 2 m T (K)")
savefig(fig, "fig7_error_maps.png")

# =============================================================================
# 10. Summary
# =============================================================================
summ = dict(t0=args.t0, steps=args.steps, data=DATA, base=args.base, obs_source=args.obs_source,
            n_stations_used=len(USE_SIDS), n_withheld=len(HOLD_SIDS),
            n_uscrn=0 if VER is None else int(VER.sid.nunique()), obs_noise_K=SIGMA_O, oi_L_km=args.oi_L, col_top_hPa=args.col_top,
            nud_tau_h=args.nud_tau, nud_alpha=float(ALPHA), nud_obs=args.nud_obs,
            regression={"b_T": BT, "R2_T": R2T, "b_Z": BZ, "R2_Z": R2Z}, oi_log=OI_LOG, checks=CHECKS)
key = SC[SC.lead_h.isin([6, 24, 48, 72])].pivot(index="arm", columns="lead_h", values="gap_t2m_conus")
summ["gap_closed_t2m"] = {a: {int(k): (None if pd.isna(v) else round(100 * float(v), 1))
                              for k, v in row.items()} for a, row in key.iterrows()}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summ, f, indent=2, default=float)

print("\n" + "=" * 76)
print("t0 DIAGNOSTICS")
print(T0DF.set_index("arm").loc[ORDER].round(3).to_string())
key_u = SC[SC.lead_h.isin([6, 24, 48, 72])].pivot(index="arm", columns="lead_h", values="rmse_t2m_uscrn")
summ["rmse_t2m_uscrn"] = {a: {int(k): (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
                          for a, row in key_u.iterrows()}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summ, f, indent=2, default=float)
print("\n2 m T RMSE vs USCRN (K)")
print(key_u.loc[[a for a in ORDER if a in key_u.index]].round(3).to_string())
LSHOW = [l for l in (6, 12, 24, 48, 72) if l in LEADS]
def _tab(col, fmt=2):
    t = SC[SC.lead_h.isin(LSHOW)].pivot(index="arm", columns="lead_h", values=col)
    return t.loc[[a for a in ORDER if a in t.index]].round(fmt).to_string()
print("\n2 m T RMSE vs WITHHELD stations (K)   <- primary for real observations")
print(_tab("rmse_t2m_withheld", 3))
print("\n2 m T error reduction vs BASE on WITHHELD stations (%)")
print(_tab("impr_t2m_withheld", 1))
if VER is not None and args.obs_source != "uscrn":
    print("\n2 m T error reduction vs BASE on USCRN (%)")
    print(_tab("impr_t2m_uscrn", 1))
print("\n2 m T error reduction vs BASE on ERA5 grid (%)   noise floor (K):",
      ", ".join(f"{int(r.lead_h)}h {r.conus_rms_2t:.3f}" for r in NOISE.itertuples() if r.lead_h in LSHOW))
print(_tab("impr_t2m_conus", 1))
if args.base == "era5":
    key = None
print("\nGAP CLOSED, CONUS 2 m T vs ERA5 (%)  [0 = BASE, 100 = ERA5 start]")
print("   (not defined for --base era5)" if args.base == "era5" else
      (100 * key.loc[[a for a in ORDER if a in key.index]]).round(1).to_string())
print("=" * 76)
print(f"Outputs in {OUT}")
