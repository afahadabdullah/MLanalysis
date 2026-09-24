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

ARMS  ERA5 | BASE | BIAS | DIR-1F | DIR-2F | COL-2F | BAL-2F | REG-2F | COL/BAL-FIX | COL/BAL-PBL |
      IAU-<type> | NUD<window>-<type>   (see OPERATIONAL_DA_PLAN.md: M0-M5)
      with --provider-file (a foreign analysis in GraphCast format, e.g. MERRA-2 from prep_merra2.py):
      M-DIR  (forecast started directly from the provider at t0-6h/t0) |
      REPLAY<H>-M / HYB<H>-M  (72 h cycling relaxed to the provider instead of ERA5, without / with stations) |
      HYB<H>-4DV-M (4D-Var on the provider-cycled background);
      with --clim-era5/--clim-provider (anomaly initialization, no retraining):
      M-MEAN (M - (clim_M - clim_E)) | M-QM (clim_E + (M - clim_M) std_E/std_M) | REPLAY<H>-MQM | HYB<H>-MQM | HYB<H>-4DV-MQM
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
  MERRA-2 as the provider analysis:  bash scripts/download_merra2.sh && python scripts/prep_merra2.py
  then add  --provider-file data/merra2/source-merra2_date-2018-01-12_res-1.0_levels-13_steps-24.nc
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
import sys
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
ap.add_argument("--max-dz", type=float, default=None, help="(deprecated; use --dz-below/--dz-above)")
ap.add_argument("--dz-below", type=float, default=400.0, help="Keep stations up to this far BELOW model orography (m) [ECMWF]")
ap.add_argument("--dz-above", type=float, default=200.0, help="Keep stations up to this far ABOVE model orography (m) [ECMWF]")
ap.add_argument("--gross", type=float, default=7.5, help="Reject |obs - background| > this (K) [ECMWF]")
ap.add_argument("--bgcheck", type=float, default=4.0, help="Reject |innovation| > bgcheck*sqrt(sb2+so2)")
ap.add_argument("--sigma-inst", type=float, default=0.5, help="Station instrument error (K)")
ap.add_argument("--sigma-repr", type=float, default=1.0, help="Station representativeness error vs 1-deg cell (K)")
ap.add_argument("--pbl-dtheta", type=float, default=1.5, help="PBL top: first level with theta > theta_2m + this (K)")
ap.add_argument("--pbl-frac", type=float, default=0.75, help="Spread increments up to this fraction of PBL depth [RAP]")
ap.add_argument("--iau-kinds", default="DIR,BAL-PBL", help="Increment types for IAU-like arms (base bg only)")
ap.add_argument("--provider-file", default=None,
                help="Foreign analysis in the same GraphCast schema/times as --data (e.g. MERRA-2 from "
                     "scripts/prep_merra2.py). Enables arms M-DIR, REPLAY<H>-M, HYB<H>-M.")
ap.add_argument("--clim-era5", default=None, help="ERA5 hour-of-day climatology (scripts/build_clim.py --source era5)")
ap.add_argument("--clim-provider", default=None,
                help="Provider (MERRA-2) climatology (build_clim.py --source merra2). With --clim-era5 enables the "
                     "anomaly-initialization arms M-MEAN, M-QM, REPLAY<H>-MQM, HYB<H>-MQM, HYB<H>-4DV-MQM")
ap.add_argument("--qm-ratio-clip", default="0.5,2.0", help="M-QM: clip of std_E/std_M")
ap.add_argument("--qm-top", type=int, default=850,
                help="M-QMS/M-BAL: QM (mean+variance) at levels >= this (hPa) and for surface fields, mean shift above")
ap.add_argument("--edelta-lag", type=int, default=24,
                help="E+DM<lag>: ERA5 start + the MERRA-2 minus ERA5 anomaly difference from <lag> h earlier (same hour)")
ap.add_argument("--arms", default="all", help="Comma list of arms to forecast (ERA5 and BASE always kept)")
ap.add_argument("--boot", type=int, default=1000, help="Bootstrap resamples over stations")
ap.add_argument("--allow-no-elev", action="store_true", help="Keep stations with unknown elevation (no height correction)")
ap.add_argument("--lapse", type=float, default=5.5, help="Lapse rate for station height correction (K/km) [ECMWF 5.5]")
ap.add_argument("--verify-max-dz", type=float, default=None,
                help="Optional extra symmetric |dz| cap for verification stations (m); default: same ECMWF window")
ap.add_argument("--regress-region", default="conus-past", choices=["conus-past", "outside"],
                help="conus-past: CONUS land background errors at t0-18h/t0-12h; outside: NH land outside CONUS at t0-6h/t0")
ap.add_argument("--fix-weights", default="1.0,0.6,0.2", help="Fixed column weights at 1000,925,850 hPa for COL-FIX/BAL-FIX")
ap.add_argument("--n-obs", type=int, default=300, help="Pseudo-stations for era5-synth")
ap.add_argument("--withheld-frac", type=float, default=0.3)
ap.add_argument("--obs-noise", type=float, default=None, help="Obs error std (K); default by source")
ap.add_argument("--oi-L", type=float, default=250.0, help="OI Gaussian length scale (km)")
ap.add_argument("--col-top", type=int, default=500, help="Top level (hPa) of column increments")
ap.add_argument("--nud-types", default="DIR,BAL-PBL,FIX", help="Increment types used by nudging arms (DIR, BAL, FIX, REG, PBL, BAL-PBL)")
ap.add_argument("--nud-windows", default="6,24", help="Nudging window(s) in hours, comma-separated (e.g. '6', '24', or '6,24')")
ap.add_argument("--nud-tau", type=float, default=6.0, help="Nudging relaxation time (h)")
ap.add_argument("--long-nud", type=int, default=0,
                help="Long nudging spin-up (h, multiple of 6), e.g. 72 = 6-hourly nudging for 3 days before t0 "
                     "(base bg only; needs an ERA5 file starting at t0-(long+6)h and stations from then)")
ap.add_argument("--long-nud-types", default="BAL", help="Increment type(s) for the long surface-only nudging arm(s)")
ap.add_argument("--hyb-tau", type=float, default=6.0,
                help="Hybrid cycling: relaxation time (h) of the FULL state toward ERA5 at every 6 h cycle "
                     "(provider-analysis stand-in); 0 disables the hybrid arms")
ap.add_argument("--hyb-variants", default="base",
                help="Comma list of hybrid-cycle variants; each is '+'-joined options: "
                     "base | LS (level-selective replay) | BC (station bias correction) | "
                     "W=<const|ramp-up|ramp-down|tri|lanczos> (weighted 4DIAU over the cycles). "
                     "e.g. 'base,LS,BC,W=ramp-up,LS+BC'")
ap.add_argument("--hyb-sfc-tau", type=float, default=24.0,
                help="LS: relaxation time (h) toward ERA5 for surface fields and levels >= --hyb-bl-top")
ap.add_argument("--hyb-bl-top", type=int, default=850, help="LS: levels at/below this pressure (hPa) use --hyb-sfc-tau")
ap.add_argument("--bias-mode", default="station-hour", choices=["station", "station-hour"],
                help="BC: bias per station, or per station and UTC hour")
ap.add_argument("--bias-gamma", type=float, default=0.2, help="BC: update weight per cycle (EW average)")
ap.add_argument("--jac-k", type=int, default=8, help="JAC: random surface perturbations for the model-Jacobian balance")
ap.add_argument("--jac-smooth-km", type=float, default=500.0, help="JAC: local-regression smoothing length (km)")
ap.add_argument("--fdv-iter", type=int, default=40, help="4DV: max L-BFGS iterations (total over restarts)")
ap.add_argument("--fdv-L", type=float, default=300.0, help="4DV: background-error correlation length (km)")
ap.add_argument("--fdv-sig", default="2m_temperature=1.5,temperature=1.0,specific_humidity=0.0005,"
                                    "u_component_of_wind=1.5,v_component_of_wind=1.5,geopotential=50",
                help="4DV: background-error std per control variable (K, kg/kg, m/s, m2/s2)")
ap.add_argument("--fdv-era5-weight", type=float, default=None,
                help="HYB-4DV: weight of the ERA5 anchor term at t0 (default: 0 when --fdv-relax-t0 1, else 1)")
ap.add_argument("--fdv-restarts", type=int, default=3,
                help="4DV: restart L-BFGS from the current point if it stops early (line-search failure)")
ap.add_argument("--fdv-solver", default="gn", choices=["gn", "lbfgs"],
                help="4DV minimizer: incremental Gauss-Newton (default; outer loops re-linearize GraphCast, inner "
                     "CG on the host with separately compiled TL (jvp) and adjoint (vjp) calls, so peak GPU memory "
                     "= one gradient, fits a 32 GB V100) or L-BFGS on the full nonlinear cost")
ap.add_argument("--fdv-outer", type=int, default=3, help="4DV-GN: outer loops (re-linearizations)")
ap.add_argument("--fdv-inner", type=int, default=40, help="4DV-GN: max inner CG iterations per outer loop (each = 1 TL + 1 adjoint run)")
ap.add_argument("--fdv-cg-tol", type=float, default=1e-3, help="4DV-GN: relative CG residual tolerance")
ap.add_argument("--fdv-sigo-scale", type=float, default=1.0,
                help="4DV: multiply the station obs-error std by this factor (Desroziers ratio printed in the log)")
ap.add_argument("--fdv-relax-t0", type=int, default=1,
                help="HYB-4DV: launch t0 frame = relax_ERA5(F(x_b)) + [F(x_a) - F(x_b)], i.e. the same ERA5 "
                     "anchoring as HYB-DIR at t0 plus the model-evolved 4D-Var increment (1), or raw F(x_a) (0)")
ap.add_argument("--jac-vars", default="all", choices=["all", "T", "TZ"],
                help="JAC: which responses to use (all variables, temperature only, or temperature+geopotential)")
ap.add_argument("--grad-selftest", type=int, default=1, help="Check the differentiable step vs the forward model")
ap.add_argument("--hyb-types", default="DIR,BAL-PBL",
                help="Station increment type(s) added on top of the ERA5-relaxed state in the hybrid arms")
ap.add_argument("--nud-obs", default="all", choices=["all", "last2"],
                help="Nudge with obs at all cycles or only last 2")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
ap.add_argument("--outdir", default=None)
ap.add_argument("--dpi", type=int, default=150)
ap.add_argument("--save-fields", type=int, default=1,
                help="Write fields.nc (2 m T, MSLP, T850, Z500, 6 h precip for every arm/lead + analyses) and "
                     "global maps (scripts/plot_global_maps.py)")
ap.add_argument("--skip-checks", action="store_true", help="Skip determinism/consistency checks")
ap.add_argument("--model", default="small", choices=["small", "large"],
                help="small = GraphCast_small (1 deg, 13 levels, ERA5 1979-2015); "
                     "large = GraphCast (0.25 deg, 37 levels, ERA5 1979-2017)")
ap.add_argument("--params", default=None, help="Explicit checkpoint .npz path (overrides --model)")
ap.add_argument("--lite", type=int, default=None,
                help="Keep only 2 m T, T850, Z500 from forecasts to save memory (default: on for --model large)")
args = ap.parse_args()

MODEL_SPEC = {
    "small": dict(res="1.0", nlev=13, tag="",
                  ckpt="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                       "mesh 2to5 - precipitation input and output.npz"),
    "large": dict(res="0.25", nlev=37, tag="_r025",
                  ckpt="GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - "
                       "mesh 2to6 - precipitation input and output.npz"),
}[args.model]
if args.lite is None:
    args.lite = 1 if args.model == "large" else 0
WANT = None if args.arms == "all" else ({"ERA5", "BASE"} | {a.strip() for a in args.arms.split(",")})


def want(name):
    """Build (possibly expensive) arm chains only if they will be forecast."""
    return WANT is None or name in WANT


T0 = np.datetime64(dt.datetime.fromisoformat(args.t0))
t_launch = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=24)
PROJ = args.proj
if args.long_nud and args.long_nud % 6:
    raise SystemExit("--long-nud must be a multiple of 6 h")
LEAD_BACK_H = max(24, args.long_nud)                        # how far before t0 the data must start (+6 h)
if args.data:
    DATA = args.data
else:
    _t_l = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=LEAD_BACK_H)
    _stem = f"res-{MODEL_SPEC['res']}_levels-{MODEL_SPEC['nlev']}"
    _ext = ".zarr" if args.model == "large" else ".nc"
    DATA = os.path.join(PROJ, "data", "era5",
                        f"source-era5_date-{_t_l:%Y-%m-%d}_{_stem}_steps-{LEAD_BACK_H // 6 + 12:02d}{_ext}")
    if not os.path.exists(DATA) and not args.long_nud:     # fall back to the standard 24 h-window file
        DATA = os.path.join(PROJ, "data", "era5",
                            f"source-era5_date-{t_launch:%Y-%m-%d}_{_stem}_steps-16{_ext}")
TAG = dt.datetime.fromisoformat(args.t0).strftime("%Y%m%dT%H")
OUT = args.outdir or os.path.join(PROJ, "runs", "exp_main", f"{TAG}_{args.obs_source}_{args.base}{MODEL_SPEC['tag']}")
if args.base == "era5":
    _w = [w for w in args.nud_windows.split(",") if w.strip() and int(w) <= 12]
    if len(_w) < len([w for w in args.nud_windows.split(",") if w.strip()]):
        print("NOTE: --base era5 uses nudging windows <= 12 h only (starts from ERA5 at t0-6h/t0-12h)")
    args.nud_windows = ",".join(_w)
    if not _w:
        args.nud_types = ""
STATION_SRC = args.obs_source in ("isd", "uscrn")
if args.obs_noise is not None:
    SIGMA_O = args.obs_noise
elif STATION_SRC:
    SIGMA_O = float(np.hypot(args.sigma_inst, args.sigma_repr))   # single-station total
else:
    SIGMA_O = {"merra2": 0.8, "merra2-field": 0.8, "era5-synth": 0.5}[args.obs_source]
T_START = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=30)
T_END = dt.datetime.fromisoformat(args.t0) + dt.timedelta(hours=6 * args.steps)
OBS_DIR = os.path.join(PROJ, "data", "obs")


T_NEED0 = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=max(30, args.long_nud))


def _default_csv(prefix):
    """Pick an observation CSV named <prefix>_<start>_<end>.csv that covers the needed period."""
    import glob, re
    best = None
    for c in sorted(glob.glob(os.path.join(OBS_DIR, f"{prefix}_*.csv"))):
        m = re.search(r"_(\d{8}T\d{2})_(\d{8}T\d{2})\.csv$", c)
        if not m:
            continue
        a, b = (dt.datetime.strptime(x, "%Y%m%dT%H") for x in m.groups())
        if a <= T_NEED0 and b >= T_END:
            if best is None or os.path.getsize(c) < os.path.getsize(best):
                best = c
    if best is None:
        print(f"   WARNING: no {prefix} file covers {T_NEED0:%Y-%m-%dT%H} .. {T_END:%Y-%m-%dT%H}")
        c = sorted(glob.glob(os.path.join(OBS_DIR, f"{prefix}_*.csv")))
        best = c[0] if c else None
    return best
os.makedirs(OUT, exist_ok=True)
PARAMS = args.params or os.path.join(PROJ, "data", "params", MODEL_SPEC["ckpt"])
if not os.path.exists(PARAMS):
    raise SystemExit(f"Checkpoint not found: {PARAMS}\n(download it into data/params/ from gs://dm_graphcast/params/)")
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


def _wrapped32(m_cfg, t_cfg):
    """Same predictor without the bfloat16 cast: used only for gradients / tangent-linear runs."""
    p = graphcast.GraphCast(m_cfg, t_cfg)
    p = normalization.InputsAndResiduals(p, **stats)
    return autoregressive.Predictor(p, gradient_checkpointing=True)


@hk.transform_with_state
def _fwd32(m_cfg, t_cfg, inputs, targets_template, forcings):
    return _wrapped32(m_cfg, t_cfg)(inputs, targets_template=targets_template, forcings=forcings)


_jit32 = jax.jit(functools.partial(
    functools.partial(_fwd32.apply, m_cfg=model_config, t_cfg=task_config),
    params=params, state={}))

# =============================================================================
# 2. Data, time indexing, masks
# =============================================================================
print("\n[2] Loading ERA5 window ...")
if not os.path.exists(DATA):
    raise SystemExit(f"ERA5 input not found: {DATA}")
if DATA.endswith(".zarr"):                      # large 0.25-deg inputs: open lazily
    DS = xr.open_zarr(DATA, decode_timedelta=True)
elif os.path.getsize(DATA) > 8e9:
    DS = xr.open_dataset(DATA, decode_timedelta=True)
else:
    try:
        DS = xr.load_dataset(DATA, decode_timedelta=True).compute()
    except Exception:
        DS = xr.load_dataset(DATA).compute()
print(f"   model = {args.model} ({os.path.basename(PARAMS)}), lite forecasts = {bool(args.lite)}")

DATETIMES = DS.coords["datetime"].values
DATETIMES = DATETIMES[0] if DATETIMES.ndim == 2 else DATETIMES
hits = np.where(DATETIMES == T0)[0]
if len(hits) != 1:
    raise SystemExit(f"t0 {T0} not found in data times {DATETIMES[0]} ... {DATETIMES[-1]}")
I0 = int(hits[0])                 # index of t0
if args.long_nud and I0 < args.long_nud // 6 + 1:
    _t_l = dt.datetime.fromisoformat(args.t0) - dt.timedelta(hours=args.long_nud)
    raise SystemExit(
        f"--long-nud {args.long_nud} needs ERA5 from t0-{args.long_nud + 6}h, but the file starts at {DATETIMES[0]}.\n"
        f"Download on the login node:\n"
        f"  python scripts/download_era5_cloud.py --date {_t_l:%Y-%m-%d} --time {_t_l:%H:%M} --steps {args.long_nud // 6 + args.steps}\n"
        f"  python scripts/download_isd_lite.py --start {(_t_l - dt.timedelta(hours=6)).replace(hour=0):%Y-%m-%dT%H} --end {T_END:%Y-%m-%dT%H}\n"
        f"  python scripts/download_uscrn_range.py --start {(_t_l - dt.timedelta(hours=6)).replace(hour=0):%Y-%m-%dT%H} --end {T_END:%Y-%m-%dT%H}")
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
    sub = DS.isel(time=slice(start, start + 2 + nsteps)).compute()
    return data_utils.extract_inputs_targets_forcings(
        sub, target_lead_times=slice("6h", f"{nsteps*6}h"), **dataclasses.asdict(task_config))


_tmpl_inputs, _, _ = window(I0 - 1, 1)
STATE_VARS = [v for v in task_config.target_variables if v in _tmpl_inputs.data_vars]
REST = {v: [d for d in _tmpl_inputs[v].dims if d not in ("batch", "time")] for v in STATE_VARS}
print(f"   state variables: {STATE_VARS}")


def era5_frame(idx):
    return {v: DS[v].isel(time=idx).isel(batch=0).transpose(*REST[v]).values.astype(np.float32)
            for v in STATE_VARS}


DSP = None
if args.provider_file:
    DSP = xr.load_dataset(args.provider_file, decode_timedelta=True)
    _pdt = DSP.coords["datetime"].values
    _pdt = _pdt[0] if _pdt.ndim == 2 else _pdt
    if len(_pdt) != len(DATETIMES) or not np.all(_pdt == DATETIMES):
        raise SystemExit(f"--provider-file times {_pdt[0]}..{_pdt[-1]} ({len(_pdt)}) do not match --data "
                         f"{DATETIMES[0]}..{DATETIMES[-1]} ({len(DATETIMES)}); rebuild it with prep_merra2.py --era5 <data>")
    print(f"   provider analysis: {os.path.basename(args.provider_file)}  ({DSP.attrs.get('source', '')[:60]})")


def provider_frame(idx):
    return {v: DSP[v].isel(time=idx).isel(batch=0).transpose(*REST[v]).values.astype(np.float32)
            for v in STATE_VARS}


CLIM_E = CLIM_M = None
if DSP is not None and args.clim_era5 and args.clim_provider:
    CLIM_E, CLIM_M = xr.load_dataset(args.clim_era5), xr.load_dataset(args.clim_provider)
    for _c in (CLIM_E, CLIM_M):
        if not (np.allclose(_c["lat"].values, LATS) and np.allclose(_c["lon"].values, LONS)
                and [int(x) for x in _c["level"].values] == LEVELS):
            raise SystemExit("climatology grid/levels do not match the input file")
    _need_h = sorted({pd.Timestamp(d).hour for d in DATETIMES})
    for _c in (CLIM_E, CLIM_M):
        _miss = [h for h in _need_h if h not in set(int(x) for x in _c["hour"].values)]
        if _miss:
            raise SystemExit(f"climatology lacks hours {_miss}")
    QM_LO, QM_HI = [float(x) for x in args.qm_ratio_clip.split(",")]
    print(f"   anomaly initialization: clim ERA5 {CLIM_E.attrs.get('years')} ({CLIM_E.attrs.get('n_days')} d), "
          f"clim provider {CLIM_M.attrs.get('years')} ({CLIM_M.attrs.get('n_days')} d)")


def _climv(ds, v, h, kind):
    return ds[f"{v}_{kind}"].sel(hour=h).transpose(*REST[v]).values.astype(np.float64)


def mapped_frame(idx, mode="QM"):
    """Provider state mapped to ERA5's climate (anomaly initialization).
    MEAN: x - (clim_M - clim_E) for every variable.
    QM  : clim_E + (x - clim_M) std_E/std_M for every variable (precipitation: mean shift).
    QMS : QM for surface fields and levels >= --qm-top, MEAN above (keeps free-troposphere balances linear).
    BAL : QMS, then geopotential rebuilt hydrostatically from the mapped virtual temperature, anchored at
          the MEAN-mapped 1000 hPa geopotential (Z consistent with T in every column)."""
    M = provider_frame(idx)
    h = pd.Timestamp(DATETIMES[idx]).hour
    out = {}
    for v in STATE_VARS:
        x = M[v].astype(np.float64)
        mE, mM = _climv(CLIM_E, v, h, "mean"), _climv(CLIM_M, v, h, "mean")
        shift = x - (mM - mE)
        if mode == "MEAN" or v == "total_precipitation_6hr":
            y = shift
        else:
            sE, sM = _climv(CLIM_E, v, h, "std"), _climv(CLIM_M, v, h, "std")
            r = np.where(sM > 1e-12, sE / np.maximum(sM, 1e-12), 1.0)
            y = mE + (x - mM) * np.clip(r, QM_LO, QM_HI)
            if mode in ("QMS", "BAL") and x.ndim == 3:            # QM only near the surface
                low = np.array([p >= args.qm_top for p in LEVELS])[:, None, None]
                y = np.where(low, y, shift)
        if v in ("total_precipitation_6hr", "specific_humidity"):
            y = np.clip(y, 0.0, None)
        out[v] = y.astype(np.float32)
    if mode == "BAL" and "geopotential" in out and "temperature" in out:
        T, Z = out["temperature"].astype(np.float64), out["geopotential"].astype(np.float64)
        q = out["specific_humidity"].astype(np.float64) if "specific_humidity" in out else 0.0 * T
        Tv = T * (1.0 + 0.608 * q)
        asc = sorted(LEVELS, reverse=True)                       # 1000 ... 50 hPa
        Znew = Z.copy()
        for pb, pt in zip(asc[:-1], asc[1:]):
            ib, it = LIDX[pb], LIDX[pt]
            Znew[it] = Znew[ib] + RD * 0.5 * (Tv[ib] + Tv[it]) * np.log(pb / pt)
        out["geopotential"] = Znew.astype(np.float32)
    return out


def edelta_pair(lag_h):
    """ERA5 start + the MERRA-2 minus ERA5 difference from lag_h earlier (same hour of day): a perturbation
    of MERRA-2 size and structure that is not today's (anomaly part only when climatologies are given)."""
    k = lag_h // 6
    pair = []
    for idx in (I0 - 1, I0):
        E, Lg = era5_frame(idx), idx - k
        Ml = mapped_frame(Lg, "MEAN") if CLIM_E is not None else provider_frame(Lg)
        El = era5_frame(Lg)
        fr = {}
        for v in STATE_VARS:
            y = E[v].astype(np.float64) + (Ml[v].astype(np.float64) - El[v])
            if v in ("total_precipitation_6hr", "specific_humidity"):
                y = np.clip(y, 0.0, None)
            fr[v] = y.astype(np.float32)
        pair.append(fr)
    return tuple(pair)


class _Lev:
    """Holds selected pressure levels of a field, indexable by the full-level index."""
    def __init__(self, d):
        self.d = d

    def __getitem__(self, i):
        return self.d[i]


def lite_forecast(preds):
    """Reduce a GraphCast rollout to the fields the scores/plots use (2 m T, T850, Z500)."""
    out = []
    for k in range(preds.sizes["time"]):
        fr = {}
        for v, levs in (("2m_temperature", None), ("temperature", (850,)), ("geopotential", (500,))):
            da = preds[v].isel(time=k)
            if "batch" in da.dims:
                da = da.isel(batch=0)
            if levs is None:
                fr[v] = np.asarray(da.transpose(*REST[v]).values, dtype=np.float32)
            else:
                fr[v] = _Lev({LIDX[p]: np.asarray(da.sel(level=p).transpose(*[d for d in REST[v] if d != "level"]).values,
                                                  dtype=np.float32) for p in levs})
        out.append(fr)
    return out


def pred_frame(preds, k):
    if isinstance(preds, list):                  # lite forecast
        return preds[k]
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


LONG_IDX = list(range(I0 - args.long_nud // 6 + 1, I0 + 1)) if args.long_nud else []
OBS_LOAD_IDX = sorted(set(OBS_IDX) | set(LONG_IDX))
OBS_TIMES = [pd.Timestamp(DATETIMES[i]) for i in OBS_LOAD_IDX]
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
        M2FIELD = {i: M2[t] for i, t in zip(OBS_LOAD_IDX, OBS_TIMES)}
else:
    path = args.obs_file or _default_csv("isd_lite" if args.obs_source == "isd" else "uscrn")
    if not path or not os.path.exists(path):
        raise SystemExit(f"Observation CSV not found ({path}); run the downloader on the login node.")
    OBSDF = load_station_csv(path)
    print(f"   {args.obs_source}: {OBSDF.sid.nunique()} stations from {os.path.basename(path)}")

# height correction to model orography, and ECMWF height window (dz = z_station - z_model)


def in_window(dz, extra=None):
    ok = np.isfinite(dz) & (dz >= -args.dz_below) & (dz <= args.dz_above)
    if args.allow_no_elev:
        ok |= ~np.isfinite(dz)
    if extra is not None:
        ok &= ~(np.abs(np.nan_to_num(dz)) > extra)
    return ok


zm = interp2(ZMOD, OBSDF.lat.values, OBSDF.lon.values)
dz = OBSDF.elev.values - zm
OBSDF["y"] = OBSDF.t2m_K.values + np.where(np.isfinite(dz), GAMMA * dz, 0.0)
_n0 = OBSDF.sid.nunique()
OBSDF = OBSDF[in_window(dz)]
print(f"   height window [-{args.dz_below:.0f}, +{args.dz_above:.0f}] m, lapse {args.lapse} K/km: "
      f"{OBSDF.sid.nunique()} of {_n0} stations kept")

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
print(f"   withheld stations usable for verification: {int(in_window(_dz, args.verify_max_dz).sum())}")

# independent verification network (USCRN) — never inserted
VER = None
if args.obs_source != "uscrn":
    vpath = args.verify_file or _default_csv("uscrn")
    if vpath and os.path.exists(vpath):
        VER = load_station_csv(vpath)
        vz = VER.elev.values - interp2(ZMOD, VER.lat.values, VER.lon.values)
        VER["y"] = VER.t2m_K.values + np.where(np.isfinite(vz), GAMMA * vz, 0.0)
        _nv = VER.sid.nunique()
        VER = VER[in_window(vz, args.verify_max_dz)]
        print(f"   verification: USCRN {VER.sid.nunique()} of {_nv} stations in the height window ({os.path.basename(vpath)})")
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


def oi_increment(bg2t, idx, tag="", bias=None):
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
    la, lo, innov, so2k, sb2, qc = qc_innovations(bg2t, d, bias=bias, hour=pd.Timestamp(DATETIMES[idx]).hour)
    if len(innov) == 0:
        return np.zeros_like(bg2t)
    Coo = np.exp(-0.5 * (gc_dist(la[:, None], lo[:, None], la[None], lo[None]) / args.oi_L) ** 2)
    Cgo = np.exp(-0.5 * (gc_dist(LATS[_box[:, 0]][:, None], LONS[_box[:, 1]][:, None],
                                 la[None], lo[None]) / args.oi_L) ** 2)
    w = np.linalg.solve(sb2 * Coo + np.diag(so2k), innov)
    inc = np.zeros_like(bg2t)
    inc[_box[:, 0], _box[:, 1]] = sb2 * (Cgo @ w)
    OI_LOG.append(dict(tag=tag, time=str(DATETIMES[idx]), n_obs=int(len(innov)), **qc,
                       mean_innov=float(innov.mean()), rms_innov=float(np.sqrt((innov ** 2).mean())),
                       sigma_b=float(np.sqrt(sb2)), sigma_o_mean=float(np.sqrt(so2k.mean()))))
    return inc.astype(np.float32)


def qc_innovations(bg2t, d, bias=None, hour=0):
    """ECMWF-style QC + 1-deg super-obs. Returns lat, lon, innovation, obs-error variance per
    super-ob, background-error variance estimate, and QC counts.
    bias: optional dict (station bias state, VarBC-lite); the current estimate is subtracted from
    each observation before QC and updated afterwards with weight --bias-gamma."""
    la, lo, y = d.lat.values, d.lon.values, d.y.values
    sids = d.sid.values
    keys = [(s_, hour) if args.bias_mode == "station-hour" else s_ for s_ in sids]
    b_now = np.array([bias.get(k, 0.0) for k in keys]) if bias is not None else np.zeros(len(y))
    innov = y - b_now - interp2(bg2t, la, lo)
    n_in = len(innov)
    keep = np.abs(innov) <= args.gross                       # gross check
    la, lo, innov = la[keep], lo[keep], innov[keep]
    keys = [k for k, kk in zip(keys, keep) if kk]; b_now = b_now[keep]
    n_gross = n_in - len(innov)
    so2_single = SIGMA_O ** 2
    sb2 = max(float(np.var(innov)) - so2_single, 0.05) if len(innov) > 1 else 1.0
    keep = np.abs(innov) <= args.bgcheck * np.sqrt(sb2 + so2_single)   # background check
    la, lo, innov = la[keep], lo[keep], innov[keep]
    keys = [k for k, kk in zip(keys, keep) if kk]; b_now = b_now[keep]
    n_bg = int((~keep).sum())
    if bias is not None:                                     # update after using the current estimate
        for k, bn, dv in zip(keys, b_now, innov):
            bias[k] = float(bn + args.bias_gamma * dv)
    n_per = np.ones(len(innov))
    if args.superob and len(innov):
        ci = np.round((la - LATS[0]) / (LATS[1] - LATS[0])).astype(int)
        cj = np.round(np.mod(lo - LONS[0], 360) / (LONS[1] - LONS[0])).astype(int)
        g = pd.DataFrame(dict(c=ci * 10000 + cj, la=la, lo=lo, d=innov, n=1.0)).groupby("c").agg(
            la=("la", "mean"), lo=("lo", "mean"), d=("d", "mean"), n=("n", "sum"))
        la, lo, innov, n_per = g.la.values, g.lo.values, g.d.values, g.n.values
    if STATION_SRC and args.obs_noise is None:
        so2k = args.sigma_inst ** 2 + args.sigma_repr ** 2 / n_per
    else:
        so2k = np.full(len(innov), SIGMA_O ** 2) / (n_per if args.superob else 1.0)
    if len(innov) > 1:
        sb2 = max(float(np.var(innov)) - float(np.mean(so2k)), 0.05)
    return la, lo, innov, so2k, sb2, dict(n_raw=n_in, rej_gross=int(n_gross), rej_bg=n_bg)


def bias_increment(bg2t, idx, tag=""):
    """M1b: uniform CONUS-land shift equal to the mean QC'd innovation (bias-only control)."""
    d = obs_at(idx)
    if d.empty or M2FIELD is not None:
        return np.zeros_like(bg2t)
    _, _, innov, _, _, _ = qc_innovations(bg2t, d)
    mb = float(innov.mean()) if len(innov) else 0.0
    OI_LOG.append(dict(tag=tag or "bias", time=str(DATETIMES[idx]), mean_innov=mb))
    return (mb * CONUS_LAND).astype(np.float32)


def _vfilter(df):
    if df is None or df.empty:
        return df
    dz = df.elev.values - interp2(ZMOD, df.lat.values, df.lon.values)
    return df[in_window(dz, args.verify_max_dz)]


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
JAC_B = None   # model-Jacobian balance coefficients {(var, level_index or None): field}
KAPPA = 0.2857


def diagnose_pbl(frame):
    """RAP-style mixed-layer depth from the model column (theta-excess method).
    Returns (weights {p: array}, depth_hPa array, surface pressure hPa).
    weight(p) = 1 - (ps - p) / (pbl_frac * (ps - p_top)), clipped to [0, 1], only above ground.
    A column whose first above-ground level already exceeds theta_2m + dtheta is stable:
    all weights are 0 (surface-only)."""
    t2 = frame["2m_temperature"]
    ps = frame["mean_sea_level_pressure"] * np.exp(-ZSFC / (RD * t2)) / 100.0   # hPa (ZSFC is geopotential)
    th2 = t2 * (1000.0 / ps) ** KAPPA
    ptop = np.full_like(t2, np.nan)
    for p in sorted(LEVELS, reverse=True):                    # 1000, 925, ... upward
        if p < 700:
            break
        th = frame["temperature"][LIDX[p]] * (1000.0 / p) ** KAPPA
        above = p < ps
        hit = above & np.isnan(ptop) & (th > th2 + args.pbl_dtheta)
        ptop[hit] = p
    ptop = np.where(np.isnan(ptop), 700.0, ptop)              # well-mixed through 700 hPa: cap
    depth = np.clip(ps - ptop, 0.0, None)
    w = {}
    for p in LEVELS:
        if p < 700:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            wp = 1.0 - (ps - p) / (args.pbl_frac * depth)
        wp = np.where((p < ps) & (depth > 0) & np.isfinite(wp), np.clip(wp, 0.0, 1.0), 0.0)
        if np.any(wp > 0):
            w[p] = wp.astype(np.float32)
    return w, depth, ps


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
    if kind in ("PBL", "BAL-PBL", "COL-PBL"):
        wts, _, _ = diagnose_pbl(frame)
        dT = {p: wp * inc for p, wp in wts.items()}
        for p, v in dT.items():
            f["temperature"][LIDX[p]] += v
        if kind == "BAL-PBL" and dT:
            for p, v in hypsometric_phi(dT).items():
                f["geopotential"][LIDX[p]] += v
    if kind == "JAC":
        if JAC_B is None:
            raise RuntimeError("JAC balance not computed")
        for (v, li), coef in JAC_B.items():
            if li is None:
                f[v] += coef * inc
            else:
                f[v][li] += coef * inc
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
if want("DIR-1F"):
    ARMS["DIR-1F"] = (BASE_A, apply_increment(BASE_B, INC_B, "DIR"))
if DSP is not None and want("M-DIR"):                  # foreign analysis used directly (no cycling)
    ARMS["M-DIR"] = (provider_frame(I0 - 1), provider_frame(I0))
if CLIM_E is not None:
    for _mode in ("MEAN", "QM", "QMS", "BAL"):
        if want(f"M-{_mode}"):
            ARMS[f"M-{_mode}"] = (mapped_frame(I0 - 1, _mode), mapped_frame(I0, _mode))
if DSP is not None and want(f"E+DM{args.edelta_lag}"):
    if I0 - 1 - args.edelta_lag // 6 < 0 or args.edelta_lag % 24:
        print(f"   E+DM{args.edelta_lag} skipped: lag must be a multiple of 24 h inside the data window")
    else:
        ARMS[f"E+DM{args.edelta_lag}"] = edelta_pair(args.edelta_lag)
for k in ("DIR", "COL", "BAL", "REG", "COL-FIX", "BAL-FIX", "COL-PBL", "BAL-PBL"):
    _nm = f"{k}-2F" if ("FIX" not in k and "PBL" not in k) else f"{k}"
    if want(_nm):
        ARMS[_nm] = (apply_increment(BASE_A, INC_A, k), apply_increment(BASE_B, INC_B, k))
# M1b: bias-only control (uniform shift by the mean innovation, both frames)
if want("BIAS"):
    ARMS["BIAS"] = (apply_increment(BASE_A, bias_increment(BASE_A["2m_temperature"], I0 - 1), "DIR"),
                    apply_increment(BASE_B, bias_increment(BASE_B["2m_temperature"], I0), "DIR"))
_w_b, _dep_b, _ps_b = diagnose_pbl(BASE_B)
_mixed_b = np.any(np.stack([w > 0 for w in _w_b.values()]), axis=0) if _w_b else np.zeros_like(_dep_b, bool)
PBL_STATS = dict(stable_frac_conus_land=float(np.mean(~_mixed_b[CONUS_LAND])),
                 mean_depth_hPa=float(np.mean(_dep_b[CONUS_LAND])))
print(f"   PBL at t0 (BASE): {100*PBL_STATS['stable_frac_conus_land']:.0f} % of CONUS land columns surface-only (stable), "
      f"mean diagnosed depth {PBL_STATS['mean_depth_hPa']:.0f} hPa")


def iau_chain(kind):
    """M4 (GEOS IAU analogue): one increment from the t0 observations, added as 1/4 after each
    6 h step from t0-24 h to t0 (base bg only)."""
    A, B = era5_frame(S0), era5_frame(S0 + 1)
    for idx in OBS_IDX:
        C = pred_frame(forecast(A, B, idx - 2, 1), 0)
        C = apply_increment(C, INC_B, kind, scale=0.25)
        A, B = B, C
    return A, B


if args.base == "bg" and M2FIELD is None:
    for k in [x.strip() for x in args.iau_kinds.split(",") if x.strip()]:
        if not want(f"IAU-{k}"):
            continue
        t_ = time.time()
        ARMS[f"IAU-{k}"] = iau_chain(k)
        print(f"   IAU-{k}: 4 x 1/4 of the t0 increment, {time.time()-t_:.1f} s")

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


def iau_weights(profile, n):
    """Weights (max 1) of the increments over n cycles (oldest first): weighted-4DIAU analogue."""
    k = np.arange(1, n + 1, dtype=float)
    if profile == "const":
        w = np.ones(n)
    elif profile == "ramp-up":
        w = k / n
    elif profile == "ramp-down":
        w = (n - k + 1) / n
    elif profile == "tri":
        w = 1.0 - np.abs((k - 0.5) / n * 2.0 - 1.0)
    elif profile == "lanczos":                     # Lanczos-windowed low-pass weights, centred
        x = (k - (n + 1) / 2) / ((n + 1) / 2)
        w = np.sinc(x) * np.sinc(x) + 1e-3
    else:
        raise ValueError(profile)
    return w / w.max()


def relax_to_era5(C, E, a_upper, a_sfc=None):
    """Full-state relaxation toward ERA5; with a_sfc, surface fields and levels >= --hyb-bl-top use a_sfc."""
    out = {}
    for v in STATE_VARS:
        a = np.full(C[v].shape[:1] if C[v].ndim == 3 else (), a_upper, dtype=np.float32)
        if a_sfc is not None:
            if C[v].ndim == 3:
                a = np.array([a_sfc if p >= args.hyb_bl_top else a_upper for p in LEVELS], dtype=np.float32)
            elif v in ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"):
                a = np.float32(a_sfc)
        if C[v].ndim == 3:
            a = a[:, None, None]
        out[v] = (C[v] + a * (E[v] - C[v])).astype(np.float32)
    return out


def hybrid_chain(kind, hours, a_era5, alpha_obs, weights="const", ls=False, bc=False,
                 end_idx=None, last_obs=True, tag=None, target=None):
    """Generalized hybrid cycling (see long_nudge_chain) with weighted-4DIAU cycle weights,
    level-selective replay (ls) and station bias correction (bc). Runs from t0-(hours+6)h up to
    frame end_idx (default t0). last_obs=False skips the station increment at end_idx."""
    end_idx = I0 if end_idx is None else end_idx
    n = hours // 6
    w = iau_weights(weights, n)
    a_sfc = (1.0 - np.exp(-6.0 / args.hyb_sfc_tau)) if ls else None
    bias = {} if bc else None
    target = target or era5_frame
    A, B = target(I0 - n - 1), target(I0 - n)
    for j, idx in enumerate(range(I0 - n + 1, end_idx + 1)):
        C = pred_frame(forecast(A, B, idx - 2, 1), 0)
        if a_era5 > 0:
            C = relax_to_era5(C, target(idx), w[j] * a_era5, None if a_sfc is None else w[j] * a_sfc)
        if alpha_obs > 0 and (last_obs or idx != end_idx):
            C = apply_increment(C, oi_increment(C["2m_temperature"], idx, tag or f"hyb-{kind}", bias=bias),
                                kind, scale=w[j] * alpha_obs)
        A, B = B, C
    if bias:
        BIAS_LOG[tag or kind] = dict(n_keys=len(bias), mean=float(np.mean(list(bias.values()))),
                                     rms=float(np.sqrt(np.mean(np.square(list(bias.values()))))))
    return A, B


BIAS_LOG = {}


def long_nudge_chain(kind, alpha, hours, era5_alpha=0.0, tag=None, target=None):
    """Operational-style cycling: cold start from ERA5 at t0-(hours+6)h / t0-hours, then 6-hourly
    GraphCast steps up to t0. After every step:
      1. (hybrid) relax the FULL state - every variable, every level - toward ERA5 at that time
         with gain era5_alpha (ERA5 = stand-in for a provider analysis / reanalysis replay), then
      2. add the station increment of type `kind` with gain alpha, computed against the relaxed state.
    era5_alpha = 0, alpha > 0 : surface-only nudging (upper air free to drift)
    era5_alpha > 0, alpha = 0 : reanalysis replay only (no own observations)
    both > 0                  : hybrid = reanalysis replay + own surface observations
    both = 0                  : free-running control
    target: frame function of the analysis to start from and relax to (default ERA5; provider_frame
    for a foreign analysis such as MERRA-2)."""
    target = target or era5_frame
    n = hours // 6
    A, B = target(I0 - n - 1), target(I0 - n)
    for idx in range(I0 - n + 1, I0 + 1):
        C = pred_frame(forecast(A, B, idx - 2, 1), 0)
        if era5_alpha > 0:
            E = target(idx)
            C = {v: (C[v] + era5_alpha * (E[v] - C[v])).astype(np.float32) for v in STATE_VARS}
        if alpha > 0:
            C = apply_increment(C, oi_increment(C["2m_temperature"], idx, tag or f"nud{hours}-{kind}"),
                                kind, scale=alpha)
        A, B = B, C
    return A, B


if args.long_nud:
    if args.base != "bg" or M2FIELD is not None:
        print("NOTE: --long-nud applies to --base bg with station/OI sources only; skipped")
    else:
        t_ = time.time()
        H = args.long_nud
        if want(f"FREE{H}"):
            ARMS[f"FREE{H}"] = long_nudge_chain("DIR", 0.0, H)
        for k in [x.strip() for x in args.long_nud_types.split(",") if x.strip()]:
            if want(f"NUD{H}-{k}"):
                ARMS[f"NUD{H}-{k}"] = long_nudge_chain(k, ALPHA, H)
        print(f"   long surface-only nudging {H} h ({H // 6} cycles, alpha={ALPHA:.2f}) "
              f"+ FREE{H} control: {time.time()-t_:.1f} s")
        if args.hyb_tau > 0:
            t_ = time.time()
            A_E = 1.0 - np.exp(-6.0 / args.hyb_tau)
            if want(f"REPLAY{H}"):
                ARMS[f"REPLAY{H}"] = long_nudge_chain("DIR", 0.0, H, era5_alpha=A_E)
            for k in [x.strip() for x in args.hyb_types.split(",") if x.strip() and x.strip() not in ("JAC", "4DV")]:
                if want(f"HYB{H}-{k}"):
                    ARMS[f"HYB{H}-{k}"] = long_nudge_chain(k, ALPHA, H, era5_alpha=A_E, tag=f"hyb{H}-{k}")
            print(f"   hybrid cycling {H} h: full state relaxed to ERA5 (alpha_ERA5={A_E:.2f}, tau={args.hyb_tau} h) "
                  f"+ stations (alpha={ALPHA:.2f}); REPLAY{H} control: {time.time()-t_:.1f} s")
            if DSP is not None and any(want(f"{a}{H}-{b}") for a in ("REPLAY", "HYB") for b in ("M", "MQM")):
                t_ = time.time()
                if want(f"REPLAY{H}-M"):
                    ARMS[f"REPLAY{H}-M"] = long_nudge_chain("DIR", 0.0, H, era5_alpha=A_E, target=provider_frame)
                if want(f"HYB{H}-M"):
                    ARMS[f"HYB{H}-M"] = long_nudge_chain("DIR", ALPHA, H, era5_alpha=A_E, tag=f"hyb{H}-m",
                                                         target=provider_frame)
                if CLIM_E is not None and want(f"REPLAY{H}-MQM"):
                    ARMS[f"REPLAY{H}-MQM"] = long_nudge_chain("DIR", 0.0, H, era5_alpha=A_E,
                                                              target=lambda i: mapped_frame(i, "QM"))
                if CLIM_E is not None and want(f"HYB{H}-MQM"):
                    ARMS[f"HYB{H}-MQM"] = long_nudge_chain("DIR", ALPHA, H, era5_alpha=A_E, tag=f"hyb{H}-mqm",
                                                           target=lambda i: mapped_frame(i, "QM"))
                print(f"   provider cycling {H} h: full state relaxed to the provider analysis (alpha={A_E:.2f}); "
                      f"REPLAY{H}-M / HYB{H}-M (+ stations): {time.time()-t_:.1f} s")

WINDOWS = [int(w.strip()) for w in args.nud_windows.split(",") if w.strip()]
for w in WINDOWS:
    for k in [x.strip() for x in args.nud_types.split(",") if x.strip()]:
        t_ = time.time()
        arm_key = f"NUD{w}-{k}" if (len(WINDOWS) > 1 or w != 24) else f"NUD-{k}"
        if not want(arm_key):
            continue
        ARMS[arm_key] = nudge_chain(k, ALPHA, window_h=w)
        print(f"   {arm_key}: window={w}h, alpha={ALPHA:.2f} (tau={args.nud_tau} h), obs={args.nud_obs}, {time.time()-t_:.1f} s")

# =============================================================================
# 6b. ML-specific methods: differentiable GraphCast step, model-Jacobian balance (JAC),
#     two-frame strong-constraint 4D-Var (4DV)
# =============================================================================
H_ = args.long_nud
HYB_VARIANTS = [v.strip() for v in args.hyb_variants.split(",") if v.strip()]
HYB_KINDS = [x.strip() for x in args.hyb_types.split(",") if x.strip()]
_need_jac = any(k == "JAC" for k in HYB_KINDS) or want("JAC-2F") or (WANT is not None and any("JAC" in a for a in WANT))
_need_4dv = want("4DV") or (H_ and want(f"HYB{H_}-4DV")) or (WANT is not None and any("4DV" in a for a in WANT))
if WANT is None:        # with --arms all, only build the gradient arms when explicitly listed in --hyb-types
    _need_jac = "JAC" in HYB_KINDS
    _need_4dv = "4DV" in HYB_KINDS
GRAD_CHECKS = {}

if _need_jac or _need_4dv:
    import jax.numpy as jnp
    import scipy.ndimage as ndi
    from scipy.optimize import minimize
    import jax.scipy.sparse.linalg  # noqa: F401  (CG for incremental 4D-Var)
    try:
        from graphcast import xarray_jax as XJ
    except ImportError:
        try:
            from weathernext.utils import xarray_jax as XJ
        except ImportError:
            import xarray_jax as XJ

    def make_step(start):
        """Differentiable one-step map (A, B) -> C on dicts of jnp arrays in REST order.
        Inputs are frames start, start+1; output is frame start+2. Float32 predictor."""
        if os.environ.get("MLDA_FAKE_STEP"):                   # test harness only
            return _fake_step
        inp_t, tgt_t, frc = window(start, 1)
        tgt_nan = tgt_t * np.nan
        perm = {v: [(["batch", "time"] + REST[v]).index(d) for d in inp_t[v].dims] for v in STATE_VARS}

        def step(A, B):
            new = {}
            for v in STATE_VARS:
                arr = jnp.stack([jnp.asarray(A[v], jnp.float32), jnp.asarray(B[v], jnp.float32)])[None]
                arr = jnp.transpose(arr, perm[v])
                new[v] = XJ.DataArray(arr, dims=inp_t[v].dims,
                                      coords={c: inp_t[v].coords[c] for c in inp_t[v].coords})
            inp = inp_t.assign(new)
            out = _jit32(rng=jax.random.PRNGKey(0), inputs=inp, targets_template=tgt_nan, forcings=frc)[0]
            C = {}
            for v in STATE_VARS:
                a = XJ.unwrap_data(out[v])
                dims = list(out[v].dims)
                a = a[tuple(0 if d in ("batch", "time") else slice(None) for d in dims)]
                rest = [d for d in dims if d not in ("batch", "time")]
                C[v] = jnp.transpose(a, [rest.index(d) for d in REST[v]])
            return C
        return step

    def _fake_step(A, B):                                     # mirrors the stub rollout (tests only)
        C = {}
        for v in STATE_VARS:
            nxt = B[v] + 0.3 * (B[v] - A[v])
            nxt = 0.9 * nxt + 0.1 * jnp.roll(nxt, 1, axis=-1)
            nxt = 0.96 * nxt + 0.04 * (jnp.roll(nxt, 1, -2) + jnp.roll(nxt, -1, -2)) / 2
            C[v] = nxt
        l1000 = LIDX[1000]
        C["temperature"] = C["temperature"].at[l1000].add(0.1 * (B["2m_temperature"] - B["temperature"][l1000]))
        return C

    def _to_np(d):
        return {v: np.asarray(a, dtype=np.float32) for v, a in d.items()}

    def grad_selftest(A, B, start):
        """(1) differentiable step vs the operational forward step; (2) JVP vs finite difference."""
        step = make_step(start)
        C_diff = _to_np(step(A, B))
        C_fwd = pred_frame(forecast(A, B, start, 1), 0)
        GRAD_CHECKS["step32_vs_fwd_bf16_conus_rms_2t_K"] = wrms(C_diff["2m_temperature"] - C_fwd["2m_temperature"], CONUS_LAND)
        v = np.zeros_like(A["2m_temperature"]); v[CONUS_LAND] = 1.0
        tA = {k: jnp.zeros_like(jnp.asarray(a)) for k, a in A.items()}; tB = dict(tA)
        tA["2m_temperature"] = jnp.asarray(v); tB["2m_temperature"] = jnp.asarray(v)
        _, jv = jax.jvp(step, (_jnp(A), _jnp(B)), (tA, tB))
        eps = 0.1
        Ap = copy_frame(A); Bp = copy_frame(B)
        Ap["2m_temperature"] = Ap["2m_temperature"] + eps * v; Bp["2m_temperature"] = Bp["2m_temperature"] + eps * v
        Cp = _to_np(step(Ap, Bp))
        fd = (Cp["temperature"][LIDX[925]] - C_diff["temperature"][LIDX[925]]) / eps
        tl = np.asarray(jv["temperature"][LIDX[925]])
        num = wrms(fd - tl, CONUS_LAND); den = max(wrms(fd, CONUS_LAND), 1e-9)
        GRAD_CHECKS["jvp_vs_fd_T925_rel_err"] = num / den
        GRAD_CHECKS["jvp_T925_response_rms_per_K"] = wrms(tl, CONUS_LAND)
        print("   gradient self-test:", {k: round(v_, 4) for k, v_ in GRAD_CHECKS.items()})
        if GRAD_CHECKS["jvp_vs_fd_T925_rel_err"] > 0.2:
            print("   WARNING: JVP and finite difference disagree by >20 % — check the xarray_jax wrapping.")

    def _jnp(d):
        return {k: jnp.asarray(a, jnp.float32) for k, a in d.items()}

    SIG_PTS = (args.jac_smooth_km / 111.0 / (LATS[1] - LATS[0]),
               args.jac_smooth_km / (111.0 * np.cos(np.deg2rad(40.0))) / (LONS[1] - LONS[0]))

    def smooth(f):
        return ndi.gaussian_filter(f, sigma=SIG_PTS, mode=("nearest", "wrap"))

    JAC_CLIP = {"temperature": 2.0, "specific_humidity": 2e-3, "u_component_of_wind": 5.0,
                "v_component_of_wind": 5.0, "geopotential": 200.0, "mean_sea_level_pressure": 300.0,
                "10m_u_component_of_wind": 5.0, "10m_v_component_of_wind": 5.0}   # per K of 2 m T

    def compute_jac_balance(A, B, start, K):
        """Model-Jacobian balance: regress the tangent-linear 6 h response of every variable/level
        on the response of 2 m T, for K random smooth 2 m T perturbations over the OI box (both
        input frames). Local regression (smoothed moments) gives spatially varying coefficients."""
        step = make_step(start)
        rng = np.random.default_rng(args.seed + 7)
        Aj, Bj = _jnp(A), _jnp(B)
        if args.jac_vars == "all":
            targets = [("2m_temperature", None), ("10m_u_component_of_wind", None), ("10m_v_component_of_wind", None),
                       ("mean_sea_level_pressure", None)]
            _vv = ("temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential")
        else:
            targets = [("2m_temperature", None)]
            _vv = ("temperature",) if args.jac_vars == "T" else ("temperature", "geopotential")
        for v in _vv:
            targets += [(v, LIDX[p]) for p in LEVELS if p >= args.col_top]
        num = {t: 0.0 for t in targets}; den = 0.0
        for k in range(K):
            pert = smooth(rng.normal(size=A["2m_temperature"].shape)) * OI_BOX
            pert = (pert / (pert[OI_BOX].std() + 1e-12)).astype(np.float32)
            tA = {x: jnp.zeros_like(a) for x, a in Aj.items()}; tB = dict(tA)
            tA["2m_temperature"] = jnp.asarray(pert); tB["2m_temperature"] = jnp.asarray(pert)
            _, jv = jax.jvp(step, (Aj, Bj), (tA, tB))
            r2 = np.asarray(jv["2m_temperature"])
            den = den + r2 * r2
            for (v, li) in targets:
                r = np.asarray(jv[v]) if li is None else np.asarray(jv[v][li])
                num[(v, li)] = num[(v, li)] + r * r2
        den_s = smooth(den) + 1e-6
        out = {}
        for (v, li) in targets:
            if v == "2m_temperature":
                continue
            c = smooth(num[(v, li)]) / den_s * OI_BOX
            lim = JAC_CLIP.get(v, None)
            if lim is not None:
                c = np.clip(c, -lim, lim)
            out[(v, li)] = c.astype(np.float32)
        prof = {LEVELS[li]: float(np.mean(out[("temperature", li)][CONUS_LAND]))
                for (v, li) in out if v == "temperature"}
        print("   JAC balance: mean dT(p)/dT2m over CONUS land:",
              {p: round(c, 3) for p, c in sorted(prof.items(), reverse=True)})
        return out, prof

    # ---- 4D-Var ---------------------------------------------------------------------------
    CTRL_SIG = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.fdv_sig.split(",")}
    _rows = np.where(OI_BOX.any(axis=1))[0]; _cols = np.where(OI_BOX.any(axis=0))[0]
    R0, R1, C0, C1 = _rows.min(), _rows.max() + 1, _cols.min(), _cols.max() + 1
    CTRL_LEV = [LIDX[p] for p in LEVELS if p >= args.col_top]

    def _gauss_kernel(sig):
        r = int(np.ceil(3 * sig)); x = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (x / sig) ** 2); return (k / k.sum()).astype(np.float32)

    _KY_1D = _gauss_kernel(args.fdv_L / 111.0 / (LATS[1] - LATS[0]))
    _KX_1D = _gauss_kernel(args.fdv_L / (111.0 * np.cos(np.deg2rad(40.0))) / (LONS[1] - LONS[0]))

    def _conv_mat(k, n):
        r = (len(k) - 1) // 2
        M = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                m = j - i + r
                if 0 <= m < len(k):
                    M[i, j] = k[m]
        return jnp.asarray(M, dtype=jnp.float32)

    _KY = _conv_mat(_KY_1D, R1 - R0)
    _KX = _conv_mat(_KX_1D, C1 - C0)

    def _bsqrt(chi):
        """B^1/2: separable Gaussian smoothing of a (..., ny, nx) control field via GEMM (avoids cuDNN)."""
        return _KY @ chi @ _KX.T

    def _interp_idx(la, lo):
        lo = np.mod(lo, 360.0)
        fi = (la - LATS[0]) / (LATS[1] - LATS[0]); fj = (lo - LONS[0]) / (LONS[1] - LONS[0])
        i0 = np.clip(np.floor(fi).astype(int), 0, len(LATS) - 2); j0 = np.floor(fj).astype(int) % len(LONS)
        wi = np.clip(fi - i0, 0, 1); wj = fj - np.floor(fj)
        return i0, j0, (j0 + 1) % len(LONS), wi, wj

    def _H(f, ii):
        i0, j0, j1, wi, wj = ii
        return ((1 - wi) * (1 - wj) * f[i0, j0] + (1 - wi) * wj * f[i0, j1]
                + wi * (1 - wj) * f[i0 + 1, j0] + wi * wj * f[i0 + 1, j1])

    def fourdvar(Ab, Bb, start, era5_anchor=False, tag="4dv", relax_t0=None, target=None):
        """Strong-constraint two-frame 4D-Var over the window [t0-12h, t0].
        Control: increments to (x_{-12}, x_{-6}) for 2 m T and T, q, u, v, Z at levels >= --col-top,
        in B^1/2 space (Gaussian correlation L=--fdv-L, std --fdv-sig), over the OI box.
        Cost: 1/2|chi|^2 + obs(t0-6h) on x_{-6} + obs(t0) on F(x_{-12}, x_{-6})
              [+ ERA5 anchor at t0 on all state variables, HYB-4DV only].
        Returns the model-consistent launch pair (x_{-6}^a, F(x_{-12}^a, x_{-6}^a))."""
        step = make_step(start)
        Aj, Bj = _jnp(Ab), _jnp(Bb)
        C_b = _to_np(step(Aj, Bj))
        obs_terms = []
        for idx, bg in ((start + 1, Bb["2m_temperature"]), (start + 2, C_b["2m_temperature"])):
            d = obs_at(idx)
            la, lo, innov, so2k, _, _ = qc_innovations(bg, d)
            ii = _interp_idx(la, lo)
            y = interp2(bg, la, lo) + innov
            so2k = so2k * args.fdv_sigo_scale ** 2
            obs_terms.append((jnp.asarray(y, jnp.float32), jnp.asarray(so2k, jnp.float32), ii))
        ctrl = [(v, None) for v in ("2m_temperature",) if v in CTRL_SIG] + \
               [(v, CTRL_LEV) for v in ("temperature", "specific_humidity", "u_component_of_wind",
                                        "v_component_of_wind", "geopotential") if v in CTRL_SIG]
        shapes = []
        for v, lev in ctrl:
            shapes.append((2,) + ((len(lev),) if lev else ()) + (R1 - R0, C1 - C0))
        sizes = [int(np.prod(sh)) for sh in shapes]
        mask_box = jnp.asarray(OI_BOX[R0:R1, C0:C1], jnp.float32)
        target = target or era5_frame
        E0 = _jnp(target(start + 2)) if era5_anchor else None

        def unpack(z):
            out, o = [], 0
            for sh, n in zip(shapes, sizes):
                out.append(z[o:o + n].reshape(sh)); o += n
            return out

        def analysed(z):
            A, B = dict(Aj), dict(Bj)
            for (v, lev), chi in zip(ctrl, unpack(z)):
                sig = CTRL_SIG[v]
                for f, X in ((0, A), (1, B)):
                    c = chi[f]
                    inc = _bsqrt(c) * sig * mask_box
                    if lev is None:
                        X[v] = X[v].at[R0:R1, C0:C1].add(inc)
                    else:
                        X[v] = X[v].at[jnp.asarray(lev), R0:R1, C0:C1].add(inc)
            return A, B

        W_ERA5 = args.fdv_era5_weight if args.fdv_era5_weight is not None else (0.0 if relax_t0 is not None else 1.0)
        N_OBS = float(sum(len(o[0]) for o in obs_terms))

        def cost_parts(z):
            A, B = analysed(z)
            C = step(A, B)
            Jb = 0.5 * jnp.sum(z * z)
            Jo = [0.5 * jnp.sum((_H(fld, ii) - y) ** 2 / so2)
                  for (y, so2, ii), fld in zip(obs_terms, (B["2m_temperature"], C["2m_temperature"]))]
            Ja = 0.0
            if E0 is not None and W_ERA5 > 0:                  # ERA5 anchor at t0 over the control region
                for v, lev in ctrl:
                    sig = CTRL_SIG[v]
                    if lev is None:
                        dv = C[v][R0:R1, C0:C1] - E0[v][R0:R1, C0:C1]
                    else:
                        dv = C[v][jnp.asarray(lev), R0:R1, C0:C1] - E0[v][jnp.asarray(lev), R0:R1, C0:C1]
                    Ja = Ja + 0.5 * W_ERA5 * jnp.sum((dv / sig) ** 2 * mask_box)
            return Jb, Jo[0], Jo[1], Ja

        def cost(z):
            Jb, Jo1, Jo2, Ja = cost_parts(z)
            return (Jb + Jo1 + Jo2 + Ja) / N_OBS                 # O(1) scaling for the optimizer

        def obs_resid(z):
            """Normalized obs residuals (H x - y)/sigma_o at t0-6h and t0 (for Desroziers diagnostics)."""
            A, B = analysed(z)
            C = step(A, B)
            return [(_H(fld, ii) - y) / jnp.sqrt(so2)
                    for (y, so2, ii), fld in zip(obs_terms, (B["2m_temperature"], C["2m_temperature"]))]

        def resid(z):
            """All weighted residuals r(z): J(z) = 1/2 |z|^2 + 1/2 |r(z)|^2 (same cost as cost_parts)."""
            A, B = analysed(z)
            C = step(A, B)
            rs = [((_H(fld, ii) - y) / jnp.sqrt(so2))
                  for (y, so2, ii), fld in zip(obs_terms, (B["2m_temperature"], C["2m_temperature"]))]
            if E0 is not None and W_ERA5 > 0:
                sm = jnp.sqrt(mask_box)
                for v, lev in ctrl:
                    sig = CTRL_SIG[v]
                    if lev is None:
                        dv = C[v][R0:R1, C0:C1] - E0[v][R0:R1, C0:C1]
                    else:
                        dv = C[v][jnp.asarray(lev), R0:R1, C0:C1] - E0[v][jnp.asarray(lev), R0:R1, C0:C1]
                    rs.append((jnp.sqrt(W_ERA5) * dv / sig * sm).ravel())
            return jnp.concatenate([r.ravel() for r in rs])

        parts = jax.jit(cost_parts)
        ores = jax.jit(obs_resid)

        def _parts(zz):
            return [float(x) for x in parts(jnp.asarray(zz, jnp.float32))]

        t_ = time.time()
        z = np.zeros(sum(sizes), np.float64)
        P0 = _parts(z)
        it_total, msgs, hist, inner_log = 0, [], [], []

        if args.fdv_solver == "gn":
            # Memory-safe incremental 4D-Var: the tangent-linear (J v, forward-mode jvp) and adjoint
            # (J^T u, reverse-mode vjp) are two separately compiled functions called one after the other,
            # so peak GPU memory = that of one gradient (same as L-BFGS). CG runs on the host in float64.
            resid_j = jax.jit(resid)
            tl_j = jax.jit(lambda zk, v: jax.jvp(resid, (zk,), (v,))[1])
            ad_j = jax.jit(lambda zk, u: jax.vjp(resid, zk)[1](u)[0])
            f32 = lambda a: jnp.asarray(a, jnp.float32)
            f64 = lambda a: np.asarray(a, np.float64)
            J_cur = sum(P0)
            for k in range(max(1, args.fdv_outer)):
                t_k = time.time()
                zk = f32(z)
                rk = f64(resid_j(zk))
                b = -(z + f64(ad_j(zk, f32(rk))))                 # -grad J at zk
                gn = float(np.linalg.norm(b))

                def hess(v):                                      # (I + J^T J) v
                    return v + f64(ad_j(zk, tl_j(zk, f32(v))))
                dz = np.zeros_like(z); r_ = b.copy(); p_ = r_.copy(); rr = float(r_ @ r_)
                n_in = 0
                for n_in in range(1, args.fdv_inner + 1):         # conjugate gradient (Hestenes-Stiefel)
                    Ap = hess(p_)
                    a_ = rr / float(p_ @ Ap)
                    dz += a_ * p_; r_ -= a_ * Ap
                    rr_new = float(r_ @ r_)
                    if n_in == 1 and k == 0:
                        print(f"      outer {k+1}: first CG iteration {time.time()-t_k:.0f} s "
                              f"(incl. compiling the TL and adjoint)", flush=True)
                    if np.sqrt(rr_new) / (gn + 1e-30) < args.fdv_cg_tol:
                        break
                    p_ = r_ + (rr_new / rr) * p_; rr = rr_new
                rel = np.sqrt(rr_new) / (gn + 1e-30)
                rl = rk + f64(tl_j(zk, f32(dz)))
                Jq = 0.5 * float((z + dz) @ (z + dz)) + 0.5 * float(rl @ rl)   # predicted (linear) cost
                if (J_cur - Jq) / max(J_cur, 1.0) < 1e-4:        # linear model predicts no further gain
                    msgs.append(f"converged: predicted decrease < 1e-4 at outer {k+1}")
                    print(f"      outer {k+1}: predicted decrease {J_cur - Jq:.2f} -> converged")
                    break
                step_len, J_new = 1.0, None
                for _bt in range(4):                              # guard against nonlinearity
                    J_try = sum(_parts(z + step_len * dz))
                    if J_try < J_cur:
                        J_new = J_try; break
                    step_len *= 0.5
                if J_new is None:
                    msgs.append(f"outer {k+1}: no decrease after backtracking (nonlinear), stopped")
                    print(f"      outer {k+1}: no decrease after backtracking; stop")
                    break
                z = z + step_len * dz
                it_total += 1
                inner_log.append(dict(outer=k + 1, J=J_new, J_pred=Jq, cg_iters=n_in, cg_rel_resid=float(rel),
                                      grad_norm=gn, step=step_len, seconds=round(time.time() - t_k, 1)))
                print(f"      outer {k+1}: J {J_cur:.1f} -> {J_new:.1f} (linear prediction {Jq:.1f}), "
                      f"{n_in} CG its, CG rel. residual {rel:.1e}, |grad| {gn:.1f}, step {step_len:g} "
                      f"({time.time()-t_k:.0f} s)", flush=True)
                done = (J_cur - J_new) / max(J_cur, 1.0) < 1e-3
                J_cur = J_new
                if done:
                    msgs.append(f"converged: relative decrease < 1e-3 at outer {k+1}")
                    break
            else:
                msgs.append(f"finished {args.fdv_outer} outer loops")
        else:
            vg = jax.jit(jax.value_and_grad(cost))

            def fun(zz):
                J, g = vg(jnp.asarray(zz, jnp.float32))
                hist.append(float(J) * N_OBS)
                return float(J), np.asarray(g, np.float64)

            for r in range(1 + max(0, args.fdv_restarts)):
                left = args.fdv_iter - it_total
                if left <= 0:
                    break
                J_before = fun(z)[0]
                res = minimize(fun, z, jac=True, method="L-BFGS-B",
                               options=dict(maxiter=left, maxcor=20, maxls=50, ftol=1e-10, gtol=1e-7))
                z = res.x; it_total += int(res.nit)
                msgs.append(str(res.message))
                gain = (J_before - float(res.fun)) / max(abs(J_before), 1e-12)
                stalled = ("ABNORMAL" in str(res.message).upper() or "FACTR" in str(res.message).upper())
                if not stalled or res.nit == 0 or gain < 1e-4:     # restart only if the last pass still made progress
                    break
            g0 = float(np.linalg.norm(fun(np.zeros_like(z))[1])); g1 = float(np.linalg.norm(fun(z)[1]))
            msgs[-1] += f" | |grad| {g0:.3g} -> {g1:.3g} ({g1 / max(g0, 1e-30):.1e} of start; <~1e-2 = truly converged)"
        P1 = _parts(z)
        A_a, B_a = analysed(jnp.asarray(z, jnp.float32))
        A_a, B_a = _to_np(A_a), _to_np(B_a)
        rb = [np.asarray(r, np.float64) for r in ores(jnp.zeros(sum(sizes), jnp.float32))]
        ra = [np.asarray(r, np.float64) for r in ores(jnp.asarray(z, jnp.float32))]
        # Desroziers: E[(y-Hx_a)(y-Hx_b)] = sigma_o^2  -> ratio of true to assumed obs-error std
        des = [float(np.sqrt(max(np.mean(a * b), 0.0))) for a, b in zip(ra, rb)]
        chi2b = [float(np.mean(b * b)) for b in rb]           # innovation variance / sigma_o^2 (expect 1 + HBH'/R)
        FDV_LOG[tag] = dict(J0=sum(P0), J_final=sum(P1), n_iter=it_total, n_eval=len(hist), solver=args.fdv_solver, outer=inner_log,
                            desroziers_sigo_ratio=des, innov_chi2=chi2b, sigo_scale=args.fdv_sigo_scale,
                            restarts=len(msgs) - 1, stop=msgs, n_obs=[int(len(o[0])) for o in obs_terms],
                            Jb=[P0[0], P1[0]], Jo_tm6=[P0[1], P1[1]], Jo_t0=[P0[2], P1[2]], J_era5=[P0[3], P1[3]],
                            era5_weight=W_ERA5, seconds=round(time.time() - t_, 1))
        print(f"   {tag} [{args.fdv_solver}]: J {sum(P0):.1f} -> {sum(P1):.1f} in {it_total} "
              f"{'outer loops' if args.fdv_solver == 'gn' else 'iterations'}, {max(len(msgs) - 1, 0)} restarts "
              f"({time.time()-t_:.0f} s); stop: {msgs[-1]}")
        print(f"      Jb 0 -> {P1[0]:.1f} | Jo(t0-6h) {P0[1]:.1f} -> {P1[1]:.1f} | Jo(t0) {P0[2]:.1f} -> {P1[2]:.1f}"
              f" | J_ERA5 {P0[3]:.1f} -> {P1[3]:.1f}   (N_obs = {int(N_OBS)}; well fitted: Jo ~ N_obs/2 per time)")
        print(f"      innovation chi2/obs (t0-6h, t0) = {chi2b[0]:.2f}, {chi2b[1]:.2f}  |  Desroziers sigma_o ratio "
              f"(true/assumed) = {des[0]:.2f}, {des[1]:.2f}  (sigo-scale {args.fdv_sigo_scale:g}; ~1 = consistent)")
        C_a = pred_frame(forecast(A_a, B_a, start, 1), 0)      # launch pair with the operational forward
        if relax_t0 is not None:                                 # HYB-4DV: same t0 anchoring as HYB-DIR
            C_bf = pred_frame(forecast(Ab, Bb, start, 1), 0)
            C_r = relax_to_era5(C_bf, target(start + 2), relax_t0)
            C_a = {v: (C_r[v] + (C_a[v] - C_bf[v])).astype(np.float32) for v in STATE_VARS}
            FDV_LOG[tag]["t0_relaxed"] = True
        return B_a, C_a

    FDV_LOG = {}
    if not args.skip_checks and args.grad_selftest:
        grad_selftest(BASE_A, BASE_B, I0 - 1)

    if _need_jac:
        t_ = time.time()
        JAC_B, JAC_PROF = compute_jac_balance(BASE_A, BASE_B, I0 - 1, args.jac_k)
        print(f"   JAC balance from {args.jac_k} tangent-linear runs: {time.time()-t_:.1f} s")
        if want("JAC-2F"):
            ARMS["JAC-2F"] = (apply_increment(BASE_A, INC_A, "JAC"), apply_increment(BASE_B, INC_B, "JAC"))
    if _need_4dv and args.base == "bg" and M2FIELD is None:
        if want("4DV"):
            ARMS["4DV"] = fourdvar(BG_CHAIN[I0 - 2], BG_CHAIN[I0 - 1], I0 - 2, tag="4DV")
        if H_ and want(f"HYB{H_}-4DV") and args.hyb_tau > 0:
            A_E = 1.0 - np.exp(-6.0 / args.hyb_tau)
            Ab, Bb = hybrid_chain("DIR", H_, A_E, ALPHA, end_idx=I0 - 1, last_obs=False, tag=f"hyb{H_}-4dvbg")
            ARMS[f"HYB{H_}-4DV"] = fourdvar(Ab, Bb, I0 - 2, era5_anchor=True, tag=f"HYB{H_}-4DV",
                                            relax_t0=A_E if args.fdv_relax_t0 else None)
        _prov_4dv = []
        if DSP is not None:
            _prov_4dv.append(("M", provider_frame))
        if CLIM_E is not None:
            _prov_4dv.append(("MQM", lambda i: mapped_frame(i, "QM")))
        for _suf, _tgt in _prov_4dv:
            if H_ and want(f"HYB{H_}-4DV-{_suf}") and args.hyb_tau > 0:
                A_E = 1.0 - np.exp(-6.0 / args.hyb_tau)
                Ab, Bb = hybrid_chain("DIR", H_, A_E, ALPHA, end_idx=I0 - 1, last_obs=False,
                                      tag=f"hyb{H_}-4dvbg-{_suf.lower()}", target=_tgt)
                ARMS[f"HYB{H_}-4DV-{_suf}"] = fourdvar(Ab, Bb, I0 - 2, era5_anchor=True, tag=f"HYB{H_}-4DV-{_suf}",
                                                      relax_t0=A_E if args.fdv_relax_t0 else None, target=_tgt)

# hybrid-cycle variants (weighted 4DIAU, level-selective replay, bias correction; any station kind incl. JAC)
if H_ and args.base == "bg" and M2FIELD is None and args.hyb_tau > 0:
    A_E = 1.0 - np.exp(-6.0 / args.hyb_tau)
    for var in HYB_VARIANTS:
        if var == "base":
            continue
        opts = var.split("+")
        wprof = next((o.split("=")[1] for o in opts if o.startswith("W=")), "const")
        ls, bc = "LS" in opts, "BC" in opts
        for k in HYB_KINDS:
            if k == "4DV":
                continue
            name = f"HYB{H_}-{k}-{var.replace('W=', 'W')}"
            if not want(name):
                continue
            t_ = time.time()
            ARMS[name] = hybrid_chain(k, H_, A_E, ALPHA, weights=wprof, ls=ls, bc=bc, tag=name)
            print(f"   {name}: weights={wprof}, level-selective={ls}, bias-corr={bc}: {time.time()-t_:.1f} s")
    for k in HYB_KINDS:                               # JAC as a plain hybrid kind
        if k == "JAC" and want(f"HYB{H_}-JAC") and JAC_B is not None:
            ARMS[f"HYB{H_}-JAC"] = hybrid_chain("JAC", H_, A_E, ALPHA, tag=f"HYB{H_}-JAC")

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
if args.arms != "all":
    keep = {"ERA5", "BASE"} | {a.strip() for a in args.arms.split(",")}
    ARMS = {k: v for k, v in ARMS.items() if k in keep}
print(f"   arms: {list(ARMS)}")
print(f"\n[8] Running {len(ARMS)} forecasts x {args.steps} steps ...")
FC, ROWS = {}, []
for name, (A, B) in ARMS.items():
    t_ = time.time()
    FC[name] = forecast(A, B, I0 - 1, args.steps)
    if args.lite:
        FC[name] = lite_forecast(FC[name])
    print(f"   {name:8s} {time.time()-t_:5.1f} s")

NOISE_FC = forecast(TRUE_A, TRUE_B, I0 - 1, args.steps)        # rerun of ERA5 arm = run-to-run noise
if args.lite:
    NOISE_FC = lite_forecast(NOISE_FC)
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


def station_err(field2t, df):
    df = _vfilter(df)
    if df is None or df.empty:
        return np.array([])
    return interp2(field2t, df.lat.values, df.lon.values) - df.y.values


BOOT = []
_rngb = np.random.default_rng(args.seed + 1)
for k, lead in enumerate(LEADS):
    tlead = pd.Timestamp(DATETIMES[I0 + 1 + k])
    nets = {"uscrn": None if VER is None else VER[VER.time == tlead],
            "withheld": ALLDF[(ALLDF.time == tlead) & ALLDF.sid.isin(HOLD_SIDS)]}
    for net, df in nets.items():
      for ref in ("BASE", "ERA5"):
        if ref == "ERA5" and args.base == "era5":
            continue
        eb = station_err(pred_frame(FC[ref], k)["2m_temperature"], df)
        if len(eb) < 5:
            continue
        idx = _rngb.integers(0, len(eb), size=(args.boot, len(eb)))
        rb = np.sqrt(np.mean(eb[idx] ** 2, axis=1))
        rb0 = np.sqrt(np.mean(eb ** 2))
        for name in ARMS:
            if name in ("BASE", ref):
                continue
            ea = station_err(pred_frame(FC[name], k)["2m_temperature"], df)
            ra = np.sqrt(np.mean(ea[idx] ** 2, axis=1))
            dpct = 100 * (ra - rb) / rb
            BOOT.append(dict(arm=name, lead_h=lead, net=net, ref=ref, n_st=len(eb),
                             d_pct=100 * (np.sqrt(np.mean(ea ** 2)) - rb0) / rb0,
                             lo=float(np.percentile(dpct, 2.5)), hi=float(np.percentile(dpct, 97.5))))
BOOT = pd.DataFrame(BOOT)
if len(BOOT):
    BOOT["sig"] = (BOOT.hi < 0) | (BOOT.lo > 0)
    BOOT.to_csv(os.path.join(OUT, "bootstrap_station_diffs.csv"), index=False)
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
ORDER = ["ERA5", "BASE", "BIAS", "DIR-1F", "DIR-2F", "COL-2F", "BAL-2F", "REG-2F", "COL-FIX", "BAL-FIX",
         "COL-PBL", "BAL-PBL", "IAU-DIR", "IAU-PBL", "IAU-BAL-PBL"] + \
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
       "NUD24-DIR": "#e377c2", "NUD24-BAL": "#2ca02c", "NUD24-FIX": "#9467bd",
       "BIAS": "#7f7f7f", "COL-PBL": "#bcbd22", "BAL-PBL": "#8c564b", "IAU-DIR": "#ff9896", "IAU-PBL": "#c49c94",
       "IAU-BAL-PBL": "#9edae5", "NUD-BAL-PBL": "#006d2c", "NUD6-BAL-PBL": "#31a354", "NUD12-BAL-PBL": "#74c476",
       "NUD24-BAL-PBL": "#006d2c", "NUD6-PBL": "#a1d99b", "NUD24-PBL": "#41ab5d",
       "NUD72-BAL": "#00441b", "NUD72-BAL-PBL": "#00441b", "NUD72-DIR": "#c51b7d", "FREE72": "#525252",
       "REPLAY72": "#08519c", "HYB72-DIR": "#e6550d", "HYB72-BAL-PBL": "#a63603", "HYB72-BAL": "#fd8d3c",
       "JAC-2F": "#6a3d9a", "4DV": "#b15928", "HYB72-JAC": "#cab2d6", "HYB72-4DV": "#000000",
       "HYB72-DIR-LS": "#fdae6b", "HYB72-DIR-BC": "#fd8d3c", "HYB72-DIR-LS+BC": "#d94801",
       "HYB72-DIR-Wramp-up": "#fdd0a2", "HYB72-DIR-Wramp-down": "#e6550d", "HYB72-DIR-Wtri": "#f16913",
       "M-DIR": "#e7298a", "REPLAY72-M": "#66a61e", "HYB72-M": "#1b9e77",
       "M-MEAN": "#fb9a99", "M-QM": "#e31a1c", "REPLAY72-MQM": "#b2df8a", "HYB72-MQM": "#33a02c",
       "HYB72-4DV-M": "#6a51a3", "HYB72-4DV-MQM": "#3f007d",
       "M-QMS": "#fdbf6f", "M-BAL": "#ff7f00", "E+DM24": "#737373"}
STY = {"ERA5": "--", "BASE": "--", "FREE72": ":", "FREE48": ":", "FREE120": ":", "REPLAY72": "-.", "REPLAY72-M": "-.",
       "M-DIR": "--"}
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
          ("Residual after DIR-2F (ERA5 − arm)", TRUE_B["2m_temperature"] - ARMS.get("DIR-2F", ARMS["BASE"])[1]["2m_temperature"]),
          ("ERA5 − BASE, T850", TRUE_B["temperature"][L850] - BASE_B["temperature"][L850]),
          ("Increment at T850: BAL-2F − BASE", ARMS.get("BAL-2F", ARMS["BASE"])[1]["temperature"][L850] - BASE_B["temperature"][L850]),
          ("Residual T850 after BAL-2F", TRUE_B["temperature"][L850] - ARMS.get("BAL-2F", ARMS["BASE"])[1]["temperature"][L850])]
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

# (8) bootstrap: % change in station RMSE vs BASE with 95 % CI
if len(BOOT):
    BOOT_ALL = BOOT
    BOOT = BOOT_ALL[BOOT_ALL.ref == "BASE"]
    nets = [n for n in ("uscrn", "withheld") if n in set(BOOT.net)]
    lsel = [l for l in (6, 12, 24, 48, 72) if l in LEADS]
    arms_b = [a for a in ORDER if a not in ("BASE",) and a in set(BOOT.arm)]
    fig, axs = plt.subplots(len(nets), 1, figsize=(max(10, 0.9 * len(arms_b) * len(lsel) / 2), 4.2 * len(nets)), squeeze=False)
    for r, net in enumerate(nets):
        ax = axs[r, 0]
        wbar = 0.8 / len(lsel)
        for j, l in enumerate(lsel):
            sub = BOOT[(BOOT.net == net) & (BOOT.lead_h == l)].set_index("arm").reindex(arms_b)
            x = np.arange(len(arms_b)) + (j - (len(lsel) - 1) / 2) * wbar
            ax.errorbar(x, sub.d_pct, yerr=[sub.d_pct - sub.lo, sub.hi - sub.d_pct], fmt="o", ms=3,
                        capsize=2, label=f"+{l} h")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(arms_b))); ax.set_xticklabels(arms_b, rotation=60, fontsize=8)
        n_st = int(BOOT[BOOT.net == net].n_st.median())
        ax.set_ylabel("% change in RMSE vs BASE"); ax.grid(alpha=0.3)
        ax.set_title(f"2 m T vs {net} stations (~{n_st}), paired bootstrap 95 % CI  (<0 = better)")
        ax.legend(fontsize=7, ncol=len(lsel))
    fig.tight_layout()
    savefig(fig, "fig8_bootstrap_station.png")

# (9) PBL diagnosis at t0 for the BASE state
fig = plt.figure(figsize=(14, 4.5))
ax, kw = mapax(fig, (1, 2, 1))
im = ax.pcolormesh(LONS, LATS, np.where(CONUS_LAND, _dep_b, np.nan), cmap="viridis", vmin=0, vmax=300, shading="auto", **kw)
plt.colorbar(im, ax=ax, shrink=0.7, label="hPa"); ax.set_title("Diagnosed mixed-layer depth at t0 (BASE)")
ax, kw = mapax(fig, (1, 2, 2))
w925 = _w_b.get(925, np.zeros_like(_dep_b))
im = ax.pcolormesh(LONS, LATS, np.where(CONUS_LAND, w925, np.nan), cmap="magma", vmin=0, vmax=1, shading="auto", **kw)
plt.colorbar(im, ax=ax, shrink=0.7); ax.set_title("COL-PBL weight at 925 hPa (0 = surface-only)")
savefig(fig, "fig9_pbl_diagnosis.png")

# =============================================================================
# 10. Summary
# =============================================================================
summ = dict(t0=args.t0, steps=args.steps, data=DATA, base=args.base, obs_source=args.obs_source,
            n_stations_used=len(USE_SIDS), n_withheld=len(HOLD_SIDS),
            n_uscrn=0 if VER is None else int(VER.sid.nunique()), obs_noise_K=SIGMA_O, oi_L_km=args.oi_L, col_top_hPa=args.col_top,
            nud_tau_h=args.nud_tau, nud_alpha=float(ALPHA), nud_obs=args.nud_obs,
            regression={"b_T": BT, "R2_T": R2T, "b_Z": BZ, "R2_Z": R2Z}, oi_log=OI_LOG, checks=CHECKS,
            qc=dict(lapse_K_per_km=args.lapse, dz_below=args.dz_below, dz_above=args.dz_above, gross_K=args.gross,
                    bgcheck=args.bgcheck, sigma_inst=args.sigma_inst, sigma_repr=args.sigma_repr),
            pbl=PBL_STATS, grad_checks=globals().get("GRAD_CHECKS", {}), fourdvar=globals().get("FDV_LOG", {}),
            station_bias=BIAS_LOG, jac_profile=globals().get("JAC_PROF", {}))
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
for _ref in ("BASE", "ERA5"):
  if not len(BOOT) or _ref not in set(BOOT_ALL.ref):
    continue
  print(f"\nPAIRED STATION BOOTSTRAP: % change in 2 m T RMSE vs {_ref}  [95 % CI]  (* = significant, <0 = better)")
  if True:
    for net in ("uscrn", "withheld"):
        sub = BOOT_ALL[(BOOT_ALL.net == net) & (BOOT_ALL.ref == _ref)]
        if sub.empty:
            continue
        lsel = [l for l in (6, 12, 24, 48, 72) if l in LEADS]
        print(f"  {net} (~{int(sub.n_st.median())} stations)")
        cell = sub.assign(txt=sub.apply(lambda r: f"{r.d_pct:+5.1f} [{r.lo:+.1f},{r.hi:+.1f}]{'*' if r.sig else ' '}", axis=1))
        t = cell[cell.lead_h.isin(lsel)].pivot(index="arm", columns="lead_h", values="txt")
        print(t.loc[[a for a in ORDER if a in t.index]].to_string())

# ---------------------------------------------------------------------------------------------
# Two-truth verification (with --provider-file): does a forecast started from the foreign analysis
# keep forecasting the foreign analysis, or does the ERA5-trained model pull it toward ERA5?
# ---------------------------------------------------------------------------------------------
if DSP is not None:
    TERR = CONUS_LAND & (ZSFC > 1000.0 * G)
    SPECS = [("t2m", "2m_temperature", None, CONUS_LAND, 1.0, "K"),
             ("mslp", "mean_sea_level_pressure", None, CONUS_LAND, 100.0, "hPa"),
             ("mslp_terrain", "mean_sea_level_pressure", None, TERR, 100.0, "hPa"),
             ("t850", "temperature", L850, CONUS, 1.0, "K"),
             ("z500", "geopotential", L500, DOWNSTREAM, G, "m")]

    def _fld(fr, v, lev):
        if v not in fr:
            return None
        return fr[v][lev] if lev is not None else fr[v]

    def _wmean(x, m):
        w = COSW * m
        return float(np.sum(w * x) / np.sum(w))

    TT = []
    for name in ARMS:
        for k in [-1] + list(range(len(LEADS))):
            idx = I0 + 1 + k
            f = ARMS[name][1] if k < 0 else pred_frame(FC[name], k)
            fe = ARMS["ERA5"][1] if k < 0 else pred_frame(FC["ERA5"], k)
            E_, M_ = era5_frame(idx), provider_frame(idx)
            row = dict(arm=name, lead_h=0 if k < 0 else LEADS[k])
            for key_, v, lev, msk, sc, _u in SPECS:
                a, ae, e, m = _fld(f, v, lev), _fld(fe, v, lev), _fld(E_, v, lev), _fld(M_, v, lev)
                if a is None or ae is None:
                    continue
                row[f"rmse_vsE_{key_}"] = wrms((a - e) / sc, msk)
                row[f"rmse_vsM_{key_}"] = wrms((a - m) / sc, msk)
                row[f"bias_vsE_{key_}"] = _wmean((a - e) / sc, msk)
                row[f"bias_vsM_{key_}"] = _wmean((a - m) / sc, msk)
                dme = m - e                                  # analysis difference at the valid time
                den = np.sum(COSW * msk * dme ** 2)
                row[f"retain_{key_}"] = float(np.sum(COSW * msk * (a - ae) * dme) / den) if den > 0 else np.nan
            TT.append(row)
    TT = pd.DataFrame(TT)
    TT.to_csv(os.path.join(OUT, "two_truth_scores.csv"), index=False)
    L2 = [0] + LSHOW

    def _t2(col, fmt=2):
        if col not in TT:
            return "   (not available)"
        t = TT[TT.lead_h.isin(L2)].pivot(index="arm", columns="lead_h", values=col)
        return t.loc[[a for a in ORDER if a in t.index]].round(fmt).to_string()

    print("\n" + "=" * 76)
    print("TWO-TRUTH VERIFICATION: each forecast scored against ERA5 AND against the provider (MERRA-2)")
    print("   lead 0 = the initial state at t0. Stations (above) stay the neutral truth.")
    for key_, _v, _l, _m, _s, u in SPECS:
        if f"rmse_vsE_{key_}" not in TT:
            continue
        lab = {"t2m": "2 m T, CONUS land", "mslp": "MSLP, CONUS land", "mslp_terrain": "MSLP, CONUS land > 1000 m",
               "t850": "T850, CONUS", "z500": "Z500, downstream"}[key_]
        print(f"\n{lab}: RMSE vs ERA5 ({u})")
        print(_t2(f"rmse_vsE_{key_}", 3))
        print(f"{lab}: RMSE vs MERRA-2 ({u})")
        print(_t2(f"rmse_vsM_{key_}", 3))
    print("\nSYSTEMATIC PART: area-mean bias of the forecast vs ERA5 | vs MERRA-2")
    for key_ in ("t2m", "mslp_terrain"):
        if f"bias_vsE_{key_}" in TT:
            print(f"   {key_} vs ERA5"); print(_t2(f"bias_vsE_{key_}", 2))
            print(f"   {key_} vs MERRA-2"); print(_t2(f"bias_vsM_{key_}", 2))
    print("\nIDENTITY RETENTION r = <F_arm - F_ERA5, M - E> / |M - E|^2 at the valid time")
    print("   1 = the forecast keeps the MERRA-2 minus ERA5 difference; 0 = it has become the ERA5-started forecast")
    for key_ in ("t2m", "mslp", "mslp_terrain", "t850", "z500"):
        if f"retain_{key_}" in TT:
            print(f"   {key_}"); print(_t2(f"retain_{key_}", 2))
    summ["two_truth"] = TT.to_dict(orient="records")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    try:
        MARMS = [a for a in ORDER if a in ("ERA5", "BASE") or a.startswith("M-") or a.startswith("E+")
                 or a.endswith("-M") or a.endswith("-MQM")]
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        for a in MARMS:
            sub = TT[TT.arm == a].sort_values("lead_h")
            kw = dict(color=COL.get(a, "k"), marker="o", ms=3)
            axs[0].plot(sub.lead_h, sub.rmse_vsE_t2m, "-", label=f"{a} vs ERA5", **kw)
            axs[0].plot(sub.lead_h, sub.rmse_vsM_t2m, ":", label=f"{a} vs MERRA-2", **kw)
            if "rmse_vsE_z500" in sub:
                axs[1].plot(sub.lead_h, sub.rmse_vsE_z500, "-", **kw)
                axs[1].plot(sub.lead_h, sub.rmse_vsM_z500, ":", **kw)
            if a != "ERA5":
                for key_, ls in (("t2m", "-"), ("mslp_terrain", "--"), ("z500", ":")):
                    if f"retain_{key_}" in sub:
                        axs[2].plot(sub.lead_h, sub[f"retain_{key_}"], ls, color=COL.get(a, "k"),
                                    label=f"{a} {key_}" if a == "M-DIR" else None)
        axs[0].set_title("2 m T RMSE, CONUS land (solid vs ERA5, dotted vs MERRA-2)"); axs[0].set_ylabel("K")
        axs[1].set_title("Z500 RMSE, downstream (solid vs ERA5, dotted vs MERRA-2)"); axs[1].set_ylabel("m")
        axs[2].axhline(1, color="0.6", lw=0.8); axs[2].axhline(0, color="0.6", lw=0.8)
        axs[2].set_title("Identity retention r (1 = keeps MERRA-2, 0 = became ERA5)")
        for ax in axs:
            ax.set_xlabel("lead (h)"); ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7, ncol=2); axs[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_two_truth.png"), dpi=130); plt.close(fig)
        print("   ✓ fig10_two_truth.png")
    except Exception as ex:
        print("   (fig10 skipped:", ex, ")")


# ---------------------------------------------------------------------------------------------
# Fields for global maps (re-plot any time: python scripts/plot_global_maps.py <OUT>/fields.nc)
# ---------------------------------------------------------------------------------------------
if args.save_fields:
    try:
        _FV = {"t2m": ("2m_temperature", None, 1.0), "mslp": ("mean_sea_level_pressure", None, 100.0),
               "t850": ("temperature", L850, 1.0), "z500": ("geopotential", L500, G),
               "tp6h": ("total_precipitation_6hr", None, 1e-3)}
        _L0 = [0] + LEADS
        _anames = list(ARMS)
        FDS = xr.Dataset(coords=dict(arm=_anames, lead_h=np.array(_L0, np.int32), lat=LATS, lon=LONS))

        def _get(fr, v, lev):
            if v not in fr:
                return None
            x = fr[v][lev] if lev is not None else fr[v]
            return np.asarray(x, np.float32)

        for key, (v, lev, sc) in _FV.items():
            arr = np.full((len(_anames), len(_L0), len(LATS), len(LONS)), np.nan, np.float32)
            for ia, a in enumerate(_anames):
                for il in range(len(_L0)):
                    x = _get(ARMS[a][1] if il == 0 else pred_frame(FC[a], il - 1), v, lev)
                    if x is not None:
                        arr[ia, il] = x / sc
            if np.isfinite(arr).any():
                FDS[key] = (("arm", "lead_h", "lat", "lon"), arr)
            for src, fn in (("era5", era5_frame), ("provider", provider_frame if DSP is not None else None)):
                if fn is None or I0 + len(_L0) - 1 >= len(DATETIMES):
                    continue
                an = np.stack([_get(fn(I0 + il), v, lev) / sc for il in range(len(_L0))])
                FDS[f"{key}_{src}"] = (("lead_h", "lat", "lon"), an.astype(np.float32))
        FDS.attrs.update(t0=args.t0, units="t2m K, mslp hPa, t850 K, z500 m, tp6h mm per 6 h",
                         provider=os.path.basename(args.provider_file) if args.provider_file else "")
        FDS.to_netcdf(os.path.join(OUT, "fields.nc"),
                      encoding={k: {"zlib": True, "complevel": 3} for k in FDS.data_vars})
        print(f"   fields.nc: {len(_anames)} arms x {len(_L0)} leads")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import plot_global_maps
        plot_global_maps.make_all(os.path.join(OUT, "fields.nc"), args.clim_era5, args.clim_provider,
                                  arm="M-DIR" if "M-DIR" in ARMS else ("M-QM" if "M-QM" in ARMS else "M-DIR"))
        if args.clim_era5 and args.clim_provider and DSP is not None:
            import score_anomalies
            score_anomalies.main(os.path.join(OUT, "fields.nc"), args.clim_era5, args.clim_provider, DATA)
    except Exception as ex:
        print("   (global maps skipped:", repr(ex)[:200], ")")

print("=" * 76)
print(f"Outputs in {OUT}")
