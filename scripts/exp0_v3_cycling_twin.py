#!/usr/bin/env python3
"""
SUPERSEDED by scripts/exp_main_real_obs.py (use --obs-source era5-synth for this twin).

Experiment 0 v3 — Cycling twin with a model background (single initialization)
==============================================================================
MAIN QUESTION
  What is the best way to insert new surface observations into a frozen,
  ERA5-trained ML model (GraphCast_small) so that the forecast improves?

WHY v3 (vs v1/v2)
  v1/v2 built the background error with the same vertical profile and the same
  hypsometric formula used by the insertion operators ("inverse crime").
  v3 lets the model and the atmosphere define the error:
    * TRUTH       = ERA5 analyses (t0-6h, t0, and every 6 h to the end lead)
    * BACKGROUND  = GraphCast's own 24 h forecast valid at t0-6h and t0,
                    started from ERA5 at t0-30h / t0-24h (an operational-style
                    background: realistic, multivariate, flow-dependent)
    * OBS         = ERA5 2 m T sampled at CONUS land grid points + noise
                    (70 % assimilated, 30 % withheld)
    * VERTICAL SPREADING is *estimated* by regressing the background error
      profile on the 2 m error over a training region OUTSIDE CONUS
      (NH mid-latitude land), never the formula used to create the error.

ARMS (all forecasts start from the pair (t0-6h, t0) and run --steps x 6 h)
  TRUTH    ERA5 pair (upper bound; still contains model error)
  BG       background pair, no observations (lower bound)
  DIR-1F   2 m T increment at t0 only
  DIR-2F   2 m T increment at t0-6h and t0
  COL-2F   + regressed column T increment (levels >= --col-top hPa)
  BAL-2F   COL + hypsometric geopotential from the column T increment
  REG-2F   2 m T + regressed column T + regressed geopotential (statistical balance)
  NUD-<K>  cycling/nudging: step GraphCast from ERA5 at t0-24h, and after every
           6 h step add alpha * (increment of type K) from that time's obs;
           the final pair (t0-6h, t0) is the model's own nudged state.
           alpha = 1 - exp(-6h / tau). NUD with alpha=0 == BG (checked).

SCORES
  * t0 state error vs ERA5 (CONUS): 2t, T and Z by level; withheld-station error
  * Forecast RMSE vs ERA5 analyses at every lead (CONUS 2t, CONUS T850,
    Z500 over CONUS+downstream box)
  * Gap closed  = (E_BG - E_arm) / (E_BG - E_TRUTH)   [0 = BG, 1 = TRUTH]
  * Retention of the t0 2 m T increment, first-step jump, hypsometric residual

DATA (one file covering t0-30h ... t0+steps*6h), download on the LOGIN node:
  python scripts/download_era5_cloud.py --date 2018-01-14 --time 12:00 --steps 16
  -> data/era5/source-era5_date-2018-01-14_res-1.0_levels-13_steps-16.nc
  (launch t0-24h = 2018-01-14 12Z, first frame t0-30h, last frame t0+72h)

RUN (GPU node):
  source activate_env.sh
  python scripts/exp0_v3_cycling_twin.py --t0 2018-01-15T12:00 --steps 12
Outputs: runs/exp0_v3/<t0>/ (scores.csv, summary.json, *.png, *.nc)
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
ap.add_argument("--n-obs", type=int, default=300, help="Pseudo-stations over CONUS land")
ap.add_argument("--withheld-frac", type=float, default=0.3)
ap.add_argument("--obs-noise", type=float, default=0.5, help="Obs error std (K)")
ap.add_argument("--oi-L", type=float, default=250.0, help="OI Gaussian length scale (km)")
ap.add_argument("--col-top", type=int, default=500, help="Top level (hPa) of column increments")
ap.add_argument("--nud-types", default="DIR,BAL", help="Increment types used by nudging arms")
ap.add_argument("--nud-tau", type=float, default=6.0, help="Nudging relaxation time (h)")
ap.add_argument("--nud-obs", default="all", choices=["all", "last2"],
                help="Nudge with obs at all 4 cycles (t0-18..t0) or only t0-6h,t0")
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
OUT = args.outdir or os.path.join(PROJ, "runs", "exp0_v3", TAG)
os.makedirs(OUT, exist_ok=True)
PARAMS = os.path.join(PROJ, "data", "params",
                      "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                      "mesh 2to5 - precipitation input and output.npz")
STATS = os.path.join(PROJ, "data", "stats")
RNG = np.random.default_rng(args.seed)
G, RD = 9.80665, 287.05

print("=" * 76)
print("EXP 0 v3 — CYCLING TWIN WITH MODEL BACKGROUND")
print("=" * 76)
print(f"t0 = {args.t0}   steps = {args.steps} ({args.steps*6} h)   data = {DATA}")
print(f"out = {OUT}")

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
print(f"   background 2t error at t0 (CONUS): {wrms(BG_B['2m_temperature']-TRUE_B['2m_temperature'], CONUS):.3f} K")

# =============================================================================
# 4. Observations and OI
# =============================================================================
print("\n[4] Pseudo-stations and optimal interpolation ...")
cand = np.argwhere(CONUS_LAND)
sel = cand[RNG.choice(len(cand), size=min(args.n_obs, len(cand)), replace=False)]
n_use = int(round((1 - args.withheld_frac) * len(sel)))
ST_USE, ST_HOLD = sel[:n_use], sel[n_use:]
OBS_IDX = [I0 - 3, I0 - 2, I0 - 1, I0]
OBS = {idx: era5_frame(idx)["2m_temperature"][sel[:, 0], sel[:, 1]]
       + RNG.normal(0, args.obs_noise, len(sel)) for idx in OBS_IDX}
print(f"   stations used {len(ST_USE)}, withheld {len(ST_HOLD)}, noise {args.obs_noise} K")

R_EARTH = 6371.0


def gc_dist(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat, dlon = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


_olat, _olon = LATS[ST_USE[:, 0]], LONS[ST_USE[:, 1]]
_C_oo = np.exp(-0.5 * (gc_dist(_olat[:, None], _olon[:, None], _olat[None], _olon[None]) / args.oi_L) ** 2)
_box = np.argwhere(OI_BOX)
_C_go = np.exp(-0.5 * (gc_dist(LATS[_box[:, 0]][:, None], LONS[_box[:, 1]][:, None],
                                _olat[None], _olon[None]) / args.oi_L) ** 2)
OI_LOG = []


def oi_increment(bg2t, idx):
    """Univariate OI of 2 m T: inc = sb2 C_go (sb2 C_oo + so2 I)^-1 d; sb2 from innovations."""
    y = OBS[idx][:n_use]
    d = y - bg2t[ST_USE[:, 0], ST_USE[:, 1]]
    so2 = args.obs_noise ** 2
    sb2 = max(float(np.var(d)) - so2, 0.05)
    w = np.linalg.solve(sb2 * _C_oo + so2 * np.eye(len(d)), d)
    inc = np.zeros_like(bg2t)
    inc[_box[:, 0], _box[:, 1]] = sb2 * (_C_go @ w)
    OI_LOG.append(dict(time=str(DATETIMES[idx]), mean_innov=float(d.mean()),
                       rms_innov=float(np.sqrt((d ** 2).mean())), sigma_b=float(np.sqrt(sb2))))
    return inc.astype(np.float32)

# =============================================================================
# 5. Vertical regression (training region outside CONUS)
# =============================================================================
print("\n[5] Estimating vertical spreading by regression outside CONUS ...")
x_list, yT, yZ = [], {p: [] for p in LEVELS}, {p: [] for p in LEVELS}
for idx in (I0 - 1, I0):
    e = {v: era5_frame(idx)[v] - BG_CHAIN[idx][v] for v in ("2m_temperature", "temperature", "geopotential")}
    x_list.append(e["2m_temperature"][TRAIN])
    for p in LEVELS:
        yT[p].append(e["temperature"][LIDX[p]][TRAIN])
        yZ[p].append(e["geopotential"][LIDX[p]][TRAIN])
X = np.concatenate(x_list); X = X - X.mean()
BT, BZ, R2T, R2Z = {}, {}, {}, {}
for p in LEVELS:
    for yy, B, R2 in ((np.concatenate(yT[p]), BT, R2T), (np.concatenate(yZ[p]), BZ, R2Z)):
        yy = yy - yy.mean()
        b = float(np.sum(X * yy) / np.sum(X * X))
        B[p] = b if p >= args.col_top else 0.0
        R2[p] = float(np.corrcoef(X, yy)[0, 1] ** 2)
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


def apply_increment(frame, inc2t, kind, scale=1.0):
    f = copy_frame(frame)
    inc = scale * inc2t
    f["2m_temperature"] += inc
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
INC_A = oi_increment(BG_A["2m_temperature"], I0 - 1)
INC_B = oi_increment(BG_B["2m_temperature"], I0)
ARMS = {"TRUTH": (TRUE_A, TRUE_B), "BG": (BG_A, BG_B),
        "DIR-1F": (BG_A, apply_increment(BG_B, INC_B, "DIR"))}
for k in ("DIR", "COL", "BAL", "REG"):
    ARMS[f"{k}-2F"] = (apply_increment(BG_A, INC_A, k), apply_increment(BG_B, INC_B, k))

ALPHA = 1.0 - np.exp(-6.0 / args.nud_tau)
nud_times = OBS_IDX if args.nud_obs == "all" else [I0 - 1, I0]


def nudge_chain(kind, alpha):
    A, B = era5_frame(S0), era5_frame(S0 + 1)
    for idx in OBS_IDX:                                  # targets t0-18, -12, -6, t0
        C = pred_frame(forecast(A, B, idx - 2, 1), 0)
        if alpha > 0 and idx in nud_times:
            C = apply_increment(C, oi_increment(C["2m_temperature"], idx), kind, scale=alpha)
        A, B = B, C
    return A, B


for k in [s.strip() for s in args.nud_types.split(",") if s.strip()]:
    t_ = time.time()
    ARMS[f"NUD-{k}"] = nudge_chain(k, ALPHA)
    print(f"   NUD-{k}: alpha={ALPHA:.2f} (tau={args.nud_tau} h), obs={args.nud_obs}, {time.time()-t_:.1f} s")

CHECKS = {}
if not args.skip_checks:
    print("\n[7] Consistency checks ...")
    A0, B0 = nudge_chain("DIR", 0.0)
    CHECKS["stepwise_vs_rollout_maxdiff_2t_K"] = float(np.abs(B0["2m_temperature"] - BG_B["2m_temperature"]).max())
    p1 = forecast(TRUE_A, TRUE_B, I0 - 1, 2)
    p2 = forecast(TRUE_A, TRUE_B, I0 - 1, 2)
    CHECKS["rerun_maxdiff_2t_K"] = float(np.abs(pred_frame(p1, 1)["2m_temperature"]
                                                - pred_frame(p2, 1)["2m_temperature"]).max())
    for k, v in CHECKS.items():
        print(f"   {k}: {v:.3e}")

# =============================================================================
# 8. Forecasts and scores
# =============================================================================
print(f"\n[8] Running {len(ARMS)} forecasts x {args.steps} steps ...")
FC, ROWS = {}, []
for name, (A, B) in ARMS.items():
    t_ = time.time()
    FC[name] = forecast(A, B, I0 - 1, args.steps)
    print(f"   {name:8s} {time.time()-t_:5.1f} s")

LEADS = [6 * (k + 1) for k in range(args.steps)]
L850, L500 = LIDX[850], LIDX[500]
for name in ARMS:
    for k, lead in enumerate(LEADS):
        f, tr = pred_frame(FC[name], k), era5_frame(I0 + 1 + k)
        ROWS.append(dict(arm=name, lead_h=lead,
                         rmse_t2m_conus=wrms(f["2m_temperature"] - tr["2m_temperature"], CONUS),
                         rmse_t850_conus=wrms(f["temperature"][L850] - tr["temperature"][L850], CONUS),
                         rmse_z500_down=wrms((f["geopotential"][L500] - tr["geopotential"][L500]) / G, DOWNSTREAM)))
SC = pd.DataFrame(ROWS)
for m in ("rmse_t2m_conus", "rmse_t850_conus", "rmse_z500_down"):
    piv = SC.pivot(index="lead_h", columns="arm", values=m)
    gap = (piv["BG"] - piv["TRUTH"]).replace(0, np.nan)
    for a in piv.columns:
        SC.loc[SC.arm == a, m.replace("rmse", "gap")] = SC.loc[SC.arm == a, "lead_h"].map(
            (piv["BG"] - piv[a]) / gap).values
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
    hold_obs = OBS[I0][n_use:]
    hold_err = B["2m_temperature"][ST_HOLD[:, 0], ST_HOLD[:, 1]] - hold_obs
    f6 = pred_frame(FC[name], 0)
    T0ROWS.append(dict(
        arm=name,
        t0_err_t2m_conus=wrms(e2, CONUS),
        t0_err_withheld_obs=float(np.sqrt(np.mean(hold_err ** 2))),
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
    if name in ("BG",):
        continue
    d0 = (ARMS[name][1]["2m_temperature"] - BG_B["2m_temperature"]) * CONUS
    nrm = np.sum(COSW * d0 ** 2)
    if nrm <= 0:
        continue
    RET[name] = [float(np.sum(COSW * (pred_frame(FC[name], k)["2m_temperature"]
                                       - pred_frame(FC["BG"], k)["2m_temperature"]) * d0) / nrm)
                 for k in range(args.steps)]

# =============================================================================
# 9. Plots
# =============================================================================
print("\n[9] Plotting ...")
ORDER = ["TRUTH", "BG", "DIR-1F", "DIR-2F", "COL-2F", "BAL-2F", "REG-2F"] + \
        [a for a in ARMS if a.startswith("NUD-")]
ORDER = [a for a in ORDER if a in ARMS]
COL = {"TRUTH": "#222222", "BG": "#9a9a9a", "DIR-1F": "#f4a3a3", "DIR-2F": "#d62728",
       "COL-2F": "#ff7f0e", "BAL-2F": "#1f77b4", "REG-2F": "#17becf",
       "NUD-DIR": "#e377c2", "NUD-COL": "#bcbd22", "NUD-BAL": "#2ca02c", "NUD-REG": "#8c564b"}
STY = {"TRUTH": "--", "BG": "--"}
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
lim = max(1.0, np.percentile(np.abs((TRUE_B["2m_temperature"] - BG_B["2m_temperature"])[CONUS]), 98))
panels = [("Background error (ERA5 − BG), 2 m T", TRUE_B["2m_temperature"] - BG_B["2m_temperature"]),
          ("OI increment at t0 (● used, ✕ withheld)", INC_B),
          ("Residual after DIR-2F (ERA5 − arm)", TRUE_B["2m_temperature"] - ARMS["DIR-2F"][1]["2m_temperature"]),
          ("Background error, T850", TRUE_B["temperature"][L850] - BG_B["temperature"][L850]),
          ("Increment at T850: BAL-2F − BG", ARMS["BAL-2F"][1]["temperature"][L850] - BG_B["temperature"][L850]),
          ("Residual T850 after BAL-2F", TRUE_B["temperature"][L850] - ARMS["BAL-2F"][1]["temperature"][L850])]
for i, (title, fld) in enumerate(panels):
    ax, kw = mapax(fig, (2, 3, i + 1))
    im = ax.pcolormesh(LONS, LATS, fld, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", **kw)
    if i == 1:
        ax.scatter(LONS[ST_USE[:, 1]], LATS[ST_USE[:, 0]], s=6, c="k", **kw)
        ax.scatter(LONS[ST_HOLD[:, 1]], LATS[ST_HOLD[:, 0]], s=12, c="m", marker="x", **kw)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7, label="K")
fig.suptitle(f"Exp 0 v3 — initial state at t0 = {args.t0} UTC", fontsize=13, weight="bold")
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
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for ax, m, ttl in zip(axs, ["rmse_t2m_conus", "rmse_t850_conus", "rmse_z500_down"],
                      ["CONUS 2 m T RMSE (K)", "CONUS T850 RMSE (K)", "Z500 RMSE, CONUS+downstream (m)"]):
    for a in ORDER:
        s = SC[SC.arm == a]
        ax.plot(s.lead_h, s[m], STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
    ax.set_title(ttl); ax.set_xlabel("Lead (h)"); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle("Forecast error vs ERA5 analyses", weight="bold")
savefig(fig, "fig3_rmse_vs_lead.png")

# (4) gap closed
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, m, ttl in zip(axs, ["gap_t2m_conus", "gap_t850_conus"], ["2 m T", "T850"]):
    for a in ORDER:
        if a in ("TRUTH", "BG"):
            continue
        s = SC[SC.arm == a]
        ax.plot(s.lead_h, 100 * s[m], "-", color=COL.get(a, "k"), marker="o", ms=3, label=a)
    ax.axhline(0, color="0.5", lw=0.8); ax.axhline(100, color="k", ls="--", lw=0.8)
    ax.set_title(f"Gap closed, CONUS {ttl}  (0 % = BG, 100 % = TRUTH)")
    ax.set_xlabel("Lead (h)"); ax.set_ylabel("%"); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
savefig(fig, "fig4_gap_closed.png")

# (5) retention of the t0 increment
fig, ax = plt.subplots(figsize=(8, 5))
for a in ORDER:
    if a in RET:
        ax.plot([0] + LEADS, [1] + RET[a], STY.get(a, "-"), color=COL.get(a, "k"), marker="o", ms=3, label=a)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("Lead (h)"); ax.set_ylabel("R(t)")
ax.set_title("Retention of the t0 2 m T increment (vs BG forecast)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
savefig(fig, "fig5_retention.png")

# (6) t0 consistency and accuracy bars
fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
metrics = [("t0_err_withheld_obs", "t0 error at WITHHELD stations (K)"),
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
show = [a for a in ["BG", "DIR-2F", "BAL-2F", "NUD-BAL", "TRUTH"] if a in ARMS]
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
summ = dict(t0=args.t0, steps=args.steps, data=DATA, n_obs_used=int(n_use), n_withheld=int(len(ST_HOLD)),
            obs_noise_K=args.obs_noise, oi_L_km=args.oi_L, col_top_hPa=args.col_top,
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
print("\nGAP CLOSED, CONUS 2 m T (%)  [0 = BG, 100 = TRUTH]")
print((100 * key.loc[[a for a in ORDER if a in key.index]]).round(1).to_string())
print("=" * 76)
print(f"Outputs in {OUT}")
