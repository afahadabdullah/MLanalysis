#!/usr/bin/env python3
"""
Experiment 0: Identical-Twin (OSSE) Insertion Test
===================================================
Answers the central research question of the project against a KNOWN TRUTH:
  "Which observation insertion method recovers the most background error,
   and does physical balance (BAL) outperform equal-heat column insertion (COL)
   and surface-only insertion (DIR)?"

Experiment Design (per PROJECT_PLAN_LEAN.md & RESULTS.md Section 8-9):
  1. Nature Run (Truth):   Unperturbed ERA5 trajectory (F_nature).
  2. Degraded Background:  Analysis with a realistic regional cold/synoptic bias
                           (ΔT = -2.0 K over CONUS, applied at both t-6h and t0).
  3. Synthetic Obs:        Station observations sampled from Nature Run over CONUS.
  4. Insertion Arms (assimilating synthetic observations):
       - BG:   Uncorrected degraded background forecast (no obs).
       - DIR:  Direct surface insertion (ΔT2m = +2.0 K at both t-6h and t0).
       - COL:  Column thermal insertion (ΔT2m + T[1000, 925, 850]) WITHOUT Z balance.
       - BAL:  Balanced column insertion (COL + hypsometric geopotential lift).
  5. Verification:
       - Directly scored against the known Nature Run across 24 hours.
       - Cleanly separates DEPTH (COL vs DIR) from BALANCE (BAL vs COL).

Outputs saved to: <PROJ>/runs/exp0_twin/

Usage:
  python scripts/test_exp0_twin.py
  python scripts/test_exp0_twin.py --error-amp 2.0 --steps 4
"""

import argparse
import dataclasses
import functools
import os
import sys
import time
import warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# Control JAX memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import haiku as hk

try:
    from graphcast import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        graphcast,
        normalization,
        rollout,
    )
except ImportError:
    from weathernext.weathernext1_graph import graphcast
    from weathernext.utils import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        normalization,
        rollout,
    )

# Headless plotting setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Experiment 0: Identical-Twin (OSSE) Insertion Test")
parser.add_argument("--steps", type=int, default=4, help="Forecast steps (6h each, default: 4 = 24h)")
parser.add_argument("--error-amp", type=float, default=2.0,
                    help="Amplitude of background cold bias error in Kelvin. Default: 2.0 K")
parser.add_argument("--lat", type=float, default=38.0, help="Center latitude for error (default: 38.0 N)")
parser.add_argument("--lon", type=float, default=265.0, help="Center longitude for error (default: 265.0 E = 95 W)")
parser.add_argument("--sigma-lat", type=float, default=4.0, help="Lat Gaussian std dev (default: 4.0 deg)")
parser.add_argument("--sigma-lon", type=float, default=6.0, help="Lon Gaussian std dev (default: 6.0 deg)")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/runs/exp0_twin)")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "runs", "exp0_twin")
os.makedirs(OUTDIR, exist_ok=True)

DATA_DIR = os.path.join(PROJ, "data")
PARAMS_FILE = os.path.join(
    DATA_DIR, "params",
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
)
STATS_DIR = os.path.join(DATA_DIR, "stats")
SAMPLE_FILE = os.path.join(
    DATA_DIR, "sample",
    "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
)

print("=" * 68)
print("EXPERIMENT 0: IDENTICAL-TWIN (OSSE) INSERTION BENCHMARK")
print("=" * 68)
print(f"Project root:         {PROJ}")
print(f"Output directory:     {OUTDIR}")
print(f"Background Cold Bias: -{args.error_amp:.1f} K centered at ({args.lat}°N, {args.lon}°E)")
print(f"Observation Recovery: +{args.error_amp:.1f} K synthetic station correction")
print(f"Forecast Horizon:     {args.steps} steps ({args.steps * 6} hours)")

# ---------------------------------------------------------------------------
# 1. Load Checkpoint and Stats
# ---------------------------------------------------------------------------
print("\n[1/6] Loading model checkpoint and normalization stats ...")
with open(PARAMS_FILE, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config

with open(os.path.join(STATS_DIR, "diffs_stddev_by_level.nc"), "rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open(os.path.join(STATS_DIR, "mean_by_level.nc"), "rb") as f:
    mean_by_level = xr.load_dataset(f).compute()
with open(os.path.join(STATS_DIR, "stddev_by_level.nc"), "rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()

# ---------------------------------------------------------------------------
# 2. Prepare Nature Run (Ground Truth)
# ---------------------------------------------------------------------------
print(f"\n[2/6] Loading Nature Run dataset ...")
with open(SAMPLE_FILE, "rb") as f:
    try:
        example_batch = xr.load_dataset(f, decode_timedelta=True).compute()
    except Exception:
        example_batch = xr.load_dataset(f).compute()

eval_inputs_nature, eval_targets_nature, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{args.steps * 6}h"),
    **dataclasses.asdict(task_config),
)

lats = eval_inputs_nature.coords["lat"].values
lons = eval_inputs_nature.coords["lon"].values
levels = eval_inputs_nature.coords["level"].values
times = eval_targets_nature.coords["time"].values
n_steps = len(times)
input_times = eval_inputs_nature.coords["time"].values
n_input_times = len(input_times)

print(f"      Input frames to perturb: {list(input_times)} (n={n_input_times})")
print(f"      Pressure levels (hPa):   {list(levels)}")

# ---------------------------------------------------------------------------
# 3. Synthesize Background Error & Observation Insertion States
# ---------------------------------------------------------------------------
print(f"\n[3/6] Generating degraded background and observation-inserted initial states ...")

# 2D Gaussian error pattern
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
d_lat = lat_grid - args.lat
d_lon = (lon_grid - args.lon + 180.0) % 360.0 - 180.0
gaussian_shape = np.exp(-0.5 * ((d_lat / args.sigma_lat) ** 2 + (d_lon / args.sigma_lon) ** 2))
gaussian_shape[gaussian_shape < 0.01] = 0.0

# Background error: Cold bias of -args.error_amp
bg_error_2m = -args.error_amp * gaussian_shape
# Observation innovation to recover: +args.error_amp
obs_correction = +args.error_amp * gaussian_shape

# --- State 1: Degraded Background Analysis (X_bg) ---
# Background has a cold bias in 2m temperature at BOTH input times
inputs_bg = eval_inputs_nature.copy(deep=True)
for t_idx in range(n_input_times):
    inputs_bg["2m_temperature"].values[..., t_idx, :, :] += bg_error_2m

# Also introduce a weak cold bias in lower levels of background
for lev, w in {1000: 1.0, 925: 0.6, 850: 0.2}.items():
    if lev in levels:
        l_idx = list(levels).index(lev)
        for t_idx in range(n_input_times):
            inputs_bg["temperature"].values[..., t_idx, l_idx, :, :] += (w * bg_error_2m)

# --- State 2: Direct Surface Insertion (DIR) ---
# Adds observation correction ONLY to 2m_temperature (applied at BOTH input times)
inputs_dir = inputs_bg.copy(deep=True)
for t_idx in range(n_input_times):
    inputs_dir["2m_temperature"].values[..., t_idx, :, :] += obs_correction

# --- State 3: Column Thermal Insertion WITHOUT Z balance (COL) ---
# Adds observation correction to 2m_temperature AND lower-level temperatures,
# but leaves geopotential Z untouched. (Isolates DEPTH from BALANCE).
inputs_col = inputs_bg.copy(deep=True)
for t_idx in range(n_input_times):
    inputs_col["2m_temperature"].values[..., t_idx, :, :] += obs_correction
    for lev, w in {1000: 1.0, 925: 0.6, 850: 0.2}.items():
        if lev in levels:
            l_idx = list(levels).index(lev)
            inputs_col["temperature"].values[..., t_idx, l_idx, :, :] += (w * obs_correction)

# --- State 4: Balanced Insertion (BAL) ---
# Adds column correction AND integrates geopotential Z hypsometrically.
inputs_bal = inputs_col.copy(deep=True)
R_d = 287.058
delta_Z_cum = np.zeros_like(obs_correction)

# Hypsometric thickness increments
if 1000 in levels and 925 in levels:
    delta_Z_cum += R_d * (0.5 * (1.0 + 0.6) * obs_correction) * np.log(1000.0 / 925.0)
    l925_idx = list(levels).index(925)
    for t_idx in range(n_input_times):
        inputs_bal["geopotential"].values[..., t_idx, l925_idx, :, :] += delta_Z_cum

if 925 in levels and 850 in levels:
    delta_Z_cum += R_d * (0.5 * (0.6 + 0.2) * obs_correction) * np.log(925.0 / 850.0)
    l850_idx = list(levels).index(850)
    for t_idx in range(n_input_times):
        inputs_bal["geopotential"].values[..., t_idx, l850_idx, :, :] += delta_Z_cum

# Lift all levels above 850 hPa uniformly
for lev in levels:
    if lev < 850:
        l_idx = list(levels).index(lev)
        for t_idx in range(n_input_times):
            inputs_bal["geopotential"].values[..., t_idx, l_idx, :, :] += delta_Z_cum

print(f"      Initial states created:")
print(f"        - Background (BG):   Injected cold bias (-{args.error_amp:.1f} K)")
print(f"        - Direct (DIR):       Obs inserted at surface only (2m T)")
print(f"        - Column (COL):      Obs inserted into column (T only, no Z balance)")
print(f"        - Balanced (BAL):    Obs inserted with hypsometric Z lift (+{np.max(delta_Z_cum)/9.80665:.2f} gpm)")

# ---------------------------------------------------------------------------
# 4. Compile Forward Operator & Execute the 4 Twin Rollouts
# ---------------------------------------------------------------------------
print(f"\n[4/6] JIT compiling and running the 4 OSSE twin rollouts on GPU ...")

def construct_wrapped_graphcast(m_cfg, t_cfg):
    predictor = graphcast.GraphCast(m_cfg, t_cfg)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(m_cfg, t_cfg, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(m_cfg, t_cfg)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

with_configs = functools.partial(run_forward.apply, m_cfg=model_config, t_cfg=task_config)
_run_forward_jitted = jax.jit(functools.partial(with_configs, params=params, state=state))

def run_forward_jitted(rng, inputs, targets_template, forcings):
    preds, _state = _run_forward_jitted(rng=rng, inputs=inputs, targets_template=targets_template, forcings=forcings)
    return preds

rollout_kw = dict(
    rng=jax.random.PRNGKey(0),
    targets_template=eval_targets_nature * np.nan,
    forcings=eval_forcings,
)

# 1. Nature Run (Truth)
print("      [1/4] Running NATURE RUN (F_nature) ...")
t_start = time.time()
preds_nature = rollout.chunked_prediction(run_forward_jitted, inputs=eval_inputs_nature, **rollout_kw)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 2. Degraded Background (No Obs)
print("      [2/4] Running UNCORRECTED BACKGROUND (F_bg) ...")
t_start = time.time()
preds_bg = rollout.chunked_prediction(run_forward_jitted, inputs=inputs_bg, **rollout_kw)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 3. Direct Surface Insertion (DIR)
print("      [3/4] Running DIRECT INSERTION (F_dir) ...")
t_start = time.time()
preds_dir = rollout.chunked_prediction(run_forward_jitted, inputs=inputs_dir, **rollout_kw)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 4. Column Insertion (COL)
print("      [4/4] Running COLUMN INSERTION without Z (F_col) ...")
t_start = time.time()
preds_col = rollout.chunked_prediction(run_forward_jitted, inputs=inputs_col, **rollout_kw)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 5. Balanced Insertion (BAL)
print("      [5/5] Running BALANCED INSERTION with Z (F_bal) ...")
t_start = time.time()
preds_bal = rollout.chunked_prediction(run_forward_jitted, inputs=inputs_bal, **rollout_kw)
print(f"            Finished in {time.time() - t_start:.2f} s")

# ---------------------------------------------------------------------------
# 5. Score Error Recovery Against Known Nature Run
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("EXPERIMENT 0 RESULTS: ERROR RECOVERY AGAINST KNOWN TRUTH")
print("=" * 68)

# Mask target CONUS region
conus_mask = (lats[:, np.newaxis] >= 25) & (lats[:, np.newaxis] <= 50) & (lons[np.newaxis, :] >= 235) & (lons[np.newaxis, :] <= 290)
weights_lat = np.cos(np.deg2rad(lats))[:, np.newaxis]
weights_2d = weights_lat * np.ones_like(lon_grid)
weights_conus = weights_2d[conus_mask]

lead_hours = []
rmse_bg = []
rmse_dir = []
rmse_col = []
rmse_bal = []

rec_dir = []
rec_col = []
rec_bal = []

for s in range(n_steps):
    hours = (s + 1) * 6
    lead_hours.append(hours)
    
    t2m_nat = preds_nature["2m_temperature"].isel(time=s).values.squeeze()
    t2m_b = preds_bg["2m_temperature"].isel(time=s).values.squeeze()
    t2m_d = preds_dir["2m_temperature"].isel(time=s).values.squeeze()
    t2m_c = preds_col["2m_temperature"].isel(time=s).values.squeeze()
    t2m_bal = preds_bal["2m_temperature"].isel(time=s).values.squeeze()
    
    # CONUS errors against known Nature Run
    err_b = (t2m_b - t2m_nat)[conus_mask]
    err_d = (t2m_d - t2m_nat)[conus_mask]
    err_c = (t2m_c - t2m_nat)[conus_mask]
    err_bal = (t2m_bal - t2m_nat)[conus_mask]
    
    r_b = float(np.sqrt(np.sum(weights_conus * err_b**2) / np.sum(weights_conus)))
    r_d = float(np.sqrt(np.sum(weights_conus * err_d**2) / np.sum(weights_conus)))
    r_c = float(np.sqrt(np.sum(weights_conus * err_c**2) / np.sum(weights_conus)))
    r_bal = float(np.sqrt(np.sum(weights_conus * err_bal**2) / np.sum(weights_conus)))
    
    rmse_bg.append(r_b)
    rmse_dir.append(r_d)
    rmse_col.append(r_c)
    rmse_bal.append(r_bal)
    
    # Error Recovery Percentage: (RMSE_bg - RMSE_m) / RMSE_bg * 100
    rec_d_pct = ((r_b - r_d) / r_b) * 100.0
    rec_c_pct = ((r_b - r_c) / r_b) * 100.0
    rec_bal_pct = ((r_b - r_bal) / r_b) * 100.0
    
    rec_dir.append(rec_d_pct)
    rec_col.append(rec_c_pct)
    rec_bal.append(rec_bal_pct)
    
    print(f"\nLead +{hours:02d}h vs. Known Nature Run:")
    print(f"  Uncorrected BG Error: {r_b:.4f} K (Baseline Error)")
    print(f"  DIR  Error:           {r_d:.4f} K  |  Error Recovered: {rec_d_pct:+6.2f}%")
    print(f"  COL  Error (No Z):    {r_c:.4f} K  |  Error Recovered: {rec_c_pct:+6.2f}%")
    print(f"  BAL  Error (With Z):  {r_bal:.4f} K  |  Error Recovered: {rec_bal_pct:+6.2f}%")
    
    # Isolate Depth vs Balance
    depth_gain = rec_c_pct - rec_d_pct
    balance_gain = rec_bal_pct - rec_c_pct
    print(f"  --> Depth Benefit (COL - DIR):    {depth_gain:+6.2f}%")
    print(f"  --> Balance Benefit (BAL - COL):  {balance_gain:+6.2f}%")

# ---------------------------------------------------------------------------
# 6. Comparative Diagnostic Figures
# ---------------------------------------------------------------------------
print(f"\n[6/6] Generating publication figures in {OUTDIR} ...")

def add_map_elements(ax):
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", facecolor="none", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", facecolor="none", zorder=3)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="dimgray", facecolor="none", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, zorder=3)
        gl.top_labels = False
        gl.right_labels = False

extent = [230, 300, 20, 60]

# --- Figure 1: Error Recovery Percentage vs. Lead Time ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lead_hours, rec_dir, "o-", color="#d62728", linewidth=2.5, label="DIR (Surface Only)")
ax.plot(lead_hours, rec_col, "^-", color="#ff7f0e", linewidth=2.5, label="COL (Column T, No Z Balance)")
ax.plot(lead_hours, rec_bal, "s-", color="#1f77b4", linewidth=2.5, label="BAL (Column T + Hypsometric Z)")
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.set_xlabel("Forecast Lead Time (hours)", fontsize=11)
ax.set_ylabel("Background Error Recovered (%)", fontsize=11)
ax.set_title("Identical-Twin Experiment: Error Recovery vs. Known Truth", fontsize=12, fontweight="bold")
ax.set_xticks(lead_hours)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=10, loc="best")
fig.tight_layout()
f1 = os.path.join(OUTDIR, "exp0_error_recovery_curve.png")
fig.savefig(f1, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f1)}")

# --- Figure 2: CONUS Absolute RMSE Progression ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lead_hours, rmse_bg, "k--", linewidth=2.2, label="Uncorrected Background (BG)")
ax.plot(lead_hours, rmse_dir, "o-", color="#d62728", linewidth=2.0, label="DIR (Surface Only)")
ax.plot(lead_hours, rmse_col, "^-", color="#ff7f0e", linewidth=2.0, label="COL (Column T Only)")
ax.plot(lead_hours, rmse_bal, "s-", color="#1f77b4", linewidth=2.0, label="BAL (Balanced)")
ax.set_xlabel("Forecast Lead Time (hours)", fontsize=11)
ax.set_ylabel("CONUS 2m Temperature RMSE vs. Truth (K)", fontsize=11)
ax.set_title("CONUS Forecast Error vs. Nature Run", fontsize=12, fontweight="bold")
ax.set_xticks(lead_hours)
ax.grid(True, alpha=0.4)
ax.legend(fontsize=10)
fig.tight_layout()
f2 = os.path.join(OUTDIR, "exp0_conus_rmse_progression.png")
fig.savefig(f2, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f2)}")

# --- Figure 3: Disentangling Depth vs. Balance ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(lead_hours))
width = 0.35

depth_gains = [c - d for c, d in zip(rec_col, rec_dir)]
balance_gains = [b - c for b, c in zip(rec_bal, rec_col)]

b1 = ax.bar(x - width/2, depth_gains, width, label="Depth Benefit (COL vs. DIR)", color="#ff7f0e")
b2 = ax.bar(x + width/2, balance_gains, width, label="Balance Benefit (BAL vs. COL)", color="#1f77b4")

ax.set_ylabel("Additional Error Recovery (%)", fontsize=11)
ax.set_title("Disentangling Insertion Mechanism: Column Depth vs. Physical Balance", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"+{h}h" for h in lead_hours])
ax.axhline(0, color="k", linewidth=0.8)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
f3 = os.path.join(OUTDIR, "exp0_depth_vs_balance_tradeoff.png")
fig.savefig(f3, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f3)}")

# --- Figure 4: Day-1 (+24h) Remaining Error Maps ---
step_24 = min(3, n_steps - 1)
hours_24 = (step_24 + 1) * 6

t2m_nat_24 = preds_nature["2m_temperature"].isel(time=step_24).values.squeeze()
err_bg_24 = np.abs(preds_bg["2m_temperature"].isel(time=step_24).values.squeeze() - t2m_nat_24)
err_dir_24 = np.abs(preds_dir["2m_temperature"].isel(time=step_24).values.squeeze() - t2m_nat_24)
err_col_24 = np.abs(preds_col["2m_temperature"].isel(time=step_24).values.squeeze() - t2m_nat_24)
err_bal_24 = np.abs(preds_bal["2m_temperature"].isel(time=step_24).values.squeeze() - t2m_nat_24)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5),
                         subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {})
vmax_err = max(0.5, np.percentile(err_bg_24, 99))

kw = dict(cmap="YlOrRd", vmin=0, vmax=vmax_err, transform=ccrs.PlateCarree() if HAS_CARTOPY else None)

im0 = axes[0].pcolormesh(lons, lats, err_bg_24, **kw)
axes[0].set_title(f"Uncorrected BG Error (+{hours_24}h)", fontsize=10, fontweight="bold")
im1 = axes[1].pcolormesh(lons, lats, err_dir_24, **kw)
axes[1].set_title(f"DIR Error (+{hours_24}h)", fontsize=10, fontweight="bold")
im2 = axes[2].pcolormesh(lons, lats, err_col_24, **kw)
axes[2].set_title(f"COL Error (+{hours_24}h)", fontsize=10, fontweight="bold")
im3 = axes[3].pcolormesh(lons, lats, err_bal_24, **kw)
axes[3].set_title(f"BAL Error (+{hours_24}h)", fontsize=10, fontweight="bold")

for ax in axes:
    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        add_map_elements(ax)

fig.suptitle(f"Day-1 Absolute Error vs. Nature Run (+{hours_24}h)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
f4 = os.path.join(OUTDIR, "exp0_day1_error_comparison_maps.png")
fig.savefig(f4, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {os.path.basename(f4)}")

# Save Summary Table
sum_txt = os.path.join(OUTDIR, "exp0_twin_summary.txt")
with open(sum_txt, "w") as f:
    f.write("EXPERIMENT 0: IDENTICAL-TWIN (OSSE) INSERTION SUMMARY\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Injected background cold bias: -{args.error_amp:.2f} K\n")
    f.write(f"Observation recovery target:   +{args.error_amp:.2f} K\n\n")
    f.write(f"{'Lead':6s} {'BG Error':10s} {'DIR Error (Rec %)':20s} {'COL Error (Rec %)':20s} {'BAL Error (Rec %)'}\n")
    f.write("-" * 85 + "\n")
    for h, rb, rd, rc, rbal, pd, pc, pbal in zip(lead_hours, rmse_bg, rmse_dir, rmse_col, rmse_bal, rec_dir, rec_col, rec_bal):
        f.write(f"+{h:02d}h   {rb:7.4f} K  {rd:7.4f} K ({pd:+5.1f}%)   {rc:7.4f} K ({pc:+5.1f}%)   {rbal:7.4f} K ({pbal:+5.1f}%)\n")
print(f"\nSummary logged to: {sum_txt}")

# Save NetCDF
out_nc = os.path.join(OUTDIR, "exp0_twin_dataset.nc")
out_ds = xr.Dataset({
    "t2m_nature": preds_nature["2m_temperature"],
    "t2m_bg": preds_bg["2m_temperature"],
    "t2m_dir": preds_dir["2m_temperature"],
    "t2m_col": preds_col["2m_temperature"],
    "t2m_bal": preds_bal["2m_temperature"],
})
out_ds.to_netcdf(out_nc)
print(f"Dataset saved to: {out_nc}")

print("\n" + "=" * 68)
print(f"Experiment 0 complete! All figures and data saved in:\n  {OUTDIR}/")
print("=" * 68)
