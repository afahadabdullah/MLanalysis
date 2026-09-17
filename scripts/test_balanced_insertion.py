#!/usr/bin/env python3
"""
GraphCast Balanced Increment Insertion (E-BAL) vs. Direct Insertion (E-DIR)
============================================================================
Implements Hypothesis H3 from PROJECT_PLAN_LEAN.md (Exp 2 / Stage S0):
  "Spreading a surface increment (ΔT2m) vertically into the lower troposphere
   (1000, 925, 850 hPa) and updating geopotential (Z) hypsometrically
   reduces initialization shock and preserves physical consistency."

This script executes a 3-way rollout comparison:
  1. Baseline (E):         Unperturbed ERA5 initial state
  2. Direct (E-DIR):       Surface-only increment: ΔT2m = +2.0 K
  3. Balanced (E-BAL):     ΔT2m coupled into T(1000, 925, 850 hPa) with weights [1.0, 0.6, 0.2]
                           and geopotential Z hydrostatically integrated from bottom up.

Metrics & Diagnostics:
  - Retention Curves:       R_dir(t) vs R_bal(t) over 24 hours
  - Initialization Shock:   First-step jump ||F(x0) - x0|| over CONUS
  - Physical Consistency:   t0 lapse-rate anomaly (T2m - T1000)
  - Vertical Profiles:      Cross-sections of column warming (p vs lon)
  - Forecast Accuracy:      CONUS RMSE vs ERA5 truth

Usage:
  python scripts/test_balanced_insertion.py
  python scripts/test_balanced_insertion.py --amplitude 2.0 --steps 4
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

# Ensure JAX memory preallocation is controlled
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

# Matplotlib setup (headless Agg)
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
parser = argparse.ArgumentParser(description="GraphCast Balanced Insertion Experiment (E-BAL vs E-DIR)")
parser.add_argument("--steps", type=int, default=4,
                    help="Forecast steps (6h each). Default: 4 (24 hours)")
parser.add_argument("--amplitude", type=float, default=2.0,
                    help="Surface anomaly amplitude in Kelvin. Default: +2.0 K")
parser.add_argument("--lat", type=float, default=38.0,
                    help="Center latitude for anomaly in degrees N. Default: 38.0 N")
parser.add_argument("--lon", type=float, default=265.0,
                    help="Center longitude in degrees E (0-360). Default: 265.0 E (95 W)")
parser.add_argument("--sigma-lat", type=float, default=4.0,
                    help="Latitudinal Gaussian std dev (deg). Default: 4.0")
parser.add_argument("--sigma-lon", type=float, default=6.0,
                    help="Longitudinal Gaussian std dev (deg). Default: 6.0")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: <PROJ>/runs/balanced)")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "runs", "balanced")
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

print("=" * 65)
print("GraphCast Experiment 2: Balanced (E-BAL) vs. Direct (E-DIR) Insertion")
print("=" * 65)
print(f"Project root:     {PROJ}")
print(f"Output directory: {OUTDIR}")
print(f"Anomaly config:   ΔT2m = {args.amplitude:+.1f} K at ({args.lat}°N, {args.lon}°E)")
print(f"Forecast horizon: {args.steps} steps ({args.steps * 6} hours)")

# ---------------------------------------------------------------------------
# 1. Load Checkpoint and Stats
# ---------------------------------------------------------------------------
print("\n[1/6] Loading checkpoint and normalization statistics ...")
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
# 2. Prepare Base Inputs
# ---------------------------------------------------------------------------
print(f"\n[2/6] Loading dataset and extracting baseline inputs ...")
with open(SAMPLE_FILE, "rb") as f:
    try:
        example_batch = xr.load_dataset(f, decode_timedelta=True).compute()
    except Exception:
        example_batch = xr.load_dataset(f).compute()

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{args.steps * 6}h"),
    **dataclasses.asdict(task_config),
)

lats = eval_inputs.coords["lat"].values
lons = eval_inputs.coords["lon"].values
times = eval_targets.coords["time"].values
n_steps = len(times)
t0_coord = eval_inputs.coords["time"].values[-1]

# ---------------------------------------------------------------------------
# 3. Create Perturbed Inputs: E-DIR and E-BAL
# ---------------------------------------------------------------------------
print(f"\n[3/6] Synthesizing Direct (E-DIR) and Balanced (E-BAL) Initial States ...")

# 2D Gaussian anomaly over (lat, lon)
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
d_lat = lat_grid - args.lat
d_lon = (lon_grid - args.lon + 180.0) % 360.0 - 180.0
delta_X = args.amplitude * np.exp(-0.5 * ((d_lat / args.sigma_lat) ** 2 + (d_lon / args.sigma_lon) ** 2))
delta_X[np.abs(delta_X) < (0.01 * np.abs(args.amplitude))] = 0.0

weights_lat = np.cos(np.deg2rad(lats))[:, np.newaxis]
delta_X_norm_sq = np.sum(delta_X**2 * weights_lat)

# --- 3A. E-DIR: Modify only 2m_temperature at t0 ---
eval_inputs_dir = eval_inputs.copy(deep=True)
t2m_curr = eval_inputs_dir["2m_temperature"].sel(time=t0_coord).values
eval_inputs_dir["2m_temperature"].loc[dict(time=t0_coord)] = t2m_curr + delta_X

# --- 3B. E-BAL: Vertical spreading + Hypsometric Geopotential Integration ---
eval_inputs_bal = eval_inputs.copy(deep=True)
eval_inputs_bal["2m_temperature"].loc[dict(time=t0_coord)] = t2m_curr + delta_X

# Pressure levels available
levels = eval_inputs_bal.coords["level"].values
print(f"      Model pressure levels (hPa): {list(levels)}")

# Gas constant for dry air (J / (kg K))
R_d = 287.058

# Vertical spreading weights from PROJECT_PLAN_LEAN.md:
# 1000 hPa: 1.0, 925 hPa: 0.6, 850 hPa: 0.2
level_weights = {1000: 1.0, 925: 0.6, 850: 0.2}

# Apply temperature spread to 3D temperature at t0
delta_T_profile = {}
for lev, w in level_weights.items():
    if lev in levels:
        t_lev = eval_inputs_bal["temperature"].sel(time=t0_coord, level=lev).values
        dT = w * delta_X
        eval_inputs_bal["temperature"].loc[dict(time=t0_coord, level=lev)] = t_lev + dT
        delta_T_profile[lev] = dT
        print(f"      Balanced T increment applied at {lev:4d} hPa (weight = {w:.1f})")

# Hydrostatic / Hypsometric Integration for Geopotential (Z)
# dPhi = -R_d * T * dln(p)  =>  Phi(p2) - Phi(p1) = R_d * T_layer * ln(p1 / p2)
# Integrating from bottom (1000 hPa) upward:
delta_Z_levels = {}
delta_Z_cum = np.zeros_like(delta_X)

# Start thickness integration from 1000 hPa upwards
# 1000 -> 925 hPa:
if 1000 in levels and 925 in levels:
    dT_layer_1000_925 = 0.5 * (delta_T_profile.get(1000, 0) + delta_T_profile.get(925, 0))
    dZ_1000_925 = R_d * dT_layer_1000_925 * np.log(1000.0 / 925.0)
    delta_Z_cum += dZ_1000_925
    delta_Z_levels[925] = delta_Z_cum.copy()

# 925 -> 850 hPa:
if 925 in levels and 850 in levels:
    dT_layer_925_850 = 0.5 * (delta_T_profile.get(925, 0) + delta_T_profile.get(850, 0))
    dZ_925_850 = R_d * dT_layer_925_850 * np.log(925.0 / 850.0)
    delta_Z_cum += dZ_925_850
    delta_Z_levels[850] = delta_Z_cum.copy()

# For all levels above 850 hPa, dT = 0, so dZ remains constant (the ridge aloft):
for lev in levels:
    if lev < 850:
        delta_Z_levels[lev] = delta_Z_cum.copy()

# Apply geopotential increment to eval_inputs_bal['geopotential'] at t0
for lev, dZ in delta_Z_levels.items():
    if lev in levels:
        z_curr = eval_inputs_bal["geopotential"].sel(time=t0_coord, level=lev).values
        eval_inputs_bal["geopotential"].loc[dict(time=t0_coord, level=lev)] = z_curr + dZ

max_z_lift = np.max(delta_Z_cum) / 9.80665  # converted to meters
print(f"      Hypsometric geopotential ridge created aloft: +{max_z_lift:.2f} gpm max at 500 hPa")

# ---------------------------------------------------------------------------
# 4. Predictor Setup & Rollout Execution
# ---------------------------------------------------------------------------
print(f"\n[4/6] JIT compiling predictor and running 3-way rollouts ...")

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

# 1. Baseline (E)
print("      [1/3] Running BASELINE rollout (E) ...")
t_start = time.time()
preds_base = rollout.chunked_prediction(
    run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
    targets_template=eval_targets * np.nan, forcings=eval_forcings,
)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 2. Direct (E-DIR)
print("      [2/3] Running DIRECT rollout (E-DIR) ...")
t_start = time.time()
preds_dir = rollout.chunked_prediction(
    run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs_dir,
    targets_template=eval_targets * np.nan, forcings=eval_forcings,
)
print(f"            Finished in {time.time() - t_start:.2f} s")

# 3. Balanced (E-BAL)
print("      [3/3] Running BALANCED rollout (E-BAL) ...")
t_start = time.time()
preds_bal = rollout.chunked_prediction(
    run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs_bal,
    targets_template=eval_targets * np.nan, forcings=eval_forcings,
)
print(f"            Finished in {time.time() - t_start:.2f} s")

# ---------------------------------------------------------------------------
# 5. Scientific Metric Computations
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("EXPERIMENT 2 COMPARATIVE EVALUATION: E-DIR vs. E-BAL")
print("=" * 65)

# Mask CONUS
conus_mask = (lats[:, np.newaxis] >= 25) & (lats[:, np.newaxis] <= 50) & (lons[np.newaxis, :] >= 235) & (lons[np.newaxis, :] <= 290)
weights_2d = weights_lat * np.ones_like(lon_grid)
weights_conus = weights_2d[conus_mask]

lead_hours = []
ret_dir = []
ret_bal = []
peak_dir = []
peak_bal = []

rmse_base_c = []
rmse_dir_c = []
rmse_bal_c = []

for s in range(n_steps):
    hours = (s + 1) * 6
    lead_hours.append(hours)
    
    t2m_b = preds_base["2m_temperature"].isel(time=s).values.squeeze()
    t2m_d = preds_dir["2m_temperature"].isel(time=s).values.squeeze()
    t2m_bal = preds_bal["2m_temperature"].isel(time=s).values.squeeze()
    truth = eval_targets["2m_temperature"].isel(time=s).values.squeeze()
    
    diff_d = t2m_d - t2m_b
    diff_bal = t2m_bal - t2m_b
    
    # Retention metrics
    r_d = np.sum(diff_d * delta_X * weights_lat) / (delta_X_norm_sq + 1e-12)
    r_bal = np.sum(diff_bal * delta_X * weights_lat) / (delta_X_norm_sq + 1e-12)
    ret_dir.append(float(r_d))
    ret_bal.append(float(r_bal))
    
    p_d = np.max(np.abs(diff_d)) / abs(args.amplitude)
    p_bal = np.max(np.abs(diff_bal)) / abs(args.amplitude)
    peak_dir.append(float(p_d))
    peak_bal.append(float(p_bal))
    
    # CONUS RMSE vs Truth
    err_b = (t2m_b - truth)[conus_mask]
    err_d = (t2m_d - truth)[conus_mask]
    err_bal = (t2m_bal - truth)[conus_mask]
    
    rmse_base_c.append(float(np.sqrt(np.sum(weights_conus * err_b**2) / np.sum(weights_conus))))
    rmse_dir_c.append(float(np.sqrt(np.sum(weights_conus * err_d**2) / np.sum(weights_conus))))
    rmse_bal_c.append(float(np.sqrt(np.sum(weights_conus * err_bal**2) / np.sum(weights_conus))))
    
    print(f"  Lead +{hours:02d}h:")
    print(f"     Retention:  E-DIR = {r_d*100:5.2f}%  |  E-BAL = {r_bal*100:5.2f}%  (Δ = {(r_bal - r_d)*100:+5.2f}%)")
    print(f"     Peak Amp:   E-DIR = {p_d*100:5.1f}%  |  E-BAL = {p_bal*100:5.1f}%")
    print(f"     CONUS RMSE: Base = {rmse_base_c[-1]:.3f} K | E-DIR = {rmse_dir_c[-1]:.3f} K | E-BAL = {rmse_bal_c[-1]:.3f} K")

# Initialization Shock Proxy: First-step jump ||F(x0) - x0|| at step 0 (+6h)
t2m_x0_b = eval_inputs["2m_temperature"].sel(time=t0_coord).values
t2m_x0_d = eval_inputs_dir["2m_temperature"].sel(time=t0_coord).values
t2m_x0_bal = eval_inputs_bal["2m_temperature"].sel(time=t0_coord).values

jump_base = np.sqrt(np.mean(((preds_base["2m_temperature"].isel(time=0).values.squeeze() - t2m_x0_b)[conus_mask])**2))
jump_dir = np.sqrt(np.mean(((preds_dir["2m_temperature"].isel(time=0).values.squeeze() - t2m_x0_d)[conus_mask])**2))
jump_bal = np.sqrt(np.mean(((preds_bal["2m_temperature"].isel(time=0).values.squeeze() - t2m_x0_bal)[conus_mask])**2))

print("\nInitialization Shock Proxy (First-step jump ||F(x0) - x0|| over CONUS):")
print(f"  Baseline jump:   {jump_base:.4f} K")
print(f"  E-DIR jump:      {jump_dir:.4f} K  (Excess jump over base: {jump_dir - jump_base:+.4f} K)")
print(f"  E-BAL jump:      {jump_bal:.4f} K  (Excess jump over base: {jump_bal - jump_base:+.4f} K)")

# Physical Inconsistency Metric: t0 Lapse Rate Anomaly (T2m - T1000)
lapse_base = t2m_x0_b - eval_inputs["temperature"].sel(time=t0_coord, level=1000).values
lapse_dir = t2m_x0_d - eval_inputs_dir["temperature"].sel(time=t0_coord, level=1000).values
lapse_bal = t2m_x0_bal - eval_inputs_bal["temperature"].sel(time=t0_coord, level=1000).values

print("\nPhysical Consistency at t0 (T2m - T1000 Lapse Rate Discrepancy at center):")
lat_c_idx = np.argmin(np.abs(lats - args.lat))
lon_c_idx = np.argmin(np.abs(lons - args.lon))
print(f"  Baseline lapse (T2m - T1000): {lapse_base[lat_c_idx, lon_c_idx]:+.2f} K")
print(f"  E-DIR lapse (T2m - T1000):    {lapse_dir[lat_c_idx, lon_c_idx]:+.2f} K  (Unphysical super-adiabatic jump: {lapse_dir[lat_c_idx, lon_c_idx] - lapse_base[lat_c_idx, lon_c_idx]:+.2f} K)")
print(f"  E-BAL lapse (T2m - T1000):    {lapse_bal[lat_c_idx, lon_c_idx]:+.2f} K  (Preserved lapse rate: {lapse_bal[lat_c_idx, lon_c_idx] - lapse_base[lat_c_idx, lon_c_idx]:+.2f} K)")

# ---------------------------------------------------------------------------
# 6. Diagnostic Visualizations
# ---------------------------------------------------------------------------
print(f"\n[5/6] Generating comparative diagnostic figures in {OUTDIR} ...")

def add_map_elements(ax):
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", facecolor="none", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", facecolor="none", zorder=3)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="dimgray", facecolor="none", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, zorder=3)
        gl.top_labels = False
        gl.right_labels = False

extent = [230, 300, 20, 60]

# --- Figure 1: Retention Decay Comparison (E-DIR vs E-BAL) ---
fig, ax = plt.subplots(figsize=(8, 5))
hours_axis = [0] + lead_hours
ax.plot(hours_axis, [100.0] + [r * 100 for r in ret_dir], "o-", color="#d62728", linewidth=2.5,
        label=r"E-DIR: Direct Surface-Only Insertion")
ax.plot(hours_axis, [100.0] + [r * 100 for r in ret_bal], "s-", color="#1f77b4", linewidth=2.5,
        label=r"E-BAL: Balanced (T-spread + Hypsometric $Z$)")
ax.plot(hours_axis, [100.0] + [p * 100 for p in peak_dir], "--", color="#d62728", alpha=0.7,
        label=r"E-DIR Peak Amplitude")
ax.plot(hours_axis, [100.0] + [p * 100 for p in peak_bal], "--", color="#1f77b4", alpha=0.7,
        label=r"E-BAL Peak Amplitude")

ax.set_xlabel("Forecast Lead Time (hours)", fontsize=11)
ax.set_ylabel("Retention Metric (%)", fontsize=11)
ax.set_title(r"Increment Retention: Direct (E-DIR) vs. Balanced (E-BAL)", fontsize=13, fontweight="bold")
ax.set_xticks(hours_axis)
ax.set_ylim(0, 110)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=9, loc="best")
fig.tight_layout()
f1 = os.path.join(OUTDIR, "retention_comparison_dir_vs_bal.png")
fig.savefig(f1, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f1)}")

# --- Figure 2: Initialization Shock and Consistency Summary ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (A) Shock / First-step jump excess
excess_shock = [0.0, jump_dir - jump_base, jump_bal - jump_base]
bar_cols = ["#2ca02c", "#d62728", "#1f77b4"]
bars = axes[0].bar(["Baseline", "E-DIR", "E-BAL"], excess_shock, color=bar_cols)
axes[0].set_ylabel("Excess Jump over Baseline (K)", fontsize=10)
axes[0].set_title(r"Initialization Shock: $\|F(x_0) - x_0\|$ Excess", fontsize=11, fontweight="bold")
axes[0].grid(True, axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:+.4f} K", ha="center", fontsize=9)

# (B) CONUS RMSE progression
axes[1].plot(lead_hours, rmse_base_c, "k--", label="Baseline (ERA5)", linewidth=1.8)
axes[1].plot(lead_hours, rmse_dir_c, "o-", color="#d62728", label="E-DIR (Direct)", linewidth=2)
axes[1].plot(lead_hours, rmse_bal_c, "s-", color="#1f77b4", label="E-BAL (Balanced)", linewidth=2)
axes[1].set_xlabel("Lead Time (hours)", fontsize=10)
axes[1].set_ylabel("CONUS 2m Temperature RMSE (K)", fontsize=10)
axes[1].set_title("CONUS Verification Error vs. Lead Time", fontsize=11, fontweight="bold")
axes[1].set_xticks(lead_hours)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

fig.suptitle("Physical Balance vs. Initialization Shock", fontsize=13, fontweight="bold")
fig.tight_layout()
f2 = os.path.join(OUTDIR, "shock_and_rmse_comparison.png")
fig.savefig(f2, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f2)}")

# --- Figure 3: Vertical Cross Section of Warming at +12h (p vs Longitude) ---
lat_target = args.lat
lat_idx = np.argmin(np.abs(lats - lat_target))
step_eval = min(1, n_steps - 1)  # step 1 = +12h
hours_eval = (step_eval + 1) * 6

t_eval_dir = (preds_dir["temperature"].isel(time=step_eval) - preds_base["temperature"].isel(time=step_eval)).values.squeeze()[:, lat_idx, :]
t_eval_bal = (preds_bal["temperature"].isel(time=step_eval) - preds_base["temperature"].isel(time=step_eval)).values.squeeze()[:, lat_idx, :]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vmax_t = max(0.5, np.max(np.abs(t_eval_bal)))

im0 = axes[0].pcolormesh(lons, levels, t_eval_dir, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t, shading="auto")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(1000, 200)
axes[0].set_ylabel("Pressure Level (hPa)", fontsize=10)
axes[0].set_xlabel("Longitude (°E)", fontsize=10)
axes[0].set_title(f"E-DIR: Induced Column ΔT at +{hours_eval}h ({lat_target}°N)", fontsize=11, fontweight="bold")
plt.colorbar(im0, ax=axes[0], label="ΔT (K)", shrink=0.8)

im1 = axes[1].pcolormesh(lons, levels, t_eval_bal, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t, shading="auto")
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(1000, 200)
axes[1].set_ylabel("Pressure Level (hPa)", fontsize=10)
axes[1].set_xlabel("Longitude (°E)", fontsize=10)
axes[1].set_title(f"E-BAL: Induced Column ΔT at +{hours_eval}h ({lat_target}°N)", fontsize=11, fontweight="bold")
plt.colorbar(im1, ax=axes[1], label="ΔT (K)", shrink=0.8)

fig.suptitle("Vertical Structure Comparison: Direct vs. Balanced Insertion", fontsize=13, fontweight="bold")
fig.tight_layout()
f3 = os.path.join(OUTDIR, "vertical_cross_section_dir_vs_bal.png")
fig.savefig(f3, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(f3)}")

# --- Figure 4: Spatial Triplet at +24h (E-DIR vs E-BAL vs Difference) ---
step_24 = min(3, n_steps - 1)
hours_24 = (step_24 + 1) * 6
t2m_d_24 = (preds_dir["2m_temperature"].isel(time=step_24) - preds_base["2m_temperature"].isel(time=step_24)).values.squeeze()
t2m_b_24 = (preds_bal["2m_temperature"].isel(time=step_24) - preds_base["2m_temperature"].isel(time=step_24)).values.squeeze()
diff_bal_dir = t2m_b_24 - t2m_d_24

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5),
                         subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {})
vmax_s = max(0.5, np.max(np.abs(t2m_b_24)))

im0 = axes[0].pcolormesh(lons, lats, t2m_d_24, cmap="RdBu_r", vmin=-vmax_s, vmax=vmax_s,
                         transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
axes[0].set_title(f"E-DIR Remaining Anomaly (+{hours_24}h)", fontsize=10, fontweight="bold")
if HAS_CARTOPY:
    axes[0].set_extent(extent, crs=ccrs.PlateCarree())
    add_map_elements(axes[0])
plt.colorbar(im0, ax=axes[0], shrink=0.7, label="K")

im1 = axes[1].pcolormesh(lons, lats, t2m_b_24, cmap="RdBu_r", vmin=-vmax_s, vmax=vmax_s,
                         transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
axes[1].set_title(f"E-BAL Remaining Anomaly (+{hours_24}h)", fontsize=10, fontweight="bold")
if HAS_CARTOPY:
    axes[1].set_extent(extent, crs=ccrs.PlateCarree())
    add_map_elements(axes[1])
plt.colorbar(im1, ax=axes[1], shrink=0.7, label="K")

dmax_bd = max(0.2, np.max(np.abs(diff_bal_dir)))
im2 = axes[2].pcolormesh(lons, lats, diff_bal_dir, cmap="PuOr_r", vmin=-dmax_bd, vmax=dmax_bd,
                         transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
axes[2].set_title(f"Difference: E-BAL − E-DIR (+{hours_24}h)", fontsize=10, fontweight="bold")
if HAS_CARTOPY:
    axes[2].set_extent(extent, crs=ccrs.PlateCarree())
    add_map_elements(axes[2])
plt.colorbar(im2, ax=axes[2], shrink=0.7, label="K")

fig.suptitle("Day-1 Spatial Persistence: Direct vs. Balanced Insertion", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
f4 = os.path.join(OUTDIR, "spatial_comparison_24h_dir_vs_bal.png")
fig.savefig(f4, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {os.path.basename(f4)}")

# Save Summary Table
sum_file = os.path.join(OUTDIR, "balanced_experiment_summary.txt")
with open(sum_file, "w") as f:
    f.write("GraphCast Balanced (E-BAL) vs Direct (E-DIR) Experiment Summary\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Injected amplitude: {args.amplitude:+.2f} K at ({args.lat}N, {args.lon}E)\n")
    f.write(f"Max geopotential lift aloft: +{max_z_lift:.2f} gpm\n\n")
    f.write("Retention Scores:\n")
    for h, rd, rb, pd, pb in zip(lead_hours, ret_dir, ret_bal, peak_dir, peak_bal):
        f.write(f"+{h:02d}h: E-DIR Ret={rd*100:5.2f}%, E-BAL Ret={rb*100:5.2f}% | E-DIR Peak={pd*100:5.1f}%, E-BAL Peak={pb*100:5.1f}%\n")
    f.write("\nInitialization Shock (Jump over CONUS):\n")
    f.write(f"Baseline: {jump_base:.4f} K | E-DIR: {jump_dir:.4f} K | E-BAL: {jump_bal:.4f} K\n")
print(f"\nSummary table logged to: {sum_file}")

# Save NetCDF differences
comp_ds = xr.Dataset({
    "diff_dir_t2m": preds_dir["2m_temperature"] - preds_base["2m_temperature"],
    "diff_bal_t2m": preds_bal["2m_temperature"] - preds_base["2m_temperature"],
    "diff_dir_temp": preds_dir["temperature"] - preds_base["temperature"],
    "diff_bal_temp": preds_bal["temperature"] - preds_base["temperature"],
    "diff_bal_z": preds_bal["geopotential"] - preds_base["geopotential"],
})
comp_nc = os.path.join(OUTDIR, "balanced_comparison.nc")
comp_ds.to_netcdf(comp_nc)
print(f"Comparative dataset saved to: {comp_nc}")

print("\n" + "=" * 65)
print(f"Experiment complete! Figures and data saved in:\n  {OUTDIR}/")
print("=" * 65)
