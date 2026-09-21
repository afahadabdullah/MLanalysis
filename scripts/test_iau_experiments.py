#!/usr/bin/env python3
"""
Experiment 2b: 4D Balance & NASA GMAO GEOS IAU Temporal Windowing
=================================================================
Evaluates spatial balance (DIR vs. COL vs. BAL) crossed with temporal
windowing (Impulse t0-only vs. Tendency-Neutral IAU across [t-6h, t0]).

Background & Physical Motivation:
  In NASA GMAO GEOS DA (Bloom et al. 1996; Takacs et al. 2018), Incremental
  Analysis Updates (IAU) insert observational increments as a continuous
  forcing across the assimilation window rather than an abrupt impulse jump,
  filtering spurious gravity waves and letting model dynamics absorb the heat.

  In GraphCast, the model infers tendencies from its two input frames (t-6h, t0).
  An impulse injection at t0 alone implies an artificial +2 K / 6h heating rate,
  distorting early rollout dynamics. An IAU window (applying the increment at both
  t-6h and t0) enforces an implied tendency of 0 K/h, presenting the anomaly as an
  established, dynamically balanced air mass.

The 6-Arm Factorial Matrix:
  1. DIR-IMP: Direct surface only at t0 (Impulse baseline)
  2. DIR-IAU: Direct surface only at both t-6h and t0 (Tendency-neutral IAU)
  3. COL-IMP: Column thermal (surface + 1000/925/850) at t0, no Z (Depth only)
  4. COL-IAU: Column thermal at both t-6h and t0, no Z (Depth + IAU)
  5. BAL-IMP: Column thermal + hypsometric Z at t0 (Spatial balance)
  6. BAL-IAU: Column thermal + hypsometric Z at both t-6h and t0 (Full 4D Balance)

Usage:
  python scripts/test_iau_experiments.py
  python scripts/test_iau_experiments.py --sample data/era5/source-era5_date-2018-01-15_res-1.0_levels-13_steps-04.nc
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GraphCast 4D Balance & IAU Experiment")
parser.add_argument("--steps", type=int, default=4, help="Forecast steps (6h each, default: 4 = 24h)")
parser.add_argument("--amplitude", type=float, default=2.0, help="Peak surface anomaly (K). Default: +2.0 K")
parser.add_argument("--lat", type=float, default=38.0, help="Center latitude (deg N). Default: 38.0")
parser.add_argument("--lon", type=float, default=265.0, help="Center longitude (deg E). Default: 265.0 (95 W)")
parser.add_argument("--sigma-lat", type=float, default=4.0, help="Lat std dev (deg). Default: 4.0")
parser.add_argument("--sigma-lon", type=float, default=6.0, help="Lon std dev (deg). Default: 6.0")
parser.add_argument("--sample", default=None, help="Sample NetCDF dataset path")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"), help="Project root")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/runs/iau)")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "runs", "iau")
os.makedirs(OUTDIR, exist_ok=True)

DATA_DIR = os.path.join(PROJ, "data")
PARAMS_FILE = os.path.join(
    DATA_DIR, "params",
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
)
STATS_DIR = os.path.join(DATA_DIR, "stats")
SAMPLE_FILE = args.sample or os.path.join(
    DATA_DIR, "era5", "source-era5_date-2018-01-15_res-1.0_levels-13_steps-04.nc"
)
if not os.path.exists(SAMPLE_FILE):
    SAMPLE_FILE = os.path.join(DATA_DIR, "sample", "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc")

print("=" * 72)
print("EXPERIMENT 2B: 4D BALANCE & NASA GMAO GEOS IAU TEMPORAL WINDOWING")
print("=" * 72)
print(f"Project root:     {PROJ}")
print(f"Dataset path:     {SAMPLE_FILE}")
print(f"Output directory: {OUTDIR}")
print(f"Anomaly bubble:   +{args.amplitude:.1f} K centered at ({args.lat}°N, {args.lon}°E)")
print(f"Forecast horizon: {args.steps} steps ({args.steps * 6} hours)")

# ---------------------------------------------------------------------------
# 1. Load Checkpoint and Stats
# ---------------------------------------------------------------------------
print("\n[1/5] Loading model checkpoint and normalization statistics ...")
with open(PARAMS_FILE, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config

try:
    diffs_stddev_by_level = xr.load_dataset(os.path.join(STATS_DIR, "diffs_stddev_by_level.nc")).compute()
    mean_by_level = xr.load_dataset(os.path.join(STATS_DIR, "mean_by_level.nc")).compute()
    stddev_by_level = xr.load_dataset(os.path.join(STATS_DIR, "stddev_by_level.nc")).compute()
except Exception:
    with open(os.path.join(STATS_DIR, "diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open(os.path.join(STATS_DIR, "mean_by_level.nc"), "rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with open(os.path.join(STATS_DIR, "stddev_by_level.nc"), "rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()

# ---------------------------------------------------------------------------
# 2. Prepare Base Inputs
# ---------------------------------------------------------------------------
print(f"\n[2/5] Loading dataset: {os.path.basename(SAMPLE_FILE)} ...")
try:
    example_batch = xr.load_dataset(SAMPLE_FILE, decode_timedelta=True).compute()
except Exception:
    try:
        example_batch = xr.load_dataset(SAMPLE_FILE).compute()
    except Exception:
        with open(SAMPLE_FILE, "rb") as f:
            example_batch = xr.load_dataset(f).compute()

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{args.steps * 6}h"),
    **dataclasses.asdict(task_config),
)

lats = eval_inputs.coords["lat"].values
lons = eval_inputs.coords["lon"].values
levels = eval_inputs.coords["level"].values
times = eval_targets.coords["time"].values
input_times = eval_inputs.coords["time"].values
n_steps = len(times)
t_minus_6 = input_times[0]
t0_coord = input_times[-1]

# ---------------------------------------------------------------------------
# 3. Construct Gaussian Increment Profiles
# ---------------------------------------------------------------------------
print(f"\n[3/5] Synthesizing spatial anomaly fields (DIR, COL, BAL) ...")
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
d_lat = lat_grid - args.lat
d_lon = (lon_grid - args.lon + 180.0) % 360.0 - 180.0

delta_sfc = args.amplitude * np.exp(-0.5 * ((d_lat / args.sigma_lat) ** 2 + (d_lon / args.sigma_lon) ** 2))
delta_sfc[np.abs(delta_sfc) < (0.01 * np.abs(args.amplitude))] = 0.0

cos_lat = np.cos(np.deg2rad(lat_grid))
delta_X_norm_sq = np.sum(delta_sfc**2 * cos_lat)

# Vertical thermal profile: weights at [1000, 925, 850] hPa
alpha_weights = {1000: 1.0, 925: 0.6, 850: 0.2}
column_warming = {}
for p, w in alpha_weights.items():
    column_warming[p] = w * delta_sfc

# Hypsometric geopotential update (thickness expansion):
# ΔZ_layer = (R_d / g) * ln(p_bot / p_top) * ΔT_v_mean
Rd = 287.05
g = 9.80665
delta_Z = np.zeros_like(delta_sfc)
p_profile = [1000, 925, 850, 700]
thickness_profile = {}
for i in range(len(p_profile) - 1):
    p_bot, p_top = p_profile[i], p_profile[i + 1]
    dT_bot = column_warming.get(p_bot, 0.0)
    dT_top = column_warming.get(p_top, 0.0)
    dT_layer = 0.5 * (dT_bot + dT_top)
    dPhi = Rd * np.log(p_bot / p_top) * dT_layer
    thickness_profile[p_top] = dPhi

# Cumulative geopotential ridge ΔΦ at levels
phi_ridge = {}
cum_phi = np.zeros_like(delta_sfc)
for p in [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]:
    if p in thickness_profile:
        cum_phi = cum_phi + thickness_profile[p]
    phi_ridge[p] = cum_phi.copy()

# Function to inject increment into eval_inputs copy
def create_perturbed_inputs(spatial_type, temporal_type):
    """
    spatial_type: 'DIR' (surface only), 'COL' (column T), 'BAL' (column T + Z)
    temporal_type: 'IMP' (t0 only), 'IAU' (both t-6h and t0)
    """
    inp = eval_inputs.copy(deep=True)
    target_times = [t0_coord] if temporal_type == "IMP" else [t_minus_6, t0_coord]

    for t_step in target_times:
        # 1. Update 2m Temperature
        t2m_orig = inp["2m_temperature"].sel(time=t_step).values
        inp["2m_temperature"].loc[dict(time=t_step)] = t2m_orig + delta_sfc

        # 2. Update Column Temperature (COL and BAL)
        if spatial_type in ["COL", "BAL"]:
            for p, dT in column_warming.items():
                if p in inp.coords["level"].values:
                    t_orig = inp["temperature"].sel(time=t_step, level=p).values
                    inp["temperature"].loc[dict(time=t_step, level=p)] = t_orig + dT

        # 3. Update Geopotential Ridge (BAL only)
        if spatial_type == "BAL":
            for p, dPhi in phi_ridge.items():
                if p in inp.coords["level"].values:
                    z_orig = inp["geopotential"].sel(time=t_step, level=p).values
                    inp["geopotential"].loc[dict(time=t_step, level=p)] = z_orig + dPhi

    return inp

# Build the 6 arms
arms = {
    "DIR-IMP": create_perturbed_inputs("DIR", "IMP"),
    "DIR-IAU": create_perturbed_inputs("DIR", "IAU"),
    "COL-IMP": create_perturbed_inputs("COL", "IMP"),
    "COL-IAU": create_perturbed_inputs("COL", "IAU"),
    "BAL-IMP": create_perturbed_inputs("BAL", "IMP"),
    "BAL-IAU": create_perturbed_inputs("BAL", "IAU"),
}

# ---------------------------------------------------------------------------
# 4. JIT Predictor & Rollout Execution
# ---------------------------------------------------------------------------
print("\n[4/5] Building GraphCast predictor & JIT compiling ...")

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
    predictions, _state = _run_forward_jitted(rng=rng, inputs=inputs, targets_template=targets_template, forcings=forcings)
    return predictions

def execute_rollout(inp, name):
    print(f"      Running rollout: {name} ...", end="", flush=True)
    t0_run = time.time()
    preds = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inp,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    print(f" done ({time.time() - t0_run:.2f} s)")
    return preds

print("\nExecuting Rollouts across all arms:")
preds_base = execute_rollout(eval_inputs, "BASELINE (E-BASE)")
results = {}
for arm_name, arm_inp in arms.items():
    results[arm_name] = execute_rollout(arm_inp, arm_name)

# ---------------------------------------------------------------------------
# 5. Compute Metrics & Save Diagnostics
# ---------------------------------------------------------------------------
print("\n[5/5] Computing retention, peak amplitude, and CONUS RMSE ...")

lat_mask = (lats >= 25.0) & (lats <= 50.0)
lon_mask = (lons >= 235.0) & (lons <= 295.0)
conus_mask = lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]
conus_weights = cos_lat * conus_mask

retention_data = {arm: [] for arm in arms}
peak_amp_data = {arm: [] for arm in arms}
conus_rmse_data = {arm: [] for arm in arms}
base_rmse_data = []

print("\n" + "=" * 78)
print(f"{'Lead':<8} | {'Metric':<14} | {'DIR-IMP':<9} | {'DIR-IAU':<9} | {'COL-IMP':<9} | {'COL-IAU':<9} | {'BAL-IMP':<9} | {'BAL-IAU':<9}")
print("=" * 78)

for step_idx in range(n_steps):
    t_val = times[step_idx]
    lead_h = int(t_val / np.timedelta64(1, "h"))
    f_base = preds_base["2m_temperature"].isel(time=step_idx).values[0]

    # Baseline CONUS RMSE
    target_true = eval_targets["2m_temperature"].isel(time=step_idx).values[0]
    base_conus_rmse = np.sqrt(np.average((f_base - target_true)**2, weights=conus_weights))
    base_rmse_data.append(base_conus_rmse)

    ret_row = {}
    peak_row = {}
    rmse_row = {}

    for arm in arms:
        f_arm = results[arm]["2m_temperature"].isel(time=step_idx).values[0]
        diff = f_arm - f_base

        # 1. Projection retention
        proj = np.sum(diff * delta_sfc * cos_lat) / delta_X_norm_sq * 100.0
        ret_row[arm] = proj
        retention_data[arm].append(proj)

        # 2. Peak amplitude
        peak = np.max(diff) / args.amplitude * 100.0
        peak_row[arm] = peak
        peak_amp_data[arm].append(peak)

        # 3. CONUS RMSE
        arm_rmse = np.sqrt(np.average((f_arm - target_true)**2, weights=conus_weights))
        rmse_row[arm] = arm_rmse
        conus_rmse_data[arm].append(arm_rmse)

    print(f"+{lead_h:02d}h     | Retention (%)  | {ret_row['DIR-IMP']:6.1f} %  | {ret_row['DIR-IAU']:6.1f} %  | {ret_row['COL-IMP']:6.1f} %  | {ret_row['COL-IAU']:6.1f} %  | {ret_row['BAL-IMP']:6.1f} %  | {ret_row['BAL-IAU']:6.1f} %")
    print(f"         | Peak Amp (%)   | {peak_row['DIR-IMP']:6.1f} %  | {peak_row['DIR-IAU']:6.1f} %  | {peak_row['COL-IMP']:6.1f} %  | {peak_row['COL-IAU']:6.1f} %  | {peak_row['BAL-IMP']:6.1f} %  | {peak_row['BAL-IAU']:6.1f} %")
    print(f"         | CONUS RMSE (K) | {rmse_row['DIR-IMP']:6.3f} K  | {rmse_row['DIR-IAU']:6.3f} K  | {rmse_row['COL-IMP']:6.3f} K  | {rmse_row['COL-IAU']:6.3f} K  | {rmse_row['BAL-IMP']:6.3f} K  | {rmse_row['BAL-IAU']:6.3f} K")
    print("-" * 78)

print("=" * 78)

# ---------------------------------------------------------------------------
# Generate Diagnostic Plots
# ---------------------------------------------------------------------------
lead_hours = [int(t / np.timedelta64(1, "h")) for t in times]
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=args.dpi)

# Panel 1: Retention Comparison
ax1 = axes[0]
colors = {"DIR": "#e41a1c", "COL": "#ff7f00", "BAL": "#377eb8"}
for s_type in ["DIR", "COL", "BAL"]:
    # IMP: Dashed
    ax1.plot([0] + lead_hours, [100.0] + retention_data[f"{s_type}-IMP"],
             linestyle="--", marker="o", color=colors[s_type], label=f"{s_type}-IMP (t0 only)")
    # IAU: Solid thick
    ax1.plot([0] + lead_hours, [100.0] + retention_data[f"{s_type}-IAU"],
             linestyle="-", linewidth=2.5, marker="s", color=colors[s_type], label=f"{s_type}-IAU (t-6h & t0)")

ax1.axhline(50, color="gray", linestyle=":", alpha=0.6, label="50% Half-Life")
ax1.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax1.set_ylabel("Area-Weighted Retention (%)", fontsize=12)
ax1.set_title("Increment Retention: Spatial Balance vs. NASA GMAO IAU", fontsize=13, fontweight="bold")
ax1.set_xticks([0, 6, 12, 18, 24])
ax1.set_ylim(0, 130)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)

# Panel 2: CONUS RMSE Trajectory
ax2 = axes[1]
ax2.plot(lead_hours, base_rmse_data, "k--", linewidth=2, label="Base ERA5")
for s_type in ["DIR", "COL", "BAL"]:
    ax2.plot(lead_hours, conus_rmse_data[f"{s_type}-IMP"], linestyle="--", marker="o", color=colors[s_type], label=f"{s_type}-IMP")
    ax2.plot(lead_hours, conus_rmse_data[f"{s_type}-IAU"], linestyle="-", linewidth=2.0, marker="s", color=colors[s_type], label=f"{s_type}-IAU")

ax2.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax2.set_ylabel("CONUS 2m Temperature RMSE (K)", fontsize=12)
ax2.set_title("Forecast Error Trajectory (CONUS)", fontsize=13, fontweight="bold")
ax2.set_xticks(lead_hours)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
fig_out = os.path.join(OUTDIR, "retention_iau_spatial_vs_temporal.png")
plt.savefig(fig_out)
plt.close()
print(f"\nFigure saved to: {fig_out}")

# Save full results NetCDF
ds_save = xr.Dataset(
    data_vars={
        "base_rmse": (["lead"], base_rmse_data),
        **{f"retention_{arm}": (["lead"], retention_data[arm]) for arm in arms},
        **{f"peak_amp_{arm}": (["lead"], peak_amp_data[arm]) for arm in arms},
        **{f"conus_rmse_{arm}": (["lead"], conus_rmse_data[arm]) for arm in arms},
    },
    coords={"lead": lead_hours},
)
nc_out = os.path.join(OUTDIR, "iau_experiment_results.nc")
ds_save.to_netcdf(nc_out)
print(f"Results NetCDF saved to: {nc_out}")
print("\n" + "=" * 72)
print("SUCCESS: 4D Balance & IAU Experiment Finished Successfully!")
print("=" * 72)
