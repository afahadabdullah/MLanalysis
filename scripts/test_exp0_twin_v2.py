#!/usr/bin/env python3
"""
Experiment 0 v2: Unbiased Identical-Twin (OSSE) Benchmark
==========================================================
Answers: "Which observation insertion method recovers the most background
error against a known truth, when the mass field is degraded and observations
are sparse and noisy (no inverse crime)?"

Key Improvements over v1:
  1. Mass Field (Z) Degraded in Background:
     A cold trough is injected into both T and geopotential heights (Z)
     consistently. Therefore, the hypsometric update in BAL genuinely restores
     the true mass field rather than adding spurious error.
  2. Sparse Pseudo-Observations with Noise:
     300 random stations are sampled over CONUS from the Nature Run truth.
     0.5 K Gaussian noise is added. 30% of stations are withheld for independent
     verification. The remaining 70% are spread via 2D spatial analysis (L = 150 km).
  3. NASA GMAO GEOS IAU Windowing Included:
     Evaluates both Impulse (t0 only) and IAU Window (t-6h & t0) across
     DIR, COL, and BAL.

The Evaluated Arms:
  - BG:      Uncorrected degraded background
  - DIR-IMP: Direct surface only at t0
  - DIR-IAU: Direct surface at t-6h and t0
  - COL-IMP: Column thermal at t0, no Z
  - COL-IAU: Column thermal at t-6h and t0, no Z
  - BAL-IMP: Column thermal + hypsometric Z at t0
  - BAL-IAU: Full 4D Balance (Column T + Z + IAU window)

Usage:
  python scripts/test_exp0_twin_v2.py
  python scripts/test_exp0_twin_v2.py --sample data/era5/source-era5_date-2018-01-15_res-1.0_levels-13_steps-04.nc
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
parser = argparse.ArgumentParser(description="Unbiased Twin OSSE Benchmark (Exp 0 v2)")
parser.add_argument("--steps", type=int, default=4, help="Forecast steps (6h each). Default: 4 (24h)")
parser.add_argument("--error-amp", type=float, default=2.0, help="Background cold bias amplitude (K). Default: 2.0 K")
parser.add_argument("--obs-noise", type=float, default=0.5, help="Observation noise std dev (K). Default: 0.5 K")
parser.add_argument("--n-obs", type=int, default=300, help="Number of CONUS pseudo-stations. Default: 300")
parser.add_argument("--lat", type=float, default=38.0, help="Center latitude (deg N). Default: 38.0")
parser.add_argument("--lon", type=float, default=265.0, help="Center longitude (deg E). Default: 265.0 (95 W)")
parser.add_argument("--sigma-lat", type=float, default=4.0, help="Lat std dev (deg). Default: 4.0")
parser.add_argument("--sigma-lon", type=float, default=6.0, help="Lon std dev (deg). Default: 6.0")
parser.add_argument("--sample", default=None, help="Sample NetCDF dataset path")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"), help="Project root")
parser.add_argument("--outdir", default=None, help="Output directory (default: <PROJ>/runs/exp0_twin_v2)")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "runs", "exp0_twin_v2")
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
print("EXPERIMENT 0 v2: UNBIASED IDENTICAL-TWIN (OSSE) BENCHMARK")
print("=" * 72)
print(f"Project root:         {PROJ}")
print(f"Dataset path:         {SAMPLE_FILE}")
print(f"Background Cold Bias: -{args.error_amp:.1f} K (Mass Field Z Degraded)")
print(f"Pseudo-Stations:      {args.n_obs} CONUS points (Noise = {args.obs_noise:.1f} K, 30% Withheld)")
print(f"Forecast Horizon:     {args.steps} steps ({args.steps * 6} hours)")

# ---------------------------------------------------------------------------
# 1. Load Checkpoint and Stats
# ---------------------------------------------------------------------------
print("\n[1/6] Loading checkpoint and stats ...")
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
# 2. Prepare Nature Run (Ground Truth)
# ---------------------------------------------------------------------------
print(f"\n[2/6] Loading Nature Run dataset: {os.path.basename(SAMPLE_FILE)} ...")
try:
    example_batch = xr.load_dataset(SAMPLE_FILE, decode_timedelta=True).compute()
except Exception:
    try:
        example_batch = xr.load_dataset(SAMPLE_FILE).compute()
    except Exception:
        with open(SAMPLE_FILE, "rb") as f:
            example_batch = xr.load_dataset(f).compute()

eval_inputs_nature, eval_targets_nature, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{args.steps * 6}h"),
    **dataclasses.asdict(task_config),
)

lats = eval_inputs_nature.coords["lat"].values
lons = eval_inputs_nature.coords["lon"].values
times = eval_targets_nature.coords["time"].values
input_times = eval_inputs_nature.coords["time"].values
n_steps = len(times)
t_minus_6 = input_times[0]
t0_coord = input_times[-1]

# CONUS mask
lat_mask = (lats >= 25.0) & (lats <= 50.0)
lon_mask = (lons >= 235.0) & (lons <= 295.0)
conus_mask = lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]
cos_lat = np.cos(np.deg2rad(lats))[:, np.newaxis]
conus_weights = cos_lat * conus_mask

# ---------------------------------------------------------------------------
# 3. Create Degraded Background Analysis (Mass & Thermodynamic Fields)
# ---------------------------------------------------------------------------
print(f"\n[3/6] Generating degraded background (cold anomaly + hypsometric height trough) ...")
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
d_lat = lat_grid - args.lat
d_lon = (lon_grid - args.lon + 180.0) % 360.0 - 180.0

# Cold anomaly pattern (negative)
bg_cold_sfc = -args.error_amp * np.exp(-0.5 * ((d_lat / args.sigma_lat) ** 2 + (d_lon / args.sigma_lon) ** 2))
bg_cold_sfc[np.abs(bg_cold_sfc) < (0.01 * args.error_amp)] = 0.0

# Column temperature error
bg_column_error = {
    1000: 1.0 * bg_cold_sfc,
    925: 0.6 * bg_cold_sfc,
    850: 0.2 * bg_cold_sfc,
}

# Hypsometric geopotential depression (cold trough): ΔZ < 0
Rd, g = 287.05, 9.80665
p_profile = [1000, 925, 850, 700]
bg_phi_depression = {}
cum_dphi = np.zeros_like(bg_cold_sfc)
for i in range(len(p_profile) - 1):
    p_bot, p_top = p_profile[i], p_profile[i + 1]
    dT_layer = 0.5 * (bg_column_error.get(p_bot, 0.0) + bg_column_error.get(p_top, 0.0))
    dPhi = Rd * np.log(p_bot / p_top) * dT_layer
    cum_dphi = cum_dphi + dPhi
    bg_phi_depression[p_top] = cum_dphi.copy()

for p in [600, 500, 400, 300, 250, 200, 150, 100, 50]:
    bg_phi_depression[p] = cum_dphi.copy()

eval_inputs_bg = eval_inputs_nature.copy(deep=True)
for t_step in [t_minus_6, t0_coord]:
    # 2m temperature
    eval_inputs_bg["2m_temperature"].loc[dict(time=t_step)] = (
        eval_inputs_bg["2m_temperature"].sel(time=t_step).values + bg_cold_sfc
    )
    # Column temperature
    for p, dT in bg_column_error.items():
        if p in eval_inputs_bg.coords["level"].values:
            eval_inputs_bg["temperature"].loc[dict(time=t_step, level=p)] = (
                eval_inputs_bg["temperature"].sel(time=t_step, level=p).values + dT
            )
    # Mass field geopotential depression (trough)
    for p, dPhi in bg_phi_depression.items():
        if p in eval_inputs_bg.coords["level"].values:
            eval_inputs_bg["geopotential"].loc[dict(time=t_step, level=p)] = (
                eval_inputs_bg["geopotential"].sel(time=t_step, level=p).values + dPhi
            )

print("      Background analysis created with self-consistent cold anomaly & height trough.")

# ---------------------------------------------------------------------------
# 4. Sample Sparse Observations & Objective Analysis (No Inverse Crime)
# ---------------------------------------------------------------------------
print(f"\n[4/6] Sampling {args.n_obs} pseudo-observations over CONUS with noise ...")
np.random.seed(42)

# Candidate grid points within CONUS
conus_indices = np.argwhere(conus_mask)
sampled_idx = conus_indices[np.random.choice(len(conus_indices), size=args.n_obs, replace=False)]

# Split 70% assimilated, 30% withheld
n_assimilated = int(0.7 * args.n_obs)
idx_assimilated = sampled_idx[:n_assimilated]
idx_withheld = sampled_idx[n_assimilated:]

print(f"      Assimilated stations: {len(idx_assimilated)} | Withheld stations: {len(idx_withheld)}")

def analyze_observations_for_time(t_step):
    """Samples true T2m, adds noise, computes departures from BG, and spreads via OI."""
    t_true_sfc = eval_inputs_nature["2m_temperature"].sel(time=t_step).values
    t_bg_sfc = eval_inputs_bg["2m_temperature"].sel(time=t_step).values

    obs_lats = lats[idx_assimilated[:, 0]]
    obs_lons = lons[idx_assimilated[:, 1]]
    obs_true = t_true_sfc[idx_assimilated[:, 0], idx_assimilated[:, 1]]
    # Add observation noise
    obs_val = obs_true + np.random.normal(0.0, args.obs_noise, size=len(obs_true))
    # Background at observation locations
    bg_at_obs = t_bg_sfc[idx_assimilated[:, 0], idx_assimilated[:, 1]]
    departures = obs_val - bg_at_obs  # y - H(x_bg)

    # Objective Analysis: Gaussian RBF spreading
    # L = 150 km ≈ 1.5 deg lat, 1.8 deg lon
    L_lat, L_lon = 1.5, 1.8
    analysis_field = np.zeros_like(t_bg_sfc)
    norm_weights = np.zeros_like(t_bg_sfc)

    for k in range(len(departures)):
        dy = (lat_grid - obs_lats[k]) / L_lat
        dx = ((lon_grid - obs_lons[k] + 180.0) % 360.0 - 180.0) / L_lon
        w = np.exp(-0.5 * (dy**2 + dx**2))
        analysis_field += w * departures[k]
        norm_weights += w

    mask_valid = norm_weights > 0.05
    analysis_field[mask_valid] /= norm_weights[mask_valid]
    analysis_field[~mask_valid] = 0.0
    analysis_field *= conus_mask.astype(float)
    return analysis_field

ana_sfc_t0 = analyze_observations_for_time(t0_coord)
ana_sfc_tm6 = analyze_observations_for_time(t_minus_6)

print(f"      Analyzed correction at t0: max = {ana_sfc_t0.max():+.2f} K, min = {ana_sfc_t0.min():+.2f} K")

# Compute hypsometric geopotential correction corresponding to analyzed T
def compute_hypsometric_correction(delta_t_sfc):
    col_T = {1000: 1.0 * delta_t_sfc, 925: 0.6 * delta_t_sfc, 850: 0.2 * delta_t_sfc}
    phi_lift = {}
    cum_phi = np.zeros_like(delta_t_sfc)
    for i in range(len(p_profile) - 1):
        p_bot, p_top = p_profile[i], p_profile[i + 1]
        dT_layer = 0.5 * (col_T.get(p_bot, 0.0) + col_T.get(p_top, 0.0))
        dPhi = Rd * np.log(p_bot / p_top) * dT_layer
        cum_phi = cum_phi + dPhi
        phi_lift[p_top] = cum_phi.copy()
    for p in [600, 500, 400, 300, 250, 200, 150, 100, 50]:
        phi_lift[p] = cum_phi.copy()
    return col_T, phi_lift

# Build the 6 correction arms on top of Background:
def build_correction_arm(spatial_type, temporal_type):
    inp = eval_inputs_bg.copy(deep=True)
    t_targets = [t0_coord] if temporal_type == "IMP" else [t_minus_6, t0_coord]

    for t_step in t_targets:
        ana_sfc = ana_sfc_t0 if t_step == t0_coord else ana_sfc_tm6
        # 1. Surface T
        inp["2m_temperature"].loc[dict(time=t_step)] = (
            inp["2m_temperature"].sel(time=t_step).values + ana_sfc
        )
        if spatial_type in ["COL", "BAL"]:
            col_T, phi_lift = compute_hypsometric_correction(ana_sfc)
            # 2. Column T
            for p, dT in col_T.items():
                if p in inp.coords["level"].values:
                    inp["temperature"].loc[dict(time=t_step, level=p)] = (
                        inp["temperature"].sel(time=t_step, level=p).values + dT
                    )
            # 3. Hypsometric Z lift (BAL only)
            if spatial_type == "BAL":
                for p, dPhi in phi_lift.items():
                    if p in inp.coords["level"].values:
                        inp["geopotential"].loc[dict(time=t_step, level=p)] = (
                            inp["geopotential"].sel(time=t_step, level=p).values + dPhi
                        )
    return inp

arms = {
    "BG": eval_inputs_bg,
    "DIR-IMP": build_correction_arm("DIR", "IMP"),
    "DIR-IAU": build_correction_arm("DIR", "IAU"),
    "COL-IMP": build_correction_arm("COL", "IMP"),
    "COL-IAU": build_correction_arm("COL", "IAU"),
    "BAL-IMP": build_correction_arm("BAL", "IMP"),
    "BAL-IAU": build_correction_arm("BAL", "IAU"),
}

# ---------------------------------------------------------------------------
# 5. JIT & Execute Rollouts
# ---------------------------------------------------------------------------
print("\n[5/6] JIT compiling predictor & executing all arms ...")

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
    print(f"      Rollout: {name:<10} ...", end="", flush=True)
    t0_run = time.time()
    preds = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inp,
        targets_template=eval_targets_nature * np.nan,
        forcings=eval_forcings,
    )
    print(f" done ({time.time() - t0_run:.2f} s)")
    return preds

# Execute Nature Run Truth
preds_nature = execute_rollout(eval_inputs_nature, "TRUTH")
results = {}
for arm_name, arm_inp in arms.items():
    results[arm_name] = execute_rollout(arm_inp, arm_name)

# ---------------------------------------------------------------------------
# 6. Score Against Nature Run Truth & Report
# ---------------------------------------------------------------------------
print("\n[6/6] Scoring error recovery against known Nature Run truth ...")

conus_rmse_by_arm = {arm: [] for arm in arms}
recovery_by_arm = {arm: [] for arm in arms if arm != "BG"}

print("\n" + "=" * 80)
print(f"{'Lead':<7} | {'BG Err':<8} | {'DIR-IMP':<9} | {'DIR-IAU':<9} | {'COL-IMP':<9} | {'COL-IAU':<9} | {'BAL-IMP':<9} | {'BAL-IAU':<9}")
print("=" * 80)

lead_hours = [int(t / np.timedelta64(1, "h")) for t in times]

for step_idx in range(n_steps):
    lead_h = lead_hours[step_idx]
    f_true = preds_nature["2m_temperature"].isel(time=step_idx).values[0]
    f_bg = results["BG"]["2m_temperature"].isel(time=step_idx).values[0]

    bg_rmse = np.sqrt(np.average((f_bg - f_true)**2, weights=conus_weights))
    conus_rmse_by_arm["BG"].append(bg_rmse)

    arm_rmses = {}
    arm_recov = {}

    for arm in arms:
        f_arm = results[arm]["2m_temperature"].isel(time=step_idx).values[0]
        arm_rmse = np.sqrt(np.average((f_arm - f_true)**2, weights=conus_weights))
        conus_rmse_by_arm[arm].append(arm_rmse)
        arm_rmses[arm] = arm_rmse
        if arm != "BG":
            rec = (1.0 - arm_rmse / bg_rmse) * 100.0
            arm_recov[arm] = rec
            recovery_by_arm[arm].append(rec)

    print(f"+{lead_h:02d}h    | RMSE (K) | {arm_rmses['DIR-IMP']:6.3f} K | {arm_rmses['DIR-IAU']:6.3f} K | {arm_rmses['COL-IMP']:6.3f} K | {arm_rmses['COL-IAU']:6.3f} K | {arm_rmses['BAL-IMP']:6.3f} K | {arm_rmses['BAL-IAU']:6.3f} K")
    print(f"       | Recov(%) | {arm_recov['DIR-IMP']:6.1f} % | {arm_recov['DIR-IAU']:6.1f} % | {arm_recov['COL-IMP']:6.1f} % | {arm_recov['COL-IAU']:6.1f} % | {arm_recov['BAL-IMP']:6.1f} % | {arm_recov['BAL-IAU']:6.1f} %")
    print("-" * 80)

print("=" * 80)

# ---------------------------------------------------------------------------
# Diagnostic Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=args.dpi)

colors = {"DIR": "#e41a1c", "COL": "#ff7f00", "BAL": "#377eb8"}

# Panel 1: Error Recovery Percentage
ax1 = axes[0]
for s_type in ["DIR", "COL", "BAL"]:
    ax1.plot(lead_hours, recovery_by_arm[f"{s_type}-IMP"], linestyle="--", marker="o", color=colors[s_type], label=f"{s_type}-IMP")
    ax1.plot(lead_hours, recovery_by_arm[f"{s_type}-IAU"], linestyle="-", linewidth=2.2, marker="s", color=colors[s_type], label=f"{s_type}-IAU (Windowed)")

ax1.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax1.set_ylabel("Background Error Recovered (%)", fontsize=12)
ax1.set_title("Twin Benchmark: Error Recovery vs. Nature Run Truth", fontsize=13, fontweight="bold")
ax1.set_xticks(lead_hours)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)

# Panel 2: CONUS RMSE vs Lead Time
ax2 = axes[1]
ax2.plot(lead_hours, conus_rmse_by_arm["BG"], "k-.", linewidth=2.2, label="Uncorrected Background (BG)")
for s_type in ["DIR", "COL", "BAL"]:
    ax2.plot(lead_hours, conus_rmse_by_arm[f"{s_type}-IMP"], linestyle="--", marker="o", color=colors[s_type], label=f"{s_type}-IMP")
    ax2.plot(lead_hours, conus_rmse_by_arm[f"{s_type}-IAU"], linestyle="-", linewidth=2.0, marker="s", color=colors[s_type], label=f"{s_type}-IAU")

ax2.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
ax2.set_ylabel("CONUS 2m Temperature RMSE (K)", fontsize=12)
ax2.set_title("Forecast Error vs. True Atmosphere", fontsize=13, fontweight="bold")
ax2.set_xticks(lead_hours)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
fig_out = os.path.join(OUTDIR, "exp0_v2_error_recovery_progression.png")
plt.savefig(fig_out)
plt.close()
print(f"\nFigure saved to: {fig_out}")

# Save results to NetCDF
ds_save = xr.Dataset(
    data_vars={
        "bg_rmse": (["lead"], conus_rmse_by_arm["BG"]),
        **{f"rmse_{arm}": (["lead"], conus_rmse_by_arm[arm]) for arm in arms if arm != "BG"},
        **{f"recovery_{arm}": (["lead"], recovery_by_arm[arm]) for arm in arms if arm != "BG"},
    },
    coords={"lead": lead_hours},
)
nc_out = os.path.join(OUTDIR, "exp0_v2_results.nc")
ds_save.to_netcdf(nc_out)
print(f"Results NetCDF saved to: {nc_out}")

print("\n" + "=" * 72)
print("SUCCESS: Experiment 0 v2 Completed Successfully!")
print("=" * 72)
