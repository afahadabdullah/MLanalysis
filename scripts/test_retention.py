#!/usr/bin/env python3
"""
GraphCast Synthetic Anomaly & Increment Retention Experiment
============================================================
Implements the core hypothesis test from PROJECT_PLAN_LEAN.md (Exp 2 / Stage S0):
  "Does GraphCast retain an inserted surface temperature increment (ΔT2m),
   or does initialization shock wipe it out in the early forecast hours?"

This script:
  1. Loads GraphCast_small checkpoint, normalization stats, and the sample dataset.
  2. Injects a localized Gaussian temperature anomaly (ΔT2m = +2.0 K) over CONUS at t0.
  3. Executes two autoregressive rollouts:
       - Baseline: F_base(t)
       - Perturbed (Direct Insertion): F_pert(t)
  4. Computes the formal increment retention metric:
       R(t) = <F_pert(t) - F_base(t), ΔX> / ||ΔX||^2  (area-weighted)
     plus peak amplitude retention and vertical response metrics.
  5. Saves the full 4D difference dataset to NetCDF (runs/retention/retention_experiment.nc).
  6. Generates comprehensive diagnostic plots:
       - retention_decay_curve.png
       - retention_spatial_evolution.png
       - vertical_response_profile.png
       - surface_wind_mslp_coupling.png

Usage:
  python scripts/test_retention.py
  python scripts/test_retention.py --amplitude 2.0 --steps 4
  python scripts/test_retention.py --lat 38.0 --lon 265.0 --sigma 5.0
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
parser = argparse.ArgumentParser(description="GraphCast Synthetic Anomaly Retention Test")
parser.add_argument("--steps", type=int, default=4,
                    help="Forecast steps (6h each). Default: 4 (24 hours)")
parser.add_argument("--amplitude", type=float, default=2.0,
                    help="Peak anomaly amplitude in Kelvin. Default: +2.0 K")
parser.add_argument("--lat", type=float, default=38.0,
                    help="Center latitude for anomaly in degrees N. Default: 38.0 N (Central US)")
parser.add_argument("--lon", type=float, default=265.0,
                    help="Center longitude in degrees E (0-360). Default: 265.0 E (95 W)")
parser.add_argument("--sigma-lat", type=float, default=4.0,
                    help="Latitudinal Gaussian standard deviation (deg). Default: 4.0")
parser.add_argument("--sigma-lon", type=float, default=6.0,
                    help="Longitudinal Gaussian standard deviation (deg). Default: 6.0")
parser.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    help="Project root directory")
parser.add_argument("--outdir", default=None,
                    help="Directory for outputs and plots (default: <PROJ>/runs/retention)")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

PROJ = args.proj
OUTDIR = args.outdir or os.path.join(PROJ, "runs", "retention")
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
print("GraphCast Synthetic Anomaly & Increment Retention Experiment")
print("=" * 65)
print(f"Project root:     {PROJ}")
print(f"Output directory: {OUTDIR}")
print(f"Anomaly config:   ΔT2m = {args.amplitude:+.1f} K at ({args.lat}°N, {args.lon}°E)")
print(f"Anomaly radius:   σ_lat = {args.sigma_lat}°, σ_lon = {args.sigma_lon}°")
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
# 2. Load Sample Data & Prepare Baseline Inputs
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

# ---------------------------------------------------------------------------
# 3. Create Perturbed Inputs (Direct Insertion of Gaussian Bubble at t0)
# ---------------------------------------------------------------------------
print(f"\n[3/6] Generating synthetic 2m temperature increment (ΔX) ...")
eval_inputs_pert = eval_inputs.copy(deep=True)

# Compute 2D Gaussian anomaly over (lat, lon) grid
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
d_lat = lat_grid - args.lat
# Longitude distance handling periodic wrap-around
d_lon = (lon_grid - args.lon + 180.0) % 360.0 - 180.0

gaussian_bubble = args.amplitude * np.exp(-0.5 * ((d_lat / args.sigma_lat) ** 2 + (d_lon / args.sigma_lon) ** 2))
# Zero out negligible perturbations below 1% of peak
gaussian_bubble[np.abs(gaussian_bubble) < (0.01 * np.abs(args.amplitude))] = 0.0

# In GraphCast, inputs have time coordinates (typically [t-6h, t0]).
# We inject the perturbation at the analysis launch time t0 (last input time).
t0_coord = eval_inputs.coords["time"].values[-1]

# Apply to eval_inputs_pert['2m_temperature']
t2m_current = eval_inputs_pert["2m_temperature"].sel(time=t0_coord).values
eval_inputs_pert["2m_temperature"].loc[dict(time=t0_coord)] = t2m_current + gaussian_bubble

# Store the exact applied increment ΔX
delta_X = gaussian_bubble  # shape (lat, lon)
delta_X_norm_sq = np.sum(delta_X**2 * np.cos(np.deg2rad(lat_grid)))
print(f"      Injected bubble max: {np.max(delta_X):+.2f} K, min: {np.min(delta_X):+.2f} K")
print(f"      Injected increment area-norm ||ΔX||: {np.sqrt(delta_X_norm_sq):.3f} K")

# ---------------------------------------------------------------------------
# 4. Construct Predictor & JIT Compile
# ---------------------------------------------------------------------------
print(f"\n[4/6] Building GraphCast predictor & JIT compiling ...")

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

with_configs = functools.partial(
    run_forward.apply,
    m_cfg=model_config,
    t_cfg=task_config,
)

_run_forward_jitted = jax.jit(
    functools.partial(with_configs, params=params, state=state)
)

def run_forward_jitted(rng, inputs, targets_template, forcings):
    predictions, _state = _run_forward_jitted(
        rng=rng, inputs=inputs, targets_template=targets_template, forcings=forcings
    )
    return predictions

# ---------------------------------------------------------------------------
# 5. Run Both Rollouts (Baseline & Perturbed)
# ---------------------------------------------------------------------------
print(f"\n[5/6] Executing forecast rollouts ({n_steps} steps / {n_steps * 6}h) on GPU ...")

t_start = time.time()
print("      Running BASELINE rollout (F_base) ...")
preds_base = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
)
t_base = time.time() - t_start
print(f"      Baseline finished in {t_base:.2f} s")

t_start = time.time()
print("      Running PERTURBED rollout (F_pert) ...")
preds_pert = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs_pert,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
)
t_pert = time.time() - t_start
print(f"      Perturbed finished in {t_pert:.2f} s")

# Compute the difference dataset
diff_ds = preds_pert - preds_base

# ---------------------------------------------------------------------------
# 6. Compute Increment Retention and Physical Coupling Metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("INCREMENT RETENTION ANALYSIS RESULTS")
print("=" * 65)

# Area weights for spherical grid
weights = np.cos(np.deg2rad(lats))[:, np.newaxis]

retention_scores = []
peak_amplitudes = []
lead_hours = []

for step_idx in range(n_steps):
    hours = (step_idx + 1) * 6
    lead_hours.append(hours)
    
    # 2m temperature response at this step
    delta_t2m = diff_ds["2m_temperature"].isel(time=step_idx).values.squeeze()
    
    # Area-weighted inner product: <delta_t2m, delta_X> / ||delta_X||^2
    inner_prod = np.sum(delta_t2m * delta_X * weights)
    retention_r = inner_prod / (delta_X_norm_sq + 1e-12)
    peak_amp = np.max(np.abs(delta_t2m))
    amp_ratio = peak_amp / np.abs(args.amplitude)
    
    retention_scores.append(float(retention_r))
    peak_amplitudes.append(float(amp_ratio))
    
    print(f"  Lead +{hours:02d}h:  Projection Retention R(t) = {retention_r * 100:6.2f}%  |  Peak Anomaly = {peak_amp:.3f} K ({amp_ratio * 100:5.1f}%)")

# Check vertical propagation into 3D temperature
vertical_response = {}
if "temperature" in diff_ds and "level" in diff_ds["temperature"].dims:
    print("\nVertical Response in Column Temperature (ΔT at bubble center):")
    levels = diff_ds.coords["level"].values
    lat_idx = np.argmin(np.abs(lats - args.lat))
    lon_idx = np.argmin(np.abs(lons - args.lon))
    
    for step_idx in range(n_steps):
        hours = (step_idx + 1) * 6
        print(f"  At +{hours:02d}h:")
        for lev in levels:
            delta_t_lev = float(diff_ds["temperature"].isel(time=step_idx).sel(level=lev).values.squeeze()[lat_idx, lon_idx])
            if abs(delta_t_lev) > 0.01:
                print(f"     {lev:4d} hPa: ΔT = {delta_t_lev:+.3f} K")

# Save summary text
summary_txt = os.path.join(OUTDIR, "retention_summary.txt")
with open(summary_txt, "w") as f:
    f.write("GraphCast Synthetic Increment Retention Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Injected anomaly: {args.amplitude:+.2f} K at ({args.lat}N, {args.lon}E)\n")
    f.write(f"Radius: sigma_lat={args.sigma_lat}, sigma_lon={args.sigma_lon}\n\n")
    for h, r, p in zip(lead_hours, retention_scores, peak_amplitudes):
        f.write(f"+{h:02d}h: Retention = {r*100:.2f}%, Peak = {p*100:.2f}%\n")
print(f"\nSummary written to: {summary_txt}")

# Save NetCDF
nc_out = os.path.join(OUTDIR, "retention_experiment.nc")
diff_ds.to_netcdf(nc_out)
print(f"Full difference dataset saved to: {nc_out}")

# ---------------------------------------------------------------------------
# 7. Diagnostic Plotting
# ---------------------------------------------------------------------------
print(f"\n[6/6] Generating publication diagnostic plots in {OUTDIR} ...")

def add_map_elements(ax):
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="k")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="grey")
        ax.add_feature(cfeature.STATES, linewidth=0.2, color="lightgrey")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False

# Plot 1: Retention Decay Curve
fig, ax = plt.subplots(figsize=(8, 5))
hours_axis = [0] + lead_hours
ret_axis = [100.0] + [r * 100 for r in retention_scores]
peak_axis = [100.0] + [p * 100 for p in peak_amplitudes]

ax.plot(hours_axis, ret_axis, "o-", color="#1f77b4", linewidth=2.5, markersize=8,
        label=r"Projection Retention: $\langle \Delta F(t), \Delta X \rangle / \|\Delta X\|^2$")
ax.plot(hours_axis, peak_axis, "s--", color="#d62728", linewidth=2.0, markersize=7,
        label=r"Peak Amplitude Retention: $\max|\Delta F(t)| / \max|\Delta X|$")

ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.set_xlabel("Forecast Lead Time (hours)", fontsize=11)
ax.set_ylabel("Retention Metric (%)", fontsize=11)
ax.set_title(f"GraphCast Increment Retention (Injected ΔT2m = {args.amplitude:+.1f} K)",
             fontsize=13, fontweight="bold")
ax.set_xticks(hours_axis)
ax.set_ylim(-10, 110)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=10, loc="best")
fig.tight_layout()
p1 = os.path.join(OUTDIR, "retention_decay_curve.png")
fig.savefig(p1, dpi=args.dpi)
plt.close(fig)
print(f"  ✓ {os.path.basename(p1)}")

# Plot 2: Spatial Evolution Multi-panel (t0, +6h, +12h, +18h, +24h)
n_panels = 1 + n_steps
fig_w = 4 * n_panels
if HAS_CARTOPY:
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
else:
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.5))

# Regional zoom over North America / CONUS
extent = [230, 300, 20, 60]  # [lon_min, lon_max, lat_min, lat_max]
vmax = abs(args.amplitude)
vmin = -vmax

# Panel 0: t=0 injection
ax0 = axes[0]
if HAS_CARTOPY:
    im0 = ax0.pcolormesh(lons, lats, delta_X, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())
    ax0.set_extent(extent, crs=ccrs.PlateCarree())
    add_map_elements(ax0)
else:
    im0 = ax0.pcolormesh(lons, lats, delta_X, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax0.set_xlim(extent[0], extent[1])
    ax0.set_ylim(extent[2], extent[3])
ax0.set_title("t = 0h (Injected ΔX)", fontsize=11, fontweight="bold")

# Panels 1..N: forecast steps
for step_idx in range(n_steps):
    ax = axes[step_idx + 1]
    hours = (step_idx + 1) * 6
    diff_2m = diff_ds["2m_temperature"].isel(time=step_idx).values.squeeze()
    
    if HAS_CARTOPY:
        ax.pcolormesh(lons, lats, diff_2m, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                      transform=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        add_map_elements(ax)
    else:
        ax.pcolormesh(lons, lats, diff_2m, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xlim(extent[0], extent[1])
        ax0.set_ylim(extent[2], extent[3])
        
    r_val = retention_scores[step_idx] * 100
    ax.set_title(f"+{hours:02d}h (Ret: {r_val:.1f}%)", fontsize=11, fontweight="bold")

fig.suptitle(r"Spatial Evolution of $\Delta T_{2m}$ Anomaly over CONUS",
             fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
p2 = os.path.join(OUTDIR, "retention_spatial_evolution.png")
fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {os.path.basename(p2)}")

# Plot 3: Vertical Response Profile (Does surface anomaly propagate aloft?)
if "temperature" in diff_ds and "level" in diff_ds["temperature"].dims:
    levels = diff_ds.coords["level"].values
    lat_idx = np.argmin(np.abs(lats - args.lat))
    lon_idx = np.argmin(np.abs(lons - args.lon))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for step_idx in range(n_steps):
        hours = (step_idx + 1) * 6
        t_col = diff_ds["temperature"].isel(time=step_idx).values.squeeze()[:, lat_idx, lon_idx]
        ax.plot(t_col, levels, "o-", label=f"+{hours:02d}h", linewidth=1.8)
        
    ax.set_ylabel("Pressure Level (hPa)", fontsize=11)
    ax.set_xlabel("Induced ΔT (K)", fontsize=11)
    ax.set_title("Column Temperature Response at Anomaly Center", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(OUTDIR, "vertical_response_profile.png")
    fig.savefig(p3, dpi=args.dpi)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(p3)}")

# Plot 4: Secondary Coupling (Wind & MSLP adjustments)
if "10m_u_component_of_wind" in diff_ds and "mean_sea_level_pressure" in diff_ds:
    step_eval = min(1, n_steps - 1)  # step 1 = +12h
    hours_eval = (step_eval + 1) * 6
    diff_mslp = diff_ds["mean_sea_level_pressure"].isel(time=step_eval).values.squeeze() / 100.0  # Pa -> hPa
    diff_u = diff_ds["10m_u_component_of_wind"].isel(time=step_eval).values.squeeze()
    diff_v = diff_ds["10m_v_component_of_wind"].isel(time=step_eval).values.squeeze()
    
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {})
    emax = max(0.2, np.max(np.abs(diff_mslp)))
    
    if HAS_CARTOPY:
        im = ax.pcolormesh(lons, lats, diff_mslp, cmap="PuOr_r", vmin=-emax, vmax=emax,
                           transform=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        add_map_elements(ax)
        # Vector subsampling for arrows
        skip = 2
        ax.quiver(lons[::skip], lats[::skip], diff_u[::skip, ::skip], diff_v[::skip, ::skip],
                  transform=ccrs.PlateCarree(), scale=25, width=0.003, color="black")
    else:
        im = ax.pcolormesh(lons, lats, diff_mslp, cmap="PuOr_r", vmin=-emax, vmax=emax)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        skip = 2
        ax.quiver(lons[::skip], lats[::skip], diff_u[::skip, ::skip], diff_v[::skip, ::skip],
                  scale=25, color="black")
        
    plt.colorbar(im, ax=ax, label="Induced ΔMSLP (hPa)", shrink=0.7)
    ax.set_title(f"Dynamic Adjustment: Induced ΔMSLP & 10m Wind at +{hours_eval}h",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p4 = os.path.join(OUTDIR, "surface_wind_mslp_coupling.png")
    fig.savefig(p4, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.basename(p4)}")

print("\n" + "=" * 65)
print(f"Experiment complete! All figures saved in:\n  {OUTDIR}/")
print("=" * 65)
