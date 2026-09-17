#!/usr/bin/env python3
"""
GraphCast Diagnostic Plotting
==============================
Generates verification plots comparing GraphCast predictions against ERA5 truth.

Outputs saved to: <PROJ>/runs/diagnostics/

Panels produced:
  1. 2m Temperature — Forecast vs ERA5 vs Error (per lead time)
  2. Geopotential 500 hPa — Forecast vs ERA5 vs Error
  3. Temperature 850 hPa — Forecast vs ERA5 vs Error
  4. Specific Humidity 700 hPa — Forecast vs ERA5 vs Error
  5. 10m Wind Speed — Forecast vs ERA5 vs Error
  6. Global RMSE by variable (bar chart)
  7. RMSE by pressure level (temperature, geopotential, wind)
  8. Zonal-mean temperature cross-section (lat × pressure)

Usage:
  python scripts/plot_diagnostics.py                              # defaults
  python scripts/plot_diagnostics.py --predictions runs/predictions.nc --truth runs/era5_truth.nc
  python scripts/plot_diagnostics.py --step 0                     # plot only first lead time
"""

import argparse
import os
import sys
import warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = os.environ.get("PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="GraphCast forecast diagnostics")
parser.add_argument("--predictions", default=os.path.join(PROJ, "runs", "predictions.nc"),
                    help="Path to predictions NetCDF (default: runs/predictions.nc)")
parser.add_argument("--truth", default=os.path.join(PROJ, "runs", "era5_truth.nc"),
                    help="Path to ERA5 truth NetCDF (default: runs/era5_truth.nc)")
parser.add_argument("--outdir", default=os.path.join(PROJ, "runs", "diagnostics"),
                    help="Output directory for plots (default: runs/diagnostics/)")
parser.add_argument("--step", type=int, default=None,
                    help="Plot only this lead-time step index (0-based). Default: all steps.")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib setup (Agg backend for headless nodes)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Try cartopy for proper map projections; fall back to plain lat/lon
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  Note: cartopy not found — maps will use plain lat/lon axes.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("GraphCast Diagnostic Plots")
print("=" * 60)

print(f"\nLoading predictions: {args.predictions}")
preds = xr.load_dataset(args.predictions)
print(f"Loading ERA5 truth:  {args.truth}")
truth = xr.load_dataset(args.truth)

# Coordinate arrays
lats = preds.coords["lat"].values
lons = preds.coords["lon"].values
times = preds.coords["time"].values
n_steps = len(times)

steps_to_plot = [args.step] if args.step is not None else list(range(n_steps))
print(f"Lead times: {n_steps} steps ({n_steps * 6}h total)")
print(f"Steps to plot: {steps_to_plot}")
print(f"Predicted variables: {sorted(preds.data_vars.keys())}")
print(f"Output directory: {args.outdir}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _lead_label(step_idx):
    """Return human-readable lead time label, e.g. '+6h', '+12h'."""
    return f"+{(step_idx + 1) * 6}h"


def _get_field(ds, var, step, level=None, batch=0):
    """Extract a 2-D (lat, lon) slice from a dataset."""
    sel = dict(time=ds.coords["time"].values[step])
    if "batch" in ds[var].dims:
        sel["batch"] = batch
    if level is not None and "level" in ds[var].dims:
        sel["level"] = level
    return ds[var].sel(**sel).values.squeeze()


def _add_map_features(ax):
    """Add coastlines and gridlines if cartopy is available."""
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="k")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="grey")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False


def _create_axes(ncols=3, figsize=(18, 5)):
    """Create figure + axes, optionally with map projection."""
    if HAS_CARTOPY:
        fig, axes = plt.subplots(1, ncols, figsize=figsize,
                                 subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return fig, axes


def plot_spatial_triplet(var, step, level=None, units="", cmap="RdYlBu_r",
                         err_cmap="RdBu_r", vmin=None, vmax=None, tag=None):
    """Plot Forecast | ERA5 Truth | Error for a single field and lead time."""
    fc = _get_field(preds, var, step, level)
    tr = _get_field(truth, var, step, level)
    err = fc - tr

    if vmin is None:
        vmin = min(np.nanmin(fc), np.nanmin(tr))
    if vmax is None:
        vmax = max(np.nanmax(fc), np.nanmax(tr))
    emax = max(abs(np.nanmin(err)), abs(np.nanmax(err)))

    lead = _lead_label(step)
    level_str = f" @ {level} hPa" if level else ""
    title_base = f"{var}{level_str} {lead}"
    tag_str = tag or var.replace(" ", "_")

    fig, axes = _create_axes()

    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    transform_kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}

    im0 = axes[0].pcolormesh(lons, lats, fc, **kw, **transform_kw)
    axes[0].set_title(f"GraphCast Forecast {lead}")
    _add_map_features(axes[0])
    plt.colorbar(im0, ax=axes[0], shrink=0.7, label=units)

    im1 = axes[1].pcolormesh(lons, lats, tr, **kw, **transform_kw)
    axes[1].set_title(f"ERA5 Truth {lead}")
    _add_map_features(axes[1])
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label=units)

    im2 = axes[2].pcolormesh(lons, lats, err, cmap=err_cmap, vmin=-emax, vmax=emax,
                              **transform_kw)
    axes[2].set_title(f"Error (FC − ERA5) {lead}")
    _add_map_features(axes[2])
    plt.colorbar(im2, ax=axes[2], shrink=0.7, label=units)

    fig.suptitle(title_base, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = f"{tag_str}_step{step}.png"
    fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def compute_rmse(pred_field, truth_field):
    """Global area-weighted RMSE."""
    err = pred_field - truth_field
    # Area weighting by cos(latitude)
    weights = np.cos(np.deg2rad(lats))
    weights = weights[:, np.newaxis]  # broadcast over lon
    weights = np.broadcast_to(weights, err.shape)
    return float(np.sqrt(np.nanmean(weights * err**2)))


# ===========================================================================
# 1–5. Spatial maps for key variables
# ===========================================================================
print("\n--- Spatial Maps ---")

SURFACE_FIELDS = [
    ("2m_temperature",           None,  "K",       "RdYlBu_r", "t2m"),
    ("10m_u_component_of_wind",  None,  "m/s",     "RdBu_r",   "u10"),
    ("10m_v_component_of_wind",  None,  "m/s",     "RdBu_r",   "v10"),
    ("mean_sea_level_pressure",  None,  "Pa",      "coolwarm",  "mslp"),
    ("total_precipitation_6hr",  None,  "m",       "YlGnBu",   "tp6h"),
]

PRESSURE_FIELDS = [
    ("geopotential",             500,   "m²/s²",   "RdYlBu_r", "z500"),
    ("temperature",              850,   "K",       "RdYlBu_r", "t850"),
    ("specific_humidity",        700,   "kg/kg",   "YlGnBu",   "q700"),
    ("u_component_of_wind",      250,   "m/s",     "RdBu_r",   "u250"),
]

ALL_FIELDS = SURFACE_FIELDS + PRESSURE_FIELDS

for step in steps_to_plot:
    for var, level, units, cmap, tag in ALL_FIELDS:
        if var not in preds.data_vars:
            continue
        if level is not None and "level" not in preds[var].dims:
            continue
        try:
            plot_spatial_triplet(var, step, level=level, units=units, cmap=cmap, tag=tag)
        except Exception as e:
            print(f"  ✗ Skipped {tag} step {step}: {e}")

# ===========================================================================
# 6. Global RMSE by variable (bar chart)
# ===========================================================================
print("\n--- Global RMSE Summary ---")

rmse_data = {}
for var in sorted(preds.data_vars.keys()):
    if var not in truth.data_vars:
        continue
    try:
        fc = preds[var].values.squeeze()
        tr = truth[var].values.squeeze()
        if fc.shape != tr.shape:
            continue
        # Flatten everything except lat/lon (last two dims)
        err = fc - tr
        weights = np.cos(np.deg2rad(lats))
        # Reshape weights to broadcast: weight over lat dimension
        w_shape = [1] * (err.ndim - 2) + [len(lats), 1]
        w = np.broadcast_to(weights.reshape(w_shape), err.shape)
        rmse_val = float(np.sqrt(np.nanmean(w * err**2)))
        rmse_data[var] = rmse_val
        print(f"  {var:40s}  RMSE = {rmse_val:.6f}")
    except Exception as e:
        print(f"  {var:40s}  skipped ({e})")

if rmse_data:
    fig, ax = plt.subplots(figsize=(max(10, len(rmse_data) * 0.8), 6))
    names = list(rmse_data.keys())
    values = [rmse_data[n] for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("Area-Weighted RMSE")
    ax.set_title("Global RMSE by Variable (all lead times)")
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fname = "rmse_by_variable.png"
    fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")

# ===========================================================================
# 7. RMSE by pressure level (for 3-D variables)
# ===========================================================================
print("\n--- RMSE by Pressure Level ---")

LEVEL_VARS = ["temperature", "geopotential", "u_component_of_wind",
              "v_component_of_wind", "specific_humidity"]

level_vars_present = [v for v in LEVEL_VARS if v in preds.data_vars and "level" in preds[v].dims]

if level_vars_present and "level" in preds.coords:
    levels = preds.coords["level"].values
    fig, ax = plt.subplots(figsize=(8, 6))
    for var in level_vars_present:
        rmse_by_level = []
        for lev in levels:
            fc = preds[var].sel(level=lev).values.squeeze()
            tr = truth[var].sel(level=lev).values.squeeze()
            err = fc - tr
            weights = np.cos(np.deg2rad(lats))
            w_shape = [1] * (err.ndim - 2) + [len(lats), 1]
            w = np.broadcast_to(weights.reshape(w_shape), err.shape)
            rmse_by_level.append(float(np.sqrt(np.nanmean(w * err**2))))
        ax.plot(rmse_by_level, levels, marker="o", label=var, linewidth=1.5)
    ax.set_ylabel("Pressure Level (hPa)")
    ax.set_xlabel("Area-Weighted RMSE")
    ax.set_title("RMSE by Pressure Level (all lead times)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "rmse_by_pressure_level.png"
    fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")

# ===========================================================================
# 8. Zonal-mean temperature cross-section (lat × pressure)
# ===========================================================================
print("\n--- Zonal-Mean Cross-Sections ---")

if "temperature" in preds.data_vars and "level" in preds["temperature"].dims:
    levels = preds.coords["level"].values
    for step in steps_to_plot:
        lead = _lead_label(step)
        fc_t = preds["temperature"].isel(time=step).mean(dim=["batch", "lon"]).values.squeeze()
        tr_t = truth["temperature"].isel(time=step).mean(dim=["batch", "lon"]).values.squeeze()
        err_t = fc_t - tr_t  # shape: (level, lat)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Forecast
        im0 = axes[0].pcolormesh(lats, levels, fc_t, cmap="RdYlBu_r", shading="auto")
        axes[0].set_title(f"GraphCast Forecast {lead}")
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_xlabel("Latitude")
        plt.colorbar(im0, ax=axes[0], label="K")

        # ERA5 Truth
        im1 = axes[1].pcolormesh(lats, levels, tr_t, cmap="RdYlBu_r", shading="auto")
        axes[1].set_title(f"ERA5 Truth {lead}")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Latitude")
        plt.colorbar(im1, ax=axes[1], label="K")

        # Error
        emax = max(abs(np.nanmin(err_t)), abs(np.nanmax(err_t)))
        im2 = axes[2].pcolormesh(lats, levels, err_t, cmap="RdBu_r",
                                  vmin=-emax, vmax=emax, shading="auto")
        axes[2].set_title(f"Error (FC − ERA5) {lead}")
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Latitude")
        plt.colorbar(im2, ax=axes[2], label="K")

        fig.suptitle(f"Zonal-Mean Temperature Cross-Section {lead}", fontsize=14,
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        fname = f"zonal_mean_temperature_step{step}.png"
        fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fname}")

# ===========================================================================
# 9. 10m Wind Speed composite (from u10 + v10)
# ===========================================================================
if ("10m_u_component_of_wind" in preds.data_vars and
        "10m_v_component_of_wind" in preds.data_vars):
    print("\n--- 10m Wind Speed ---")
    for step in steps_to_plot:
        lead = _lead_label(step)
        u_fc = _get_field(preds, "10m_u_component_of_wind", step)
        v_fc = _get_field(preds, "10m_v_component_of_wind", step)
        spd_fc = np.sqrt(u_fc**2 + v_fc**2)

        u_tr = _get_field(truth, "10m_u_component_of_wind", step)
        v_tr = _get_field(truth, "10m_v_component_of_wind", step)
        spd_tr = np.sqrt(u_tr**2 + v_tr**2)

        err = spd_fc - spd_tr
        vmax = max(np.nanmax(spd_fc), np.nanmax(spd_tr))
        emax = max(abs(np.nanmin(err)), abs(np.nanmax(err)))

        fig, axes = _create_axes()
        transform_kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}

        im0 = axes[0].pcolormesh(lons, lats, spd_fc, cmap="YlOrRd", vmin=0, vmax=vmax,
                                  **transform_kw)
        axes[0].set_title(f"GraphCast Wind Speed {lead}")
        _add_map_features(axes[0])
        plt.colorbar(im0, ax=axes[0], shrink=0.7, label="m/s")

        im1 = axes[1].pcolormesh(lons, lats, spd_tr, cmap="YlOrRd", vmin=0, vmax=vmax,
                                  **transform_kw)
        axes[1].set_title(f"ERA5 Wind Speed {lead}")
        _add_map_features(axes[1])
        plt.colorbar(im1, ax=axes[1], shrink=0.7, label="m/s")

        im2 = axes[2].pcolormesh(lons, lats, err, cmap="RdBu_r", vmin=-emax, vmax=emax,
                                  **transform_kw)
        axes[2].set_title(f"Error (FC − ERA5) {lead}")
        _add_map_features(axes[2])
        plt.colorbar(im2, ax=axes[2], shrink=0.7, label="m/s")

        fig.suptitle(f"10m Wind Speed {lead}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fname = f"wind_speed_10m_step{step}.png"
        fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fname}")

# ===========================================================================
# 10. RMSE Growth vs. Lead Time (Error Curves)
# ===========================================================================
if n_steps > 1:
    print("\n--- RMSE vs. Lead Time Curves ---")
    lead_hours = [(s + 1) * 6 for s in range(n_steps)]
    weights_lat = np.cos(np.deg2rad(lats))[:, np.newaxis]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 2m Temperature
    if "2m_temperature" in preds and "2m_temperature" in truth:
        rmse_t2m = []
        for s in range(n_steps):
            e = _get_field(preds, "2m_temperature", s) - _get_field(truth, "2m_temperature", s)
            rmse_t2m.append(float(np.sqrt(np.nanmean(weights_lat * e**2))))
        axes[0, 0].plot(lead_hours, rmse_t2m, "o-", color="#d62728", linewidth=2)
        axes[0, 0].set_title("2m Temperature (K)")
        axes[0, 0].set_ylabel("Area-Weighted RMSE")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(lead_hours)

    # 2. Z500 (converted to geopotential height in meters)
    if "geopotential" in preds and "geopotential" in truth and "level" in preds["geopotential"].dims:
        rmse_z500 = []
        for s in range(n_steps):
            e = (_get_field(preds, "geopotential", s, level=500) - 
                 _get_field(truth, "geopotential", s, level=500)) / 9.80665  # m2/s2 -> gpm
            rmse_z500.append(float(np.sqrt(np.nanmean(weights_lat * e**2))))
        axes[0, 1].plot(lead_hours, rmse_z500, "s-", color="#1f77b4", linewidth=2)
        axes[0, 1].set_title("500 hPa Geopotential Height (m)")
        axes[0, 1].set_ylabel("Area-Weighted RMSE")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(lead_hours)

    # 3. T850
    if "temperature" in preds and "temperature" in truth and "level" in preds["temperature"].dims:
        rmse_t850 = []
        for s in range(n_steps):
            e = _get_field(preds, "temperature", s, level=850) - _get_field(truth, "temperature", s, level=850)
            rmse_t850.append(float(np.sqrt(np.nanmean(weights_lat * e**2))))
        axes[1, 0].plot(lead_hours, rmse_t850, "^-", color="#ff7f0e", linewidth=2)
        axes[1, 0].set_title("850 hPa Temperature (K)")
        axes[1, 0].set_xlabel("Lead Time (hours)")
        axes[1, 0].set_ylabel("Area-Weighted RMSE")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(lead_hours)

    # 4. 10m Wind Speed
    if "10m_u_component_of_wind" in preds and "10m_u_component_of_wind" in truth:
        rmse_w10 = []
        for s in range(n_steps):
            u_fc = _get_field(preds, "10m_u_component_of_wind", s)
            v_fc = _get_field(preds, "10m_v_component_of_wind", s)
            u_tr = _get_field(truth, "10m_u_component_of_wind", s)
            v_tr = _get_field(truth, "10m_v_component_of_wind", s)
            e = np.sqrt(u_fc**2 + v_fc**2) - np.sqrt(u_tr**2 + v_tr**2)
            rmse_w10.append(float(np.sqrt(np.nanmean(weights_lat * e**2))))
        axes[1, 1].plot(lead_hours, rmse_w10, "d-", color="#2ca02c", linewidth=2)
        axes[1, 1].set_title("10m Wind Speed (m/s)")
        axes[1, 1].set_xlabel("Lead Time (hours)")
        axes[1, 1].set_ylabel("Area-Weighted RMSE")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(lead_hours)

    fig.suptitle("Forecast Error Growth vs. Lead Time", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fname = "rmse_vs_lead_time.png"
    fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")

n_plots = len([f for f in os.listdir(args.outdir) if f.endswith(".png")])
print(f"\n{'=' * 60}")
print(f"Done — {n_plots} diagnostic plots saved to:")
print(f"  {args.outdir}/")
print("=" * 60)
