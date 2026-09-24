#!/usr/bin/env python3
"""
Global maps for the foreign-analysis (MERRA-2) experiments
=========================================================
Reads the fields.nc that exp_main_real_obs.py writes (--save-fields 1, default): for every arm and
lead (0 = initial state at t0, then 6 h ... ) the fields 2 m T, MSLP, T850, Z500 and 6 h precipitation,
plus the ERA5 and provider (MERRA-2) analyses at the same valid times. Makes:

  fig11_global_difference_propagation.png
      rows = fields; columns = the MERRA-2 - ERA5 analysis difference at t0, the forecast difference
      F(M-DIR) - F(ERA5 start) at +24 h and +72 h, and the analysis difference M - E at +72 h.
      If the forecast difference keeps looking like the analysis difference, the model carries MERRA-2's
      state; if it fades or changes pattern, the model has pulled it toward its ERA5 world.
  fig12_global_skill_vs_era5start.png
      rows = fields (Z500, T850, MSLP, 2 m T); columns = arms. Mean over +24 ... +72 h of
      |F_arm - ERA5| - |F_ERA5start - ERA5|  (blue = the arm is closer to the ERA5 analyses than the
      forecast started from ERA5, red = farther).
  fig13_global_climatology_difference.png          (with --clim-era5/--clim-provider)
      clim_MERRA-2 - clim_ERA5 at 00 and 12 UTC and the std ratio: the systematic part that anomaly
      initialization removes.

Usage (re-plot without re-running the experiment):
  python scripts/plot_global_maps.py runs/exp_main/<run>/fields.nc
  python scripts/plot_global_maps.py runs/exp_main/<run>/fields.nc --clim-era5 <f> --clim-provider <f> --arm M-QM
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

FIELDS = [("mslp", "MSLP", "hPa"), ("t2m", "2 m T", "K"), ("t850", "T850", "K"), ("z500", "Z500", "m"),
          ("tp6h", "6 h precip", "mm")]   # fields.nc may also hold w850 (Pa/s) and ws10 (m/s), used by fig15
CLIM_VARS = {"mslp": ("mean_sea_level_pressure", None, 100.0), "t2m": ("2m_temperature", None, 1.0),
             "t850": ("temperature", 850, 1.0), "z500": ("geopotential", 500, 9.80665),
             "tp6h": ("total_precipitation_6hr", None, 1e-3)}


def _axes(fig, nr, nc, i):
    if HAS_CARTOPY:
        ax = fig.add_subplot(nr, nc, i, projection=ccrs.Robinson(central_longitude=0))
        ax.coastlines(linewidth=0.4, color="0.25")
        ax.set_global()
        return ax, dict(transform=ccrs.PlateCarree())
    ax = fig.add_subplot(nr, nc, i)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90); ax.set_xticks([]); ax.set_yticks([])
    return ax, {}


def _lim(*arrs, q=98):
    v = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrs if a is not None])
    lim = float(np.percentile(v, q)) if v.size else 1.0
    return lim if lim > 0 else 1e-9          # an all-zero field then plots as neutral white


def _draw(ax, kw, lat, lon, f, vmax, cmap="RdBu_r", vmin=None):
    lon_c = np.concatenate([lon, [lon[0] + 360.0]])
    f_c = np.concatenate([f, f[:, :1]], axis=1)
    return ax.pcolormesh(lon_c, lat, f_c, cmap=cmap, vmin=-vmax if vmin is None else vmin, vmax=vmax,
                         shading="nearest", **kw)


def _rms(f, lat):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(f)
    m = np.isfinite(f)
    return float(np.sqrt(np.sum((w * f ** 2)[m]) / np.sum(w[m])))


def fig_difference_propagation(ds, out, arm="M-DIR", leads=(24, 72)):
    if arm not in ds.arm.values or "ERA5" not in ds.arm.values or "t2m_provider" not in ds:
        print(f"   (fig11 skipped: needs arms {arm} and ERA5 and the provider analyses)")
        return
    lat, lon = ds.lat.values, ds.lon.values
    leads = [l for l in leads if l in ds.lead_h.values]
    lmax = max(leads) if leads else int(ds.lead_h.values[-1])
    cols = [("analysis M - E, t0", lambda k: (ds[f"{k}_provider"] - ds[f"{k}_era5"]).sel(lead_h=0).values)]
    for L in leads:
        cols.append((f"forecast {arm} - ERA5 start, +{L} h",
                     lambda k, L=L: (ds[k].sel(arm=arm, lead_h=L) - ds[k].sel(arm="ERA5", lead_h=L)).values))
    cols.append((f"analysis M - E, +{lmax} h", lambda k: (ds[f"{k}_provider"] - ds[f"{k}_era5"]).sel(lead_h=lmax).values))
    rows = [f for f in FIELDS if f[0] in ds and f"{f[0]}_provider" in ds]
    fig = plt.figure(figsize=(4.2 * len(cols), 2.5 * len(rows) + 0.8))
    for r, (k, name, unit) in enumerate(rows):
        fields = [c[1](k) for c in cols]
        vmax = _lim(*fields)
        for c, ((title, _), f) in enumerate(zip(cols, fields)):
            ax, kw = _axes(fig, len(rows), len(cols), r * len(cols) + c + 1)
            im = _draw(ax, kw, lat, lon, f, vmax)
            ax.set_title(f"{name}: {title}\nrms {_rms(f, lat):.2f} {unit}", fontsize=8)
        cb = fig.colorbar(im, ax=fig.axes[-len(cols):], shrink=0.8, pad=0.01)
        cb.set_label(unit, fontsize=8)
    fig.suptitle(f"Does GraphCast keep MERRA-2's state? MERRA-2 - ERA5 difference at t0 vs the forecast difference "
                 f"({arm} minus the ERA5-started forecast)", fontsize=11)
    p = os.path.join(out, "fig11_global_difference_propagation.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("   ✓", os.path.basename(p))


def fig_skill_maps(ds, out, arms=None, lmin=24):
    if "ERA5" not in ds.arm.values or "t2m_era5" not in ds:
        print("   (fig12 skipped: needs the ERA5 arm and analyses)")
        return
    pref = ["M-DIR", "M-MEAN", "M-QM", "REPLAY72-M", "REPLAY72-MQM", "HYB72-M", "HYB72-MQM", "HYB72-4DV-MQM",
            "HYB72-DIR", "HYB72-4DV"]
    arms = arms or [a for a in pref if a in ds.arm.values][:6]
    if not arms:
        print("   (fig12 skipped: no arms to show)")
        return
    lat, lon = ds.lat.values, ds.lon.values
    L = [l for l in ds.lead_h.values if l >= lmin]
    rows = [f for f in FIELDS if f[0] in ("z500", "t850", "mslp", "t2m") and f[0] in ds]
    ref = {k: np.abs(ds[k].sel(arm="ERA5", lead_h=L) - ds[f"{k}_era5"].sel(lead_h=L)).mean("lead_h").values
           for k, _, _ in rows}
    fig = plt.figure(figsize=(3.9 * len(arms), 2.4 * len(rows) + 0.8))
    for r, (k, name, unit) in enumerate(rows):
        fields = [np.abs(ds[k].sel(arm=a, lead_h=L) - ds[f"{k}_era5"].sel(lead_h=L)).mean("lead_h").values - ref[k]
                  for a in arms]
        vmax = _lim(*fields)
        for c, (a, f) in enumerate(zip(arms, fields)):
            ax, kw = _axes(fig, len(rows), len(arms), r * len(arms) + c + 1)
            im = _draw(ax, kw, lat, lon, f, vmax)
            w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(f)
            ax.set_title(f"{a}: {name}\nmean {np.sum(w * f) / np.sum(w):+.2f} {unit}", fontsize=8)
        cb = fig.colorbar(im, ax=fig.axes[-len(arms):], shrink=0.8, pad=0.01)
        cb.set_label(unit, fontsize=8)
    fig.suptitle(f"|arm - ERA5| - |ERA5 start - ERA5|, mean over +{min(L)} ... +{max(L)} h   "
                 f"(blue = closer to the ERA5 analyses than the forecast started from ERA5)", fontsize=11)
    p = os.path.join(out, "fig12_global_skill_vs_era5start.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("   ✓", os.path.basename(p))


def fig_clim_difference(clim_e, clim_m, out):
    E, M = xr.open_dataset(clim_e), xr.open_dataset(clim_m)
    lat, lon = E.lat.values, E.lon.values
    hours = [h for h in (0, 12) if h in E.hour.values]
    cols = [(f"mean M - E, {h:02d} UTC", "diff", h) for h in hours] + [("std ratio E / M, 12 UTC" if 12 in hours
                                                                         else "std ratio E / M", "ratio", hours[-1])]
    rows = [f for f in FIELDS if f"{CLIM_VARS[f[0]][0]}_mean" in E]
    fig = plt.figure(figsize=(4.2 * len(cols), 2.5 * len(rows) + 0.8))
    for r, (k, name, unit) in enumerate(rows):
        v, lev, sc = CLIM_VARS[k]

        def g(ds, stat, h):
            da = ds[f"{v}_{stat}"].sel(hour=h)
            return (da.sel(level=lev) if lev is not None else da).transpose("lat", "lon").values

        diffs = [(g(M, "mean", h) - g(E, "mean", h)) / sc for (_, kind, h) in cols if kind == "diff"]
        vmax = _lim(*diffs)
        for c, (title, kind, h) in enumerate(cols):
            ax, kw = _axes(fig, len(rows), len(cols), r * len(cols) + c + 1)
            if kind == "diff":
                f = (g(M, "mean", h) - g(E, "mean", h)) / sc
                im = _draw(ax, kw, lat, lon, f, vmax)
                ax.set_title(f"{name}: {title}\nrms {_rms(f, lat):.2f} {unit}", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
            else:
                f = g(E, "std", h) / np.maximum(g(M, "std", h), 1e-12)
                im = _draw(ax, kw, lat, lon, np.log2(np.clip(f, 0.25, 4)), 1.0, cmap="PuOr_r")
                ax.set_title(f"{name}: {title}\n(log2 scale, ±1 = ×2)", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    fig.suptitle(f"Systematic MERRA-2 - ERA5 difference ({E.attrs.get('years', '')}, month {E.attrs.get('month', '')}): "
                 f"what anomaly initialization removes", fontsize=11)
    p = os.path.join(out, "fig13_global_climatology_difference.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("   ✓", os.path.basename(p))


def fig_shock_maps(ds, out):
    """Where each imbalanced start adjusts: rows = arms, columns = the change of the arm - ERA5-start MSLP
    difference over the first 6 h, and the arm - ERA5-start difference of omega850, 6 h precipitation and
    10 m wind at +6 h (+ omega850 at +24 h)."""
    pref = ["DIR-1F", "MX-SFC-1F", "MX-SFC", "MX-SFC-QM", "MX-UA", "M-DIR", "M-QM"]
    arms = [a for a in pref if a in ds.arm.values][:6]
    if not arms or "ERA5" not in ds.arm.values or 6 not in ds.lead_h.values:
        print("   (fig15 skipped: needs ERA5 and at least one shock/foreign arm)")
        return
    lat, lon = ds.lat.values, ds.lon.values

    def d(k, a, L):
        return (ds[k].sel(arm=a, lead_h=L) - ds[k].sel(arm="ERA5", lead_h=L)).values

    cols = []
    if "mslp" in ds:
        cols.append(("MSLP: change of (arm - ERA5) over 0-6 h", "hPa", lambda a: d("mslp", a, 6) - d("mslp", a, 0)))
    if "w850" in ds:
        cols.append(("omega850: arm - ERA5 at +6 h", "Pa/s", lambda a: d("w850", a, 6)))
    if "tp6h" in ds:
        cols.append(("6 h precip: arm - ERA5 at +6 h", "mm", lambda a: d("tp6h", a, 6)))
    if "ws10" in ds:
        cols.append(("10 m wind speed: arm - ERA5 at +6 h", "m/s", lambda a: d("ws10", a, 6)))
    if "w850" in ds and 24 in ds.lead_h.values:
        cols.append(("omega850: arm - ERA5 at +24 h", "Pa/s", lambda a: d("w850", a, 24)))
    if not cols:
        return
    fig = plt.figure(figsize=(3.9 * len(cols), 2.8 * len(arms) + 1.0))
    for c, (title, unit, fn) in enumerate(cols):
        fields = [fn(a) for a in arms]
        vmax = _lim(*fields)
        col_axes = []
        for r, (a, f) in enumerate(zip(arms, fields)):
            ax, kw = _axes(fig, len(arms), len(cols), r * len(cols) + c + 1)
            im = _draw(ax, kw, lat, lon, f, vmax)
            ax.set_title(f"{a}: {title}\nglobal rms {_rms(f, lat):.3g} {unit}", fontsize=7.5)
            col_axes.append(ax)
        cb = fig.colorbar(im, ax=col_axes, shrink=0.6, pad=0.02, orientation="horizontal", aspect=30)
        cb.set_label(unit, fontsize=8)
    fig.suptitle("Initialization shock: where each start adjusts in the first hours (differences from the "
                 "ERA5-started forecast)", fontsize=11)
    p = os.path.join(out, "fig15_shock_maps.png")
    fig.savefig(p, dpi=105, bbox_inches="tight"); plt.close(fig)
    print("   ✓", os.path.basename(p))


def make_all(fields_nc, clim_era5=None, clim_provider=None, arm="M-DIR", out=None):
    ds = xr.open_dataset(fields_nc)
    out = out or os.path.dirname(os.path.abspath(fields_nc))
    fig_difference_propagation(ds, out, arm=arm)
    fig_skill_maps(ds, out)
    fig_shock_maps(ds, out)
    if clim_era5 and clim_provider:
        fig_clim_difference(clim_era5, clim_provider, out)
    ds.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Global maps from an exp_main_real_obs.py fields.nc")
    ap.add_argument("fields")
    ap.add_argument("--clim-era5", default=None)
    ap.add_argument("--clim-provider", default=None)
    ap.add_argument("--arm", default="M-DIR", help="Arm whose forecast difference is shown in fig11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make_all(a.fields, a.clim_era5, a.clim_provider, a.arm, a.out)
