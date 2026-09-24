#!/usr/bin/env python3
"""
Can a forecast started from a foreign analysis (MERRA-2) predict that foreign analysis?
=====================================================================================
Truth = the raw MERRA-2 analysis at the valid time (full fields, not anomalies).

GraphCast was trained on ERA5, so a forecast lives (or drifts) into ERA5's climate. Each forecast is
therefore scored twice against MERRA-2:

  raw        F                                     (what the model outputs)
  back-mapped F* = inverse of the arm's input mapping, i.e. put back into MERRA-2's climate
             arms with QM mapping (*QM*):  F* = clim_M + (F - clim_E) * std_M / std_E
             all other arms:               F* = F - clim_E + clim_M
             (hour-of-day climatologies of the valid time; M-MEAN therefore starts exactly at MERRA-2)

The ERA5-started forecast is back-mapped the same way, so it competes on equal terms:
"does starting from MERRA-2 predict MERRA-2 better than starting from ERA5 and mapping the result?"
ACC against MERRA-2 anomalies (analysis - clim_M vs F* - clim_M) is also reported.

Inputs: fields.nc from exp_main_real_obs.py (--save-fields 1), the two climatologies (build_clim.py),
the ERA5 input file (land-sea mask, orography). Runs in seconds on any node:
  python scripts/score_anomalies.py runs/exp_main/<run>/fields.nc \\
      --clim-era5 data/clim/clim_era5_m01_2011-2017.nc --clim-provider data/clim/clim_merra2_m01_2011-2017.nc
Writes foreign_truth_scores.csv and fig14_forecast_merra2.png next to fields.nc.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665
CLIM_VARS = {"mslp": ("mean_sea_level_pressure", None, 100.0), "t2m": ("2m_temperature", None, 1.0),
             "t850": ("temperature", 850, 1.0), "z500": ("geopotential", 500, G)}
# (key, field, region, label, unit)
SPECS = [("t2m_conus", "t2m", "conus_land", "2 m T, CONUS land", "K"),
         ("mslp_conus", "mslp", "conus_land", "MSLP, CONUS land", "hPa"),
         ("mslp_terrain", "mslp", "conus_terr", "MSLP, CONUS land > 1000 m", "hPa"),
         ("t850_conus", "t850", "conus", "T850, CONUS", "K"),
         ("t850_nhx", "t850", "nhx", "T850, NH 20-90N", "K"),
         ("z500_nhx", "z500", "nhx", "Z500, NH 20-90N", "m"),
         ("z500_down", "z500", "down", "Z500, N America-Atlantic", "m")]
ORDER = ["ERA5", "BASE", "E+DM24", "M-DIR", "M-MEAN", "M-QM", "M-QMS", "M-BAL", "REPLAY72-M", "REPLAY72-MQM", "HYB72-M", "HYB72-MQM",
         "HYB72-4DV-M", "HYB72-4DV-MQM", "HYB72-DIR", "HYB72-4DV", "REPLAY72"]


ERA5_WORLD = ("ERA5", "BASE", "HYB72-DIR", "HYB72-4DV", "REPLAY72")


def is_era5_world(a):
    return a in ERA5_WORLD or a.startswith("E+")


def main(fields, clim_e, clim_p, template=None, out=None, leads_show=(0, 6, 12, 24, 48, 72), quiet=False,
         ratio_clip=(0.5, 2.0), qm_top=850):
    F = xr.open_dataset(fields)
    CE, CM = xr.open_dataset(clim_e), xr.open_dataset(clim_p)
    out = out or os.path.dirname(os.path.abspath(fields))
    lat, lon = F.lat.values, F.lon.values
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    W = np.cos(np.deg2rad(LA))
    lsm, zs = None, None
    if template and os.path.exists(template):
        t = xr.open_dataset(template)
        lsm = t["land_sea_mask"].values
        lsm = lsm[0] if lsm.ndim == 3 else lsm
        zs = t["geopotential_at_surface"].values
        zs = (zs[0] if zs.ndim == 3 else zs) / G
    land = (lsm > 0.5) if lsm is not None else np.ones_like(LA, bool)
    high = (zs > 1000.0) if zs is not None else np.zeros_like(LA, bool)
    conus = (LA >= 25) & (LA <= 50) & (LO >= 235) & (LO <= 295)
    REG = {"conus_land": conus & land, "conus": conus, "conus_terr": conus & land & high, "nhx": LA >= 20,
           "down": (LA >= 25) & (LA <= 60) & (LO >= 230) & (LO <= 340)}
    t0 = pd.Timestamp(F.attrs["t0"])
    arms = [a for a in ORDER if a in F.arm.values] + [a for a in F.arm.values if a not in ORDER]

    def clim(ds, k, hour, stat="mean"):
        v, lev, sc = CLIM_VARS[k]
        da = ds[f"{v}_{stat}"].sel(hour=hour)
        if lev is not None:
            da = da.sel(level=lev)
        return da.transpose("lat", "lon").values / sc

    rows = []
    for L in F.lead_h.values:
        hour = (t0 + pd.Timedelta(hours=int(L))).hour
        for key, k, reg, _, _ in SPECS:
            if k not in F or f"{k}_provider" not in F:
                continue
            m = REG[reg]
            if not m.any():
                continue
            w = W * m
            cE, cM = clim(CE, k, hour), clim(CM, k, hour)
            r_inv = np.clip(clim(CM, k, hour, "std") / np.maximum(clim(CE, k, hour, "std"), 1e-12), *ratio_clip)
            M = F[f"{k}_provider"].sel(lead_h=L).values
            Ev = F[f"{k}_era5"].sel(lead_h=L).values if f"{k}_era5" in F else None
            aM = M - cM
            lev = CLIM_VARS[k][1]
            for a in arms:
                f = F[k].sel(arm=a, lead_h=L).values
                if not np.isfinite(f[m]).all():
                    continue
                if "QMS" in a or "BAL" in a:                  # QM only for surface / levels >= qm_top
                    use_qm = lev is None or lev >= qm_top
                else:
                    use_qm = "QM" in a
                fb = cM + (f - cE) * r_inv if use_qm else f - cE + cM
                r = dict(arm=a, lead_h=int(L), field=key)
                for tag, x in (("raw", f), ("back", fb)):
                    d = x - M
                    r[f"rmse_{tag}"] = float(np.sqrt(np.sum(w * d ** 2) / np.sum(w)))
                    r[f"bias_{tag}"] = float(np.sum(w * d) / np.sum(w))
                if Ev is not None:
                    r["rmse_vsE"] = float(np.sqrt(np.sum(w * (f - Ev) ** 2) / np.sum(w)))
                r["rmse_own"] = r.get("rmse_vsE", np.nan) if is_era5_world(a) else r["rmse_back"]
                r["own_truth"] = "ERA5" if is_era5_world(a) else "MERRA-2"
                aF = fb - cM
                den = np.sqrt(np.sum(w * aF ** 2) * np.sum(w * aM ** 2))
                r["acc_M"] = float(np.sum(w * aF * aM) / den) if den > 0 else np.nan
                rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "foreign_truth_scores.csv"), index=False)

    def tab(field, col, fmt):
        t = S[(S.field == field) & S.lead_h.isin(leads_show)].pivot(index="arm", columns="lead_h", values=col)
        return t.loc[[a for a in arms if a in t.index]].round(fmt).to_string()

    if not quiet:
        print("\n" + "=" * 76)
        print("FORECASTING MERRA-2: truth = raw MERRA-2 analysis at the valid time")
        print("   raw = model output; back-mapped = forecast put back into MERRA-2's climate (inverse of the")
        print("   arm's input mapping; ERA5 start: F - clim_E + clim_M). Best practical score = back-mapped.")
        for key, _, _, lab, unit in SPECS:
            if key not in set(S.field):
                continue
            fmt = 3 if unit == "K" else 2
            print(f"\n{lab}: RMSE vs MERRA-2, back-mapped ({unit})")
            print(tab(key, "rmse_back", fmt))
            print(f"{lab}: RMSE vs MERRA-2, raw ({unit})")
            print(tab(key, "rmse_raw", fmt))
            if key in ("t2m_conus", "mslp_terrain"):
                print(f"{lab}: bias vs MERRA-2, back-mapped ({unit})")
                print(tab(key, "bias_back", 2))
            print(f"{lab}: ACC vs MERRA-2 anomalies")
            print(tab(key, "acc_M", 3))

    if not quiet and "rmse_own" in S:
        print("\n" + "=" * 76)
        print("OWN-WORLD SKILL: each start scored against its own reanalysis")
        print("   ERA5-world arms (ERA5, BASE, HYB72-DIR, E+DM*) vs ERA5 (raw); MERRA-2 arms vs MERRA-2 (back-mapped).")
        print("   E+DM<lag> = ERA5 + a MERRA-2-sized difference from another time: if it grows like the MERRA-2")
        print("   starts, MERRA-2 - ERA5 differences act like ordinary analysis errors; if the MERRA-2 starts grow")
        print("   faster, the MERRA-2 states are foreign to GraphCast.")
        for key, _, _, lab, unit in SPECS:
            if key in ("t2m_conus", "mslp_terrain", "t850_nhx", "z500_nhx", "z500_down") and key in set(S.field):
                print(f"\n{lab}: RMSE vs own reanalysis ({unit})")
                print(tab(key, "rmse_own", 3 if unit == "K" else 2))

    fl = [s for s in SPECS if s[0] in ("t2m_conus", "mslp_terrain", "t850_nhx", "z500_nhx") and s[0] in set(S.field)]
    if fl:
        cmap = plt.get_cmap("tab10")
        col = {a: ("k" if a == "ERA5" else "0.55" if a == "BASE" else cmap(i % 10)) for i, a in enumerate(arms)}
        fig, axs = plt.subplots(1, len(fl), figsize=(4.4 * len(fl), 4.2), squeeze=False)
        for j, (key, _, _, lab, unit) in enumerate(fl):
            ax = axs[0, j]
            for a in arms:
                s = S[(S.field == key) & (S.arm == a)].sort_values("lead_h")
                if s.empty:
                    continue
                ax.plot(s.lead_h, s.rmse_back, "-", color=col[a], marker="o", ms=3,
                        lw=2.2 if a.startswith("HYB") or a == "ERA5" else 1.4, label=a)
                ax.plot(s.lead_h, s.rmse_raw, ":", color=col[a], lw=1.0)
            ax.set_title(f"{lab}: RMSE vs MERRA-2 ({unit})\nsolid = back-mapped, dotted = raw", fontsize=9)
            ax.set_xlabel("lead (h)")
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle("Can a MERRA-2 start predict MERRA-2? (ERA5 start back-mapped to MERRA-2's climate for comparison)",
                     fontsize=10)
        fig.tight_layout()
        p = os.path.join(out, "fig14_forecast_merra2.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        if not quiet:
            print("   ✓", os.path.basename(p))
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score forecasts against the raw MERRA-2 analyses (foreign truth)")
    ap.add_argument("fields")
    ap.add_argument("--clim-era5", required=True)
    ap.add_argument("--clim-provider", required=True)
    ap.add_argument("--era5-template", default=os.path.join(os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    "data", "era5", "source-era5_date-2018-01-12_res-1.0_levels-13_steps-24.nc"),
                    help="ERA5 input file, for the land-sea mask and orography")
    ap.add_argument("--qm-ratio-clip", default="0.5,2.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.fields, a.clim_era5, a.clim_provider, a.era5_template, a.out,
         ratio_clip=tuple(float(x) for x in a.qm_ratio_clip.split(",")))
