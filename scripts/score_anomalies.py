#!/usr/bin/env python3
"""
Anomaly verification: which arm best predicts the MERRA-2 (or ERA5) weather anomalies?
====================================================================================
Scores in anomaly space so that neither reanalysis's climate is favoured:

  truth anomaly     a_M = MERRA-2 analysis - clim_MERRA-2(hour)      (and a_E = ERA5 - clim_ERA5(hour))
  forecast anomaly  a_F = forecast - clim_ERA5(hour)                 (GraphCast lives in ERA5's climate)

  ACC  = sum w a_F a_T / sqrt(sum w a_F^2 * sum w a_T^2)   (uncentred, cos-lat weights)
  aRMSE = RMS(a_F - a_T)

Against a_M this is the same as verifying against MERRA-2 mean-mapped into ERA5's climate
(M - clim_M + clim_E), so M-MEAN starts with ACC = 1 by construction; the question is which arm
keeps MERRA-2's anomalies best as the forecast runs, and whether a forecast started from ERA5
predicts MERRA-2's anomalies as well as the MERRA-2-started ones do.

Inputs: the fields.nc written by exp_main_real_obs.py (--save-fields 1), the two climatologies
(scripts/build_clim.py) and the ERA5 input file (land-sea mask). Runs in seconds on any node:
  python scripts/score_anomalies.py runs/exp_main/<run>/fields.nc \\
      --clim-era5 data/clim/clim_era5_m01_2011-2017.nc --clim-provider data/clim/clim_merra2_m01_2011-2017.nc
Writes anomaly_scores.csv and fig14_anomaly_skill.png next to fields.nc and prints the tables.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLIM_VARS = {"mslp": ("mean_sea_level_pressure", None, 100.0), "t2m": ("2m_temperature", None, 1.0),
             "t850": ("temperature", 850, 1.0), "z500": ("geopotential", 500, 9.80665)}
# (key, field, region, label, unit)
SPECS = [("t2m_conus", "t2m", "conus_land", "2 m T, CONUS land", "K"),
         ("mslp_conus", "mslp", "conus", "MSLP, CONUS", "hPa"),
         ("t850_nhx", "t850", "nhx", "T850, NH 20-90N", "K"),
         ("z500_nhx", "z500", "nhx", "Z500, NH 20-90N", "m"),
         ("t850_conus", "t850", "conus", "T850, CONUS", "K"),
         ("z500_down", "z500", "down", "Z500, N America-Atlantic", "m")]
ORDER = ["ERA5", "BASE", "M-DIR", "M-MEAN", "M-QM", "REPLAY72-M", "REPLAY72-MQM", "HYB72-M", "HYB72-MQM",
         "HYB72-4DV-M", "HYB72-4DV-MQM", "HYB72-DIR", "HYB72-4DV", "REPLAY72"]


def main(fields, clim_e, clim_p, template=None, out=None, leads_show=(0, 6, 12, 24, 48, 72), quiet=False):
    F = xr.open_dataset(fields)
    CE, CM = xr.open_dataset(clim_e), xr.open_dataset(clim_p)
    out = out or os.path.dirname(os.path.abspath(fields))
    lat, lon = F.lat.values, F.lon.values
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    W = np.cos(np.deg2rad(LA))
    lsm = None
    if template and os.path.exists(template):
        t = xr.open_dataset(template)
        lsm = t["land_sea_mask"].values
        lsm = lsm[0] if lsm.ndim == 3 else lsm
    land = (lsm > 0.5) if lsm is not None else np.ones_like(LA, bool)
    REG = {"conus_land": (LA >= 25) & (LA <= 50) & (LO >= 235) & (LO <= 295) & land,
           "conus": (LA >= 25) & (LA <= 50) & (LO >= 235) & (LO <= 295),
           "nhx": LA >= 20,
           "down": (LA >= 25) & (LA <= 60) & (LO >= 230) & (LO <= 340)}
    t0 = pd.Timestamp(F.attrs["t0"])
    arms = [a for a in ORDER if a in F.arm.values] + [a for a in F.arm.values if a not in ORDER]

    def clim(ds, k, hour):
        v, lev, sc = CLIM_VARS[k]
        da = ds[f"{v}_mean"].sel(hour=hour)
        if lev is not None:
            da = da.sel(level=lev)
        return da.transpose("lat", "lon").values / sc

    rows = []
    for L in F.lead_h.values:
        hour = (t0 + pd.Timedelta(hours=int(L))).hour
        for key, k, reg, _, _ in SPECS:
            if k not in F or f"{k}_provider" not in F or f"{k}_era5" not in F:
                continue
            cE, cM = clim(CE, k, hour), clim(CM, k, hour)
            aM = F[f"{k}_provider"].sel(lead_h=L).values - cM
            aE = F[f"{k}_era5"].sel(lead_h=L).values - cE
            m = REG[reg]
            w = W * m
            for a in arms:
                aF = F[k].sel(arm=a, lead_h=L).values - cE
                if not np.isfinite(aF[m]).all():
                    continue
                r = dict(arm=a, lead_h=int(L), field=key)
                for tag, aT in (("M", aM), ("E", aE)):
                    num = np.sum(w * aF * aT)
                    den = np.sqrt(np.sum(w * aF ** 2) * np.sum(w * aT ** 2))
                    r[f"acc_{tag}"] = float(num / den) if den > 0 else np.nan
                    r[f"armse_{tag}"] = float(np.sqrt(np.sum(w * (aF - aT) ** 2) / np.sum(w)))
                rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "anomaly_scores.csv"), index=False)

    def tab(field, col, fmt):
        t = S[(S.field == field) & S.lead_h.isin(leads_show)].pivot(index="arm", columns="lead_h", values=col)
        return t.loc[[a for a in arms if a in t.index]].round(fmt).to_string()

    if not quiet:
        print("\n" + "=" * 76)
        print("ANOMALY VERIFICATION (forecast - clim_ERA5 vs analysis - own climatology, same hour)")
        print("   ACC vs MERRA-2 anomalies = how well each arm predicts MERRA-2's weather; vs ERA5 anomalies = ERA5's")
        for key, _, _, lab, unit in SPECS:
            if key not in set(S.field):
                continue
            print(f"\n{lab}: ACC vs MERRA-2 anomalies")
            print(tab(key, "acc_M", 3))
            print(f"{lab}: ACC vs ERA5 anomalies")
            print(tab(key, "acc_E", 3))
            print(f"{lab}: anomaly RMSE vs MERRA-2 ({unit})")
            print(tab(key, "armse_M", 3 if unit == "K" else 2))

    # figure: ACC vs lead, vs MERRA-2 (top) and vs ERA5 (bottom)
    fl = [s for s in SPECS[:4] if s[0] in set(S.field)]
    if fl:
        cmap = plt.get_cmap("tab10")
        col = {a: ("k" if a == "ERA5" else "0.55" if a == "BASE" else cmap(i % 10)) for i, a in enumerate(arms)}
        fig, axs = plt.subplots(2, len(fl), figsize=(4.2 * len(fl), 7), squeeze=False)
        for j, (key, _, _, lab, _) in enumerate(fl):
            for i, tag in enumerate(("M", "E")):
                ax = axs[i, j]
                for a in arms:
                    s = S[(S.field == key) & (S.arm == a)].sort_values("lead_h")
                    if s.empty:
                        continue
                    ls = "--" if a in ("ERA5", "BASE") else "-"
                    ax.plot(s.lead_h, s[f"acc_{tag}"], ls, color=col[a], marker="o", ms=3,
                            lw=2.2 if a.startswith("HYB") else 1.4, label=a)
                ax.set_title(f"{lab}\nACC vs {'MERRA-2' if tag == 'M' else 'ERA5'} anomalies", fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_xlabel("lead (h)")
        axs[0, 0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = os.path.join(out, "fig14_anomaly_skill.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        if not quiet:
            print("   ✓", os.path.basename(p))
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Anomaly-space verification against MERRA-2 and ERA5")
    ap.add_argument("fields")
    ap.add_argument("--clim-era5", required=True)
    ap.add_argument("--clim-provider", required=True)
    ap.add_argument("--era5-template", default=os.path.join(os.environ.get("PROJ", "/home/afahad/project/MLanalysis"),
                    "data", "era5", "source-era5_date-2018-01-12_res-1.0_levels-13_steps-24.nc"),
                    help="ERA5 input file, for the land-sea mask")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.fields, a.clim_era5, a.clim_provider, a.era5_template, a.out)
