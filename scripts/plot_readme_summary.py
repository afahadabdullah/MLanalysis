"""README summary figure (2018-01-15 12 UTC case, GraphCast_small 1 deg, ISD stations).

Numbers are copied from the run log of runs/exp_main/20180115T12_isd_bg_4dv_lbfgs2
(identical to the v3 run; RESULTS.md section 27.2):
  panel a: paired station bootstrap, % change in 2 m T RMSE vs the ERA5 start, withheld ISD stations
           (BASE is the plain RMSE ratio vs ERA5, it is not part of the paired test)
  panel b: t0 error vs +6 h error at the withheld stations (t0 DIAGNOSTICS and RMSE tables)
Usage: python scripts/plot_readme_summary.py [out.png]
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LEADS = [6, 12, 24, 48, 72]
# arm: (mean, lo, hi) per lead, % vs ERA5 start, withheld ISD (~635 stations)
VS_ERA5 = {
    "DIR-1F":    [(16.1, 10.5, 22.8), (9.9, 4.0, 16.4), (4.4, 0.2, 9.1), (5.3, 1.8, 8.9), (3.5, -0.6, 7.7)],
    "4DV":       [(8.6, 5.0, 13.2), (3.1, -0.2, 7.0), (2.5, -0.0, 5.0), (4.7, 2.0, 7.5), (2.1, -1.2, 5.5)],
    "REPLAY72":  [(0.0, -0.7, 0.8), (-1.5, -2.2, -0.8), (-0.8, -1.6, -0.0), (-0.3, -1.4, 0.6), (-0.9, -1.9, 0.0)],
    "HYB72-DIR": [(2.8, -0.7, 6.5), (-0.4, -3.4, 2.9), (-2.7, -4.8, -0.3), (-2.6, -4.2, -1.0), (-3.6, -4.8, -2.3)],
    "HYB72-4DV": [(-1.0, -2.6, 0.7), (-3.1, -4.5, -1.6), (-2.0, -3.1, -0.9), (-1.6, -2.8, -0.4), (-2.4, -3.6, -1.3)],
}
RMSE_ERA5 = [1.742, 1.765, 2.324, 2.768, 2.918]
RMSE_BASE = [1.985, 1.917, 2.429, 2.966, 3.055]
BASE_PCT = [100 * (b / e - 1) for b, e in zip(RMSE_BASE, RMSE_ERA5)]

# t0 error vs +6 h error at withheld stations (K)
T0_VS_6H = {
    "ERA5":      (1.930, 1.742),
    "BASE":      (2.241, 1.985),
    "DIR-1F":    (1.784, 2.023),
    "4DV":       (2.014, 1.892),
    "REPLAY72":  (1.960, 1.742),
    "HYB72-DIR": (1.730, 1.791),
    "HYB72-4DV": (1.832, 1.723),
}

# fixed categorical order (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6 on #fcfcfb)
COL = {"HYB72-DIR": "#2a78d6", "HYB72-4DV": "#eb6834", "REPLAY72": "#1baf7a",
       "4DV": "#eda100", "DIR-1F": "#e87ba4"}
NEUTRAL = {"ERA5": "#52514e", "BASE": "#8f8e89"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
DESC = {"HYB72-DIR": "ERA5 replay + stations, 72 h cycling",
        "HYB72-4DV": "same cycling, final step by 4D-Var",
        "REPLAY72": "ERA5 replay only, no stations",
        "4DV": "4D-Var through GraphCast, no cycling",
        "DIR-1F": "direct insertion at t0 only"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.40))

# ---- panel a: % change vs ERA5 by lead -------------------------------------------------------
ax.axhline(0, color=NEUTRAL["ERA5"], lw=1.2, zorder=1)
ax.text(36, 0.4, "start from ERA5", color=INK2, fontsize=9.5, ha="center", va="bottom")
ax.text(4.0, -6.7, "↓ better than ERA5", color=INK2, fontsize=9.5, ha="left", va="bottom")
ax.plot(LEADS, BASE_PCT, color=NEUTRAL["BASE"], lw=1.6, ls=(0, (4, 3)), zorder=2)
for arm in ["DIR-1F", "4DV", "REPLAY72", "HYB72-DIR", "HYB72-4DV"]:
    m = [v[0] for v in VS_ERA5[arm]]
    lo = [v[1] for v in VS_ERA5[arm]]
    hi = [v[2] for v in VS_ERA5[arm]]
    hero = arm.startswith("HYB")
    if hero:
        ax.fill_between(LEADS, lo, hi, color=COL[arm], alpha=0.13, lw=0, zorder=2)
    ax.plot(LEADS, m, color=COL[arm], lw=2.6 if hero else 2.0, zorder=4 if hero else 3,
            marker="o", ms=6.5, mfc=COL[arm], mec=SURF, mew=1.8)
# direct labels at the right end (identity is never color-alone)
end_lab = {"DIR-1F": 3.5, "4DV": 2.1, "BASE": 4.7, "REPLAY72": -0.9, "HYB72-4DV": -2.4, "HYB72-DIR": -3.6}
nudge = {"BASE": 6.4, "DIR-1F": 4.2, "4DV": 2.0, "REPLAY72": -0.4, "HYB72-4DV": -2.3, "HYB72-DIR": -4.5}
for arm, y in end_lab.items():
    c = COL.get(arm, NEUTRAL.get(arm))
    ax.plot([72.9, 75.0], [y, nudge[arm]], color=GRID, lw=0.8, zorder=1, clip_on=False)
    ax.text(77.3, nudge[arm], arm, color=INK, fontsize=9.5,
            va="center", fontweight="bold" if arm.startswith("HYB") else "normal", clip_on=False)
    ax.plot(75.9, nudge[arm], marker="s", ms=7, color=c, clip_on=False, zorder=5, mec=SURF)
ax.set_xlim(3, 72.8)
ax.set_ylim(-7, 18)
ax.set_xticks(LEADS)
ax.set_xlabel("forecast lead (h)")
ax.set_ylabel("2 m T RMSE change vs ERA5 start (%)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("a   Error vs the ERA5 start, 652 withheld stations", loc="left", fontsize=12, fontweight="bold", pad=10)
ax.text(0.0, -0.19, "Shading: 95 % paired station-bootstrap interval (cycling arms). BASE = 24 h GraphCast background.",
        transform=ax.transAxes, fontsize=8.8, color=INK2)

# ---- panel b: consistency beats closeness ----------------------------------------------------
lim = (1.68, 2.30)
off = {"ERA5": (-9, -2), "BASE": (-8, 8), "DIR-1F": (8, 4), "4DV": (8, -3), "REPLAY72": (9, -4),
       "HYB72-DIR": (9, 8), "HYB72-4DV": (8, -9)}
ha = {"BASE": "right", "ERA5": "right"}
for arm, (x0, y6) in T0_VS_6H.items():
    c = COL.get(arm, NEUTRAL.get(arm))
    bx.scatter([x0], [y6], s=85, color=c, edgecolor=SURF, linewidth=2, zorder=3)
    bx.annotate(arm, (x0, y6), xytext=off[arm], textcoords="offset points", fontsize=9.5, color=INK,
                ha=ha.get(arm, "left"), va="center", fontweight="bold" if arm.startswith("HYB") else "normal")
# the DIR-1F shock arrow
bx.text(1.797, 1.998, "closest to the stations at t0,\nbut worse than BASE by +6 h (shock)", fontsize=8.8, color=INK2, va="top")
bx.set_xlim(*lim)
bx.set_ylim(1.68, 2.08)
bx.set_xlabel("error at t0 (K)   ← closer to stations")
bx.set_ylabel("error at +6 h (K)")
bx.grid(color=GRID, lw=0.8)
bx.set_axisbelow(True)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.set_title("b   Consistency beats closeness", loc="left", fontsize=12, fontweight="bold", pad=10)
bx.text(0.0, -0.19, "2 m T RMSE at the withheld stations, t0 vs +6 h forecast.", transform=bx.transAxes,
        fontsize=8.8, color=INK2)

fig.suptitle("Inserting surface stations into GraphCast (1°, 2018-01-15 12 UTC)\n"
             "Cycling with ERA5 + stations beats the ERA5 start; direct insertion shocks the model",
             x=0.075, ha="left", y=1.04, fontsize=12.5, color=INK, linespacing=1.5)
out = sys.argv[1] if len(sys.argv) > 1 else "docs/figs/readme_summary.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURF)
print("wrote", out)
