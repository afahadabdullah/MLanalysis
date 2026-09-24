"""README figure for the foreign-reanalysis (MERRA-2) experiments (2018-01-15 12 UTC, GraphCast_small 1 deg).

Numbers are copied from the run logs (RESULTS.md sections 28-31):
  a: paired station bootstrap, % change in 2 m T RMSE vs the ERA5 start, withheld ISD (runs isd_merra2_anom / _v2)
  b: RMSE vs the MERRA-2 analyses, M-QM start relative to the ERA5 start, both back-mapped to MERRA-2's
     climate (score_anomalies.py, section 30)
  c: Z500 (N America-Atlantic) RMSE of each start against its own reanalysis (sections 30-31)
  d: identity retention of the raw MERRA-2 start (section 28)
Usage: python scripts/plot_readme_merra2.py [out.png]
"""
import sys
import matplotlib.pyplot as plt

LEADS = [6, 12, 24, 48, 72]
L0 = [0] + LEADS
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
NEU, NEU2 = "#52514e", "#8f8e89"

# a) % vs ERA5 start, withheld ISD: (mean, lo, hi)
STN = {
    "M-DIR": ([20.9, 15.9, 2.9, 8.4, 14.9], MAGENTA),
    "M-QM": ([12.2, 10.2, 9.8, 8.1, 11.1], ORANGE),
    "REPLAY72-MQM": ([7.5, 4.8, 6.0, 7.1, 8.1], AQUA),
    "HYB72-MQM": ([7.4, 5.9, 2.7, 3.5, 5.5], BLUE),
    "HYB72-DIR (ERA5)": ([2.8, -0.4, -2.7, -2.6, -3.6], NEU2),
}
HYB_CI = ([3.3, 1.8, -0.2, 0.9, 2.0], [11.6, 10.5, 5.7, 6.8, 9.6])
# b) M-QM vs ERA5 start, RMSE vs MERRA-2 (back-mapped), %
FCM = {
    "MSLP, CONUS land": ([-52.8, -42.1, -36.8, 27.7, -2.4], ORANGE),
    "2 m T, CONUS land": ([-26.4, -25.2, -11.1, 1.8, 6.6], BLUE),
    "T850, NH 20-90N": ([-38.1, -15.6, 1.7, 14.1, 12.0], AQUA),
    "Z500, NH 20-90N": ([-25.2, 4.2, 38.5, 32.0, 18.4], YELLOW),
}
# c) own-world Z500 downstream (m)
OWN = {
    "ERA5 start vs ERA5": ([0.0, 2.715, 3.734, 4.900, 12.061, 20.502], NEU, "--"),
    "M-DIR vs MERRA-2": ([0.0, 3.672, 5.472, 8.840, 17.028, 26.645], MAGENTA, "-"),
    "M-QM vs MERRA-2": ([0.0, 3.50, 5.55, 8.44, 16.99, 29.52], ORANGE, "-"),
    "E+DM24 vs ERA5": ([3.981, 4.884, 7.069, 9.806, 18.784, 30.655], BLUE, "-"),
}
# d) retention of M-DIR
RET = {
    "MSLP > 1000 m": ([1.00, 0.93, 0.83, 0.55, 0.46, 0.33], ORANGE),
    "T850": ([1.00, 0.54, 0.13, 0.03, 0.16, 0.14], AQUA),
    "2 m T": ([1.00, 0.42, 0.14, 0.06, 0.07, 0.10], BLUE),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axs = plt.subplots(2, 2, figsize=(13.6, 9.6), gridspec_kw=dict(hspace=0.5, wspace=0.62))


def style(ax, xt, xlabel="forecast lead (h)"):
    ax.set_xticks(xt)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def endlabels(ax, items, xs, x_end, pad=2.5, fs=9.3, bold=()):
    """items: list of (label, y_end, y_label, color). Leader line to the label column."""
    for lab, y, yl, c in items:
        ax.plot([x_end + 0.6, x_end + pad * 0.8], [y, yl], color=GRID, lw=0.8, clip_on=False)
        ax.plot(x_end + pad, yl, marker="s", ms=6.5, color=c, mec=SURF, clip_on=False)
        ax.text(x_end + pad + 1.6, yl, lab, va="center", fontsize=fs, color=INK, clip_on=False,
                fontweight="bold" if lab in bold else "normal")


# ---- a) stations -------------------------------------------------------------------------
ax = axs[0, 0]
ax.axhline(0, color=NEU, lw=1.1)
ax.text(36, -0.5, "start from ERA5", color=INK2, fontsize=8.8, ha="center", va="top")
ax.fill_between(LEADS, *HYB_CI, color=BLUE, alpha=0.13, lw=0)
for name, (v, c) in STN.items():
    ref = "ERA5" in name
    ax.plot(LEADS, v, "--" if ref else "-", color=c, lw=1.6 if ref else (2.6 if name == "HYB72-MQM" else 1.9),
            marker="o", ms=6, mec=SURF, mew=1.6)
endlabels(ax, [("M-DIR", 14.9, 16.0, MAGENTA), ("M-QM", 11.1, 11.8, ORANGE), ("REPLAY72-MQM", 8.1, 7.8, AQUA),
               ("HYB72-MQM", 5.5, 3.9, BLUE), ("HYB72-DIR (ERA5)", -3.6, -3.6, NEU2)], LEADS, 72,
          pad=3.5, bold=("HYB72-MQM",))
ax.set_xlim(3, 73); ax.set_ylim(-6, 24)
ax.set_ylabel("2 m T RMSE change vs ERA5 start (%)")
style(ax, LEADS)
ax.set_title("a   Stations: cost of starting from MERRA-2", loc="left", fontsize=11.5, fontweight="bold", pad=8)
ax.text(0, -0.24, "652 withheld ISD stations; shading = 95 % bootstrap interval for HYB72-MQM.\n"
        "M-QM = MERRA-2 mapped to ERA5's climate; REPLAY/HYB = 72 h cycling toward it (HYB + stations).",
        transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

# ---- b) forecasting MERRA-2 ------------------------------------------------------------
ax = axs[0, 1]
ax.axhspan(-60, 0, color="#eef4fb", zorder=0)
ax.axhline(0, color=NEU, lw=1.1)
ax.text(4, -57, "MERRA-2 start predicts MERRA-2 better", color="#1c5cab", fontsize=8.8, va="bottom")
ax.text(4, 53, "ERA5 start (mapped) predicts MERRA-2 better", color=INK2, fontsize=8.8, va="top")
for name, (v, c) in FCM.items():
    ax.plot(LEADS, v, "-", color=c, lw=2.0, marker="o", ms=6, mec=SURF, mew=1.6)
endlabels(ax, [("Z500, NH", 18.4, 22.0, YELLOW), ("T850, NH", 12.0, 12.0, AQUA),
               ("2 m T, CONUS", 6.6, 3.0, BLUE), ("MSLP, CONUS", -2.4, -6.0, ORANGE)], LEADS, 72, pad=3.5)
ax.set_xlim(3, 73); ax.set_ylim(-60, 55)
ax.set_ylabel("RMSE vs MERRA-2: M-QM start relative\nto ERA5 start (%)")
style(ax, LEADS)
ax.set_title("b   Forecasting MERRA-2 itself (truth = MERRA-2)", loc="left", fontsize=11.5, fontweight="bold", pad=8)
ax.text(0, -0.24, "Both forecasts mapped back into MERRA-2's climate before scoring.",
        transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

# ---- c) own-world error growth -----------------------------------------------------------
ax = axs[1, 0]
for name, (v, c, ls) in OWN.items():
    ax.plot(L0, v, ls, color=c, lw=2.4 if name.startswith("E+") else 1.9, marker="o", ms=5.5, mec=SURF, mew=1.5)
endlabels(ax, [("E+DM24 vs ERA5", 30.655, 32.2, BLUE), ("M-QM vs MERRA-2", 29.52, 28.6, ORANGE),
               ("M-DIR vs MERRA-2", 26.645, 25.0, MAGENTA), ("ERA5 start vs ERA5", 20.502, 20.5, NEU)], L0, 72,
          pad=3.5, bold=("E+DM24 vs ERA5",))
ax.set_xlim(-2, 73); ax.set_ylim(0, 34)
ax.set_ylabel("Z500 RMSE, N America-Atlantic (m)")
style(ax, L0)
ax.set_title("c   Not foreign, just a less accurate start", loc="left", fontsize=11.5, fontweight="bold", pad=8)
ax.text(0, -0.24, "Each start scored against its own reanalysis. E+DM24 = ERA5 + a MERRA-2-sized difference\n"
        "from 24 h earlier: it grows as fast as the MERRA-2 starts.", transform=ax.transAxes, fontsize=8.5,
        color=INK2, va="top")

# ---- d) retention ------------------------------------------------------------------------
ax = axs[1, 1]
ax.axhline(0, color=NEU, lw=1.1)
for name, (v, c) in RET.items():
    ax.plot(L0, v, "-", color=c, lw=2.0, marker="o", ms=5.5, mec=SURF, mew=1.5)
endlabels(ax, [("MSLP > 1000 m", 0.33, 0.36, ORANGE), ("T850", 0.14, 0.20, AQUA), ("2 m T", 0.10, 0.04, BLUE)],
          L0, 72, pad=3.5)
ax.text(24, 0.93, "1 = keeps MERRA-2's state\n0 = has become the ERA5-started forecast", color=INK2, fontsize=8.8,
        va="top")
ax.set_xlim(-2, 73); ax.set_ylim(-0.1, 1.05)
ax.set_ylabel("retention of the MERRA-2 - ERA5 difference")
style(ax, L0)
ax.set_title("d   The model pulls MERRA-2 toward ERA5", loc="left", fontsize=11.5, fontweight="bold", pad=8)
ax.text(0, -0.24, "Raw MERRA-2 start (M-DIR). The terrain pressure offset is a sea-level-reduction artifact\n"
        "that the model keeps for days.", transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

fig.suptitle("GraphCast (ERA5-trained, frozen) started from MERRA-2, 2018-01-15 12 UTC",
             x=0.075, ha="left", y=0.985, fontsize=13, color=INK)
out = sys.argv[1] if len(sys.argv) > 1 else "docs/figs/readme_merra2.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote", out)
