"""Model Shock Comparison: Direct Observation Insertion vs Dynamically Imbalanced States.

Synthesizes data from RESULTS.md (§21-27 station insertion, §28-32 foreign reanalyses, §33 shock anatomy):
  Panel a: Error at t0 vs Error at +6 h at withheld stations (The "Closeness vs Consistency" scatter)
           Contrasting direct single-frame insertion vs balanced DA vs mismatched states.
  Panel b: Physical Initialization Shock Index across variables (0-6 h adjustment relative to ERA5 = 1.00).

Usage:
  python3 scripts/plot_shock_comparison.py [docs/figs/model_shock_comparison.png]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#d9383a"
NEU, NEU2 = "#52514e", "#8f8e89"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlecolor": INK,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.2, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.38))


def style(axis):
    axis.grid(color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)


# ==============================================================================
# Panel a: Scatter of t0 error vs +6 h error at withheld stations
# ==============================================================================
# Categories:
# 1. Direct Obs Insertion (unbalanced obs jump): red/pink
# 2. Balanced DA / Cycling: blue/green/orange
# 3. Mismatched / Spliced States (ablation): yellow/magenta
# 4. Baselines: gray

SCATTER_DATA = {
    # arm: (t0_err, 6h_err, category, label_offset, ha, va, bold)
    "ERA5": (1.930, 1.742, "baseline", (-10, -6), "right", "center", False),
    "BASE": (2.241, 1.985, "baseline", (-10, 8), "right", "center", False),
    "DIR-1F": (1.784, 2.023, "direct_obs", (-10, 6), "right", "center", True),
    "M-QM+DIR": (1.880, 2.030, "direct_obs", (10, 4), "left", "center", False),
    "4DV": (2.014, 1.892, "balanced_da", (10, -2), "left", "center", False),
    "HYB72-DIR": (1.730, 1.791, "balanced_da", (-10, -6), "right", "center", True),
    "HYB72-4DV": (1.832, 1.723, "balanced_da", (10, -6), "left", "center", True),
    "HYB72-MQM": (1.760, 1.870, "balanced_da", (-10, 6), "right", "center", False),
    "M-DIR": (2.528, 2.105, "mismatched", (-10, 6), "right", "center", False),
    "M-QM": (2.458, 1.953, "mismatched", (-10, -6), "right", "center", False),
    "MX-SFC": (2.528, 2.001, "mismatched", (10, -2), "left", "center", False),
    "MX-UA": (1.930, 1.835, "mismatched", (10, 6), "left", "center", False),
    "MX-SFC-QM": (2.458, 1.893, "mismatched", (-10, 6), "right", "center", True),
}

CAT_COLORS = {
    "direct_obs": RED,
    "balanced_da": BLUE,
    "mismatched": YELLOW,
    "baseline": NEU,
}

CAT_MARKERS = {
    "direct_obs": "X",
    "balanced_da": "o",
    "mismatched": "s",
    "baseline": "D",
}

for arm, (x0, y6, cat, (dx, dy), ha, va, bold) in SCATTER_DATA.items():
    c = CAT_COLORS[cat]
    m = CAT_MARKERS[cat]
    ax.scatter([x0], [y6], s=90 if bold else 70, color=c, marker=m, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.annotate(
        arm,
        (x0, y6),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.0,
        ha=ha,
        va=va,
        color=INK if bold else INK2,
        fontweight="bold" if bold else "normal",
        zorder=5,
    )

# Shock vector annotation for DIR-1F
ax.annotate(
    "",
    xy=(1.784, 2.015),
    xytext=(1.784, 1.80),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, ls="--"),
    zorder=3,
)
ax.text(
    1.70,
    2.08,
    "Direct obs insertion:\nclosest at t0, degrades at +6 h\n(severe tendency shock)",
    color=RED,
    fontsize=8.5,
    fontweight="bold",
    va="bottom",
)

# Reference boundary for smooth assimilation
ax.text(
    2.15,
    1.73,
    "Balanced DA & cycling:\nmaintains or improves skill",
    color=BLUE,
    fontsize=8.5,
    ha="center",
)

# Diagonal 1:1 reference line
ref_x = np.linspace(1.68, 2.25, 50)
ax.plot(ref_x, ref_x, ls=":", color=NEU2, lw=1.0, zorder=1)
ax.text(2.18, 2.20, "1:1 line", color=NEU2, fontsize=8.0, rotation=42)

ax.set_xlim(1.65, 2.68)
ax.set_ylim(1.68, 2.22)
ax.set_xlabel("RMSE at t0 (K)   ← closer to stations")
ax.set_ylabel("RMSE at +6 h forecast (K)")
style(ax)
ax.set_title("a   Forecast jump: Direct insertion vs Mismatched states", loc="left", fontsize=11.5, fontweight="bold", pad=8)

# Custom legend for categories
leg_handles = [
    plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=RED, ms=8, label="Direct Obs Insertion (single-frame shock)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=YELLOW, ms=8, label="Mismatched / Spliced Analysis (balance test)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=8, label="Balanced DA / Replay Cycling (smooth)"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=NEU, ms=8, label="Baselines (ERA5, BASE)"),
]
ax.legend(handles=leg_handles, loc="upper right", fontsize=8.0, frameon=True, facecolor=SURF, edgecolor=GRID)


# ==============================================================================
# Panel b: Physical Initialization Shock Index (0-6 h RMS change / ERA5 reference)
# ==============================================================================
VARS = ["ws10\n(10 m wind)", "MSLP\n(mass field)", "w850\n(vert. motion)", "tp6h\n(precip spin-up)"]
ARMS_SHOCK = {
    "DIR-1F (obs insertion)": ([0.90, 0.98, 0.78, 1.05], RED, "//"),
    "M-DIR (raw foreign)": ([1.11, 1.05, 0.96, 1.07], MAGENTA, ""),
    "MX-SFC (raw sfc + ERA5 UA)": ([1.07, 0.99, 0.99, 0.97], ORANGE, ""),
    "MX-UA (ERA5 sfc + foreign UA)": ([1.02, 1.05, 0.96, 0.99], AQUA, ""),
    "MX-SFC-QM (mapped sfc + ERA5 UA)": ([1.00, 0.99, 0.99, 0.97], BLUE, ""),
}

x = np.arange(len(VARS))
n_arms = len(ARMS_SHOCK)
width = 0.15

bx.axhline(1.00, color=NEU, lw=1.2, ls="--", zorder=2)
bx.text(len(VARS) - 0.5, 1.005, "1.00 = ERA5 reference", color=NEU, fontsize=8.2, ha="right", va="bottom")

for i, (name, (vals, col, hatch)) in enumerate(ARMS_SHOCK.items()):
    pos = x - (n_arms - 1) * width / 2 + i * width
    bars = bx.bar(pos, vals, width, label=name, color=col, edgecolor=SURF, hatch=hatch, zorder=3)

bx.set_xticks(x)
bx.set_xticklabels(VARS, fontsize=9.2)
bx.set_ylabel("Shock Index (RMS change / ERA5 reference)")
bx.set_ylim(0.70, 1.16)
style(bx)
bx.set_title("b   Physical Shock Index across fields (0–6 h)", loc="left", fontsize=11.5, fontweight="bold", pad=8)
bx.legend(loc="lower left", fontsize=7.8, frameon=True, facecolor=SURF, edgecolor=GRID)

# Highlight annotations in panel b
bx.annotate("Wind shock\n(+11% in M-DIR)", xy=(0, 1.11), xytext=(0.15, 1.13),
            arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.0), fontsize=7.8, color=MAGENTA)
bx.annotate("Cured by QM\n(index = 1.00)", xy=(0, 1.00), xytext=(-0.25, 0.90),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0), fontsize=7.8, color=BLUE)

fig.suptitle(
    "Initialization Shock Diagnostics: Direct Observation Insertion vs Dynamically Imbalanced States\n"
    "Direct obs insertion triggers localized tendency shock; mismatched analyses trigger field-specific shock (cured by QM)",
    x=0.07, ha="left", y=1.03, fontsize=12.0, color=INK, fontweight="bold"
)

out = sys.argv[1] if len(sys.argv) > 1 else "docs/figs/model_shock_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURF)
print("wrote", out)
