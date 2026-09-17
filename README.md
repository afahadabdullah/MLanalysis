# MLanalysis: Observation Insertion & Initial-Condition Sensitivity in ML Weather Forecasting

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Model: GraphCast](https://img.shields.io/badge/model-GraphCast__small-green.svg)](https://github.com/google-deepmind/graphcast)
[![Target: NCCS Prism](https://img.shields.io/badge/platform-NCCS%20Prism%20GPUs-orange.svg)](SETUP_NCCS_PRISM.md)

An empirical research framework investigating whether and how off-the-shelf, pretrained machine learning weather prediction (MLWP) models can assimilate dense observational data into existing global reanalyses—and whether physical or model consistency governs forecast retention.

---

## 🔬 Scientific Overview

Pretrained, ERA5-trained ML weather models (e.g. DeepMind's GraphCast) are typically initialized directly from gridded global reanalyses. Operational weather centers and meteorological organizations often wish to inject their own high-density regional observations (such as surface mesonets) into these initial states without retraining the model.

### Central Research Question
> **Can an ERA5-trained ML forecast model make use of observational information that is not in its initial analysis, and does the insertion strategy (direct vs. balanced vs. nudged) determine whether that information helps or hurts?**

### Hypotheses & Core Questions

| # | Question | Hypothesis | Experiment |
|---|---|---|---|
| **Q1** | Is ERA5 the practical ceiling for initial conditions, and why does a foreign analysis (MERRA-2) score worse? | The MERRA-2 penalty is predominantly distribution mismatch, which statistical climatology mapping to ERA5 can largely alleviate. | **Exp 1** |
| **Q2** | Does adding mesonet 2 m temperature improve short-range forecasts over the ERA5 ceiling against independent reference stations? | Direct insertion improves $t_0$ fit, but rapid increment decay occurs within early steps. Physically/model-consistent insertion retains more value. | **Exp 2** |
| **Q3** | Does consistency matter (balanced or nudged vs. direct)? | Direct surface increments conflict with boundary-layer vertical profiles and are rejected or distorted. Balanced and nudged states preserve information longer. | **Exp 2, Exp 4** |
| **Q4** | Is forecast degradation caused by physical imbalance or by the arrival of new information? | Inserting already-assimilated stations (ASOS/METAR) introduces imbalance with minimal new information, isolating the imbalance penalty. | **Exp 3, Exp 4** |

---

## 🛠️ Insertion Strategies

All mesonet observations are processed into super-observations averaged to the 1° grid with representativeness error estimates before insertion:

1. **Direct Insertion (`-DIR`)**:
   - Univariate Optimal Interpolation (OI) applied strictly to the 2 m temperature (`2t`) channel over CONUS.
   - All other vertical and surface channels remain unchanged.
2. **Balanced Increment (`-BAL`)**:
   - The surface temperature increment $\Delta T_{\text{2m}}$ is propagated vertically into lower tropospheric levels (1000, 925, 850 hPa) using boundary-layer weights.
   - Geopotential ($z$) is hypsometrically re-integrated through the column to enforce hydrostatic balance.
3. **Model-Nudged (`-NUD`)**:
   - The ML model is integrated forward from $t_0 - 24\,\text{h}$ with continuous relaxation (analysis nudging) toward observation-adjusted states.
   - Allows the ML model's internal dynamics to construct its own dynamically consistent response across all state channels.

---

## 🧪 Experimental Roadmap

```
Stage S0 (1 Case, 6 runs)   ──▶ Stage S1 (Pilot, 10 dates)  ──▶ Stage S2 (Tuning, 24 dates) ──▶ Stage S3 (Evaluation, 80 dates)
[Sanity & Pipeline Verification]   [Effect Size & S/N Ratio]     [Freeze Hyperparameters]          [Final Block Bootstrap Tests]
```

- **Experiment 1: Baselines & Ceiling Gap**
  - Compares native ERA5 (`E`), MERRA-2 (`M`), nudged controls (`E-NUD0`, `M-NUD0`), and climatology-mapped MERRA-2 (`M-CLIM`) to separate distribution gap from true information gap.
- **Experiment 2: Mesonet Insertion (Core)**
  - Tests `-DIR`, `-BAL`, and `-NUD` increments on ERA5 and MERRA-2 baselines against independent USCRN stations up to 72 hours.
- **Experiment 3: Imbalance vs. New Information**
  - Replaces mesonet observations with already-assimilated ASOS/METAR observations to isolate numerical shock from observational gain.
- **Experiment 4: Gridded Swaps & Dynamic Rebalance**
  - **S1 (Surface):** Swapping CONUS `2t` with high-resolution NOAA URMA.
  - **S2 (Aloft):** Swapping full-column temperature with JRA-3Q, evaluating hypsometric and geostrophic rebalancing.

---

## 📊 Data Sources & Observing Systems

| Dataset | Role | Grid / Resolution | Details |
|---|---|---|---|
| **ERA5** | In-distribution baseline | 1°, 13 pressure levels | Native training distribution (Google Cloud ARCO-ERA5) |
| **MERRA-2** | Foreign reanalysis baseline | 1° regridded | NASA GMAO reanalysis (`inst3_3d_asm_Np`, `inst1_2d_asm_Nx`) |
| **MADIS Mesonets** | Inserted observation data | Point stations (CONUS) | High-density non-GTS surface networks (unassimilated) |
| **USCRN** | Ground truth verification | ~140 reference stations | NOAA US Climate Reference Network (**never inserted**) |
| **ASOS / METAR** | Imbalance control | CONUS network | Standard surface reports (already assimilated in analyses) |
| **URMA** | Exp 4 Surface Swap | 2.5 km regridded to 1° | NOAA Unrestricted Mesoscale Analysis |
| **JRA-3Q** | Exp 4 Upper-air Swap | 1.25° regridded to 1° | JMA reanalysis temperature fields |
| **IGRA2** | Upper-air verification | Radiosonde soundings | CONUS 850 / 700 hPa temperature verification at lead times |

---

## 📂 Repository Structure & Documentation

```text
MLanalysis/
├── README.md               # Project overview and scientific guide (this file)
├── PROJECT_PLAN_LEAN.md    # Consolidated lean project specification (Rev 6)
├── PROJECT_PLAN.md         # Comprehensive proposal, theory, and background
├── SETUP_NCCS_PRISM.md     # HPC setup and execution instructions for NASA NCCS Prism
├── DATA_SOURCES.md         # Reference and parked observation sources
├── .gitignore              # Git ignore rules for data, models, and environments
│
└── [Planned Modules]
    ├── config.yaml         # Experiment configuration and hyperparameter settings
    ├── prep_era5.py        # ERA5 acquisition and regridding pipeline
    ├── prep_merra2.py      # MERRA-2 adapter and regridding pipeline
    ├── clim_map.py         # Statistical climatology mapping (MERRA-2 -> ERA5)
    ├── obs_madis.py        # MADIS QC, station filtering, and super-obbing
    ├── insert.py           # Direct and balanced insertion algorithms
    ├── nudge.py            # ML analysis nudging routines
    ├── swap.py             # Gridded variable swap & balance routines
    ├── run.py              # Forecast execution orchestrator (GraphCast)
    ├── score.py            # Verification metrics (RMSE, bias, retention) vs. USCRN/IGRA2
    └── diagnostics.py      # Shock metrics, vertical response profiles, and energy spectra
```

---

## 🚀 Getting Started

### High-Performance Computing Setup
This project runs frozen `GraphCast_small` (1°, 13 pressure levels) using JAX and CUDA 12 on GPU clusters.

For step-by-step instructions on setting up the environment, compiling JAX with CUDA, allocating nodes, and downloading GraphCast weights on the NASA NCCS Prism cluster, see:
📖 **[SETUP_NCCS_PRISM.md](SETUP_NCCS_PRISM.md)**

### Running Stage S0 Verification
Before scaling experiments, Stage S0 executes a single winter 12 UTC CONUS test case across six baseline configurations (`E`, `M`, `E-NUD0`, `E-DIR`, `E-BAL`, `E-NUD`) to verify:
1. Input contract compliance with official DeepMind GraphCast checkpoints.
2. Exact numerical reproducibility ($\alpha=1$ identity matches base analysis).
3. Increment retention and plausibility of downstream response.

---

## 📑 References & Documentation
- Detailed scientific proposal: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
- Consolidated lean roadmap: [`PROJECT_PLAN_LEAN.md`](PROJECT_PLAN_LEAN.md)
- Additional data sources: [`DATA_SOURCES.md`](DATA_SOURCES.md)
- Cluster setup guide: [`SETUP_NCCS_PRISM.md`](SETUP_NCCS_PRISM.md)