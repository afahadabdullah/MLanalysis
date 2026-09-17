# Can dense surface observations improve an off-the-shelf ML weather forecast, and how should they be inserted?

**Plan version:** 17 Sep 2026, revision 6 (consolidated). Supersedes earlier revisions of this file. `PROJECT_PLAN.md` is kept as the long background proposal.
**Scope:** a one-person side project, 5–8 h/week, about 5–6 months.
**Model:** GraphCast_small, frozen (NCCS Prism GPUs; see `SETUP_NCCS_PRISM.md`). **Test case:** 2 m temperature from US mesonets over CONUS.

---

## 0. One-paragraph summary

Organizations run pretrained, ERA5-trained ML weather models from a gridded analysis and want to add their own dense observations, typically surface stations. This project inserts US mesonet 2 m temperature (which ERA5 and MERRA-2 do not assimilate) into ERA5 and MERRA-2 initial states in three ways: directly (univariate), with a vertically consistent balanced increment, and by nudging the ML model. It then measures whether the extra information survives into 6–72 h forecasts, verified against independent reference stations. Baseline experiments first establish how far a non-training analysis (MERRA-2) is from the ERA5 "ceiling", and whether that gap comes from distribution mismatch or from information content.

---

## 1. Questions and hypotheses

**Main question**
> Can an ERA5-trained ML forecast model make use of observational information that is not in its initial analysis, and does the way the observations are inserted (direct vs. balanced vs. nudged) decide whether that information helps or hurts?

| # | Sub-question | Hypothesis | Experiment |
|---|---|---|---|
| **Q1** | Is ERA5 the practical ceiling for initial conditions, and why does a foreign analysis (MERRA-2) score worse? | The MERRA-2 penalty is mainly distribution mismatch, which statistical mapping to ERA5 climatology can largely remove | Exp 1 |
| **Q2** | Does adding mesonet 2 m T improve short-range forecasts, and can it beat ERA5 initial conditions against independent observations? | Direct insertion improves t0 fit but the gain decays within the first steps. Consistent insertion keeps more of it | Exp 2 |
| **Q3** | Does consistency matter: balanced or nudged vs. direct? | The direct surface increment conflicts with the boundary-layer profile and is rejected or distorted. Balanced and nudged insertion retain more information | Exp 2, **Exp 4** |
| **Q4** | Is any penalty caused by imbalance itself or by new information? | Inserting already-assimilated stations (ASOS/METAR) adds imbalance with little information. Mesonets add both | Exp 3, **Exp 4** |

**Why 2 m temperature, and not precipitation.** GraphCast_small takes 6 h precipitation as an *input*, but precipitation is not a physical prognostic state variable. The operational GraphCast checkpoint omits it as an input altogether. Changing it is unlikely to test balance or information use, so precipitation is left unchanged from each base analysis.

---

## 2. Terms used consistently

| Term | Meaning in this project |
|---|---|
| **Analysis / reanalysis** | ERA5 and MERRA-2 only. They are produced by full multivariate data assimilation. **This project does not produce a reanalysis.** |
| **Variable-swapped state** | MERRA-2 (the balanced reference) with **one variable replaced wholesale** by another gridded source. Every other variable stays MERRA-2, so the state is internally inconsistent by construction. It is closer to observations only if shown to be (§5.5) |
| **Observation-adjusted state** | A base analysis with mesonet observations inserted. It is only "closer to the inserted observations". Whether it is closer to the truth is tested with withheld and reference stations. |
| **Direct insertion** | Univariate update of the `2t` channel only |
| **Balanced insertion** | The `2t` increment extended vertically into the lowest temperature levels, with geopotential updated hypsometrically. This is a physically consistent increment built without the model. |
| **Nudged (model-consistent) state** | The ML model relaxed toward the observation-adjusted state over a window (analysis nudging). The model builds its own consistent response. It is not a proof of dynamical balance. |
| **Distribution gap / information gap** | The part of an initial-condition penalty caused by "looking foreign" to the model, vs. the part caused by being farther from the true atmosphere |

---

## 3. Model

**GraphCast_small** (Google DeepMind, JAX)
- 1°, 13 pressure levels, 6 h step, trained on ERA5 1979–2015, so 2018 onward is out of sample.
- Input: two states (t0−6 h, t0).
  - **Upper air, 13 levels:** `t`, `q`, `u`, `v`, `z`, `w`.
  - **Surface:** `2t`, `10u`, `10v`, `msl`, `tp` (6 h).
  - **Static and forcing fields:** orography, land-sea mask, TOA radiation, time.
- Weights stay frozen. There is no retraining.
- Run 5-day forecasts. The signal of interest is at 6–72 h, with the downstream response at days 3–5.
- **Interface:** check whether NVIDIA Earth2Studio's GraphCast wrapper can run single steps on a modified state. Otherwise use the official `graphcast` repository demo code.
- **Hardware:** one GPU (cloud, Colab, or a lab/NCCS GPU). Benchmark in week 1.

**Caveat:** at 1° (about 100 km), mesonet information is represented only as grid-cell means. A 0.25° GraphCast repeat is an optional follow-on.

---

## 4. Data sources

### 4.1 Base initial states
| Data | Use | Access | Notes |
|---|---|---|---|
| **ERA5** | Base state (in-distribution); benchmark; climatology for mapping | ARCO-ERA5 (Google Cloud Zarr), coarsened to 1° | Same regrid path as MERRA-2 |
| **MERRA-2** | Base state (foreign analysis) | NASA GES DISC: `inst3_3d_asm_Np` (H, T, U, V, QV, OMEGA), `inst1_2d_asm_Nx` (T2M, U10M, V10M, SLP), `tavg1_2d_flx_Nx` (PRECTOT) | Conservative regrid to 1°. Fill below-ground points as ERA5 does. Use ERA5 static fields for all runs |

### 4.1b Gridded replacement sources (Exp 4)
| Data | Variable swapped into MERRA-2 | Access | Notes |
|---|---|---|---|
| **URMA** (NOAA Unrestricted Mesoscale Analysis, 2.5 km, hourly, CONUS) | `2t` over CONUS | NOAA open data on AWS / NOMADS archive (*confirm period and bucket*) | Observation-dense 2D analysis (it ingests mesonets) with an HRRR/NAM background, so no ERA5. Conservatively coarsened to 1°, with a 3° cosine taper at the CONUS edge. A gridded shortcut for Exp 2 if the MADIS OI route fails |
| **JRA-3Q** (JMA reanalysis, 1.25°) | `t` at all 13 levels, global | NCAR RDA (ds640.0) | A different model and DA system from MERRA-2. The swap breaks hypsometric and geostrophic consistency with MERRA-2's z, u, v. Its "better observed" status is checked against radiosondes on 2018 (it may not be better; either result is reported) |

### 4.2 Observations
| Data | Role | Assimilated in ERA5 / MERRA-2? | Access |
|---|---|---|---|
| **MADIS mesonet** (non-METAR providers) 2 m T, hourly | **Inserted** (new information) | Not assimilated in either (*confirm*; mesonets generally aren't on the GTS) | NOAA MADIS archive (account needed; some providers restricted). **Fallback:** Synoptic Data API (academic access) or state mesonets (Oklahoma, New York) |
| **ASOS/METAR** 2 m T (MADIS METAR or NOAA ISD) | **Inserted** in Exp 3 (already-assimilated control) | ERA5: yes, via the screen-level analysis (*confirm*); MERRA-2: *confirm* | NOAA ISD / MADIS |
| **USCRN** (~140 reference stations) | **Verification only, never inserted** | Not assimilated (*confirm*) | NOAA NCEI, hourly files |
| Withheld 20 % of mesonet stations (random per date) | t0 check: "closer to observations" | — | Same as the mesonet data |
| **IGRA2** radiosondes (CONUS) | Verification of 850/700 hPa T at +12–72 h | Assimilated (used only at later leads, so independent of t0) | NOAA NCEI |

**Why this makes a clean test:**
- Mesonets add information that neither analysis has.
- USCRN is an independent, high-quality reference.
- ASOS is the same variable and region but already known to the analyses.

---

## 5. Methods

### 5.1 Observation processing
1. **QC:** gross limits, MADIS QC flags, a buddy check against neighbors, and removal of stations with <80 % availability or a persistent bias vs. neighbors (computed on 2018).
2. **Height correction:** move each station's temperature to model orography with a 6.5 K/km lapse rate; reject stations with >300 m height difference.
3. **Super-observations:** average stations within each 1° cell, and record the count and standard deviation. This is the representativeness error estimate.
4. **Time matching:** use observations within ±30 min of the analysis time (00/06/12/18 UTC).
5. **Station split per start date:** 80 % inserted, 20 % withheld. USCRN is always excluded from insertion.

### 5.2 Insertion methods (the same super-observations for every method)
| ID suffix | Method | Details |
|---|---|---|
| **-DIR** | Direct | Univariate OI on the `2t` channel over CONUS land cells: Gaussian horizontal correlation (L ≈ 150 km at 1°, tuned on 2018), obs/background error ratio from 2018 departure statistics. No other channel changes |
| **-BAL** | Balanced increment | The -DIR `2t` increment ΔT₂ₘ is spread into `t` at 1000/925/850 hPa with weights (1.0, 0.6, 0.2) × ΔT₂ₘ, only at levels above ground. `z` is updated at all levels by hypsometric integration. Test weights on 2018 separately for 00 and 12 UTC (stable morning vs. mixed afternoon boundary layers) |
| **-NUD** | Nudged | Start from the base analysis at t0−24 h. At each 6 h cycle, form that cycle's -DIR adjusted state (mesonet obs at that time) and update `x = F(x) + α_k·(X_k − F(x))` on all channels, with the tapered gain (α_k → 0 at the last update). Other regions and channels therefore relax to the base analysis. The launch pair is the model's own states |
| **-SMO** | Smoothing control | The base analysis with a 2t/t spectral filter matched to the -NUD t0 spectrum over CONUS |

- **Nudging parameters:** α = 1 − exp(−6h/τ). Tune on 2018 with τ ∈ {6, 12, 24 h} and W ∈ {12, 24 h}. Freeze before evaluating.
- **Fairness:** -NUD uses observations from several cycles, while -DIR/-BAL use only t0−6 h and t0. Control: a W = 12 h -NUD run uses only those same two times.
- **Sanity check:** α = 1 with no observations reproduces the base-analysis forecast exactly.

### 5.3 Climatology mapping (Exp 1)
MERRA-2 is mapped to ERA5 statistics by removing the monthly-mean difference and rescaling the standard deviation, per grid point, level, and channel. Statistics come from 2010–2017. No ERA5 data from forecast days is used.

### 5.4 Diagnostics
| Diagnostic | What it shows |
|---|---|
| t0 error vs. withheld mesonet and USCRN | Is the adjusted state closer to independent observations? |
| **Increment retention:** CONUS `2t` increment remaining at +6/+12/+24/+48 h, as `⟨F_X − F_base, ΔX⟩ / ‖ΔX‖²` | Does the model keep or reject the information? |
| Vertical response at +6 h (Δt at 1000–700 hPa) | Does the model spread a surface increment realistically? |
| First-step jump `‖F(x0) − x0‖` over CONUS vs. the base analysis | Initialization-shock proxy |
| Hypsometric residual (1000–850 hPa) and 2t−t1000 lapse anomaly | Physical inconsistency at t0 |
| Downstream Z500 and T850 difference at days 3–5 (N. Atlantic) | Does a regional increment propagate? |

### 5.5 Verification and statistics
- **Primary metric:** CONUS 2 m T RMSE and bias vs. **USCRN** at +6, +12, +24, +48, +72 h, separately for 00 and 12 UTC starts.
- **Secondary metrics:**
  - 2 m T vs. withheld mesonets.
  - 850/700 hPa T vs. IGRA2 at +12–72 h.
  - Global Z500/T850 RMSE vs. ERA5 (a check that the global forecast is not degraded).
- **Statistics:** paired differences by start date, block bootstrap by week, 95 % intervals. The primary endpoints are fixed in advance: +24 h and +48 h USCRN RMSE.

---

## 6. Experiment design

**Staged sampling. Start with a single initial condition.** Sample size grows only when each stage works.

| Stage | Starts | Purpose | What can be concluded |
|---|---|---|---|
| **S0: one case** | **1** (a 2018 date) | Pipeline, code correctness, visible response | Nothing statistical. Only: does the machinery work, and is the response large enough to be worth chasing? |
| S1: pilot | 10 (2018) | Effect size and its spread across dates | A rough signal-to-noise estimate, and the sample size needed |
| S2: tuning | 24 (2018) | Freeze OI/BAL/nudging parameters | Parameter choice only |
| S3: evaluation | 80 (2019–2020) | Paired statistics, block bootstrap | The publishable result |

**Stage S0 protocol (do this first)**
1. **Pick one date from 2018**, not from the evaluation years. A winter 12 UTC CONUS case is suggested: stable nocturnal boundary layer, strong surface temperature gradients, and the largest mesonet-vs-analysis differences. Confirm by checking the ERA5-minus-mesonet spread on a few candidate days.
2. **Run six forecasts:** E, M, E-NUD0, E-DIR, E-BAL, E-NUD.
3. **Verify the machinery:**
   - The input contract matches the official sample (variable order, units, levels, latitude direction).
   - α = 1 with no observations reproduces E exactly (to numerical precision).
   - Re-running the same configuration gives bit-identical output, so any difference is a real difference and not run-to-run noise.
   - The inserted increment appears where the observations are, with a sensible amplitude.
4. **Look at, per run:** the t0 increment map and its vertical profile; the inconsistency metrics; the increment retention at +6/+12/+24 h; forecast 2 m T against USCRN at +6 to +72 h (as an anecdote); and 500 hPa difference maps at days 3–5 to confirm the change spreads plausibly and does not blow up.
5. **Decide from S0 only:** is the day-1 2 m T difference between methods a meaningful fraction of the typical error? As rough guidance, 1° super-obbed increments are expected to be around 0.5–2 K, while day-1 2 m T RMSE against stations is around 1.5–2.5 K. A method difference of a few hundredths of a kelvin means the effect is below the noise, and no realistic number of dates will rescue it.

**Sample size, decided after S1, not now.** From the paired day-1/day-2 differences across the 10 pilot dates, compute the standard deviation σ_d and the smallest difference Δ worth detecting, then N ≈ (1.96 σ_d / Δ)². If the required N is far above 80, either narrow the target (a region, a season, a start hour) or report the result as a null.

**Date pools (fixed before any evaluation run)**
- **Dev / tuning:** 2018, 24 starts (monthly, at 00 and 12 UTC). S0 and S1 dates come from this pool.
- **Evaluation:** 2019–2020, **80 starts**. Balance 00 and 12 UTC and all four seasons, ~10 days apart.

### Exp 1: baselines and the ERA5 ceiling (answers Q1)
| Run | Initial state | Question |
|---|---|---|
| **E** | ERA5 | In-distribution benchmark |
| **M** | MERRA-2 | Foreign-analysis baseline |
| **E-NUD0** | Model nudged toward ERA5, with no observations | Does model-consistent preparation beat even ERA5? Is the ceiling ERA5, or the model's own version of ERA5? |
| **M-NUD0** | Model nudged toward MERRA-2, with no observations | Control for M-based runs in Exp 2 |
| **M→E** | Start MERRA-2 at t0−12 h, nudge toward ERA5 | How quickly does the model forget the source? |
| **M-CLIM** | MERRA-2 mapped to ERA5 climatology | Distribution gap vs. information gap |

Gap split: distribution part ≈ Err(M) − Err(M-CLIM); information part ≈ Err(M-CLIM) − Err(E). This is scored against USCRN/IGRA2 as well as ERA5.

### Exp 2: mesonet insertion (answers Q2 and Q3). The core experiment
| | Direct | Balanced | Nudged |
|---|---|---|---|
| **ERA5 base** | E-DIR | E-BAL | E-NUD |
| **MERRA-2 base** | M-DIR | M-BAL | M-NUD |
Plus **E-SMO** (smoothing control).

**Key comparisons:**
- E-DIR/BAL/NUD vs. E: can the observations beat ERA5 initial conditions against USCRN?
- DIR vs. BAL vs. NUD: does the insertion method matter?
- E-NUD vs. E-NUD0: separates observation information from the nudging effect itself.
- M-based vs. E-based gains: is the value of the observations larger for a foreign analysis?

### Exp 3: information vs. imbalance (answers Q4)
| Run | Inserted data |
|---|---|
| E-DIR-ASOS, E-NUD-ASOS | ASOS/METAR 2 m T (already in ERA5) |
Compare these with E-DIR and E-NUD from Exp 2, using the same variable, region, and methods.
- **ASOS direct worse than E:** imbalance or over-fitting penalty alone.
- **Mesonet direct better than ASOS direct:** new information is present.
- **Nudged ≈ E for ASOS but better than E for mesonets:** consistent insertion extracts only the new information.

### Exp 4: MERRA-2 as the balanced state vs. a single-variable swap (answers Q3 and Q4)

**Idea:** MERRA-2 is the consistent (balanced) reference. Replacing **one** variable with a gridded product from another source gives a state that is plausibly closer to observations for that variable but inconsistent with everything else. No OI is needed; the swap *is* the direct insertion.

| Run | Initial state | Role |
|---|---|---|
| **M** | MERRA-2 | Balanced reference (shared with Exp 1) |
| **M-NUD0** | Nudged toward MERRA-2 | Nudging control (shared with Exp 1) |
| **S-DIR** | MERRA-2 with variable V from source S | Direct swap: closer to observations(?) but imbalanced |
| **S-BAL** | S-DIR, then rebalanced: `z` recomputed hypsometrically from the swapped `t` (anchored at MERRA-2 1000 hPa z / surface); for S2 also geostrophic `u,v` increments poleward of 20° (tapered) | Physically consistent version |
| **S-NUD** | Model nudged toward S-DIR (tapered gain, W = 24 h) | Model-consistent version |

**Two swaps:**

| Swap | V ← S | Region | What it isolates |
|---|---|---|---|
| **S1** | `2t` ← URMA | CONUS | Surface/boundary-layer inconsistency. Directly comparable with the mesonet OI of Exp 2 (gridded swap vs. station insertion) |
| **S2** | `t` (13 levels) ← JRA-3Q | Global | Free-troposphere **dynamical** imbalance (T vs. z, and T vs. thermal wind), the textbook case |

**Before forecasting, check at t0 (dev 2018):**
- Is S closer to observations than MERRA-2 for V? S1 vs. withheld USCRN; S2 vs. IGRA2. Both sources assimilate some of the same stations, so this is a relative check.
- Inconsistency metrics for M, S-DIR, S-BAL, S-NUD: hypsometric residual, geostrophic residual (S2), 2t−t1000 lapse anomaly (S1), first-step jump.

**Reading the results:**

| Result | Meaning |
|---|---|
| S-DIR closer to observations at t0, but its forecast is no better than (or worse than) M | Imbalance cancels the observational gain (the classic paradox, now for ML) |
| S-BAL ≈ S-NUD, both better than S-DIR and M | Physical consistency is what matters; a cheap rebalance step suffices |
| S-NUD better than S-BAL | The model needs its own consistent state, beyond textbook balance |
| S-DIR ≈ S-BAL ≈ S-NUD | The ML model tolerates this kind of imbalance (a contrast with dynamical models) |
| S1 penalty ≪ S2 penalty | Imbalance matters for the dynamics aloft, not near the surface |
| S-DIR not closer to observations | The swap is an imbalance-only (over-fitting) test, which answers Q4 directly |

**Verification:**
- S1 is scored as in Exp 2 (USCRN 2 m T, +6–72 h).
- S2 is scored against IGRA2 T850/T500 and Z500 at +24–120 h, and global Z500/T850 vs. ERA5 (the benchmark only).

---

## 6.5 Experiments sorted by effort (build in this order)

| Order | Experiment | New code needed | New data | Effort | Why here |
|---|---|---|---|---|---|
| **1** | **Exp 1 core: E vs. M** | MERRA-2 adapter only | MERRA-2, ERA5 | **Low** | Needs no observations. Proves the whole pipeline and gives the ceiling gap |
| **2** | **Exp 1: M-CLIM** (climatology mapping) | ~50 lines | 2010–2017 monthly statistics | Low | Answers Q1 (distribution vs. information) cheaply |
| **3** | **Exp 1: nudging controls** (E-NUD0, M-NUD0, M→E) | `nudge.py` + α=1 check | none | Low–moderate | Reusable by every later experiment |
| **4** | **Exp 4 / S2: JRA-3Q temperature swap** | `swap.py` + hypsometric/geostrophic rebalance | JRA-3Q (NCAR RDA) | Moderate | No station handling, no OI. Cleanest dynamical-balance test |
| **5** | **Exp 4 / S1: URMA 2 m T swap** | regrid + taper | URMA (AWS/NOMADS) | Moderate | Gridded, so no QC or OI. A fallback for Exp 2 if MADIS access fails |
| **6** | **Exp 2: mesonet OI insertion** | QC, super-obs, OI, BAL weights | MADIS, USCRN | **High** | The most valuable experiment, and the most station work |
| **7** | **Exp 3: ASOS control** | reuses Exp 2 code | ISD / MADIS METAR | Low, once 6 exists | Isolates imbalance from information |
| **8** | Parked follow-ons (moisture, 0.25°, NeuralGCM, fine-tuning) | — | — | High | Only if the results justify them |

Each of steps 1–5 is publishable-adjacent on its own, and none of them depends on station data arriving.

---

## 7. Size and compute

| Block | Runs × starts | Forecasts (5 days) |
|---|---|---|
| **S0: single case** | 6 × 1 | **6** |
| S1: pilot | 6 × 10 | 60 |
| Dev tuning (OI L, BAL weights, τ/W) | ~10 × 24 | ~240 |
| Exp 1 | 6 × 80 | 480 |
| Exp 2 | 7 × 80 | 560 |
| Exp 3 | 2 × 80 | 160 |
| Exp 4 | 2 swaps × 3 × 80 (+ ~60 dev) | ~540 |
| **Total** | | **~1,980** five-day forecasts, plus ≤4 nudging steps each |

Storage:
- Save the prepared initial conditions over CONUS and globally.
- Save CONUS `2t` and `t` (1000–700 hPa) at 6 h intervals to 72 h, and global Z500/T850 daily.
- Compute scores during the run. Keep code in `MLanalysis/` and data on scratch, not OneDrive.

---

## 8. Timeline and go/no-go gates

| Weeks | Work | Gate |
|---|---|---|
| 1–2 | Environment set up; reproduce the official GraphCast_small sample; benchmark | Sample reproduced |
| 3–4 | ERA5 and MERRA-2 prep; channel-by-channel checks; E and M on 10 dev starts | Both give sane forecasts |
| 5–6 | MADIS/USCRN/ISD download, QC, super-obs; confirm assimilation status | **Gate A:** enough mesonet super-obs (≳300 CONUS cells) and USCRN coverage |
| 7 | -DIR / -BAL / -NUD insertion code; t0 check on withheld stations | **Gate B:** E-DIR closer to withheld mesonet and USCRN than E? If not, fix QC/OI; if it still fails, the observations add no usable information at 1° |
| 8 | **Stage S0: one case, six runs** (E, M, E-NUD0, E-DIR, E-BAL, E-NUD), full diagnostics | **Gate C1:** machinery correct (α = 1 identity, reproducible runs) and the method differences are a non-trivial fraction of the typical error |
| 9 | Stage S1: the same six runs on 10 dev dates; effect size, spread, required N | **Gate C2:** is there a consistent 6–24 h response? If not, write a short "information discarded" note and stop |
| 10–12 | Tune on 2018 (24 starts) and freeze parameters | Parameters written down before any evaluation run |
| 13–16 | Evaluation runs (Exp 1–3) and scoring | — |
| 17–19 | Exp 4: URMA and JRA-3Q prep, swap/rebalance code, t0 checks, runs | S-DIR t0 check reported (closer to observations or not) |
| 20–24 | Figures and paper (GRL / AIES / Weather and Forecasting) | — |

---

## 9. Expected figures

1. Map of the mesonet super-ob increment for a case, and its vertical and temporal response under DIR/BAL/NUD.
2. **Increment retention vs. lead time** for DIR/BAL/NUD (E and M bases).
3. **USCRN 2 m T RMSE difference vs. lead time** relative to E, with bootstrap bands for all Exp 2 runs.
4. Exp 1 gap bar: M → M-CLIM → E, split into distribution and information parts.
5. Exp 3: ASOS vs. mesonet insertion (imbalance vs. information).
6. t0 inconsistency metrics vs. early forecast error across runs.
7. Exp 4: t0 obs error vs. forecast error for M, S-DIR, S-BAL, S-NUD (S1 surface vs. S2 upper air side by side).

---

## 10. Possible outcomes and what each means for users

| Outcome | Message |
|---|---|
| NUD or BAL beats E at 6–48 h vs. USCRN, while DIR does not | Frozen ML models *can* use extra observations, but only with consistent insertion. Operational guidance: do not overwrite channels directly |
| DIR ≈ BAL ≈ NUD ≈ E | Surface information is discarded within one to two steps. Using local observations needs fine-tuning or an observation-aware model |
| DIR worse than E, and ASOS-DIR also worse | Direct insertion of any surface data hurts through imbalance. A clear warning for users |
| E-NUD0 beats E | Even the training analysis is not the model's native state. Model-consistent preparation is a general improvement |
| M-CLIM ≈ E | The foreign-analysis penalty is mainly distributional. A cheap mapping approaches the ERA5 ceiling without fine-tuning |
| M-CLIM ≈ M | The penalty is information content. Fine-tuning on the new analysis would not remove it |

---

## 11. Minimal repository

```text
MLanalysis/
  PROJECT_PLAN_LEAN.md     PROJECT_PLAN.md (background)
  config.yaml              # dates, regions, OI/BAL/nudging parameters
  prep_era5.py  prep_merra2.py  clim_map.py
  obs_madis.py  obs_uscrn.py  obs_isd.py  superob.py
  insert.py                # dir, bal, smo
  swap.py  prep_urma.py  prep_jra3q.py   # Exp 4 swaps + rebalance
  nudge.py                 # nud, nud0, m→e
  run.py  score.py  diagnostics.py
  analysis.ipynb  results/  runs.csv
```

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MADIS historical mesonet access is restricted | Synoptic Data academic access, or research-grade state mesonets (Oklahoma, New York) with a smaller region |
| 1° grid smooths out mesonet value | Report super-ob representativeness; optional 0.25° GraphCast follow-on |
| JRA-3Q T or URMA 2t turns out no closer to observations than MERRA-2 | Report it; the swap then serves as a pure imbalance test (Q4) |
| Assimilation status of ASOS 2 m T or USCRN in ERA5/MERRA-2 is unclear | Confirm from ECMWF and GMAO documentation before Exp 3; if USCRN is assimilated, verify on withheld mesonets instead |
| Diurnal boundary-layer dependence | Stratify all results by 00/12 UTC start and season |
| Nudging "gain" is only blur | E-SMO and E-NUD0 controls |

---

## 13. Parked (possible follow-ons)
- Mesonet dewpoint → `q` at 1000/925 hPa (the moisture version of the same test)
- GraphCast 0.25° or Aurora repeat for resolution dependence
- NeuralGCM repeat, to support the dynamical-balance claim
- Fine-tuning GraphCast_small on MERRA-2 (only if Exp 1 shows a large distribution gap)
- Upper-air insertion (radiosonde T/q, AMSR2 TPW; see `DATA_SOURCES.md`)
- Precipitation-input replacement (IMERG): low priority, since precipitation is not a prognostic state variable
