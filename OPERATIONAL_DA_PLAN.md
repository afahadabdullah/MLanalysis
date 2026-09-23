# Operational-style observation ingestion for a frozen ML weather model

**Version:** 22 Sep 2026. Companion to `PROJECT_PLAN_LEAN.md` and `RESULTS.md` (§16–19).
**Model:** GraphCast_small, frozen (1°, 13 levels, 6 h step, trained on ERA5 1979–2015).
**Data inserted:** real surface 2 m temperature (NOAA ISD-Lite ASOS/AWOS). Later: radiosondes.
**Question:** if an organisation runs an off-the-shelf ML model and has its own observations, how should it put them in, the way operational NWP centres do, so that forecasts improve?

---

## 1. What operational centres do, and what we copy

| Centre / system | How surface temperature enters | What we copy |
|---|---|---|
| **ECMWF IFS, 4D-Var** | Screen-level T is assimilated in the upper-air 4D-Var: 5.5 K/km height correction; stations kept only from 400 m below to 200 m above model orography; departures > 7.5 K rejected; observations limited to the first 6 h of the 12 h window ([ECMWF FUG §2.1.4.9](https://confluence.ecmwf.int/display/FUG/Section+2.1.4.9+Assimilation+and+modelling+2m+temperature)). Only since after Cy48; earlier cycles (ERA5 = Cy41r2) used screen-level T only in a separate surface analysis. The change improved 2 m T forecasts ([ECMWF 2024 upgrade](https://www.ecmwf.int/en/newsletter/178/earth-system-science/improved-two-metre-temperature-forecasts-2024-upgrade)) | Observation QC rules (§3); adjoint-based 4D-Var with GraphCast's own gradients (§5, method M6) |
| **NOAA RAP/HRRR, hourly hybrid EnVar** | Surface T and dewpoint innovations extended upward as pseudo-innovations every 20 hPa up to 75 % of the boundary-layer top; model's flux-based 2 m diagnostic compared with the observation; 75 % ensemble / 25 % static covariances; digital-filter initialization each cycle ([Benjamin et al. 2016, MWR](https://journals.ametsoc.org/view/journals/mwr/144/4/mwr-d-15-0242.1.xml)) | Boundary-layer-dependent vertical spreading (method M2) |
| **NASA GEOS, IAU** | The analysis increment is applied gradually as a forcing over the 6 h window (Bloom et al. 1996) | Increment split over the steps before launch (method M4) |
| **All centres, cycling** | Background = previous short forecast; analysis every cycle; forecasts launched from the latest analysis | Continuous 6 h cycling with the ML model as the forecast model (method M5, operational mode B) |

**ML + DA literature this sits next to:** [FengWu-4DVar](https://arxiv.org/abs/2312.12455) (4D-Var with ML auto-differentiation; simulated observations); [FourCastNet DA case study, AIES 2025](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0050.1.xml); [Slivinski et al. 2025, GRL](https://arxiv.org/abs/2412.18016) (EnKF with real surface pressure: deterministic ML models accumulate small-scale noise and diverge; spectral filtering stabilizes them at a cost; poor cross-variable covariances).
**Our gap:** real surface temperature, one frozen off-the-shelf model, and a head-to-head of simple insertion, boundary-layer-aware insertion, gradual insertion, nudged cycling and 4D-Var, scored against independent stations.

---

## 2. Two operational modes

| | **Mode A: provider analysis + own observations** | **Mode B: own cycling** |
|---|---|---|
| What it is | Each cycle, take a provider's global analysis and add local observations before launching the ML forecast | The ML model's own previous forecast is the background; observations are assimilated every cycle; no provider analysis after cold start |
| Who does this | Most companies running an off-the-shelf model today | A centre running its own analysis |
| Base state in this project | ERA5 (research stand-in for a real-time analysis such as GFS/GDAS or IFS open data); MERRA-2 as a foreign-analysis variant | GraphCast 6 h forecast, cold-started from ERA5 |
| Observations needed | Surface T only is enough to test | Surface T **plus upper-air** (radiosondes; later aircraft), otherwise the free atmosphere drifts |
| Our evidence so far | `isd/era5` (§19): −1 to −3 % RMSE at USCRN from 24 to 72 h; deep insertion hurts at 6 h | `isd/bg` (§18): all arms beat the background by 1–4.5 %; nudging best at 24–48 h |
| Priority | **First** (cheapest, most realistic) | Second |

Real-time caveat: ERA5 has a ~5-day latency, so an operational Mode A would use GFS/GDAS or IFS open data. Those are foreign analyses for an ERA5-trained model; the MERRA-2 runs (Exp 1) measure that penalty.

---

## 3. Observation processing (operational rules)

Applied identically in every method, so methods differ only in how the information is spread.

| Step | Rule | Source |
|---|---|---|
| Time window | Hourly ISD within ±30 min of the analysis time (and each nudging/4D-Var time) | — |
| Height correction | T_adj = T_obs + Γ (z_station − z_model), **Γ = 5.5 K/km** | ECMWF |
| Height window | keep −400 m ≤ z_station − z_model ≤ +200 m | ECMWF |
| Gross check | reject \|y − H(x_b)\| > 7.5 K | ECMWF |
| Background check | reject \|d\| > 4 √(σ_b² + σ_o²) | standard |
| Super-observations | average within each 1° cell; σ_o per super-ob = √(σ_inst² + σ_repr²/n) with σ_inst = 0.5 K, σ_repr ≈ 1.0 K | standard |
| Operator H | bilinear interpolation of the model 2 m T to the station | — |
| Bias monitoring | log mean innovation per cycle and time of day; **do not** remove it silently (the −0.7 K night-time mean innovation in §19 is signal, not noise) | — |
| Networks | inserted: ISD (70 % of stations); verification: remaining 30 % ISD + **USCRN** (never inserted) | — |

---

## 4. Boundary-layer diagnosis from the GraphCast state

GraphCast has no PBL-height variable, so it is diagnosed from the background profile on its pressure levels (1000, 925, 850, 700 hPa), per grid column:

1. Potential temperature θ at 2 m (with surface pressure from MSLP and orography) and at each level above ground.
2. **Boundary-layer top** = lowest level where θ(level) > θ(2 m) + 1.5 K (parcel/θ-excess method); if the 2 m–1000 hPa layer already exceeds it, the layer is **stable** and the top is the surface.
3. Pseudo-innovations are placed at levels below **75 % of the PBL depth** (RAP rule), with weight 1 at the surface decreasing linearly to 0 at 75 % of the top.
4. Stable columns (typical winter night): surface-only increment. Well-mixed columns (afternoon, summer): increment through the mixed layer, which at 13 levels means 1000/925 hPa, sometimes 850 hPa.

---

## 5. Insertion methods compared (all use §3 observations)

| ID | Method | Operational analogue | New code | Effort |
|---|---|---|---|---|
| **M0** | No observations (control) | — | exists | — |
| **M1** | Surface-only OI increment (DIR-1F / DIR-2F) | simple nudging of a 2 m field | exists | — |
| **M1b** | **Bias-only:** subtract the domain-mean innovation uniformly | bias correction | ~10 lines | hours |
| **M2** | **Boundary-layer-aware OI (COL-PBL):** surface increment + pseudo-increments through the diagnosed mixed layer; hypsometric Z update | RAP/HRRR | ~80 lines | days |
| **M3** | **Statistical multivariate (NMC-B):** increments to T(p), q(p), u, v, MSLP regressed on the 2 m increment, from 48 h − 24 h GraphCast forecast differences over ~30 dates, stratified by 00/12 UTC and season | static 3D-Var B | ~150 lines + 30 dates of runs | 2–3 weeks |
| **M4** | **IAU-like:** one increment from t0 observations, added as ¼ after each of the 4 steps from t0−24 h to t0 | GEOS IAU | ~40 lines | days |
| **M5** | **Nudged cycling:** recomputed increment (M2-type) added with gain α every 6 h for 24, 48 or 120 h | continuous cycling / nudging | exists (24 h); extend | days |
| **M6** | **4D-Var with GraphCast gradients:** control = (t0−6 h, t0) state in B^½ space (B from M3); cost = background term + surface observation term at t0−6 h and t0 (ECMWF QC, first-half window); minimize with L-BFGS using `jax.grad`; launch from the analysed pair | ECMWF 4D-Var | ~300 lines | 3–5 weeks |
| M7 | EnKF / hybrid | RAP/HRRR, GFS | — | parked (see Slivinski et al.) |

**Why this order:** M1b–M2–M4–M5 reuse the current script and directly test what §18–19 showed (early depth penalty in stable conditions; nudging best at 24–48 h). M3 is needed by M6. M6 is the headline comparison: does letting the model's own adjoint decide the 3-D increment beat hand-built rules?

### 5.1 M6 details (4D-Var) worth deciding up front
- **Window:** observations at t0−6 h and t0 (one GraphCast step), later t0−12 h … t0 (two steps).
- **Control variable:** full state pair or only the lowest levels + surface (smaller, safer). Start with T, q, u, v, Z at 1000–700 hPa + all surface fields.
- **B:** from M3, with Gaussian horizontal correlation (L ≈ 250 km) as the square-root operator.
- **Precision:** gradients in float32 (GraphCast runs bf16 internally); check gradient against finite differences on a few directions.
- **Cost:** ~20–50 iterations × (forward + backward of 1–2 steps) — minutes on one GPU; within a 6 h cycle.
- **Noise:** apply a mild spectral filter to the increment if small-scale noise appears (the Slivinski et al. failure mode).

---

## 6. Verification protocol

| Level | Reference | Use |
|---|---|---|
| **Primary** | **USCRN** 2 m T, stations within the ECMWF height window, +6 … +72 h | Independent of insertion and of ERA5's atmosphere |
| Secondary | Withheld 30 % ISD stations | Same network, independent subset |
| Upper air | IGRA radiosondes, T850 / Z500 at 00/12 UTC (to add) | Checks that surface insertion does not damage the free atmosphere |
| Large scale | ERA5 grid, T850 / Z500 / MSLP | Guardrail only; not truth for 2 m T |

- **Metrics:** RMSE, bias, and % change vs the control (M0) — paired by station and time.
- **Significance:** paired bootstrap over stations (1000 resamples) within each date; block bootstrap over dates across the sample. Report 95 % intervals.
- **Stratify** by initialization hour (00/12 UTC ≈ evening/morning over CONUS), season, and terrain (flat vs complex).
- **Compare at the same time of day** (t0, +24, +48, +72 h) to avoid diurnal error-floor artefacts.

---

## 7. Experiment matrix

| Stage | Dates | Methods | Mode | Purpose |
|---|---|---|---|---|
| **S1** | 1 (2018-01-15 12 UTC, done) + 2018-01-15 00 UTC + one July date | M0, M1, M1b, M2, M4, M5(24/48 h) | A and B | Wiring, sign of effects, stable vs mixed PBL |
| **S2** | 20: 5 per season, 00 and 12 UTC, 2018 | same | A and B | Ranking with confidence intervals |
| **S3** | 30 dates of GraphCast runs (for NMC-B, not evaluation) | — | — | Build B for M3/M6 |
| **S4** | same 20 dates | + M3, M6 | A | Headline: rule-based vs statistical vs 4D-Var |
| **S5** | 10-day continuous cycle (winter) and (summer) | M5 vs M6 cycled, with radiosondes | B | Does own cycling stay stable and beat Mode A? |

Hold out 2019 dates for a final confirmation of whichever method wins S4.

---

## 8. Acceptance criteria ("operationally useful")

A method is recommended if, over ≥20 dates:
1. USCRN 2 m T RMSE improves vs M0 by ≥ 2 % at +6 … +48 h with 95 % intervals excluding zero,
2. no lead in 6 … 72 h is significantly worse than M0,
3. T850 / Z500 errors (radiosondes, ERA5) are not significantly degraded,
4. runtime fits a 6 h cycle on one GPU (all methods except possibly M6 are seconds; M6 must be ≤ 30 min).

---

## 9. Implementation map (`scripts/exp_main_real_obs.py`)

| Item | Change |
|---|---|
| QC | `--lapse 5.5`, asymmetric height window `--dz-below 400 --dz-above 200`, gross check 7.5 K, background check, super-ob σ_o |
| M1b | arm `BIAS` |
| M2 | `diagnose_pbl()` + arm `COL-PBL` (and `BAL-PBL` with Z) |
| M4 | arm `IAU-4` (fixed increment in 4 parts from t0−24 h) |
| M5 | `--nud-hours 24,48,120` with analysis every 6 h; ERA5 data window extended accordingly |
| Verification | paired station bootstrap; bias/RMSE by station; same-time-of-day summary |
| Multi-date driver | `scripts/run_dates.py` looping dates × sources × bases, aggregating `scores.csv` |
| M3 | `scripts/build_nmc_b.py` (runs GraphCast 24/48 h over 30 dates, saves regression / covariance stats) |
| M6 | `scripts/fourdvar.py` (JAX cost, gradient check, L-BFGS, writes analysed pair) |

---

## 10. What we already know that shapes this plan (single case each)

- Real ISD stations improve the t0 state at independent stations **beyond ERA5** (withheld ISD 2.14 → 1.92 K; USCRN 2.79 → 2.64 K), but most of that lead is lost in the first 6 h step (§18).
- Inserted into ERA5, they improve USCRN forecasts by 1–3 % from 24 to 72 h (§19) — small, not yet tested for significance.
- Deep insertion hurts at +6 h in the stable winter morning case; surface-only is best there. This is exactly the regime RAP handles by restricting depth to the mixed layer (→ M2).
- Nudged cycling is the most consistent across leads when inserting into the GraphCast background (§18) (→ M5).
- The hypsometric Z update has had no measurable effect so far.
- The mean innovation vs ERA5 is about −0.7 K at 06/12 UTC (ERA5 warm in the winter inversion); a bias-only control (M1b) is needed to separate bias correction from spatial information.

---

## 11. Implemented for the 2018-01-15 case (22 Sep 2026)

`scripts/exp_main_real_obs.py` now contains M0, M1, M1b, M2, M4 and M5 (6/12/24 h), the ECMWF QC rules, the boundary-layer diagnosis and a paired station bootstrap. Everything runs on the data already downloaded (ERA5 window t0−30 h … t0+72 h, ISD and USCRN 14–19 Jan).

| Item | Flag / arm | Default |
|---|---|---|
| Lapse rate | `--lapse` | **5.5 K/km** |
| Height window | `--dz-below` / `--dz-above` | **400 / 200 m** (insertion *and* verification) |
| Gross / background check | `--gross` / `--bgcheck` | 7.5 K / 4σ |
| Super-ob obs error | `--sigma-inst` / `--sigma-repr` | 0.5 / 1.0 K, σ_o² = σ_inst² + σ_repr²/n |
| M1b bias-only | arm `BIAS` | mean QC'd innovation added uniformly over CONUS land |
| M2 boundary-layer-aware | arms `COL-PBL`, `BAL-PBL` | θ-excess 1.5 K, spread to 75 % of depth (`--pbl-dtheta`, `--pbl-frac`) |
| M4 IAU-like | arms `IAU-DIR`, `IAU-BAL-PBL` (`--iau-kinds`) | t0 increment added as ¼ after each step t0−24 h … t0 (base bg only) |
| M5 nudging | `NUD<w>-<type>` (`--nud-windows`, `--nud-types`) | windows 6, 24 h; types DIR, BAL-PBL, FIX; base era5 uses ≤12 h |
| Bootstrap | `--boot` | 1000 station resamples; `bootstrap_station_diffs.csv`, `fig8_bootstrap_station.png` |
| PBL map | — | `fig9_pbl_diagnosis.png`; stable fraction and mean depth in `summary.json` |
| Arm subset | `--arms DIR-1F,COL-PBL,...` | all |

**Runs for Stage S1 on existing data:**
```bash
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd                 # Mode B-like (GraphCast background)
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --base era5     # Mode A (provider analysis + stations)
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source era5-synth          # twin control
```
Not possible yet on this data: nudging windows > 24 h (ERA5 starts at t0−30 h, ISD at t0−36 h) and any 00 UTC or summer case — those need new downloads.

### 11.1 Added: 3-day 6-hourly nudging spin-up (`--long-nud 72`)

Operational-style cycling before the forecast: cold start from ERA5 at t0−78 h / t0−72 h, then 12 GraphCast steps with a nudging increment from that time's ISD stations after every step (α = 0.63, τ = 6 h), up to t0; the 72 h forecast starts from the nudged pair.

| Arm | Meaning |
|---|---|
| `NUD72-BAL` | 3-day nudging with the best nudging type so far (BAL: surface + regressed column + hypsometric Z; `--long-nud-types` for others, e.g. `BAL,BAL-PBL`) |
| `FREE72` | Same 3-day chain with no observations (a 72 h GraphCast forecast): separates what the nudging adds from what the older starting point loses |

Read it against `BASE` (24 h background) and `NUD24-*`: does nudging 12 cycles instead of 4 keep more station information, or does the background drift (FREE72) outweigh it?

**Data to download (login node) for t0 = 2018-01-15 12 UTC:**
```bash
python scripts/download_era5_cloud.py  --date 2018-01-12 --time 12:00 --steps 24   # frames t0-78 h ... t0+72 h
python scripts/download_isd_lite.py    --start 2018-01-12T00 --end 2018-01-18T12
python scripts/download_uscrn_range.py --start 2018-01-12T00 --end 2018-01-18T12
```
→ `data/era5/source-era5_date-2018-01-12_res-1.0_levels-13_steps-24.nc` (~2× the current file), `data/obs/isd_lite_20180112T00_20180118T12.csv`, `data/obs/uscrn_20180112T00_20180118T12.csv`. The script finds them automatically (it picks the observation files whose date range covers the run); if the ERA5 file is missing it stops and prints these commands.

**Run:**
```bash
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72
# quicker, only the arms needed for this comparison:
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 \
    --nud-windows 24 --nud-types BAL --arms DIR-1F,NUD-BAL,FREE72,NUD72-BAL
```
The same longer ERA5 file also serves every other arm, so it can replace the steps-16 file.

### 11.2 Fixed: hybrid cycling — full state relaxed to ERA5 + surface stations (`--long-nud 72`, `--hyb-tau`)

§20 showed that surface-only nudging for 3 days lets the free atmosphere drift (t0 T850 error 2.0 K vs 0.75 K for the 24 h background). Operational systems avoid this by constraining the whole atmosphere every cycle. The long chain now has two extra arms; after **every** 6 h GraphCast step:

1. **Every variable at every level** (T, Z, u, v, w, q on 13 levels, and 2 m T, 10 m winds, MSLP, precipitation) is relaxed toward ERA5 at that time: `x ← x + α_E (ERA5 − x)`, α_E = 1 − exp(−6 h/τ_E), default τ_E = 6 h (α_E = 0.63). ERA5 stands in for the provider analysis; this is the reanalysis *replay* idea from the original plan.
2. The **station increment** (type DIR or BAL-PBL) is then computed against that relaxed state and added with α = 0.63.

| Arm | ERA5 relaxation | Stations | Question it answers |
|---|---|---|---|
| `FREE72` | no | no | drift of a 3-day ML forecast |
| `NUD72-BAL` | no | yes | surface-only cycling (drifts aloft, §20) |
| **`REPLAY72`** | **yes** | no | ERA5 replay alone: model-consistent version of ERA5, no own data |
| **`HYB72-DIR`**, **`HYB72-BAL-PBL`** | **yes** | **yes** | replay + own surface observations: the operational target |

Key comparisons: **HYB72 vs REPLAY72** = value of the stations inside a well-anchored cycle; **REPLAY72 vs ERA5 start** = does a model-consistent (replayed) ERA5 forecast better than raw ERA5 (the original project question); **HYB72 vs NUD24-* / NUD6-*** = does 3 days of anchored cycling keep more station information than a short window.

`--hyb-tau` sets how tightly the state follows ERA5 (6 h = strong; 12–24 h = weaker, leaves more room for the model and the stations). `--hyb-types` sets the station increment type(s).

**Run** (uses the steps-24 ERA5 file and the 12–18 Jan ISD/USCRN files already downloaded):
```bash
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 \
    --nud-windows 6,24 --nud-types DIR,BAL-PBL \
    --arms DIR-1F,NUD6-DIR,NUD24-BAL-PBL,FREE72,NUD72-BAL,REPLAY72,HYB72-DIR,HYB72-BAL-PBL
# sensitivity to the ERA5 anchoring strength:
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 --hyb-tau 12 \
    --arms REPLAY72,HYB72-DIR,HYB72-BAL-PBL --outdir runs/exp_main/20180115T12_isd_bg_hybtau12
```

### 11.3 Resolution test: full GraphCast at 0.25° / 37 levels (`--model large`)

**Question:** does the same experiment give better results with the 0.25° model? Expected effects: less representativeness error between 2 m T grid values and stations (terrain, coasts, valleys), station increments at their own scale instead of smeared into 1° cells, and a better-resolved boundary layer (37 levels: 1000, 975, 950, 925, 900, 875, 850 … hPa).

**What differs besides resolution** (so the comparison is of the *system*, not resolution alone): checkpoint `GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz` (trained to 2017, so 2018 is still out of sample; the "operational" 0.25° checkpoint is fine-tuned on HRES 2016-2021 and must **not** be used for 2018); finer model orography, so the ECMWF height window keeps more stations. Compare each resolution's arms **against its own ERA5 start and BASE**, then compare those relative gains across resolutions.

**Arms (4 + references):** `ERA5`, `BASE`, `DIR-1F` (simple insertion), `NUD6-DIR` (best short nudging), `REPLAY72` (ERA5 replay), `HYB72-DIR` (replay + stations). Only the selected arms are built, which matters at 0.25° (~1 GB per model state).

**Data and hardware**
| Item | Command / note |
|---|---|
| Checkpoint | `gs://dm_graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz` → `data/params/` (stats files are the same as for small) |
| ERA5 0.25°, 37 levels | `python scripts/download_era5_arco025.py --date 2018-01-12 --time 12:00 --steps 24` → `data/era5/source-era5_date-2018-01-12_res-0.25_levels-37_steps-24.zarr` (from Google ARCO-ERA5; ~26 frames, **~15–25 GB**; written frame by frame; put `data/era5` on nobackup first) |
| Stations | the ISD and USCRN files for 12–18 Jan already downloaded |
| GPU | ≥40 GB: try the DGX A100 node (`salloc -G1 -p dgx`); if out of memory, an H100 96 GB node (`-p grace`, ARM — needs an aarch64 JAX build) |
| Host RAM | ~40–60 GB peak (a 12-step rollout of 37-level fields); request `--mem=128G` |
| Time | ~120 model steps plus compilation; expect tens of minutes |

**Run:**
```bash
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --model large --long-nud 72 \
    --nud-windows 6 --nud-types DIR --arms DIR-1F,NUD6-DIR,REPLAY72,HYB72-DIR
# (outputs: runs/exp_main/20180115T12_isd_bg_r025/)
# same arms at 1 deg for a like-for-like table:
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 \
    --nud-windows 6 --nud-types DIR --arms DIR-1F,NUD6-DIR,REPLAY72,HYB72-DIR --outdir runs/exp_main/20180115T12_isd_bg_r1_4arms
```
Optional sensitivity at 0.25°: `--oi-L 150` (station increments at a scale the finer grid can hold).

`--lite` (default on for `large`) keeps only 2 m T, T850 and Z500 from each forecast; scores are identical to full storage (checked at 1°).

---

## 12. GMAO IAU variants, other operational methods, and ML-specific methods (22 Sep 2026)

### 12.1 What GMAO uses and how it maps to a 6-hourly ML cycle
| GMAO method | Description | Analogue here |
|---|---|---|
| IAU (Bloom et al. 1996), GEOS-5 / MERRA-2 | 6 h analysis increment added as a constant forcing during a model re-run (low-pass filter on fast modes); *replay* = IAU with increments from an existing analysis | `IAU-*`, `REPLAY72` |
| 4DIAU (GEOS Hybrid 4D-EnVar; Todling & El Akkraoui 2018, GMAO TM vol. 50) | Hourly time-varying increments; "nudged 4DIAU" moving to "nearest-time 4DIAU modulated with a digital filter" | HYB cycle with **cycle weights** (below) |
| Digital-filter-weighted IAU (Polavarapu et al. 2004) | Non-uniform weights = incremental digital filter; weights set the damped frequencies | `W=ramp-up / ramp-down / tri / lanczos` |

GraphCast steps 6 h, so continuous forcing is impossible; what transfers is the **weighting of increments across cycles** and the **time-varying increments**.

### 12.2 Implemented now (all 1°, `--long-nud 72`, base bg)

| Option | What it does | Arm name |
|---|---|---|
| **Weighted 4DIAU** `W=<profile>` | Scales both the ERA5 relaxation and the station increment of cycle *k* by w_k (const, ramp-up, ramp-down, tri, lanczos; max 1) | `HYB72-DIR-Wramp-up` … |
| **Level-selective replay** `LS` | Surface fields and levels ≥ `--hyb-bl-top` (850 hPa) relax to ERA5 with `--hyb-sfc-tau` (24 h), the rest with `--hyb-tau` (6 h): the upper air stays anchored, the boundary layer keeps station information | `HYB72-DIR-LS` |
| **Station bias correction** `BC` (VarBC-lite) | Per-station (or station × UTC hour, `--bias-mode`) bias, EW-updated each cycle (`--bias-gamma` 0.2) and removed before QC/OI; bias statistics in `summary.json` | `HYB72-DIR-BC`, combinations like `LS+BC` |
| **Model-Jacobian balance (method 2)** `JAC` | K (`--jac-k` 8) random smooth 2 m T perturbations over the OI box, propagated with GraphCast's **tangent-linear model** (`jax.jvp`, float32) for one 6 h step; local regression of every variable/level response on the 2 m T response (smoothed, `--jac-smooth-km` 500) gives spatially varying, flow-dependent coefficients; the 2 m T station increment is spread with them | `JAC-2F` (single insertion), `HYB72-JAC` (inside the cycle) |
| **Two-frame 4D-Var (method 3)** `4DV` | Strong-constraint 4D-Var over [t0−12 h, t0]: control = increments to *both* GraphCast input frames (x₋₁₂, x₋₆) for 2 m T and T, q, u, v, Z (levels ≥ 500 hPa), in B^½ space (Gaussian L = `--fdv-L` 300 km, std `--fdv-sig`); cost = background + stations at t0−6 h on x₋₆ + stations at t0 on F(x₋₁₂, x₋₆); gradient by `jax.value_and_grad` through GraphCast, L-BFGS (`--fdv-iter` 20). Launch pair (x₋₆ᵃ, F(x₋₁₂ᵃ, x₋₆ᵃ)) is model-consistent by construction | `4DV` (on the 24 h background) |
| **Hybrid + 4D-Var** | 72 h hybrid cycle to t0−6 h (stations through t0−12 h), then the 4D-Var window with an ERA5 anchor at t0 (`--fdv-era5-weight`) | `HYB72-4DV` |
| Gradient self-test | Compares the float32 differentiable step with the operational forward step, and the JVP with a finite difference (printed; >20 % disagreement warns) | `grad_checks` in `summary.json` |

Not yet implemented: **method 1** (gradient-tuned assimilation settings). It uses the same differentiable step but must be trained on a set of development dates to avoid fitting one case; it comes after the multi-date runs.

### 12.3 Run (1°, data already downloaded)
```bash
# operational variants of the hybrid cycle
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 \
    --hyb-types DIR --hyb-variants base,LS,BC,LS+BC,W=ramp-up,W=ramp-down,W=tri \
    --arms REPLAY72,HYB72-DIR,HYB72-DIR-LS,HYB72-DIR-BC,HYB72-DIR-LS+BC,HYB72-DIR-Wramp-up,HYB72-DIR-Wramp-down,HYB72-DIR-Wtri \
    --outdir runs/exp_main/20180115T12_isd_bg_hybvariants

# ML-specific methods (JAC and 4D-Var)
python scripts/exp_main_real_obs.py --t0 2018-01-15T12:00 --obs-source isd --long-nud 72 \
    --hyb-types DIR,JAC,4DV --nud-windows 6 --nud-types DIR \
    --arms DIR-1F,NUD6-DIR,JAC-2F,4DV,HYB72-DIR,HYB72-JAC,HYB72-4DV,REPLAY72 \
    --outdir runs/exp_main/20180115T12_isd_bg_mlda
```
Check first in the log: the **gradient self-test** (`step32_vs_fwd_bf16_conus_rms_2t_K` should be small, ~0.01–0.1 K; `jvp_vs_fd_T925_rel_err` < 0.2), the **JAC profile** (mean dT(p)/dT₂ₘ over CONUS land — the model's own vertical spreading, to compare with the regression 0.44/0.24/0 and the fixed 1.0/0.6/0.2), and the **4D-Var cost reduction** (J₀ → J_final).

---

## 13. Roadmap after the 4D-Var / JAC runs (23 Sep 2026)

Order: finish §12 (4D-Var/JAC with the t0-relax fix) → **Step 5** static covariances → **Step 6** full 4D-Var → **Step 7** a second reanalysis (MERRA-2) as provider → Step 8 multi-date confirmation → Step 9 ensembles.

### Step 5 — Static multivariate covariances from GraphCast (NMC method)
- Run GraphCast_small 48 h and 24 h forecasts valid at the same times for ~30 dates (Dec–Feb 2017/18 and 2016/17, 00 and 12 UTC); the 48 h − 24 h differences sample background error.
- Estimate: (a) vertical/multivariate regressions of T(p), q(p), u, v, Z(p), MSLP on 2 m T (stratified by 00/12 UTC and stable/mixed PBL); (b) horizontal correlation length per variable; (c) per-variable std for the 4D-Var control (`--fdv-sig`) and the OI (`--oi-L`).
- Replaces the 2-time regression (R² ≤ 0.33) and the hand-set 4D-Var B. New script `scripts/build_nmc_b.py`; output `data/nmc_b_1deg.nc`, read with `--b-file`.

### Step 6 — Full 4D-Var with GraphCast gradients
- B from Step 5; window extended to two model steps (t0−18 h … t0, observations at 3 times); ECMWF rule of using observations mainly in the first part of the window tested as an option.
- Cycled: 4D-Var every 6 h inside the 72 h hybrid chain (not only at the last window), with the ERA5/provider relaxation kept for the upper air.
- Gradient-tuned settings (method 1): learn α, L, ERA5-relaxation strength per level group, and the B scaling by backpropagating 6–24 h withheld-station error through GraphCast over the development dates.

---

## 14. Step 7 — A second reanalysis (MERRA-2) as the provider analysis, no fine-tuning

**Why:** in practice an organisation cannot start an ERA5-trained model from ERA5 in real time; it uses a *different* analysis (GFS/GDAS, IFS, or at NASA **GEOS-FP**, the real-time sibling of MERRA-2). MERRA-2 is the research stand-in: same GEOS model/GSI family as GEOS-FP, but a foreign analysis for GraphCast. Questions:

1. How large is the **foreign-analysis penalty** (MERRA-2 start vs ERA5 start)?
2. How much of it is **distributional** (mean/variance/diurnal-cycle differences) vs **information** (a genuinely different estimate of the weather)?
3. Does **model-consistent replay toward MERRA-2** remove the penalty without fine-tuning?
4. Can **MERRA-2 + ERA5 together** (two analyses) or **MERRA-2 + own stations** **match or beat the ERA5 start**?

### 14.1 Arms (all 1°, same stations, same verification)

| Arm | Initial state | Tests |
|---|---|---|
| `E` | ERA5 start | reference |
| `M-DIR` | MERRA-2 mapped to GraphCast inputs, used directly | foreign-analysis penalty |
| `M-MEAN` | MERRA-2 − [clim_M − clim_E] (mean anomaly transplant) | distribution: mean only |
| `M-QM` | ERA5 climatology + (MERRA-2 − clim_M) · σ_E/σ_M (anomaly + variance mapping) | distribution: mean + variance — the **anomaly-initialization** approach |
| `REPLAY72-M` | 72 h GraphCast cycle relaxed (full state, τ = 6 h) toward MERRA-2 | does model-consistent replay remove the penalty? |
| `REPLAY72-MQM` | same, toward M-QM | replay + distribution mapping |
| `HYB72-MQM` | REPLAY72-MQM + ISD stations each cycle | own obs on a foreign-analysis base |
| `REPLAY72-BLEND` | relax toward w·ERA5 + (1−w)·M-QM, w = 0.5 (and w tuned on dev dates) | two analyses: can an ensemble of reanalyses beat ERA5? |
| `HYB72-BLEND` | BLEND + ISD stations | best-possible combination |
| `M-DIR-OBC` (output-side) | M-DIR forecast minus the lead-dependent mean difference F(M) − F(E) estimated on dev dates | anomaly **forecast** (debias the output instead of the input) |

"Anomaly forecast" is covered two ways: **anomaly initialization** (M-MEAN, M-QM: keep MERRA-2's anomalies, use ERA5's climate, so the state is in-distribution for GraphCast) and **output debiasing** (M-DIR-OBC). Neither retrains the model.

### 14.2 Verification
- 2 m T at USCRN and withheld ISD (primary), T850/Z500 against ERA5 **and** against MERRA-2 (so ERA5 is not favoured by construction), radiosondes when added.
- Report each arm vs `E` (can it match/beat ERA5?) and vs `M-DIR` (how much of the foreign penalty is recovered?).
- Decomposition: distribution part ≈ Err(M-DIR) − Err(M-QM); information part ≈ Err(M-QM) − Err(E); replay part ≈ Err(M-QM) − Err(REPLAY72-MQM).

### 14.3 Data
| Item | Source | Notes |
|---|---|---|
| MERRA-2 3-D | `inst3_3d_asm_Np` (T, U, V, QV, H, OMEGA; 42 levels incl. all 13 GraphCast levels; 3-hourly, 0.5°×0.625°) | on NCCS locally (check ADAPT `/css/merra2/` or Discover `/discover/nobackup/projects/gmao/merra2/`); else GES DISC. H→Z = g·H; OMEGA (Pa/s) = ERA5 `vertical_velocity` units |
| MERRA-2 2-D | `inst1_2d_asm_Nx` (T2M, U10M, V10M, SLP), `tavg1_2d_flx_Nx` (PRECTOT → 6 h sum) | |
| Climatologies | ERA5: WeatherBench-2 ERA5 6-hourly climatology (day-of-year × hour); MERRA-2: same statistics from `inst3_3d_asm_Np` 2010–2017, ±15 days around the case date, 00/06/12/18 UTC | mean and std per grid point, level, hour; needed for M-MEAN/M-QM |
| Window | 12–18 Jan 2018 (same as the ERA5 file) | 26 frames |

Adapter (`scripts/prep_merra2.py`): subset 13 levels, conservative regrid to 1° (0.25° later), **below-ground fill** (MERRA-2 is undefined below the surface; ERA5 extrapolates: fill T with a 6.5 K/km extrapolation from the lowest valid level and Z hypsometrically, q/u/v from the lowest valid level), precipitation to 6 h accumulation, GraphCast schema identical to the ERA5 file, static fields from ERA5. **Validate channel by channel against ERA5 for one date before any forecast** (a unit or fill error looks exactly like a "foreign-analysis penalty").

Code changes in `exp_main_real_obs.py`: `--provider {era5, merra2, qm, blend}` for the relaxation target of REPLAY/HYB, `--provider-file`, `--clim-files`, `--blend-w`; new direct arms `M-DIR`, `M-MEAN`, `M-QM`; output-debias arm.

### 14.4 Expected outcomes and what they mean
- **M-QM ≈ E:** the foreign penalty is mostly distributional → anomaly initialization lets any organisation use its own analysis with an ERA5-trained model, no fine-tuning needed.
- **REPLAY72-M ≈ E but M-DIR ≪ E:** replay (model-consistent insertion) removes the penalty → the IAU/replay principle is the fix for foreign analyses (the original project question, now with a foreign analysis where the effect should be larger than with ERA5).
- **REPLAY72-BLEND or HYB72-BLEND < E:** two analyses plus the model's own consistency beat the training analysis → the most practical route to "better than ERA5" for a real-time system (e.g. GEOS-FP + GFS + own stations).
- **Nothing recovers the penalty:** information difference dominates → fine-tuning would not help either; only better analyses/observations would.

### 14.5 Real-time follow-on
Repeat the best configuration with **GEOS-FP** (real-time GEOS analyses on NCCS) as the provider for a recent period, i.e. an actual real-time Mode A system at NASA.

### 14.6 Revisit DIR-1F shock with MERRA-2: multivariate propagation analysis (to do with Step 7)
**Why:** DIR-1F (single-frame direct insertion at t0) is the clearest shock case we have. In the v3 run (RESULTS §27.2) it fits the withheld stations best at t0 (1.784 K vs 2.241 K for BASE), but by +6 h it is worse than BASE (2.023 vs 1.985 K, +1.9 %; +16.1 %* vs ERA5). Its first-step jump |F(x0) − x0| for 2 m T is 6.19 K, vs 5.91 K for BASE and 6.04 K for ERA5. JAC-2F behaves the same way (+4.2 % at +6 h, §24). The cause: the t0 frame is changed but the t0−6h frame is not, so GraphCast reads the mismatch as a false tendency. So far we have only looked at 2 m T.

**What to do:** when the MERRA-2 arms are built (M-DIR is the MERRA-2 analogue of DIR-1F: a foreign state dropped into one frame), analyse how the DIR-1F / M-DIR shock propagates across variables, in the same style as the earlier dynamical analysis (RESULTS §3–4: spatial advection, vertical coupling, winds/MSLP adjustment):
- **Variables:** 2 m T, **total_precipitation_6hr**, MSLP, 10 m u/v, T/Z/q/u/v/ω on levels (esp. 1000–700 hPa and 500 hPa).
- **Diagnostics:** increment maps F(x_arm) − F(x_BASE) at 0/6/12/24/48 h; first-step jump per variable; vertical profiles of the response; ω and divergence (gravity-wave-like adjustment); precipitation spin-up/spin-down vs ERA5 and vs MERRA-2 precip; retention and advection of the increment.
- **Compare:** DIR-1F vs 4DV vs HYB72-DIR/HYB72-4DV (ERA5 base) and M-DIR vs M-QM vs REPLAY72-M/-MQM (MERRA-2 base). Does two-frame / replay / 4D-Var insertion remove the shock signal in precip and ω, not only in 2 m T?
- **Precip verification:** MERRA-2 PRECTOT and (if added) Stage IV / IMERG over CONUS.
