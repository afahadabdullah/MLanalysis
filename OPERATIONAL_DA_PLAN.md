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
