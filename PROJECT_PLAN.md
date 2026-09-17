# Initial-condition compatibility in machine-learning weather forecasts

**Working title:** Can nudging toward the same reanalysis improve AI weather forecasts? Physical balance, temporal consistency, and model compatibility.

**Project plan prepared:** 17 September 2026  
**Status:** Scientific and implementation proposal. No forecasting models have been installed or run for this plan. Model availability and interfaces are based on the official sources linked below; pin and validate a specific checkpoint before production experiments.

**Primary experiment:** Hold the reanalysis source and pretrained forecast model fixed. Compare direct initialization from ERA5 at launch with initialization produced by running that same model toward the same ERA5 history using nudging or, where implementable, incremental analysis updates. Stop all updates at launch and evaluate the subsequent free forecast. Cross-reanalysis comparisons and the existing GEOS replay states are supporting experiments.

## 1. Scientific motivation and connection to the existing project

The GEOS–MITgcm initialization project compares forecasts launched from directly inserted MERRA-2 atmospheric and ECCO ocean states with forecasts launched from coupled incremental analysis update (IAU)/replay states. The experimental description uses three months of preceding coupled replay, 6-hour atmospheric and 5-day oceanic update windows, and 15 May–June starts during 2005–2009. Its central process question concerns how initialization affects early adjustment, moisture convergence, convection, tropical waves, and subsequent forecast evolution.

The ML extension asks whether this principle transfers to models that learn atmospheric evolution from data, and whether physical balance is the relevant mechanism. A state that is closer to observations may still be difficult for a particular forecasting model to evolve accurately. Conversely, an initialization treatment may improve forecasts by removing useful small-scale variance or correcting statistical biases, rather than by improving dynamical balance.

The project must therefore test, rather than assume, the proposition:

> Improving initial-state accuracy improves forecast skill only to the extent that the forecasting model can retain and correctly evolve the added information.

The word “only” here describes information utilization, not a universal requirement for IAU. Physically balanced initialization is one candidate route to better information utilization.

Relevant existing material:

- [Current GEOS–MITgcm manuscript](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/IAU_initilization/article.tex).
- [Existing direct moisture-convergence calculation](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/IAU_initilization/IAUinit_paper/paper_figs/mse_budget_wp_2wk.py).
- [Existing spike and moisture-budget diagnostics](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/IAU_initilization/IAUinit_paper/paper_figs/spike_budget_stats.py).

These are sources of scientific context and reusable analysis ideas. Their assumptions must be reassessed for each ML model's variables, temporal sampling, and grid.

## 2. Questions and falsifiable hypotheses

### Primary question

Can initialization prepared by nudging or incremental updates toward the same reanalysis improve the subsequent free ML forecast relative to direct insertion? If it can, is the benefit explained by physical consistency, temporal consistency, smoothing, or model-specific bias adjustment?

The associated attribution question is whether reducing specified physical inconsistencies improves early adjustment and days 3–10 skill at comparable initial observational error.

### Secondary questions

1. Does a GEOS IAU/replay state improve forecasts in a different model, or is its benefit specific to GEOS–MITgcm?
2. Does consistency between the previous and current input states matter independently of the accuracy of the launch-time state?
3. Does compatibility with ERA5 or another training dataset matter more than physical balance?
4. Do ML models amplify initial imbalance, remove it harmlessly, or remove it while losing useful meteorological information?
5. Are tropical moisture and convection more sensitive than large-scale extratropical circulation?
6. Does initialization affect forecast uncertainty as well as the ensemble mean?
7. Does joint atmosphere–SST preparation improve S2S forecasts beyond preparing either component alone?

| Hypothesis | Supporting result | Result that weakens the hypothesis |
|---|---|---|
| H1: Physical inconsistency limits forecast skill | Physical adjustment improves skill beyond matched smoothing and statistical controls, while reducing prespecified residuals | Smoothing or statistical adjustment explains the same benefit, or residual reductions do not improve skill |
| H2: Training-data mismatch dominates | Correcting source-dependent bias improves forecasts with little change in physical diagnostics | Physical adjustment transfers across data sources and architectures without distribution adjustment |
| H3: Inconsistent input history causes early error | Changing the previous input while holding the launch state fixed changes subsequent skill systematically | Results are insensitive to reasonable history changes |
| H4: Moisture and thermodynamic consistency explain tropical sensitivity | Moisture/stability interventions change early convergence and later tropical errors in a repeatable sequence | The response is confined to dry dynamics or disappears after controlling for smoothing |
| H5: The model discards beneficial analysis increments | Initially useful increments disappear rapidly, together with their forecast advantage | Added information persists and improves skill regardless of initialization treatment |
| H6: Cross-component consistency matters | Joint atmosphere–SST adjustment has a beneficial interaction beyond either component alone | Only one component matters, or the interaction is negligible |

“Balance” must be defined through the constraints actually tested. Hydrostatic consistency, approximate extratropical wind–mass balance, moist thermodynamic consistency, and coupled consistency are related but distinct. Real ageostrophic circulation, divergence, and gravity waves are not automatically errors.

## 3. Recommended model strategy

Use frozen pretrained weights initially. Training a new foundation model is not necessary to answer the scientific question. Initial-condition interventions, rather than model retraining, should be the primary experimental factor.

### 3.1 Atmospheric models

| Model/checkpoint | Role | Inputs and practical interface | Important limitations |
|---|---|---|---|
| **GraphCast_small** | Low-cost screening and differentiable sensitivity experiments | Official JAX code; 1° grid, 13 pressure levels; ERA5 training through 2015; normally two atmospheric states six hours apart, plus checkpoint-specific inputs and forcings | Requires precipitation inputs; initialization changes must respect both input times. The coarse checkpoint is a mechanism tool, not a replacement for high-resolution validation |
| **Pangu-Weather** | Independent pure-ML architecture and early-adjustment experiments | Official ONNX inference on CPU or GPU; single input state; separate 1-, 3-, 6-, and 24-hour checkpoints; 0.25° grid and 13 levels | Standard outputs lack precipitation, vertical velocity, and surface fluxes. Use a fixed stepping schedule in all treatments |
| **Aurora 0.25° Pretrained** | Main foundation-model confirmation | PyTorch; pretrained on multiple datasets; this checkpoint is recommended for ERA5-like applications; input history, upper-air variables, surface variables, and static fields | Larger GPU requirement. Do not interchange its results with those of an HRES-fine-tuned checkpoint without treating checkpoint choice as another factor |
| **Aurora 1.5** | Optional extension with precipitation and finer output sampling | Public model with expanded surface fields and hourly sub-step outputs; targeted at HRES T0 initialization | More demanding input preparation. Intermediate hourly outputs are not independently advanced hourly dynamical states; only the main 6-hour output is fed back during rollout |
| **NeuralGCM 1.4° deterministic; optionally 1.4° stochastic** | Hybrid bridge between learned forecasts and dynamical models | Official JAX code; learned encoder, dynamical integration with learned physics, and decoder; prescribed SST and sea ice | Encoder/decoder changes are part of initialization. Atmospheric model only; output budgets depend on checkpoint. Do not assume all checkpoints output precipitation and evaporation separately |

Sources: [GraphCast checkpoint documentation](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_graph/README.md), [Pangu repository](https://github.com/198808xc/Pangu-Weather), [Aurora models](https://github.com/microsoft/aurora/blob/main/docs/models.md), [Aurora usage](https://microsoft.github.io/aurora/usage.html), [NeuralGCM checkpoints](https://neuralgcm.readthedocs.io/en/latest/checkpoints.html), [NeuralGCM model interface](https://neuralgcm.readthedocs.io/en/latest/deepdive_into_models.html).

GraphCast and Pangu are specialized weather models; Aurora supplies the direct foundation-model test. NeuralGCM supplies the hybrid comparison. Describe them accurately rather than calling every pretrained weather network a foundation model.

**Recommended sequence:** GraphCast_small plus NeuralGCM for the pilot; Pangu for an independent architecture and selected short-step cases; Aurora for the main foundation-model confirmation. If existing computing expertise makes Pangu easier to start, use Pangu in place of GraphCast_small and postpone the input-history experiment until Aurora or GraphCast is available.

The regular Aurora global 0.25° configuration is documented at approximately 40 GB GPU memory; Aurora 1.5 is documented at approximately 32 GB. AuroraSmallPretrained is recommended by its authors for debugging only. Budget for a Linux GPU system and benchmark each actual configuration; no throughput estimate is assumed here. CPU support for an example does not imply that hundreds of global forecasts are practical on a desktop. [Aurora hardware guidance](https://microsoft.github.io/aurora/usage.html).

### 3.2 Coupled and S2S options

| Model | Useful experiment | Scope of the conclusion |
|---|---|---|
| **FuXi-S2S** | Daily ensemble evolution of SST, atmospheric circulation, precipitation, and MJO-related fields through 42 days | Joint SST/atmospheric prediction; not a resolved subsurface ocean or a complete air–sea energy-budget model |
| **DLESyM** | Factorial atmosphere–SST initialization and longer coupled adjustment | Coupled learned atmosphere and prognostic SST; reduced state limits attribution to detailed ocean processes |

FuXi-S2S provides public ONNX weights and sample data. Its input contains two consecutive daily means on a 1.5° grid. DLESyM provides code, model checkpoints through Git LFS, and associated data. Neither is a direct substitute for the ocean state in GEOS–MITgcm. [FuXi-S2S repository](https://github.com/tpys/FuXi-S2S), [FuXi-S2S input conventions](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/FuXiS2S/), [DLESyM repository](https://github.com/AtmosSci-DLESM/DLESyM), [DLESyM paper](https://arxiv.org/abs/2409.16247).

## 4. Data inventory and access plan

### 4.1 Initial conditions and reference fields

| Data | Purpose | Access and preparation |
|---|---|---|
| **ERA5 pressure-level and single-level fields** | Training-compatible baseline, perturbation reference, and standard verification | Obtain selected dates through Copernicus CDS, WeatherBench 2, or ARCO-ERA5; verify actual coordinate coverage and variables before extraction |
| **MERRA-2** | Alternative atmospheric analysis and direct connection to the GEOS experiment | NASA GMAO/GES DISC; some access routes require Earthdata authentication. Distinguish instantaneous analysis collections from time-averaged products |
| **Existing GEOS coupled IAU/replay output and restarts** | Test whether preparation for a dynamical coupled model transfers to ML | Locate full global atmospheric states at launch and all required preceding times; existing figures alone are insufficient |
| **HRES T0 / HRES analysis where accessible** | Checkpoint-matched Aurora operational tests and cross-analysis comparisons | WeatherBench 2 includes HRES resources. T0 and analysis are not interchangeable; verify the required product and archive coverage |
| **SST and sea-ice fields** | NeuralGCM boundary conditions and coupled experiments | Use checkpoint-compatible fields, held identical across atmospheric treatments unless boundary conditions are the stated experimental factor |
| **ECCO and existing ocean replay fields** | Connection to the coupled project | Use for the GEOS comparison and relevant SST extraction; the selected ML model may not accept subsurface temperature, salinity, or currents |

Primary access references: [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), [ARCO-ERA5](https://github.com/google-research/arco-era5), [MERRA-2 overview](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/).

Download only selected start windows and verification periods at first. Do not download the complete ERA5 archive. Record the exact store, collection, product version, selected variables, timestamps, and retrieval date. Inspect dataset coordinates directly: a filename or catalog summary is not sufficient evidence of complete annual coverage.

### 4.2 Observation-based verification

| Product | What it tests | Qualification |
|---|---|---|
| **IGRA radiosondes** | Temperature, humidity, and winds at available observed levels | Apply quality control and collocation; some observations may have entered the underlying analyses |
| **IMERG precipitation** | Rainfall timing, accumulation, spatial structure, and extremes | Aggregate forecasts and observations to the same spatial/temporal support; use one pinned product version |
| **IBTrACS** | Tropical cyclone track and intensity | Use a consistent agency/variable convention and a validated tracking method |
| **Surface station archive, selected during implementation** | Near-surface temperature and wind | Terrain, height, exposure, and representativeness errors require care |
| **Observation-based OLR product, selected during implementation** | Tropical convection and wave evolution where the model predicts comparable radiation | Match sign, integration period, and units; unavailable OLR cannot be replaced silently by an unrelated proxy |

Sources: [NOAA IGRA](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive), [NASA IMERG](https://gpm.nasa.gov/data/imerg), [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

Call these observation-based checks, not perfectly independent truth. Document assimilation overlap and measurement uncertainty. A “better analysis” must be better against prespecified observational metrics; ERA5–MERRA-2 disagreement alone does not establish which is better.

### 4.3 Time splits

Maintain three separate cohorts:

1. **Legacy bridge:** the 2005–2009 GEOS cases. These test transfer and process correspondence, but overlap training periods of several ML checkpoints.
2. **Development/validation:** dates used to choose perturbation amplitudes, filter scales, nudging strengths, and statistical corrections.
3. **Held-out evaluation:** dates after the selected checkpoint's last training/fine-tuning date and outside development windows, allowing buffers for overlapping forecasts.

Determine dates checkpoint by checkpoint. Prefer a common held-out period for model comparisons; where that is impossible, report within-model results and the sampling difference. Never choose tuning settings from evaluation forecast errors.

## 5. Input preparation: avoid creating the artifact being studied

Build a shared physical dataset and a separate adapter for each model. Apply physical perturbations in physical units before each model's normalization. Preserve both raw and transformed states.

Required preparation checks:

- Exact variable names, ordering, levels, grid coordinates, latitude direction, longitude convention, and time spacing.
- Geopotential versus geopotential height; temperature in kelvin; pressure in the correct units; specific humidity versus mixing ratio or relative humidity.
- Horizontal wind rotation when converting from model grids.
- Surface pressure versus mean sea-level pressure. They are not substitutes.
- Native hybrid/model-level to pressure-level interpolation, with documented below-ground treatment.
- A common mask for physical diagnostics over below-ground levels. Forecast input filling must reproduce the checkpoint's expected convention rather than use that mask indiscriminately.
- Static orography and land–sea masks compatible with the checkpoint; do not replace them casually when changing the analysis source.
- Accumulation periods, interval-ending timestamps, and sign conventions for precipitation and radiation.
- Regridding methods appropriate to the field, with identical methods across treatments. Recompute physical residuals after regridding because interpolation can introduce inconsistency.
- No accidental mixture of analysis sources or valid times among variables, except in an explicitly designed experiment.

Example: Pangu expects five upper-air fields on 13 levels and four surface fields. Its repository defines precise ordering and units. Use that ordering in its adapter rather than relying on alphabetical dataset ordering. For GraphCast and Aurora, construct all required historical inputs. For NeuralGCM, use its documented grid preparation and encode/decode interfaces. [Pangu input specification](https://github.com/198808xc/Pangu-Weather), [NeuralGCM quick start](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html).

FuXi-S2S needs calendar-day means, not midnight instantaneous fields. Its precipitation and radiation aggregation conventions differ from a simple average of same-date timestamps. Its timestamp labels the daily aggregate; define the actual forecast issue time after the last input day has completed. Follow the official preprocessing exactly. [FuXi-S2S conventions](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/FuXiS2S/).

Perform two reference checks before scientific runs: reproduce an official sample inference, and show that the project adapter reproduces that same inference from equivalent physical data within expected numerical tolerance.

## 6. Experiment A: transfer of the existing initialization result

This is a supporting transfer experiment. The same-reanalysis nudging experiment in Section 8 is the primary study and can proceed without the GEOS archive.

At each legacy start, prepare ERA5, direct MERRA-2, and GEOS IAU/replay atmospheric inputs. Supply consistent preceding histories where required. Run frozen ML models to day 10 with identical checkpoint, stepping schedule, forcings, and output times within each model.

For the legacy GEOS comparison, “raw” means the direct atmospheric analysis mapped to the ML interface; “replay” means the atmosphere from the coupled replay state. Ocean information enters only through variables that the ML model actually accepts.

Compare initial observation departures, physical residuals, early increments, and forecast errors. This experiment establishes transfer, not causality: replay changes multiple properties at once. Include a table of all initial differences so the interpretation is not reduced to a binary balanced/unbalanced label.

If launch-time global states or history fields are absent from the archive, this experiment requires additional extraction or rerunning initialization. Continue the ERA5-based controlled experiments while resolving that dependency.

## 7. Experiment B: controlled changes to the analysis

### 7.1 Perturbation families

Begin with a state x and construct x + δx. Use localized, smoothly tapered perturbations with several amplitudes and spatial scales. Choose small amplitudes relative to analysis differences or ensemble uncertainty measured on development dates. Use both signs and multiple locations. Large unrealistic changes are separate stress tests, not evidence about routine analysis improvements.

**Dry extratropical pair:** prescribe a smooth geopotential increment δΦ with a documented vertical structure. Construct an approximately consistent perturbation using:

```text
δu_g = -(1/f) ∂δΦ/∂y
δv_g =  (1/f) ∂δΦ/∂x
δT_v = -(1/R_d) ∂δΦ/∂ln(p)
```

Use spherical derivatives in implementation, restrict this construction to extratropical regions, and smoothly taper it before low latitudes. If specific humidity is held fixed, derive δT consistently from virtual temperature. Compare with a deliberately inconsistent perturbation in which specified cross-variable relationships are broken. These equations define an approximate perturbation balance, not complete nonlinear balance of the total atmospheric state.

The simplest height-only versus height-plus-wind comparison is a demonstration, but changes both balance and total perturbation magnitude. Strengthen it with a family of covariance-controlled or balanced/imbalanced modal perturbations matched as closely as feasible in variable-wise error and spatial spectra. Report residual differences in those matching properties; do not claim exact isolation when it is not achieved.

**Tropical moisture pair:** add a tapered humidity anomaly in a specified layer and region. Compare humidity-only modification with a jointly adjusted temperature/geopotential state under a stated thermodynamic constraint. Hold the horizontal pattern and moisture amount comparable, diagnose relative humidity and stability, and record any humidity bounding. Treat this as moist thermodynamic consistency, not geostrophic balance. Additional changes in precipitation are model responses, not prescribed outcomes.

**Information-retention pair:** create alternative analysis increments from real source differences, assess which increments improve agreement with available observations at launch, and track whether that advantage survives the first model steps. Blending toward another analysis is not guaranteed to improve accuracy; determine the actual observation departures for each blend.

### 7.2 Match initial error and spatial structure

Use a documented weighted perturbation norm, for example a sum over variables and levels of area-weighted squared increments divided by fixed development-set variance scales. Also report variable-wise norms, spectra, and geographical placement. A scalar norm cannot capture dynamically sensitive error structure.

For real forecasts, compare treatments at similar initial observational error or report the trade-off between initial error and later forecast error. For a model-generated reference trajectory, the exact perturbation error is known; this makes a useful controlled experiment but remains a test of that model's dynamics, not atmospheric truth.

Run a small amplitude sweep to test dose response. A repeatable relationship among imposed inconsistency, early adjustment, and later skill is stronger evidence than one large perturbation case.

## 8. Experiment C: practical initialization treatments

### 8.0 Primary same-reanalysis experiment: prepare with ERA5, forecast freely

Start with ERA5 as the single analysis source. Keep model weights, static fields, forecast stepping, external forcings, and verification fixed. The question is whether changing how the same information enters the model improves the forecast.

```text
                         Forecast launch t0
ERA5 at t0/history -----------------|--> DIRECT: free forecast to day 10
ERA5 from t0-48 h to t0             |
    model + repeated nudging ------|--> NUDGED: free forecast to day 10
    model + distributed increments |--> IAU-LIKE: free forecast, if implemented
    model, no updates -------------|--> FREE control: free forecast to day 10
```

All branches are scored at the same valid times measured from t0. No branch receives post-launch atmospheric targets. For a two-state ML model, DIRECT receives its required raw ERA5 pair; NUDGED receives the final pair generated during preparation. Retain x(t0) from preparation rather than overwriting it with raw ERA5 just before launch.

Nudging must be toward the time-varying reanalysis, not toward a fixed future launch-time field throughout the window. At six-hour cycling intervals, a first implementation is:

```text
x_forecast[k] = F(prepared_history[k-1])
x_prepared[k] = x_forecast[k] + alpha * (x_ERA5[k] - x_forecast[k])
alpha = 1 - exp(-delta_t / tau)
```

This gain is motivated by linear relaxation; the split forecast/update scheme remains an approximation. Test a small development grid such as a 24- or 48-hour preparation window and relaxation times of 6, 12, and 24 hours for six-hour updates. These are candidate settings, not validated recommendations. Initially use one common gain on compatible prognostic atmospheric fields; treat diagnostic accumulations, moisture bounds, and unavailable variables explicitly rather than blending all arrays indiscriminately. Later test separate humidity/wind gains or scale-selective relaxation if the diagnostics motivate them.

For deterministic models whose complete state is the supplied input history, alpha=1 at every required final input time must recover the raw-history forecast, within numerical precision. This is a useful implementation check: an unrecognized hidden state, forcing difference, or input mismatch must not masquerade as a nudging effect. NeuralGCM's internal state and encoder require a separate check because repeated encoding may itself change the solution.

**Nudging versus IAU:** nudging recalculates a correction from the current model–analysis difference. Classical IAU calculates an analysis increment against a defined background and distributes that increment as an added tendency over an assimilation window. They share the idea of gradual information insertion but are not identical. [Original IAU paper](https://journals.ametsoc.org/view/journals/mwre/124/6/1520-0493_1996_124_1256_dauiau_2_0_co_2.xml), [GEOS-5 IAU description](https://gmao.gsfc.nasa.gov/media/publications/zbly36ziNFDFbmYmvhQeVqPhUo/Rienecker369.pdf).

A schematic discrete increment-distribution experiment is:

```text
fixed increment delta_x = analysis_at_anchor - background_at_anchor
within the defined window:
    x[k+1] = F(x[k]) + w[k] * delta_x
sum(w[k]) = 1
```

Weights summing to one specify the total imposed increment; they do not guarantee that the evolving final state equals the analysis. Declare the anchor time, background trajectory, state coordinates, increment units, any rerun of the window, and final cutoff. For centered retrospective windows, use only analysis anchors and windows whose applied updates finish by t0, and distinguish retrospective reconstruction from operational availability. A more faithful IAU implementation needs sufficiently short native model steps. The formula above is a proposed analogue, not an existing GraphCast or Aurora feature.

NeuralGCM is the strongest candidate for a later tendency-level implementation, but that requires inspecting and modifying the integration code. Its standard boundary-forcing API is not a general analysis-increment input. Repeated decode/nudge/re-encode is easier to prototype but combines nudging with encoding losses; include an identical encode/decode cycling control without nudging. Short-step Pangu provides another discrete experiment, while its checkpoint schedule must stay fixed between treatments. Do not invent substeps for a model trained only at a fixed lead time.

The primary result is NUDGED minus DIRECT forecast skill across held-out starts. Then compare NUDGED with FREE, SMOOTH, PHYS, and STAT to identify the cause. A state with lower physical residuals is a candidate for improved balance, not proof of a better forecast. A useful result can also occur when the prepared state has slightly larger initial observation error but smaller later error; report both quantities and test whether the apparent gain sacrifices real extremes.

### Treatment matrix

Apply the following treatments to the same underlying source and start dates. Add treatments in stages rather than launching every combination immediately.

| ID | Treatment | Purpose |
|---|---|---|
| RAW | Unmodified analysis after required input conversion | Baseline |
| PHYS | Minimal specified physical adjustment | Test the role of selected physical inconsistencies |
| SMOOTH | Smoothing control with comparable scale-dependent attenuation | Test whether variance removal explains the gain |
| STAT | Source-to-training-dataset statistical correction | Test systematic distribution mismatch |
| HIST | Model integration and nudging over the preceding analysis sequence | Test temporal and model-state consistency |
| FREE | Same preceding integration window without nudging | Control for spin-up, information age, and model smoothing |
| REPLAY | Existing GEOS coupled IAU/replay state | Test transfer of the original method |

### 8.1 Physical adjustment

Develop a constrained adjustment that minimizes the change to the analysis while reducing selected residuals. A conceptual objective is:

```text
J(x') = ||x' - x_analysis||^2_B
      + λ_h ||hydrostatic_residual(x')||^2
      + λ_m ||specified_mass_wind_residual(x')||^2
      + λ_q ||specified_moist_constraint_residual(x')||^2
```

The B-weighted norm represents fixed variable scaling or a regularized background covariance. Each residual needs compatible normalization before weighting. Begin with a transparent hydrostatic adjustment and a separate extratropical wind–mass experiment rather than a poorly understood global optimizer. State the geopotential boundary anchor, integration convention, and any mass/moisture constraints.

Select strengths using development dates and inspect observation departures. Do not minimize total ageostrophic wind or divergence globally. Report a sensitivity curve: forecast improvement versus amount of analysis modification. Gradient-based IC optimization is optional and must not use future verification fields.

### 8.2 Statistical and smoothing controls

Estimate seasonal, location-, variable-, and level-dependent source biases from a separate training interval. Start with a simple mean correction; more complex covariance adjustment is a later extension. Freeze all correction parameters before evaluation. Diagnose how statistical correction itself changes balance.

Match the smoothing control to the physical treatment's approximate spectral attenuation and perturbation magnitude. If exact matching is not possible, report both and use several filter strengths. A treatment that lowers RMSE solely by suppressing variability is not sufficient evidence for a balance mechanism.

### 8.3 Initialization from preceding analyses: an ML analogue of replay

Use a 24–48-hour window ending at launch as the first candidate. At each supported forecast step:

```text
predicted state at t_k = model(previous prepared input state/history)
prepared state at t_k = predicted state + α_k W_k (analysis at t_k - predicted state)
```

Here W_k selects or weights variables/scales and α_k controls the nudging fraction. Construct the required history sequentially, always using already prepared states. Stop nudging at launch and save the entire final input history. Forecast freely from the true launch time onward.

This is discrete nudging, not automatically equivalent to continuous IAU. Relaxing toward an inconsistent analysis can introduce new imbalance. Test α, variable selection, and window length; diagnose all resulting changes. No forecast after launch is fed back as an initialization target. Include the FREE treatment, and label its older information content clearly.

Retrospective reanalyses may themselves assimilate observations from a window extending beyond their nominal timestamps. Thus “no post-launch targets in the algorithm” does not by itself establish operational real-time feasibility. A later operational demonstration must respect actual data availability and assimilation windows.

### 8.4 Input-history ablation

For models using states at t0-6 h and t0, hold x(t0) fixed and compare coherent versus deliberately mismatched preceding states, using realistic amplitudes. In a separate experiment, modify both times coherently. The history-only comparison isolates the influence of inferred temporal evolution. It is not applicable to standard single-state Pangu inference.

## 9. Experiment D: what NeuralGCM changes before integration

Save the external input, encoded internal state, and immediately decoded state before advancing. Quantify the changes in temperature, winds, humidity, pressure, and physical residuals. Then track their evolution during the first steps.

Use the standard encoder in all primary comparisons. Encoder bypass or replacement is a separate architectural ablation that may violate the pretrained model's intended state representation. Do not assume encode/decode is a perfect balance projection. Its documentation describes a lossy mapping between pressure-level and internal coordinates. [NeuralGCM interface](https://neuralgcm.readthedocs.io/en/latest/deepdive_into_models.html).

For the stochastic checkpoint, pair random seeds where supported to reduce comparison noise, and report ensemble uncertainty. A deterministic checkpoint is simpler for the initial mechanism tests.

## 10. Diagnostics, forecast skill, and attribution

### 10.1 Initial conditions and first 48 hours

- Observation departures at launch, separately for each variable, level, and region.
- Hydrostatic consistency, with residual definition and vertical discretization documented.
- Extratropical wind–mass diagnostics; do not interpret all ageostrophic flow as imbalance.
- Divergence/vorticity and kinetic-energy spectra, and their changes with lead time.
- Increment magnitude and spatial structure at each model step.
- Humidity, relative humidity, lapse rate, column water, and moisture-flux convergence.
- Surface-pressure tendency only where true surface pressure is available; mean sea-level pressure is a different diagnostic.
- Model projection/adjustment: how far the first forecast step moves from the input relative to the observed subsequent evolution.

Absence of visible spikes in 6-hourly ML output does not establish absence of fast adjustment. Use Pangu short-step experiments or appropriate NeuralGCM sampling for selected cases. Keep their integration schedules fixed across treatments.

### 10.2 Moisture and tropical diagnostics

Where fields support it, use:

```text
W  = (1/g) ∫ q dp
MC = -(1/g) ∫ ∇h · (q vh) dp
P  = E + MC - ∂W/∂t
```

Evaluate terms at consistent time intervals and report a closure residual. Pressure-level truncation, below-ground treatment, and finite differencing introduce error. Without evaporation and precipitation outputs, evaluate W and MC as partial diagnostics; do not claim a closed water budget. Full moist-static-energy attribution also requires the relevant radiative and surface-flux terms.

Reuse the GEOS project's tropical regions and event-composite logic, but define thresholds on a common observed or fixed reference climatology. Treatment-specific thresholds can conceal changes in variance. Test sensitivity to event thresholds and distinguish removal of false extremes from suppression of real extremes.

For S2S work, evaluate tropical propagation and MJO-related circulation. Use RMM only when the required comparable OLR and wind fields are available. Do not concatenate unrelated short forecasts to form a Wheeler–Kiladis spectrum; obtain sufficiently long, consistently sampled trajectories or use suitable shorter-window diagnostics with explicit resolution limits.

### 10.3 Forecast skill

- Days 1–2: adjustment and near-term skill.
- Days 3–10: area-weighted RMSE, anomaly correlation, bias, and regional skill for Z500, temperature, winds, humidity, and available surface fields.
- Rainfall and extremes: accumulation errors, event frequency, spatial displacement, quantiles, and retained variance; only for models producing the relevant field.
- Ensembles: CRPS, reliability, spread–error relationships, and event probabilities.
- Weeks 2–6: weekly anomalies and tropical/SST evolution with an S2S-capable model, rather than an unsupported long extension of a weather checkpoint.

Use identical verification masks, climatology periods, valid times, and observational operators across treatments. Report both native-scale process diagnostics and skill on a common verification grid. Regrid model outputs for verification without pretending that this changes the model's native information content.

### 10.4 Statistical design

Pair forecast differences by initialization date. Use block bootstrap confidence intervals across dates or weather episodes, accounting for overlapping forecasts and repeated events. Grid cells and time steps are not independent replicates. Prespecify a small set of primary metrics and treat broad maps and many-variable comparisons as secondary analyses with appropriate field-significance or multiple-testing treatment.

The strongest attribution combines: a controlled intervention, matching of initial error and spectra, a dose response, the expected early physical response, later independent skill improvement, and replication in another model. Correlation or mediation analysis alone does not prove the mechanism.

## 11. Coupled/S2S extension

After the atmospheric result is understood, test four atmosphere–SST combinations with DLESyM or a suitable joint forecast system:

| | SST from raw preparation | SST from joint adjusted preparation |
|---|---|---|
| Atmosphere from raw preparation | A0O0 | A0O1 |
| Atmosphere from joint adjusted preparation | A1O0 | A1O1 |

Define the adjusted pair using a preceding coupled preparation experiment and preserve each model's required histories. Swapping components then tests whether their joint consistency matters. Mixed pairs are deliberate interventions, not assumed physically plausible initial states.

For an error metric E, evaluate the interaction E11 - E10 - E01 + E00 with uncertainty. A negative interaction suggests extra error reduction from the joint treatment, subject to the limitations of the adjustment procedure and matching of initial accuracy.

Use FuXi-S2S for a complementary daily ensemble experiment emphasizing precipitation, SST, and tropical propagation. Daily averages cannot resolve the original subdaily shocks. DLESyM's SST-only ocean cannot test ocean stratification, currents, or subsurface heat-content balance. Avoid claiming the same resolved ocean mechanism as in GEOS–MITgcm.

For atmospheric NeuralGCM runs, use the same persisted or independently forecast SST/ice boundary policy in every atmospheric treatment. Future observed SST is a separately labeled perfect-boundary sensitivity experiment, not the operational baseline.

## 12. Implementation layout and run records

Suggested future repository structure; this plan does not create these components:

```text
MLanalysis/
  PROJECT_PLAN.md
  configs/                 # Model, data, experiment, and verification settings
  manifests/               # Checkpoints, cases, data provenance, completed runs
  src/
    data/                  # Download/subset, regrid, unit and time handling
    adapters/              # GraphCast, Pangu, Aurora, NeuralGCM interfaces
    initialization/        # Physical, statistical, smoothing, history treatments
    diagnostics/           # Balance, spectra, moisture, information retention
    verification/          # Paired skill, observations, uncertainty
  scripts/                 # Reproducible stage entry points
  notebooks/               # Exploration and figure inspection
  results/                 # Small summaries and publication figures
```

Keep model environments separate because JAX, PyTorch, ONNX, and accelerator dependencies differ. Store large input and forecast archives on suitable project storage, preferably outside a desktop cloud-sync folder. Keep code, configurations, and small summaries in MLanalysis.

Every run manifest should include model release/commit, checkpoint hash and training cutoff, source data version, actual launch/valid times, input-history timestamps, interpolation settings, static-field versions, treatment and parameters, seed, stepping schedule, boundary conditions, runtime/memory, and output locations. Hash or otherwise identify prepared IC files so every forecast is traceable to its exact input.

Conceptual execution sequence:

```python
# Pseudocode: these functions are proposed project interfaces, not existing APIs.
for case in cases:
    raw_history = load_physical_analysis_history(case)
    for treatment in treatments:
        prepared_history = prepare_initial_conditions(raw_history, treatment)
        save_input_diagnostics(prepared_history, raw_history)
        for model in selected_models:
            model_input = adapt_to_checkpoint(prepared_history, model)
            verify_input_contract(model_input, model)
            forecast = run_frozen_checkpoint(model_input, model, case)
            save_forecast_and_manifest(forecast, case, treatment, model)
            compute_paired_verification(forecast, case, treatment, model)
```

Model-dependent history preparation may require a separate prepared history for each forecasting model. Cross-model transfer is then a distinct experiment: initialize model B from a state prepared with model A. Do not mix those two designs in a single treatment label.

## 13. Staged work plan and feasible pilot

### Stage 0: establish one reproducible forecast

1. Confirm GPU allocation, storage, and data credentials where needed.
2. Pin one checkpoint and reproduce its official sample.
3. Run one ERA5-initialized 10-day forecast and inspect fields, units, and timings.
4. Benchmark wall time, peak memory, and output volume, including compilation separately.
5. Implement one transparent perturbation and verify that it reaches the model input as intended.

### Stage 1: pilot mechanism study

Use 24 starts spanning several seasons and regimes, with two models and four core treatments: RAW, HIST, FREE, and SMOOTH. This gives **192 forecasts** for one frozen HIST setting, before gain/window sensitivity runs and other controls. RAW versus HIST is the primary same-ERA5 comparison. Add PHYS and STAT for attribution once the core experiment is working; add IAU-like integration only after its update scheme is validated. Begin with a few cases before completing this matrix. Save dense first-48-hour diagnostics and selected later fields through day 10.

Include both midlatitude and tropical cases. Select dates without examining treatment forecast outcomes. Add the explicit history ablation when the pipeline is stable. Run the legacy bridge when the GEOS global IC archive is available.

A pilot is successful if the inputs are reproducible, interventions measurably alter the intended property, and treatment differences exceed numerical artifacts. A null effect is informative; it is not a reason to keep searching parameters on the evaluation set.

### Stage 2: confirm and generalize

Expand provisionally to 100–200 starts, with final sample size informed by pilot variance and the effect size of scientific interest. Add Aurora and/or Pangu, freeze treatment parameters, and use held-out dates. Include observation-based verification and the controls needed by the pilot's competing explanations.

### Stage 3: S2S and coupled test

Proceed only after an atmospheric mechanism is sufficiently clear. Use ensembles and enough dates to characterize sampling uncertainty. Separate the problem of developing a new coupled preparation method from evaluating an already defined one.

No calendar duration or GPU-hour total is committed before the first benchmark. Forecast generation may be fast while data preparation, output storage, compilation, and observation collocation dominate the workflow.

## 14. Expected scientific products

Candidate main figures:

1. Initial observation error versus forecast error, showing the trade-off for each treatment.
2. Controlled perturbation response: physical residuals and error growth versus lead time and perturbation amplitude.
3. Physical adjustment versus smoothing and statistical controls at matched scales.
4. Tropical moisture/convergence response and subsequent observed forecast errors.
5. Transfer across pure-ML, foundation, and hybrid models.
6. Optional atmosphere–SST interaction and weeks 2–6 skill.

The primary paper should establish a mechanism and a reproducible initialization intervention. It should not be only a ranking of forecast models initialized from several datasets.

Three publishable outcomes are possible: physical balance transfers across architectures; training-data or history compatibility dominates in purely learned models; or the model suppresses imbalance while also erasing useful increments. Each result changes how analyses should be prepared or how future models should be trained.

## 15. Relation to existing literature and novelty

- [Hakim and Masanam, dynamical tests of Pangu](https://arxiv.org/abs/2309.10867): idealized perturbations and dynamical adjustment are established research directions. The proposed advance is attribution under realistic analysis uncertainty and connection to independent forecast value.
- [Bonavita, physical limitations of ML weather models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL107377): motivates evaluating physical fidelity and retained variability alongside conventional forecast error.
- [Vonich and Hakim, optimal ICs for the Pacific Northwest heatwave](https://doi.org/10.1029/2024GL110651): optimizing against a future target demonstrates potential forecast sensitivity. The practical method here must use only initialization information, with evaluation data kept separate.
- [GraphCast adaptation to Canadian analyses](https://arxiv.org/abs/2408.14587): shows why training-data compatibility is a serious alternative explanation. Fine-tuning should be a later experiment with its own data split, not an uncontrolled change during IC comparisons.

Before making a priority or “first demonstration” claim, perform a focused literature update on ML initialization, balance-aware perturbations, and analysis-source adaptation. This proposal identifies a testable contribution; it does not establish that every component is novel.

## 16. Immediate implementation decisions

The first execution task should resolve four practical items: available accelerator and memory; exact checkpoint; access to global GEOS replay states and histories; and a small date list outside checkpoint training where required. These do not prevent starting the ERA5-based controlled experiment.

The recommended first scientific deliverable is a paired same-ERA5 experiment showing whether model preparation by nudging improves the free day-3-to-day-10 forecast compared with direct insertion, followed by tests distinguishing physical consistency from smoothing and model adaptation. Controlled perturbations explain the mechanism. This supplies a defensible foundation for the broader foundation-model and coupled study.
