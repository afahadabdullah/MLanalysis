# GraphCast Quick-Run Guide

Step-by-step instructions for running GraphCast on NCCS Prism — from activating
the environment through running a forecast and inspecting the output.

> [!NOTE]
> All paths assume the project root is `/home/afahad/project/MLanalysis` (already
> on `nobackup`, symlinked into `$HOME`). If your layout differs, set `export PROJ=<your path>`.

---

## 1. Get onto a GPU node

```bash
# From your workstation
ssh adapt.nccs.nasa.gov
ssh gpulogin1                                # Prism GPU login node

# Request an interactive GPU session
salloc -G1 -t 120 -c8 --mem=64G             # V100 (default)
# or for an A100:
# salloc -G1 -t 120 -p dgx -c16 --mem=100G
```

## 2. Activate the environment

```bash
module load miniforge
source activate /home/afahad/project/MLanalysis/envs/gc
export PYTHONNOUSERSITE=1                    # prevent ~/.local packages from leaking in
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # stop JAX from grabbing the entire GPU
cd /home/afahad/project/MLanalysis
```

**Quick sanity check:**

```bash
nvidia-smi                                   # verify the GPU is visible
python -c "import jax; print(jax.devices())" # must show [CudaDevice(id=0)]
```

> [!IMPORTANT]
> Always set `PYTHONNOUSERSITE=1` before running any script. Without it, stale
> packages in `~/.local/lib/python3.12/` can shadow the conda env and cause
> import errors (particularly `xarray`).

## 3. Run the test forecast (24h / 4 steps)

```bash
python scripts/test_forecast.py                  # runs 4 steps (24h) by default
# or specify a different horizon:
# python scripts/test_forecast.py --steps 2      # 2 steps (12h)
```

This script:

1. Loads the **GraphCast_small** checkpoint (1° resolution, 13 pressure levels).
2. Loads normalization statistics (`diffs_stddev`, `mean`, `stddev` by level).
3. Loads the official ERA5 sample dataset (2022-01-01, 4 steps).
4. Builds the wrapped predictor and JIT-compiles on the GPU.
5. Runs a **4-step (24 h) autoregressive rollout** (+6h, +12h, +18h, +24h).
6. Validates that predictions contain no NaNs and prints physical sanity checks.
7. Saves outputs to `runs/`:

| File | Contents |
|------|----------|
| `runs/predictions.nc` | GraphCast forecast (xarray Dataset, 4 steps) |
| `runs/era5_truth.nc` | ERA5 target fields (for verification) |
| `runs/test_forecast_summary.txt` | Timing, device info, basic stats |

**Expected output:**

```
[5/5] Running 4-step (24h) forecast rollout on GPU ...
      Forecast completed in ~11 seconds (includes initial JIT compile).

  ✓ '2m_temperature' predicted shape: (4, 1, 181, 360)
  ✓ Mean 2m temperature: 276.81 K
  ✓ Min/Max: 218.45 K / 319.12 K
  ✓ NaN count: 0
  ✓ SUCCESS: No NaNs in predicted output!
```

> [!TIP]
> The first run is slow (~10 s) because JAX JIT-compiles the graph. Subsequent
> forecasts in the same process reuse the compiled graph and finish in ~1-2 seconds.

## 4. Generate diagnostic plots

After `test_forecast.py` has saved `runs/predictions.nc` and `runs/era5_truth.nc`:

```bash
python scripts/plot_diagnostics.py
```

Plots are saved to `runs/diagnostics/`.

### What you get

| Plot file | Description |
|-----------|-------------|
| `rmse_vs_lead_time.png` | **Forecast error growth vs. lead time** (Z500, T850, T2m, 10m wind) |
| `t2m_step*.png` | 2 m temperature — Forecast / ERA5 / Error (steps 0..3) |
| `z500_step*.png` | 500 hPa geopotential height |
| `t850_step*.png` | 850 hPa temperature |
| `q700_step*.png` | 700 hPa specific humidity |
| `u250_step*.png` | 250 hPa u-wind (jet stream level) |
| `wind_speed_10m_step*.png` | 10 m wind speed composite |
| `mslp_step*.png` | Mean sea-level pressure |
| `rmse_by_variable.png` | Global area-weighted RMSE bar chart |
| `rmse_by_pressure_level.png` | RMSE vs. pressure level profile |
| `zonal_mean_temperature_step*.png` | Zonal-mean T cross-section (lat × pressure) |

---

## 5. Synthetic Anomaly & Increment Retention Experiment (Exp 2 / S0)

Tests the core scientific hypothesis from `PROJECT_PLAN_LEAN.md`:
> *"Does GraphCast retain an inserted surface temperature increment ($\Delta T_{2m}$), or does initialization shock wipe it out in the early forecast hours?"*

```bash
python scripts/test_retention.py
```

Options:
```bash
# Custom amplitude or location:
python scripts/test_retention.py --amplitude 2.0 --lat 38.0 --lon 265.0 --steps 4
```

### What it produces in `runs/retention/`:

| File / Plot | What it shows |
|---|---|
| `retention_decay_curve.png` | **Retention metric $R(t) = \frac{\langle \Delta F(t), \Delta X \rangle}{\|\Delta X\|^2}$** over lead time (0h to 24h) |
| `retention_spatial_evolution.png` | 5-panel evolution ($t=0\text{h}$, $+6\text{h}$, $+12\text{h}$, $+18\text{h}$, $+24\text{h}$) showing anomaly advection and dispersion |
| `vertical_response_profile.png` | Column temperature response $\Delta T(p)$ showing boundary layer coupling |
| `surface_wind_mslp_coupling.png` | Induced geostrophic/thermal wind and $\Delta\text{MSLP}$ adjustments |
| `retention_experiment.nc` | Full 4D difference dataset ($\Delta = F_{\text{pert}} - F_{\text{base}}$) |
| `retention_summary.txt` | Quantitative retention table by lead hour |

---

## 6. Balanced Increment Insertion Experiment (E-BAL vs. E-DIR)

Tests Hypothesis H3 from `PROJECT_PLAN_LEAN.md`:
> *"Does spreading a surface increment vertically into 1000/925/850 hPa and balancing geopotential hypsometrically reduce initialization shock and increase retention compared to direct surface insertion?"*

```bash
python scripts/test_balanced_insertion.py
```

### What it produces in `runs/balanced/`:

| File / Plot | Description |
|---|---|
| `retention_comparison_dir_vs_bal.png` | Direct vs. Balanced retention decay curves ($R_{\text{dir}}$ vs $R_{\text{bal}}$) |
| `shock_and_rmse_comparison.png` | (A) Initialization shock / first-step jump comparison, (B) CONUS RMSE error curve |
| `vertical_cross_section_dir_vs_bal.png` | Side-by-side vertical cross section ($p$ vs lon) of column warming |
| `spatial_comparison_24h_dir_vs_bal.png` | Day-1 (+24h) spatial comparison: $E\text{-DIR}$ vs $E\text{-BAL}$ vs difference |
| `balanced_comparison.nc` | NetCDF difference arrays for all 3 runs |
| `balanced_experiment_summary.txt` | Quantitative metrics, shock jump values, and lapse rate anomalies |

---

## 7. Submit a batch job



For longer runs, use the SLURM batch script instead of an interactive session:

```bash
sbatch scripts/gpu_job.sh
```

Monitor with:

```bash
squeue -u $USER                # check queue status
tail -f gc_run_<jobid>.out     # watch live output
```

## 6. One-liner: full pipeline

Copy-paste this block to run everything end-to-end on a GPU node:

```bash
module load miniforge && source activate /home/afahad/project/MLanalysis/envs/gc
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/afahad/project/MLanalysis
python scripts/test_forecast.py && python scripts/plot_diagnostics.py
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'graphcast'`

The upstream package was renamed to `weathernext`. The test script handles this
automatically via a fallback import. If you still see this error, make sure you
activated the correct conda env:

```bash
which python   # should show .../envs/gc/bin/python
```

### `ValueError: argument name 'arithmetic_compat' is not in the set of valid options`

A stale `xarray` in `~/.local/` is shadowing the conda env version. Fix:

```bash
export PYTHONNOUSERSITE=1       # quick fix
# or permanent fix:
rm -rf ~/.local/lib/python3.12/site-packages/xarray*
```

### `No GPU device detected`

You are on the login node (CPU-only). Get a GPU with `salloc -G1 ...` first.

### JIT compilation warnings (`slow_operation_alarm`)

Normal on V100s — XLA constant-folding takes a few seconds on the first call.
These warnings are harmless and do not appear on subsequent forecasts.

### `pip install` or `wget` fails on a compute node

Compute nodes (`gpu004`, etc.) have **no outbound internet**. Do all downloads
and pip installs on `gpulogin1`, then `salloc` to a compute node.

---

## File reference

```text
scripts/
  setup_env.sh         # Create the conda env (run once on login node)
  download_data.sh     # Download weights, stats, sample data (login node)
  test_forecast.py     # Run & validate a 12 h forecast (GPU node)
  plot_diagnostics.py  # Generate verification plots from saved outputs
  gpu_job.sh           # SLURM batch template
```
