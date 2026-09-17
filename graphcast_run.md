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

## 3. Run the test forecast

```bash
python scripts/test_forecast.py
```

This script:

1. Loads the **GraphCast_small** checkpoint (1° resolution, 13 pressure levels).
2. Loads normalization statistics (`diffs_stddev`, `mean`, `stddev` by level).
3. Loads the official ERA5 sample dataset (2022-01-01, 4 steps).
4. Builds the wrapped predictor and JIT-compiles on the GPU.
5. Runs a **2-step (12 h) autoregressive rollout**.
6. Validates that predictions contain no NaNs and prints physical sanity checks.
7. Saves outputs to `runs/`:

| File | Contents |
|------|----------|
| `runs/predictions.nc` | GraphCast forecast (xarray Dataset) |
| `runs/era5_truth.nc` | ERA5 target fields (for verification) |
| `runs/test_forecast_summary.txt` | Timing, device info, basic stats |

**Expected output:**

```
[5/5] Running 2-step (12h) forecast rollout on GPU ...
      Forecast completed in ~10 seconds (includes initial JIT compile).

  ✓ '2m_temperature' predicted shape: (2, 1, 181, 360)
  ✓ Mean 2m temperature: 276.76 K
  ✓ Min/Max: 219.60 K / 318.42 K
  ✓ NaN count: 0
  ✓ SUCCESS: No NaNs in predicted output!
```

> [!TIP]
> The first run is slow (~10 s) because JAX JIT-compiles the graph. Subsequent
> forecasts in the same process reuse the compiled graph and finish much faster.

## 4. Generate diagnostic plots

After `test_forecast.py` has saved `runs/predictions.nc` and `runs/era5_truth.nc`:

```bash
python scripts/plot_diagnostics.py
```

Plots are saved to `runs/diagnostics/`.

### What you get

| Plot file | Description |
|-----------|-------------|
| `t2m_step*.png` | 2 m temperature — Forecast / ERA5 / Error |
| `z500_step*.png` | 500 hPa geopotential height |
| `t850_step*.png` | 850 hPa temperature |
| `q700_step*.png` | 700 hPa specific humidity |
| `u250_step*.png` | 250 hPa u-wind (jet stream level) |
| `wind_speed_10m_step*.png` | 10 m wind speed composite |
| `mslp_step*.png` | Mean sea-level pressure |
| `rmse_by_variable.png` | Global area-weighted RMSE bar chart |
| `rmse_by_pressure_level.png` | RMSE vs. pressure level profile |
| `zonal_mean_temperature_step*.png` | Zonal-mean T cross-section (lat × pressure) |

### Useful options

```bash
# Plot only the first lead time (+6 h)
python scripts/plot_diagnostics.py --step 0

# Use custom input files
python scripts/plot_diagnostics.py \
    --predictions runs/my_experiment/predictions.nc \
    --truth runs/my_experiment/era5_truth.nc \
    --outdir runs/my_experiment/diagnostics/

# Higher-resolution figures for publication
python scripts/plot_diagnostics.py --dpi 300
```

## 5. Submit a batch job

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
