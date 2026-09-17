#!/usr/bin/env python3
"""Run a test forecast using GraphCast_small on the official sample dataset."""

import os
import sys
import time
import functools
import dataclasses
import numpy as np
import xarray as xr

print("=" * 60)
print("GraphCast Test Forecast Verification")
print("=" * 60)

# Ensure JAX memory preallocation is controlled
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
print(f"JAX version: {jax.__version__}")
devices = jax.devices()
print(f"JAX devices available: {devices}")
if not any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices):
    print("WARNING: No GPU device detected. Forecast will run on CPU (very slow).")

import haiku as hk
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)

PROJ = os.environ.get("PROJ", "/home/afahad/project/MLanalysis")
DATA_DIR = os.path.join(PROJ, "data")
PARAMS_FILE = os.path.join(
    DATA_DIR,
    "params",
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
)
STATS_DIR = os.path.join(DATA_DIR, "stats")
SAMPLE_FILE = os.path.join(
    DATA_DIR,
    "sample",
    "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
)

# 1. Check prerequisites
missing = []
for p, label in [
    (PARAMS_FILE, "Model checkpoint (.npz)"),
    (os.path.join(STATS_DIR, "diffs_stddev_by_level.nc"), "diffs_stddev_by_level.nc"),
    (os.path.join(STATS_DIR, "mean_by_level.nc"), "mean_by_level.nc"),
    (os.path.join(STATS_DIR, "stddev_by_level.nc"), "stddev_by_level.nc"),
    (SAMPLE_FILE, "Sample dataset (.nc)"),
]:
    if not os.path.exists(p):
        missing.append(f"  - {label}: {p}")

if missing:
    print("\nERROR: Missing required data files:")
    print("\n".join(missing))
    print("\nPlease run the download script first:")
    print("  bash scripts/download_data.sh\n")
    sys.exit(1)

# 2. Load model checkpoint
print(f"\n[1/5] Loading checkpoint: {os.path.basename(PARAMS_FILE)} ...")
t0 = time.time()
with open(PARAMS_FILE, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config
print(f"      Loaded checkpoint in {time.time() - t0:.2f}s")
print(f"      Resolution: {model_config.resolution} deg | Levels: {len(task_config.pressure_levels)}")

# 3. Load normalization statistics
print("\n[2/5] Loading normalization statistics ...")
with open(os.path.join(STATS_DIR, "diffs_stddev_by_level.nc"), "rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open(os.path.join(STATS_DIR, "mean_by_level.nc"), "rb") as f:
    mean_by_level = xr.load_dataset(f).compute()
with open(os.path.join(STATS_DIR, "stddev_by_level.nc"), "rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()
print("      Stats loaded successfully.")

# 4. Load sample batch and extract inputs, targets, forcings
print(f"\n[3/5] Loading sample dataset: {os.path.basename(SAMPLE_FILE)} ...")
with open(SAMPLE_FILE, "rb") as f:
    example_batch = xr.load_dataset(f).compute()

# We test a 2-step forecast (2 x 6h = 12 hours)
STEPS = 2
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{STEPS * 6}h"),
    **dataclasses.asdict(task_config),
)
print(f"      Extracted inputs: batch={eval_inputs.dims.get('batch')}, time={eval_inputs.dims.get('time')}")
print(f"      Target lead times: {eval_targets.coords['time'].values}")

# 5. Build predictor and run forward rollout
print("\n[4/5] Constructing GraphCast predictor & JIT compiling ...")

def construct_wrapped_graphcast(m_cfg, t_cfg):
    predictor = graphcast.GraphCast(m_cfg, t_cfg)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(m_cfg, t_cfg, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(m_cfg, t_cfg)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

with_configs = functools.partial(
    run_forward.apply,
    m_cfg=model_config,
    t_cfg=task_config,
)

run_forward_jitted = jax.jit(
    functools.partial(with_configs, params=params, state=state)
)

print(f"\n[5/5] Running {STEPS}-step ({STEPS * 6}h) forecast rollout on GPU ...")
t_start = time.time()

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
)

t_end = time.time()
total_time = t_end - t_start

print(f"      Forecast completed in {total_time:.2f} seconds (includes initial JIT compile).")

# Verification checks
print("\n" + "=" * 60)
print("Forecast Validation Results:")
print("=" * 60)

if "2m_temperature" in predictions:
    t2m = predictions["2m_temperature"].values
    nan_count = np.isnan(t2m).sum()
    print(f"  ✓ '2m_temperature' predicted shape: {t2m.shape}")
    print(f"  ✓ Mean 2m temperature: {np.nanmean(t2m):.2f} K")
    print(f"  ✓ Min/Max: {np.nanmin(t2m):.2f} K / {np.nanmax(t2m):.2f} K")
    print(f"  ✓ NaN count: {nan_count}")
    if nan_count == 0:
        print("  ✓ SUCCESS: No NaNs in predicted output!")
    else:
        print("  ✗ WARNING: Unexpected NaNs detected in predictions.")
else:
    print(f"  ✓ Predicted variables: {list(predictions.data_vars.keys())}")

out_dir = os.path.join(PROJ, "runs")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "test_forecast_summary.txt")
with open(out_file, "w") as f:
    f.write(f"GraphCast_small test forecast completed successfully.\n")
    f.write(f"JAX Device: {devices}\n")
    f.write(f"Forecast Steps: {STEPS} ({STEPS * 6}h)\n")
    f.write(f"Elapsed Time: {total_time:.2f} s\n")
    if "2m_temperature" in predictions:
        f.write(f"2m_temperature Mean: {np.nanmean(t2m):.2f} K\n")

print(f"\nSummary logged to: {out_file}")
print("GraphCast_small is ready for scientific experiments!")
