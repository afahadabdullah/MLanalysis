# Getting GraphCast_small running on NCCS Prism (do this before any experiment)

Version 17 Sep 2026. Commands from the NCCS "Using Prism" page; verify against current NCCS docs, since details change.

## 0. What Prism gives you

| Node set | Hardware | Arch | Use for this project |
|---|---|---|---|
| `gpu[001-022]` | 40 cores, 4× V100 32 GB | x86 | **Default.** Plenty for GraphCast_small (1°, 13 levels) |
| `gpu100` (DGX, `-p dgx`) | 128 cores, 8× A100 40 GB | x86 | Faster, and needed only if you later try 0.25° GraphCast |
| `gh[001-062]` (`-p grace`) | 72 cores, 1× H100 96 GB | **ARM (aarch64)** | Avoid at first: JAX/CUDA wheels for ARM are a separate build. Use only via an NVIDIA ARM container |

Account required (NCCS new-user process). Working storage: your ADAPT/`nobackup` area for code and data; `/lscratch/$USER` on the node for scratch (deleted after the job).

## 1. Log in
```bash
ssh <user>@adapt.nccs.nasa.gov       # or adaptlogin.nccs.nasa.gov
ssh gpulogin1                        # Prism GPU login node
```

## 2. Pick a work area and check quotas
```bash
# confirm your actual paths with NCCS docs / `showquota`
export PROJ=/explore/nobackup/people/$USER/mlanalysis    # verify this path exists for your account
mkdir -p $PROJ/{code,data,runs,envs}
mkdir -p /lscratch/$USER
```
Keep the git repo and code in `$PROJ/code`, inputs in `$PROJ/data`, and forecasts in `$PROJ/runs`. Never put large data in `$HOME`.

## 3. Interactive GPU session (for setup and the first runs)
```bash
salloc -G1 -t 120 -n1 -c8 --mem=64G          # a V100 node
# or, for an A100:
salloc -G1 -t 120 -p dgx -c16 --mem=100G
nvidia-smi                                    # confirm the GPU is visible
```

## 4. Build the Python environment
```bash
module load miniforge
conda create -p $PROJ/envs/gc python=3.11 -y
conda activate $PROJ/envs/gc

# JAX with CUDA 12 (V100 = sm_70, A100 = sm_80; both supported)
pip install --upgrade "jax[cuda12]"
python -c "import jax; print(jax.devices())"   # must list a CUDA device, not CPU

# GraphCast and its dependencies
pip install dm-haiku chex jraph trimesh xarray netcdf4 zarr gcsfs dask cartopy pandas scipy
pip install git+https://github.com/google-deepmind/graphcast.git
```
If a package fails to build, use an NVIDIA JAX container instead:
```bash
export SINGULARITY_TMPDIR=/lscratch/$USER SINGULARITY_CACHEDIR=/lscratch/$USER
singularity exec --nv -B $PROJ jax.sif python -c "import jax; print(jax.devices())"
```

**Network note:** compute nodes may have no outbound internet. Do every download (pip, model weights, ERA5/MERRA-2 data) on the login node, then run offline. If pip is blocked even there, ask NCCS about the proxy settings.

## 5. Get the model weights and sample data
From the public `dm_graphcast` bucket (login node):
```bash
cd $PROJ/data && mkdir -p params stats sample
BASE=https://storage.googleapis.com/dm_graphcast
# exact filenames: list the bucket first, they include the config in the name
pip install gsutil && gsutil ls gs://dm_graphcast/params/ | grep -i small
wget -P params "$BASE/params/<GraphCast_small ... .npz>"
for f in diffs_stddev_by_level.nc mean_by_level.nc stddev_by_level.nc; do wget -P stats "$BASE/stats/$f"; done
gsutil ls gs://dm_graphcast/dataset/ | grep -i "res-1.0" | head      # a matching 1° sample
```
Record the checkpoint filename, its SHA256, and the download date in `runs.csv`. This is the pinned checkpoint for the whole project.

## 6. Reproduce the official example (the real milestone)
1. Run the GraphCast demo notebook logic as a script, on the downloaded 1° sample, for a few steps.
2. Save the output and confirm it matches the packaged example within numerical tolerance.
3. **Only then** feed your own ERA5-derived inputs, and check that your adapter reproduces the same forecast from equivalent physical data.

## 7. Benchmark before planning any run count
Record, for one 5-day forecast (20 steps):
- compile time vs. run time (JAX compiles on the first call; time the second call separately),
- peak GPU memory (`nvidia-smi --query-gpu=memory.used --format=csv -l 5`),
- output size per forecast,
- V100 vs. A100 wall time.

Roughly 2,000 five-day forecasts are planned, so per-forecast cost decides whether they run one at a time or batched.

## 8. Batch template
```bash
#!/bin/bash
#SBATCH --job-name=gc_run
#SBATCH -G1
#SBATCH -c8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
module load miniforge
conda activate /explore/nobackup/people/$USER/mlanalysis/envs/gc
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # avoids JAX grabbing the whole GPU
cd $PROJ/code
srun python run.py --config config.yaml --dates $DATE_LIST --treatments E,M,E-DIR,E-BAL,E-NUD
```
Notes:
- Keep one process per GPU. Loop dates *inside* the process so the model compiles once.
- Write to `/lscratch/$USER` during the job, then copy the small output to `$PROJ/runs` at the end.
- Use job arrays (`#SBATCH -a 0-9`) across date blocks, not across single forecasts.

## 9. Checklist before Experiment S0
- [ ] `jax.devices()` shows a GPU on a compute node
- [ ] Official sample reproduced, tolerance recorded
- [ ] Checkpoint filename + hash + date in `runs.csv`
- [ ] Your ERA5 adapter reproduces that same forecast
- [ ] Timing and memory recorded for V100 and A100
- [ ] A 5-day forecast runs end-to-end from a batch script
- [ ] Data download path decided (login node vs. proxy)

Source: NCCS, [Using Prism](https://www.nccs.nasa.gov/using-prism/) and [Prism GPU Cluster](https://www.nccs.nasa.gov/systems/ADAPT/Prism).
