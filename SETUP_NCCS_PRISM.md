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

## 2. Everything lives in the repo

The project root is `/home/afahad/project/MLanalysis` (already hosted on `nobackup` and linked into `$HOME`):

```bash
export PROJ=/home/afahad/project/MLanalysis
cd $PROJ
```

Layout (created by `scripts/setup_env.sh`):

```text
MLanalysis/
  envs/gc/                   conda environment (git-ignored)
  .conda_pkgs/ .pip_cache/   caches, kept inside repo (git-ignored)
  data/params  data/stats    model weights + official example
  data/era5  data/merra2     inputs
  runs/  results/  logs/     outputs (git-ignored)
  configs/  src/  scripts/   code (committed)
```

`.gitignore` excludes `envs/`, caches, `data/`, `runs/`, and logs, so only code and documentation are committed.

**Storage note:** Because `/home/afahad/project/MLanalysis` is physically located on `nobackup`, all environments, large model weights, intermediate arrays, and forecast runs sit directly inside `/home/afahad/project/MLanalysis` without risking `$HOME` quotas. No separate symlinks to `nobackup` are needed.


## 3. Build the environment (login node, needs internet)

```bash
ssh adapt.nccs.nasa.gov     # then: ssh gpulogin1
cd /home/afahad/project/MLanalysis
bash scripts/setup_env.sh
```

The script creates `envs/gc` with Python 3.11, JAX (CUDA 12), GraphCast and the data stack, and keeps all caches inside the repo. Activate it later with:

```bash
module load miniforge
source activate /home/afahad/project/MLanalysis/envs/gc
```

On the login node `jax.devices()` shows CPU only. That is expected.

## 4. Get the weights and the official sample (login node)

```bash
bash scripts/download_data.sh
```

It lists the exact GraphCast_small checkpoint, the three normalization files and the matching 1° / 13-level sample in the public `dm_graphcast` bucket, then prints the `wget` lines to run. Afterwards, pin the checkpoint:

```bash
sha256sum data/params/*.npz | tee data/params/CHECKSUMS.txt
```

Record the filename, hash and date in `runs.csv`. This is the checkpoint for the whole project.

**Network note:** compute nodes (like `gpu004`) may have no outbound internet. Do all downloads — pip, weights, ERA5, MERRA-2, station data — on the login node, then run offline.

## 5. Get a GPU and check it

```bash
salloc -G1 -t 120 -c8 --mem=64G          # V100 node
# or an A100:  salloc -G1 -t 120 -p dgx -c16 --mem=100G
module load miniforge && source activate $PROJ/envs/gc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi
python -c "import jax; print(jax.devices())"   # must list a CUDA device here
```

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
`scripts/gpu_job.sh` is ready to submit (`sbatch scripts/gpu_job.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=gc_run
#SBATCH -G1
#SBATCH -c8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
export PROJ=/home/afahad/project/MLanalysis
module load miniforge
source activate $PROJ/envs/gc
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # avoids JAX grabbing the whole GPU
cd $PROJ
srun python src/run.py --config configs/config.yaml --dates $DATE_LIST --treatments E,M,E-DIR,E-BAL,E-NUD
```
Notes:
- Keep one process per GPU. Loop dates *inside* the process so the model compiles once.
- Write bulky intermediates to `/lscratch/$USER` during the job, then copy the small output into `runs/` at the end.
- Use job arrays (`#SBATCH -a 0-9`) across date blocks, not across single forecasts.

## 9. Checklist before Experiment S0
- [ ] `jax.devices()` shows a GPU on a compute node
- [ ] Official sample reproduced, tolerance recorded
- [ ] Checkpoint filename + hash + date in `runs.csv` (`data/params/CHECKSUMS.txt`)
- [ ] Your ERA5 adapter reproduces that same forecast
- [ ] Timing and memory recorded for V100 and A100
- [ ] A 5-day forecast runs end-to-end from a batch script
- [ ] Data download path decided (login node vs. proxy)

Source: NCCS, [Using Prism](https://www.nccs.nasa.gov/using-prism/) and [Prism GPU Cluster](https://www.nccs.nasa.gov/systems/ADAPT/Prism).
