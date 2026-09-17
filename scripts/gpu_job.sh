#!/bin/bash
#SBATCH --job-name=gc_run
#SBATCH -G1
#SBATCH -c8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail
PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
module load miniforge
source activate "$PROJ/envs/gc"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PROJ
mkdir -p "$PROJ/logs"
cd "$PROJ"
srun python src/run.py --config configs/config.yaml "$@"
