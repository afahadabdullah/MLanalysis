#!/bin/bash
#SBATCH --job-name=gc_025_large
#SBATCH -p dgx
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out

set -euo pipefail
PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
cd "$PROJ"
mkdir -p logs

module load miniforge 2>/dev/null || true
source activate "$PROJ/envs/gc" 2>/dev/null || conda activate "$PROJ/envs/gc"
export PYTHONNOUSERSITE=1
export PROJ
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

echo "Running on host: $(hostname)"
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/exp_main_real_obs.py \
    --t0 2018-01-15T12:00 \
    --obs-source isd \
    --model large \
    --long-nud 72 \
    --nud-windows 6 \
    --nud-types DIR \
    --arms DIR-1F,NUD6-DIR,REPLAY72,HYB72-DIR
