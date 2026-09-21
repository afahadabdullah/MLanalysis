#!/bin/bash
module load miniforge 2>/dev/null || true
source activate /home/afahad/project/MLanalysis/envs/gc 2>/dev/null || conda activate /home/afahad/project/MLanalysis/envs/gc
export PYTHONNOUSERSITE=1
export PROJ=/home/afahad/project/MLanalysis
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Environment active: $(which python)"
