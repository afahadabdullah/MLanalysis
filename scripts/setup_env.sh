#!/bin/bash
# Build the GraphCast environment INSIDE the repo.
# Run on the Prism GPU login node (internet), not a compute node:
#   ssh adapt.nccs.nasa.gov -> ssh gpulogin1 -> cd ~/project/MLanalysis && bash scripts/setup_env.sh
set -euo pipefail

PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
echo "Project root: $PROJ"

# keep every cache inside the repo so nothing lands in $HOME
export CONDA_PKGS_DIRS="$PROJ/.conda_pkgs"
export PIP_CACHE_DIR="$PROJ/.pip_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$PROJ"/{data,runs,results,configs}
mkdir -p "$PROJ"/data/{params,stats,sample,era5,merra2,obs}

module load miniforge
conda create -p "$PROJ/envs/gc" python=3.11 -y
# shellcheck disable=SC1091
source activate "$PROJ/envs/gc"

python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"
python -m pip install dm-haiku chex jraph trimesh dm-tree \
    xarray netcdf4 h5netcdf zarr gcsfs dask pandas scipy matplotlib cartopy \
    xesmf esmpy pyarrow tqdm pyyaml
python -m pip install git+https://github.com/google-deepmind/graphcast.git

echo
echo "JAX devices seen on THIS node (login node = CPU only, that is expected):"
python -c "import jax; print(jax.devices())"
echo
echo "Done. Activate later with:"
echo "  module load miniforge && source activate $PROJ/envs/gc"
