#!/bin/bash
# Download GraphCast_small weights, normalization stats, and official sample dataset
# into the repo's data/ directory. Run on the LOGIN node (compute nodes may have no internet).
set -euo pipefail
PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
DATA_DIR="$PROJ/data"
BASE="https://storage.googleapis.com/dm_graphcast"

mkdir -p "$DATA_DIR"/{params,stats,sample}

echo "=== [1/3] Downloading GraphCast_small Checkpoint ==="
PARAM_FILE="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
if [ ! -f "$DATA_DIR/params/$PARAM_FILE" ]; then
    echo "Downloading weights (~330 MB) ..."
    wget -c -O "$DATA_DIR/params/$PARAM_FILE" \
      "$BASE/params/GraphCast_small%20-%20ERA5%201979-2015%20-%20resolution%201.0%20-%20pressure%20levels%2013%20-%20mesh%202to5%20-%20precipitation%20input%20and%20output.npz"
else
    echo "Params file already exists: $DATA_DIR/params/$PARAM_FILE"
fi

echo "=== [2/3] Downloading Normalization Statistics ==="
for f in diffs_stddev_by_level.nc mean_by_level.nc stddev_by_level.nc; do
    if [ ! -f "$DATA_DIR/stats/$f" ]; then
        echo "Downloading stats/$f ..."
        wget -c -O "$DATA_DIR/stats/$f" "$BASE/stats/$f"
    else
        echo "Stats file already exists: $DATA_DIR/stats/$f"
    fi
done

echo "=== [3/3] Downloading Sample Dataset (1.0 deg, 13 levels) ==="
SAMPLE_FILE="source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
if [ ! -f "$DATA_DIR/sample/$SAMPLE_FILE" ]; then
    echo "Downloading sample dataset (~130 MB) ..."
    wget -c -O "$DATA_DIR/sample/$SAMPLE_FILE" "$BASE/dataset/$SAMPLE_FILE"
else
    echo "Sample file already exists: $DATA_DIR/sample/$SAMPLE_FILE"
fi


echo "=== Generating Checksums ==="
sha256sum "$DATA_DIR"/params/*.npz | tee "$DATA_DIR/params/CHECKSUMS.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >> "$DATA_DIR/params/CHECKSUMS.txt"

echo
echo "=== All data downloaded and verified successfully! ==="
echo "You can now run a test forecast on a GPU node:"
echo "  python scripts/test_forecast.py"
