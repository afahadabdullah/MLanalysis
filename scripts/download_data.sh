#!/bin/bash
# Download GraphCast_small weights, normalization stats, and one official sample
# into the repo's data/ directory. Run on the LOGIN node (compute nodes may have no internet).
set -euo pipefail
PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
cd "$PROJ/data"
BASE="https://storage.googleapis.com/dm_graphcast"

echo "== listing available checkpoints (pick the GraphCast_small one) =="
python - <<'PY'
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
for f in fs.ls('dm_graphcast/params'):
    if 'small' in f.lower():
        print(f)
print('--- stats ---')
print(*fs.ls('dm_graphcast/stats'), sep='\n')
print('--- sample datasets (1.0 deg, 13 levels) ---')
for f in fs.ls('dm_graphcast/dataset'):
    if 'res-1.0' in f and 'levels-13' in f:
        print(f)
PY

cat <<'MSG'

Now copy the exact names printed above into the wget lines below, e.g.:

  wget -P params "https://storage.googleapis.com/dm_graphcast/params/<exact-name>.npz"
  for f in diffs_stddev_by_level.nc mean_by_level.nc stddev_by_level.nc; do
      wget -P stats "https://storage.googleapis.com/dm_graphcast/stats/$f"
  done
  wget -P sample "https://storage.googleapis.com/dm_graphcast/dataset/<exact-sample>.nc"

Then pin the checkpoint:
  sha256sum params/*.npz | tee params/CHECKSUMS.txt
  date -u +%Y-%m-%dT%H:%M:%SZ >> params/CHECKSUMS.txt
MSG
