#!/bin/bash
# Download (or link) the MERRA-2 files needed to build GraphCast inputs from MERRA-2 (Step 7,
# OPERATIONAL_DA_PLAN.md §14). Run on the LOGIN node (compute nodes may have no internet).
#
#   bash scripts/download_merra2.sh                          # default window 2018-01-12 .. 2018-01-18
#   bash scripts/download_merra2.sh 2018-01-12 2018-01-18    # any start/end date (inclusive)
#
# Files (one per day, all 0.5° x 0.625°):
#   inst3_3d_asm_Np  (M2I3NPASM, 3-hourly, 42 p-levels): T, U, V, QV, H, OMEGA, SLP, PS
#   inst1_2d_asm_Nx  (M2I1NXASM, hourly):                T2M, U10M, V10M, SLP
#   tavg1_2d_flx_Nx  (M2T1NXFLX, hourly means):          PRECTOT, PRECTOTCORR  (-> 6 h precip)
#   const_2d_asm_Nx  (M2C0NXASM, one file):              PHIS, FRLAND           (below-ground fill)
#
# 1) If MERRA-2 is on local NCCS storage (MERRA2_all/Y%Y/M%m layout, e.g. /css/merra2/MERRA2_all on Prism,
#    files named MERRA2.<collection>.<date>.nc4), they are symlinked under the standard GES DISC name, no download.
#    Override the search with MERRA2_LOCAL=/path/to/MERRA2_all.
# 2) Otherwise they are downloaded from NASA GES DISC. This needs a (free) Earthdata login:
#      - account at https://urs.earthdata.nasa.gov, and in your profile approve the application
#        "NASA GESDISC DATA ARCHIVE"
#      - ~/.netrc with:  machine urs.earthdata.nasa.gov login <USER> password <PASS>   (chmod 600 ~/.netrc)
set -uo pipefail
PROJ="${PROJ:-/home/afahad/project/MLanalysis}"
OUT="$PROJ/data/merra2"
START="${1:-2018-01-12}"
END="${2:-2018-01-18}"
mkdir -p "$OUT"
touch ~/.urs_cookies

LOCAL_CANDIDATES=("${MERRA2_LOCAL:-}" /css/merra2/MERRA2_all /discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all)
LOCAL=""
for c in "${LOCAL_CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -d "$c" ] && { LOCAL="$c"; break; }
done
[ -n "$LOCAL" ] && echo "Local MERRA-2 found: $LOCAL (will link when a file exists there)" \
                || echo "No local MERRA-2 copy found; downloading from GES DISC"

stream() {   # MERRA-2 production stream by year (2020-2021 has some 401 reprocessing; tried as fallback)
  local y=$1
  if   [ "$y" -le 1991 ]; then echo 100
  elif [ "$y" -le 2000 ]; then echo 200
  elif [ "$y" -le 2010 ]; then echo 300
  else echo 400; fi
}

check() {    # file opens and is not an HTML error page
  python - "$1" <<'PY' >/dev/null 2>&1
import sys, netCDF4
netCDF4.Dataset(sys.argv[1]).close()
PY
}

get() {      # $1 host  $2 collection dir  $3 file name  $4 relative dir (YYYY/MM or '')
  local host=$1 coll=$2 name=$3 rel=$4 dest="$OUT/$3"
  if [ -s "$dest" ] && check "$dest"; then echo "   ok      $name"; return 0; fi
  if [ -n "$LOCAL" ]; then
    local y=${rel%%/*} m=${rel##*/}
    local dir="$LOCAL/Y$y/M$m"; [ -z "$rel" ] && dir="$LOCAL"
    local short="MERRA2.${name#MERRA2_*.}"          # NCCS copies drop the stream number: MERRA2.<coll>.<date>.nc4
    for lp in "$dir/$name" "$dir/$short"; do
      if [ -f "$lp" ]; then ln -sf "$lp" "$dest"; echo "   linked  $name  <- $lp"; return 0; fi
    done
  fi
  local url="https://$host.gesdisc.eosdis.nasa.gov/data/$coll/${rel:+$rel/}$name"
  wget -q --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies \
       --auth-no-challenge=on -c -O "$dest" "$url"
  if [ -s "$dest" ] && check "$dest"; then echo "   got     $name ($(du -h "$dest" | cut -f1))"; return 0; fi
  rm -f "$dest"; return 1
}

echo "=== MERRA-2 $START .. $END -> $OUT ==="
fail=0
d="$START"
while [ "$(date -d "$d" +%s)" -le "$(date -d "$END" +%s)" ]; do
  Y=$(date -d "$d" +%Y); M=$(date -d "$d" +%m); D=$(date -d "$d" +%Y%m%d)
  S=$(stream "$Y")
  echo "$d"
  for spec in "goldsmr5 MERRA2/M2I3NPASM.5.12.4 inst3_3d_asm_Np" \
              "goldsmr4 MERRA2/M2I1NXASM.5.12.4 inst1_2d_asm_Nx" \
              "goldsmr4 MERRA2/M2T1NXFLX.5.12.4 tavg1_2d_flx_Nx"; do
    set -- $spec
    get "$1" "$2" "MERRA2_$S.$3.$D.nc4" "$Y/$M" \
      || get "$1" "$2" "MERRA2_401.$3.$D.nc4" "$Y/$M" \
      || { echo "   FAILED  MERRA2_$S.$3.$D.nc4"; fail=1; }
  done
  d=$(date -d "$d + 1 day" +%Y-%m-%d)
done

echo "constants"
get goldsmr4 MERRA2_MONTHLY/M2C0NXASM.5.12.4/1980 MERRA2_101.const_2d_asm_Nx.00000000.nc4 "" \
  || { echo "   FAILED  const_2d_asm_Nx"; fail=1; }

echo "=== done: $(ls "$OUT"/*.nc4 2>/dev/null | wc -l) files, $(du -shL "$OUT" | cut -f1) in $OUT ==="
if [ $fail -ne 0 ]; then
  echo "Some files failed. Check ~/.netrc (urs.earthdata.nasa.gov), that 'NASA GESDISC DATA ARCHIVE' is"
  echo "approved in your Earthdata profile, and that you are on the login node."
  exit 1
fi
