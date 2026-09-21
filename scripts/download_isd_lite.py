#!/usr/bin/env python3
"""
Download NOAA ISD-Lite hourly 2 m temperature for CONUS stations (real observations).
======================================================================================
ISD-Lite = hourly subset of the Integrated Surface Database (ASOS/AWOS/SYNOP), values
already on the hour. Real station data for inserting into GraphCast initial states.

Run on the LOGIN node (internet):
  python scripts/download_isd_lite.py --start 2018-01-14T00 --end 2018-01-19T00
Output:
  data/obs/isd_lite_<start>_<end>.csv   columns: sid, lat, lon, elev, time, t2m_K
"""
import argparse, gzip, io, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2018-01-14T00")
ap.add_argument("--end", default="2018-01-19T00")
ap.add_argument("--bbox", default="24,50,-125,-66", help="lat_min,lat_max,lon_min,lon_max")
ap.add_argument("--max-stations", type=int, default=0, help="0 = all")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
args = ap.parse_args()

t_start, t_end = pd.Timestamp(args.start), pd.Timestamp(args.end)
la0, la1, lo0, lo1 = map(float, args.bbox.split(","))
OUT = os.path.join(args.proj, "data", "obs"); RAW = os.path.join(OUT, "isd_lite_raw")
os.makedirs(RAW, exist_ok=True)

print("[1] Station history ...")
hist_path = os.path.join(OUT, "isd-history.csv")
if not os.path.exists(hist_path):
    urllib.request.urlretrieve("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv", hist_path)
h = pd.read_csv(hist_path, dtype=str)
for c in ("LAT", "LON", "ELEV(M)"):
    h[c] = pd.to_numeric(h[c], errors="coerce")
h = h[(h.CTRY == "US") & h.LAT.between(la0, la1) & h.LON.between(lo0, lo1)]
h = h[(h.BEGIN <= t_start.strftime("%Y%m%d")) & (h.END >= t_end.strftime("%Y%m%d"))]
h = h[(h.USAF != "999999")]
if args.max_stations:
    h = h.sample(n=min(args.max_stations, len(h)), random_state=0)
print(f"    {len(h)} candidate stations")

COLS = ["year", "month", "day", "hour", "t", "td", "slp", "wd", "ws", "sky", "p1", "p6"]


def fetch(row, year):
    sid = f"{row.USAF}-{row.WBAN}"
    local = os.path.join(RAW, str(year), f"{sid}-{year}.gz")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{sid}-{year}.gz"
        try:
            urllib.request.urlretrieve(url, local)
        except Exception:
            return None
    try:
        with gzip.open(local, "rt") as f:
            d = pd.read_csv(f, sep=r"\s+", header=None, names=COLS, usecols=range(6))
    except Exception:
        return None
    d["time"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day, hour=d.hour))
    d = d[(d.time >= t_start) & (d.time <= t_end) & (d.t > -9999)]
    if d.empty:
        return None
    return pd.DataFrame(dict(sid=sid, lat=row.LAT, lon=row.LON, elev=row["ELEV(M)"],
                             time=d.time.values, t2m_K=d.t.values / 10.0 + 273.15))


print("[2] Downloading ISD-Lite files ...")
jobs, parts = [], []
with ThreadPoolExecutor(args.threads) as ex:
    for _, row in h.iterrows():
        for y in range(t_start.year, t_end.year + 1):
            jobs.append(ex.submit(fetch, row, y))
    for i, j in enumerate(as_completed(jobs)):
        r = j.result()
        if r is not None:
            parts.append(r)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(jobs)} files")
if not parts:
    sys.exit("No observations retrieved.")
obs = pd.concat(parts, ignore_index=True).sort_values(["time", "sid"])
out = os.path.join(OUT, f"isd_lite_{t_start:%Y%m%dT%H}_{t_end:%Y%m%dT%H}.csv")
obs.to_csv(out, index=False)
print(f"[3] {len(obs)} obs from {obs.sid.nunique()} stations -> {out}")
