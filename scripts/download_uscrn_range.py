#!/usr/bin/env python3
"""
Download NOAA USCRN hourly 2 m temperature for a time range (independent verification).
=======================================================================================
USCRN (~115 CONUS reference stations) is not assimilated operationally and is used here
only for VERIFICATION (never inserted, unless --obs-source uscrn is chosen).

Run on the LOGIN node:
  python scripts/download_uscrn_range.py --start 2018-01-14T00 --end 2018-01-19T00
Output:
  data/obs/uscrn_<start>_<end>.csv   columns: sid, lat, lon, elev, time, t2m_K
"""

import argparse
import concurrent.futures
import io
import os
import re
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2018-01-14T00")
ap.add_argument("--end", default="2018-01-19T00")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--proj", default=os.environ.get("PROJ", "/home/afahad/project/MLanalysis"))
args = ap.parse_args()

t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
OUT = os.path.join(args.proj, "data", "obs")
os.makedirs(OUT, exist_ok=True)

print("=" * 68)
print(f"NOAA USCRN Range Downloader: {t0} to {t1}")
print("=" * 68)

# 1. Fetch Station Elevation Table (cached; WBAN keys zero-padded; ELEVATION is in feet)
elev = {}
st_path = os.path.join(OUT, "uscrn_stations.tsv")
for attempt in range(3):
    try:
        if not os.path.exists(st_path):
            req = urllib.request.Request("https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(st_path, "wb") as f:
                f.write(r.read())
        st = pd.read_csv(st_path, sep="\t", dtype=str)
        st.columns = [c.strip().upper() for c in st.columns]
        for _, r in st.iterrows():
            try:
                elev[str(r["WBAN"]).strip().zfill(5)] = float(r["ELEVATION"]) * 0.3048
            except (ValueError, TypeError, KeyError):
                pass
        break
    except Exception as e:
        print(f"[1] attempt {attempt + 1}: station table unavailable ({e})")
        if os.path.exists(st_path) and os.path.getsize(st_path) == 0:
            os.remove(st_path)
        time.sleep(3)
print(f"[1] Station elevations loaded for {len(elev)} stations ({st_path})")
if not elev:
    print("    WARNING: without elevations the main script drops USCRN (or use --allow-no-elev).")

# 2. Collect Target Dates
target_dates = set((t0 + pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range((t1 - t0).days + 2))
print(f"[2] Target dates: {sorted(list(target_dates))}")

# 3. Stream Stations Across Years
all_lines = []
for year in range(t0.year, t1.year + 1):
    base_url = f"https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{year}/"
    print(f"[3] Scanning {base_url} ...")
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
            files = re.findall(r'href=[\"\'](CRNH0203-[^\"\']+\.txt)[\"\']', html)
        files = sorted(list(set(files)))
        print(f"    Found {len(files)} station files for {year}")
    except Exception as e:
        print(f"    Error accessing directory for {year}: {e}")
        continue

    def fetch_station(fname):
        url = base_url + fname
        matched = []
        try:
            req_s = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_s, timeout=12) as resp:
                for line in resp:
                    parts = line.decode("utf-8", errors="ignore").split()
                    if len(parts) > 2 and parts[1] in target_dates:
                        matched.append(line.decode("utf-8", errors="ignore"))
        except Exception:
            pass
        return matched

    t_s = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        res = list(ex.map(fetch_station, files))
    year_lines = [l for sub in res for l in sub]
    all_lines.extend(year_lines)
    print(f"    Retrieved {len(year_lines)} lines for {year} in {time.time() - t_s:.1f} s")

if not all_lines:
    sys.exit("ERROR: No USCRN observations retrieved for specified range.")

# 4. Parse DataFrame
df = pd.read_csv(
    io.StringIO("".join(all_lines)),
    sep=r"\s+",
    header=None,
    usecols=[0, 1, 2, 6, 7, 8],
    dtype={0: str, 1: str, 2: str, 6: float, 7: float, 8: float},
)
df.columns = ["sid", "date", "hhmm", "lon", "lat", "t"]

# Filter invalid values
df = df[df["t"] > -90.0].copy()

# Filter CONUS
df = df[df.lat.between(24, 50) & df.lon.between(-125, -66)].copy()

# Convert time handling 2400
hh = df.hhmm.str.zfill(4).str[:2].astype(int)
base = pd.to_datetime(df.date, format="%Y%m%d")
df["time"] = base + pd.to_timedelta(hh, unit="h")
df = df[(df.time >= t0) & (df.time <= t1)].copy()

# Add elevations and temperatures in Kelvin
df["elev"] = df.sid.str.strip().str.zfill(5).map(elev).astype(float)
print(f"[4] Elevation matched for {df.dropna(subset=['elev']).sid.nunique()} of {df.sid.nunique()} stations")
df["t2m_K"] = (df.t + 273.15).round(3)

obs = df[["sid", "lat", "lon", "elev", "time", "t2m_K"]].sort_values(["time", "sid"])
out = os.path.join(OUT, f"uscrn_{t0:%Y%m%dT%H}_{t1:%Y%m%dT%H}.csv")
obs.to_csv(out, index=False)

print("=" * 68)
print(f"SUCCESS: {len(obs)} obs from {obs.sid.nunique()} USCRN stations -> {out}")
print("=" * 68)
