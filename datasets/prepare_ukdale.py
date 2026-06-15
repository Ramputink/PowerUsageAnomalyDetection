"""
prepare_ukdale.py — Builds the processed UK-DALE subset used by NB13.

The raw UK-DALE dataset is ~112 GB (16 kHz + per-appliance channels). For FDI
detection on the AGGREGATE consumption only the `mains` signal (channel_1) of
each house is needed, resampled to 1 minute. This script extracts that subset
(a few MB) from the 5 houses and saves it as a reproducible, GitHub-uploadable
file. The raw data is NOT versioned (see datasets/README.md to download it).

Usage:  python prepare_ukdale.py
Output: data/ukdale_multihouse_1min.csv.gz  (columns: house, Global_active_power[kW])
"""
import os, time
import numpy as np
import pandas as pd

UKDALE_ROOT = '/Volumes/Extreme Pro Particion 1TB/TFG/UKRC/ukdale'
OUT = '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine/data/ukdale_multihouse_1min.csv.gz'
HOUSES = [1, 2, 3, 4, 5]

def load_mains_1min(house):
    """channel_1.dat = aggregate mains (unix_ts watts, 6 s) -> 1 min mean in kW."""
    path = os.path.join(UKDALE_ROOT, f'house_{house}', 'channel_1.dat')
    df = pd.read_csv(path, sep=' ', header=None, names=['ts', 'watts'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    s = df.set_index('datetime')['watts'].resample('1T').mean().dropna()
    out = s.to_frame('Global_active_power')
    out['Global_active_power'] /= 1000.0  # kW
    out['house'] = f'house_{house}'
    return out

frames = []
for h in HOUSES:
    t0 = time.time()
    d = load_mains_1min(h)
    frames.append(d.reset_index())
    print(f'  house_{h}: {len(d):>8,d} min  '
          f'[{d.index.min()} -> {d.index.max()}]  ({time.time()-t0:.1f}s)')

full = pd.concat(frames, ignore_index=True)
full.to_csv(OUT, index=False, compression='gzip')
mb = os.path.getsize(OUT) / 1048576
print(f'\nSaved: {OUT}  ({mb:.1f} MB, {len(full):,} rows, {full.house.nunique()} houses)')
