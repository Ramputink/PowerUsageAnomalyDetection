"""
prepare_ukdale.py — Genera el subconjunto procesado de UK-DALE para NB13.

UK-DALE crudo pesa ~112 GB (16 kHz + canales de electrodomésticos). Para la
detección de FDI sobre consumo AGREGADO solo se necesita el `mains` (channel_1)
de cada casa, resampleado a 1 minuto. Este script extrae ese subconjunto
(unos pocos MB) de las 5 casas y lo guarda como parquet, reproducible y subible
a GitHub. El crudo NO se versiona (ver datasets/README.md para descargarlo).

Uso:  python prepare_ukdale.py
Salida: data/ukdale_multihouse_1min.parquet  (columnas: house, Global_active_power[kW])
"""
import os, time
import numpy as np
import pandas as pd

UKDALE_ROOT = '/Volumes/Extreme Pro Particion 1TB/TFG/UKRC/ukdale'
OUT = '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine/data/ukdale_multihouse_1min.csv.gz'
HOUSES = [1, 2, 3, 4, 5]

def load_mains_1min(house):
    """channel_1.dat = mains agregado (unix_ts watts, 6 s) -> 1 min mean en kW."""
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
print(f'\nGuardado: {OUT}  ({mb:.1f} MB, {len(full):,} filas, {full.house.nunique()} casas)')
