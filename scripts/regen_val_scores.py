"""
regen_val_scores.py — Regenera scores de VALIDACION (y TEST) para los detectores
point-wise desde los modelos guardados, para poder recalibrar el umbral SOLO en
validacion (protocolo correcto, sin fuga de test) en NB11.

Verifica cada regeneracion comparando el score de TEST recreado contra la columna
de score ya guardada en predictions_*.csv. Solo se confia en los que validan.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tfg_common as C

DATA = C.DATA_DIR
cfg = C.load_config()
FCOLS = cfg['feature_cols']
df_train, df_val, df_test = C.load_raw()
y_val = df_val['label'].values
y_test = df_test['label'].values

results = {}   # name -> dict(val=..., test=..., verify_corr=...)

def verify(name, test_scores, pred_file, score_col):
    """Correlacion entre score recreado y guardado (alineado por longitud)."""
    p = os.path.join(DATA, pred_file)
    saved = pd.read_csv(p, usecols=[score_col])[score_col].values
    n = min(len(saved), len(test_scores))
    corr = np.corrcoef(saved[:n], test_scores[:n])[0, 1]
    print(f'  [{name}] verify corr(test_regen, saved)={corr:.5f}  '
          f'(len regen={len(test_scores)}, saved={len(saved)})')
    return corr

# --------------------------------------------------------------------- #
# 1) ISOLATION FOREST  (17 full, point-wise, score = -score_samples)
# --------------------------------------------------------------------- #
try:
    sc = joblib.load(os.path.join(DATA, 'scaler_isolation_forest.pkl'))
    m = joblib.load(os.path.join(DATA, 'model_isolation_forest.pkl'))
    M = C.build_matrices(mode='full', scaler=sc)
    sv = -m.score_samples(M['X_val'])
    st = -m.score_samples(M['X_test'])
    corr = verify('IF', st, 'predictions_isolation_forest.csv', 'if_score')
    results['IF'] = dict(val=sv, test=st, corr=corr)
except Exception as e:
    print('  [IF] ERROR', e)

# --------------------------------------------------------------------- #
# 2) KMEANS + IF  (estrategia A: IF por cluster, 17 full)
# --------------------------------------------------------------------- #
try:
    sc = joblib.load(os.path.join(DATA, 'scaler_kmeans_if.pkl'))
    bundle = joblib.load(os.path.join(DATA, 'model_kmeans_if.pkl'))
    km = bundle['kmeans']; ifs = bundle['isolation_forests']
    M = C.build_matrices(mode='full', scaler=sc)
    def kmif_score(X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        cl = km.predict(X)
        s = np.empty(len(X))
        for c, model_c in (ifs.items() if isinstance(ifs, dict) else enumerate(ifs)):
            mask = cl == c
            if mask.any():
                s[mask] = -model_c.score_samples(X[mask])
        return s
    sv = kmif_score(M['X_val']); st = kmif_score(M['X_test'])
    corr = verify('KMeans+IF', st, 'predictions_kmeans_if.csv', 'kmif_score')
    results['KMeans+IF'] = dict(val=sv, test=st, corr=corr)
except Exception as e:
    print('  [KMeans+IF] ERROR', e)

# --------------------------------------------------------------------- #
# 3) DENSE-AE  (17 full, weighted reconstruction MSE)
# --------------------------------------------------------------------- #
try:
    from tensorflow import keras
    sc = joblib.load(os.path.join(DATA, 'scaler_dense_autoencoder.pkl'))
    w = np.load(os.path.join(DATA, 'feature_weights_dense_ae.npy'))
    ae = keras.models.load_model(os.path.join(DATA, 'model_dense_autoencoder.keras'),
                                 compile=False)
    M = C.build_matrices(mode='full', scaler=sc)
    def dae_score(X):
        rec = ae.predict(X, batch_size=8192, verbose=0)
        se = (rec - X) ** 2
        # probar dos convenciones y quedarnos con la que valide
        return (se * w).sum(axis=1), np.average(se, axis=1, weights=w)
    sv_sum, sv_avg = dae_score(M['X_val'])
    st_sum, st_avg = dae_score(M['X_test'])
    c_sum = verify('Dense-AE[sum]', st_sum, 'predictions_dense_autoencoder.csv', 'dae_score')
    c_avg = verify('Dense-AE[avg]', st_avg, 'predictions_dense_autoencoder.csv', 'dae_score')
    if c_avg >= c_sum:
        results['Dense-AE'] = dict(val=sv_avg, test=st_avg, corr=c_avg)
    else:
        results['Dense-AE'] = dict(val=sv_sum, test=st_sum, corr=c_sum)
except Exception as e:
    print('  [Dense-AE] ERROR', e)

# --------------------------------------------------------------------- #
# 4) LIGHTGBM  (27 features con lags, pipeline propio de NB09)
# --------------------------------------------------------------------- #
def lgb_features(df):
    f = df[FCOLS].copy()
    f['VI_residual'] = df['Global_active_power'] - df['Voltage'] * df['Global_intensity'] / 1000.0
    h = df.index.hour + df.index.minute / 60.0
    f['hour_sin'] = np.sin(2*np.pi*h/24); f['hour_cos'] = np.cos(2*np.pi*h/24)
    d = df.index.dayofweek
    f['dow_sin'] = np.sin(2*np.pi*d/7); f['dow_cos'] = np.cos(2*np.pi*d/7)
    f['gap_diff1'] = df['Global_active_power'].diff().fillna(0)
    f['vi_res_abs'] = f['VI_residual'].abs()
    f['vi_res_roll15_mean'] = f['vi_res_abs'].rolling(15, min_periods=1).mean()
    f['gap_intensity_ratio'] = df['Global_active_power'] / (df['Global_intensity'] + 0.01)
    f['sm_gap_ratio'] = ((df['Sub_metering_1']+df['Sub_metering_2']+df['Sub_metering_3'])/1000.0
                         / (df['Global_active_power'] + 0.01))
    f['gap_lag1'] = df['Global_active_power'].shift(1).fillna(method='bfill').fillna(0)
    f['gap_lag5'] = df['Global_active_power'].shift(5).fillna(method='bfill').fillna(0)
    f['gap_diff5'] = df['Global_active_power'].diff(5).fillna(0)
    f['gap_diff15'] = df['Global_active_power'].diff(15).fillna(0)
    f['gap_roll5_mean'] = df['Global_active_power'].rolling(5, min_periods=1).mean()
    f['gap_roll5_std'] = df['Global_active_power'].rolling(5, min_periods=1).std().fillna(0)
    f['gap_roll30_mean'] = df['Global_active_power'].rolling(30, min_periods=1).mean()
    f['gap_roll30_std'] = df['Global_active_power'].rolling(30, min_periods=1).std().fillna(0)
    f['v_roll5_std'] = df['Voltage'].rolling(5, min_periods=1).std().fillna(0)
    f['i_roll5_std'] = df['Global_intensity'].rolling(5, min_periods=1).std().fillna(0)
    return f.fillna(0)
try:
    import lightgbm as lgb
    sc = joblib.load(os.path.join(DATA, 'scaler_lightgbm.pkl'))
    m = lgb.Booster(model_file=os.path.join(DATA, 'model_lightgbm.txt'))
    Xv = sc.transform(lgb_features(df_val).values)
    Xt = sc.transform(lgb_features(df_test).values)
    sv = m.predict(Xv); st = m.predict(Xt)
    sv = np.asarray(sv).ravel(); st = np.asarray(st).ravel()
    corr = verify('LightGBM', st, 'predictions_lightgbm.csv', 'score')
    results['LightGBM'] = dict(val=sv, test=st, corr=corr)
except Exception as e:
    print('  [LightGBM] ERROR', e)

# --------------------------------------------------------------------- #
# Guardar scores verificados (corr > 0.99)
# --------------------------------------------------------------------- #
print('\n=== RESUMEN VERIFICACION ===')
ok = {}
for name, r in results.items():
    status = 'OK' if r['corr'] > 0.99 else 'DUDOSO'
    print(f'  {name:14s} corr={r["corr"]:.4f}  -> {status}')
    if r['corr'] > 0.99:
        ok[name] = r
        np.save(os.path.join(DATA, f'scores_val_{name.replace("+","_").lower()}.npy'), r['val'])
        np.save(os.path.join(DATA, f'scores_test_{name.replace("+","_").lower()}.npy'), r['test'])

np.save(os.path.join(DATA, 'y_val_aligned.npy'), y_val)
np.save(os.path.join(DATA, 'y_test_aligned.npy'), y_test)
print(f'\nDetectores verificados y guardados: {list(ok.keys())}')