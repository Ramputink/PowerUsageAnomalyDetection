"""
tfg_common.py — Shared utilities for notebooks NB15-NB20.

Centralizes the data CONTRACT and the evaluation PROTOCOL defined in NB02-NB10
so that the new notebooks (generative models, federated learning, adversarial
robustness, edge compression) are strictly comparable with the earlier
detectors:

  * Same temporal split (train_clean / val_with_attacks / test_with_attacks).
  * Same features (base/medium/full modes, identical to NB02/NB04).
  * Threshold and (W, K) calibration ONLY on the validation set.
  * Same metrics: F1/F2/AUC-ROC/AUC-PR + recall by type x severity
    + per-episode detection latency.

Reuses NB04's inline functions (compute_features, calibrate_threshold,
temporal_vote_vectorized, compute_detection_latency) to avoid reimplementing
them.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

BASE_DIR = '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine/'
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
WARMUP = 30  # warm-up rows so the rolling features start up correctly


# --------------------------------------------------------------------------- #
#  Configuration / data
# --------------------------------------------------------------------------- #
def load_config():
    with open(os.path.join(DATA_DIR, 'pipeline_config.json')) as f:
        return json.load(f)


def load_raw():
    """Loads the three official (immutable) splits."""
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_clean.csv'),
                           index_col='datetime', parse_dates=True)
    df_val = pd.read_csv(os.path.join(DATA_DIR, 'val_with_attacks.csv'),
                         index_col='datetime', parse_dates=True)
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test_with_attacks.csv'),
                          index_col='datetime', parse_dates=True)
    return df_train, df_val, df_test


def compute_features(df, feature_cols, mode='full'):
    """Identical to NB02/NB04.  base=8, medium=13, full=17 features."""
    f = df[feature_cols].copy()
    f['VI_residual'] = (df['Global_active_power']
                        - df['Voltage'] * df['Global_intensity'] / 1000.0)
    if mode in ('medium', 'full'):
        h = df.index.hour + df.index.minute / 60.0
        f['hour_sin'] = np.sin(2 * np.pi * h / 24)
        f['hour_cos'] = np.cos(2 * np.pi * h / 24)
        d = df.index.dayofweek
        f['dow_sin'] = np.sin(2 * np.pi * d / 7)
        f['dow_cos'] = np.cos(2 * np.pi * d / 7)
        f['gap_diff1'] = df['Global_active_power'].diff().fillna(0)
    if mode == 'full':
        f['vi_res_abs'] = f['VI_residual'].abs()
        f['vi_res_roll15_mean'] = f['vi_res_abs'].rolling(15, min_periods=1).mean()
        f['gap_intensity_ratio'] = (df['Global_active_power']
                                    / (df['Global_intensity'] + 0.01))
        sm_sum = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
        gap_wh = df['Global_active_power'] * 1000 / 60 + 1e-8
        f['sm_gap_ratio'] = np.clip(sm_sum / gap_wh, 0, 3)
    return f


def build_matrices(mode='full', scaler=None):
    """
    Returns a dict with scaled X_train/X_val/X_test + labels + metadata.
    Uses warm-up from the tail of the previous split (no information leakage)
    and a RobustScaler fitted ONLY on train, exactly like NB04.
    """
    cfg = load_config()
    feature_cols = cfg['feature_cols']
    df_train, df_val, df_test = load_raw()

    X_train_df = compute_features(df_train, feature_cols, mode)
    _val_buf = pd.concat([df_train.iloc[-WARMUP:], df_val])
    X_val_df = compute_features(_val_buf, feature_cols, mode).iloc[WARMUP:]
    _test_buf = pd.concat([df_val.iloc[-WARMUP:], df_test])
    X_test_df = compute_features(_test_buf, feature_cols, mode).iloc[WARMUP:]

    feat_names = list(X_train_df.columns)
    if scaler is None:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_df.values)
    else:
        X_train = scaler.transform(X_train_df.values)
    X_val = scaler.transform(X_val_df.values)
    X_test = scaler.transform(X_test_df.values)

    nz = lambda a: np.nan_to_num(a, nan=0.0).astype('float32')
    return {
        'X_train': nz(X_train), 'X_val': nz(X_val), 'X_test': nz(X_test),
        'y_val': df_val['label'].values, 'y_test': df_test['label'].values,
        'feat_names': feat_names, 'n_features': X_train.shape[1],
        'scaler': scaler, 'feature_cols': feature_cols,
        'attack_types': cfg['attack_types'],
        'df_val': df_val, 'df_test': df_test,
    }


# --------------------------------------------------------------------------- #
#  Reusable base models
# --------------------------------------------------------------------------- #
def build_dense_ae(n_features, bottleneck=4, lr=5e-4):
    """Dense-AE identical to NB04 (for edge-footprint comparability)."""
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense, LeakyReLU
    from tensorflow.keras.models import Model
    inp = Input(shape=(n_features,), name='input')
    x = Dense(64, name='enc1')(inp); x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(32, name='enc2')(x);   x = LeakyReLU(negative_slope=0.1)(x)
    enc = Dense(bottleneck, activation='relu', name='bottleneck')(x)
    x = Dense(32, name='dec1')(enc); x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(64, name='dec2')(x);   x = LeakyReLU(negative_slope=0.1)(x)
    out = Dense(n_features, activation='linear', name='output')(x)
    model = Model(inp, out, name='dense_autoencoder')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


# --------------------------------------------------------------------------- #
#  Evaluation protocol (identical to NB04)
# --------------------------------------------------------------------------- #
def calibrate_threshold(y_true, scores, beta=1.0):
    """Threshold that maximizes F_beta over the precision-recall curve (on val)."""
    p, r, thr = precision_recall_curve(y_true, scores)
    p, r = p[:-1], r[:-1]
    fbeta = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-10)
    i = int(np.argmax(fbeta))
    return thr[i], float(fbeta[i])


def temporal_vote(y_pred, W, K):
    """Causal O(n) voting filter with cumsum (identical to NB04)."""
    cum = np.concatenate([[0], np.cumsum(y_pred)])
    ends = np.arange(1, len(y_pred) + 1)
    starts = np.maximum(0, ends - W)
    return (cum[ends] - cum[starts] >= K).astype(int)


def calibrate_WK(y_val, y_pred_val_raw,
                 W_candidates=(3, 5, 7, 9, 11, 15, 21, 30)):
    """Searches for the (W, K) that maximizes F1 on validation."""
    best = (0.0, 3, 2)
    for W in W_candidates:
        for K in range(2, W + 1):
            f1 = f1_score(y_val, temporal_vote(y_pred_val_raw, W, K))
            if f1 > best[0]:
                best = (f1, W, K)
    return best[1], best[2]


def evaluate(y_true, scores, threshold, W=None, K=None):
    """Returns a dict of raw metrics and (if W,K given) smoothed ones."""
    y_raw = (scores > threshold).astype(int)
    out = {
        'raw_f1': float(f1_score(y_true, y_raw)),
        'raw_f2': float(fbeta_score(y_true, y_raw, beta=2)),
        'raw_precision': float(precision_score(y_true, y_raw, zero_division=0)),
        'raw_recall': float(recall_score(y_true, y_raw)),
        'auc_roc': float(roc_auc_score(y_true, scores)),
        'auc_pr': float(average_precision_score(y_true, scores)),
    }
    if W is not None and K is not None:
        y_sm = temporal_vote(y_raw, W, K)
        out.update({
            'f1': float(f1_score(y_true, y_sm)),
            'f2': float(fbeta_score(y_true, y_sm, beta=2)),
            'precision': float(precision_score(y_true, y_sm, zero_division=0)),
            'recall': float(recall_score(y_true, y_sm)),
            'temporal_W': W, 'temporal_K': K,
        })
    return out


def recall_by_type_severity(df_test, y_pred_smooth, attack_types):
    """Recall by type x severity (smoothed)."""
    d = df_test.copy()
    d['pred'] = y_pred_smooth
    rows = []
    for t in attack_types:
        row = {'attack_type': t}
        for sev in ['low', 'medium', 'high']:
            m = (d['attack_type'] == t) & (d['severity'] == sev) & (d['label'] == 1)
            row[sev] = float(d.loc[m, 'pred'].mean()) if m.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def detection_latency(df_test, pred_col):
    """Per-episode detection latency (identical to NB04)."""
    d = df_test[df_test['episode_id'] >= 0]
    res = []
    for ep, g in d.groupby('episode_id'):
        g = g.sort_index()
        det = g[g[pred_col] > 0.5]
        res.append({
            'episode_id': ep, 'type': g['attack_type'].iloc[0],
            'severity': g['severity'].iloc[0], 'duration_min': len(g),
            'latency_min': ((det.index[0] - g.index[0]).total_seconds() / 60
                            if len(det) else np.nan),
            'detected': len(det) > 0,
        })
    return pd.DataFrame(res)


def edge_timing(predict_fn, X, n_repeat=5):
    """ms/sample and throughput of an inference function."""
    _ = predict_fn(X[:100])
    times = []
    for _ in range(n_repeat):
        t0 = time.time(); predict_fn(X); times.append(time.time() - t0)
    ms = float(np.mean(times) / len(X) * 1000)
    return ms, 1000.0 / ms


def save_metrics(name, metrics):
    path = os.path.join(DATA_DIR, f'metrics_{name}.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return path
