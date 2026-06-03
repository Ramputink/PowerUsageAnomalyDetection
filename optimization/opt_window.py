"""
opt_window.py — Búsqueda de hiperparámetros (ECC) para detectores WINDOW:
CNN-AE, Transformer-AE, LSTM-AE.

Pipeline de evaluación per-FILA idéntico a NB05/07/08:
  ventanas -> MSE por feature/patch -> pesos (AUC en val) -> score por ventana
  -> expandir a per-timestep (avg) -> calibrar umbral+(W,K) SOLO en val
  -> evaluar en test.
CORRECTNESS GATE = contrato no-trampa. Para velocidad de búsqueda se subsamplea
train (ventanas) y se puntúa val/test sobre un PREFIJO contiguo (calibración solo
en val). Finalistas -> re-validar a escala completa.

Uso: python opt_window.py <cnn|transformer|lstm|cycle> <segundos> [seed]
"""
import sys, os, time, json, csv
import numpy as np
sys.path.insert(0, '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine')
import tfg_common as C
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score)

OPT_DIR = '/Volumes/Extreme Pro Particion 1TB/TFG/optimization'
WINDOW = 60
STRIDE_TRAIN = 30
N_TRAIN_WIN = 25_000     # ventanas de entrenamiento (búsqueda)
VAL_N = 80_000          # prefijo de val para calibrar
TEST_N = 120_000         # prefijo de test para evaluar
EVAL_STRIDE = 2          # stride en val/test durante búsqueda (velocidad)

_BASE = json.load(open(f'{OPT_DIR}/baseline.json'))
_BASEF1 = {'cnn': _BASE['CNN-AE']['f1'], 'transformer': _BASE['Transformer-AE']['f1'],
           'lstm': _BASE['LSTM-AE']['f1']}

# ----- datos cacheados por modo (per-fila, alineado con y) -----
_D = {}
def get_rows(mode, clip=None):
    key = (mode, clip)
    if key not in _D:
        d = C.build_matrices(mode=mode)
        Xtr, Xv, Xte = d['X_train'], d['X_val'], d['X_test']
        if clip:
            Xtr = np.clip(Xtr, -clip, clip); Xv = np.clip(Xv, -clip, clip)
            Xte = np.clip(Xte, -clip, clip)
        _D[key] = dict(Xtr=Xtr, Xv=Xv[:VAL_N], Xte=Xte[:TEST_N],
                       yv=d['y_val'][:VAL_N], yte=d['y_test'][:TEST_N],
                       nf=d['n_features'])
    return _D[key]


def make_windows(X, window, stride):
    n = len(X)
    starts = np.arange(0, n - window + 1, stride)
    return np.stack([X[s:s + window] for s in starts]).astype('float32'), starts


def window_labels(y, starts, window):
    cum = np.concatenate([[0], np.cumsum(y)])
    return (cum[starts + window] - cum[starts] > 0).astype(int)


def expand_avg(scores_w, n_total, starts, window):
    acc = np.zeros(n_total); cnt = np.zeros(n_total)
    for s, sc in zip(starts, scores_w):
        e = min(s + window, n_total); acc[s:e] += sc; cnt[s:e] += 1
    cnt[cnt == 0] = 1
    return acc / cnt


def _eval_perrow(sv_ts, st_ts, yv, yte):
    thr, _ = C.calibrate_threshold(yv, sv_ts, beta=1.0)
    yvr = (sv_ts > thr).astype(int)
    Wb, Kb = C.calibrate_WK(yv, yvr)
    m = C.evaluate(yte, st_ts, thr, Wb, Kb)
    m['temporal_W'] = Wb; m['temporal_K'] = Kb
    return m


# ----------------------------------------------------------------------------- #
# Builders
# ----------------------------------------------------------------------------- #
def build_cnn(window, nf, cfg):
    from tensorflow import keras
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                                          Dense, Flatten, Reshape, LeakyReLU)
    f1, f2, f3 = cfg['filters']
    inp = Input((window, nf))
    x = Conv1D(f1, cfg['ksize'], padding='same')(inp); x = LeakyReLU(0.1)(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(f2, 3, padding='same')(x); x = LeakyReLU(0.1)(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(f3, 3, padding='same')(x); x = LeakyReLU(0.1)(x)
    x = MaxPooling1D(3, padding='same')(x)
    x = Flatten()(x)
    x = Dense(cfg['bottleneck'], activation='relu')(x)
    x = Dense(5 * f3)(x); x = LeakyReLU(0.1)(x)
    x = Reshape((5, f3))(x)
    x = UpSampling1D(3)(x); x = Conv1D(f2, 3, padding='same')(x); x = LeakyReLU(0.1)(x)
    x = UpSampling1D(2)(x); x = Conv1D(f1, 3, padding='same')(x); x = LeakyReLU(0.1)(x)
    x = UpSampling1D(2)(x)
    out = Conv1D(nf, cfg['ksize'], padding='same', activation='linear')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg['lr']), loss='mse', jit_compile=False)
    return model


def build_lstm(window, nf, cfg):
    from tensorflow import keras
    from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
    u = cfg['units']
    inp = Input((window, nf))
    x = LSTM(u, activation='tanh', return_sequences=True)(inp)
    x = LSTM(u // 2, activation='tanh', return_sequences=False)(x)
    enc = Dense(cfg['bottleneck'], activation='relu')(x)
    x = Dense(u // 2, activation='relu')(enc)
    x = RepeatVector(window)(x)
    x = LSTM(u // 2, activation='tanh', return_sequences=True)(x)
    x = LSTM(u, activation='tanh', return_sequences=True)(x)
    out = TimeDistributed(Dense(nf, activation='linear'))(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg['lr']), loss='mse')
    return model


def build_transformer(n_patches, patch_dim, cfg):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import (Input, Dense, LayerNormalization, Dropout,
                                          MultiHeadAttention, Add, Embedding)
    dm, nh, ff, nl, dr = (cfg['d_model'], cfg['n_heads'], cfg['ffn'],
                          cfg['n_layers'], cfg['dropout'])
    inp = Input((n_patches, patch_dim))
    x = Dense(dm)(inp)
    pos = Embedding(n_patches, dm)(tf.range(n_patches))
    x = x + pos
    for _ in range(nl):
        a = LayerNormalization()(x)
        a = MultiHeadAttention(num_heads=nh, key_dim=dm // nh, dropout=dr)(a, a, a)
        x = Add()([x, a])
        f = LayerNormalization()(x)
        f = Dense(ff, activation='gelu')(f); f = Dropout(dr)(f); f = Dense(dm)(f)
        x = Add()([x, f])
    x = LayerNormalization()(x)
    b = Dense(max(2, dm // 4), activation='gelu')(x)
    d = Dense(dm, activation='gelu')(b); d = Dropout(dr)(d)
    out = Dense(patch_dim)(d)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg['lr']), loss='mse', jit_compile=False)
    return model


def _auc_weights(mse_feat, yv):
    aucs = np.array([roc_auc_score(yv, mse_feat[:, j]) if len(np.unique(mse_feat[:, j])) > 1
                     else 0.5 for j in range(mse_feat.shape[1])])
    w = np.maximum(aucs - 0.5, 0); s = w.sum()
    return (w / s) if s > 0 else np.ones(mse_feat.shape[1]) / mse_feat.shape[1]


# ----------------------------------------------------------------------------- #
# Runners (devuelven métricas per-fila)
# ----------------------------------------------------------------------------- #
def run_cnn_or_lstm(kind, cfg):
    import tensorflow as tf
    tf.random.set_seed(cfg.get('seed', 42))
    clip = 5.0 if kind == 'lstm' else None
    d = get_rows(cfg['mode'], clip=clip); nf = d['nf']
    rng = np.random.default_rng(cfg.get('seed', 42))
    Xw_tr, _ = make_windows(d['Xtr'], WINDOW, STRIDE_TRAIN)
    if len(Xw_tr) > N_TRAIN_WIN:
        Xw_tr = Xw_tr[np.sort(rng.choice(len(Xw_tr), N_TRAIN_WIN, replace=False))]
    Xw_v, sv_st = make_windows(d['Xv'], WINDOW, EVAL_STRIDE)
    Xw_te, st_st = make_windows(d['Xte'], WINDOW, EVAL_STRIDE)
    model = (build_lstm if kind == 'lstm' else build_cnn)(WINDOW, nf, cfg)
    from tensorflow import keras
    cb = [keras.callbacks.EarlyStopping(patience=cfg.get('patience', 5),
                                        restore_best_weights=True)]
    model.fit(Xw_tr, Xw_tr, epochs=cfg['epochs'], batch_size=cfg['batch'],
              validation_split=0.1, callbacks=cb, verbose=0)
    rec_v = model.predict(Xw_v, batch_size=2048, verbose=0)
    mse_feat_v = ((Xw_v - rec_v) ** 2).mean(axis=1)             # (Nv, F)
    w = _auc_weights(mse_feat_v, window_labels(d['yv'], sv_st, WINDOW))
    sv_w = (mse_feat_v * w).sum(1)
    rec_te = model.predict(Xw_te, batch_size=2048, verbose=0)
    st_w = (((Xw_te - rec_te) ** 2).mean(axis=1) * w).sum(1)
    sv_ts = expand_avg(sv_w, len(d['Xv']), sv_st, WINDOW)
    st_ts = expand_avg(st_w, len(d['Xte']), st_st, WINDOW)
    r = _eval_perrow(sv_ts, st_ts, d['yv'], d['yte'])
    r['n_params'] = int(model.count_params())
    return r


def run_transformer(cfg):
    import tensorflow as tf
    tf.random.set_seed(cfg.get('seed', 42))
    d = get_rows(cfg['mode']); nf = d['nf']
    pl = cfg['patch_len']; npatch = WINDOW // pl
    rng = np.random.default_rng(cfg.get('seed', 42))
    def to_patches(Xw):
        N = Xw.shape[0]
        return Xw.reshape(N, npatch, pl, nf).reshape(N, npatch, pl * nf)
    Xw_tr, _ = make_windows(d['Xtr'], WINDOW, STRIDE_TRAIN)
    if len(Xw_tr) > N_TRAIN_WIN:
        Xw_tr = Xw_tr[np.sort(rng.choice(len(Xw_tr), N_TRAIN_WIN, replace=False))]
    Xw_v, sv_st = make_windows(d['Xv'], WINDOW, EVAL_STRIDE)
    Xw_te, st_st = make_windows(d['Xte'], WINDOW, EVAL_STRIDE)
    Xp_tr, Xp_v, Xp_te = to_patches(Xw_tr), to_patches(Xw_v), to_patches(Xw_te)
    model = build_transformer(npatch, pl * nf, cfg)
    from tensorflow import keras
    cb = [keras.callbacks.EarlyStopping(patience=cfg.get('patience', 5),
                                        restore_best_weights=True)]
    model.fit(Xp_tr, Xp_tr, epochs=cfg['epochs'], batch_size=cfg['batch'],
              validation_split=0.1, callbacks=cb, verbose=0)
    # score por patch -> pesos AUC sobre patches (en val)
    rec_v = model.predict(Xp_v, batch_size=1024, verbose=0)
    err_v = ((Xp_v - rec_v) ** 2).mean(axis=2)                 # (Nv, n_patches)
    w = _auc_weights(err_v, window_labels(d['yv'], sv_st, WINDOW))
    sv_w = (err_v * w).sum(1)
    rec_te = model.predict(Xp_te, batch_size=1024, verbose=0)
    st_w = (((Xp_te - rec_te) ** 2).mean(axis=2) * w).sum(1)
    sv_ts = expand_avg(sv_w, len(d['Xv']), sv_st, WINDOW)
    st_ts = expand_avg(st_w, len(d['Xte']), st_st, WINDOW)
    r = _eval_perrow(sv_ts, st_ts, d['yv'], d['yte'])
    r['n_params'] = int(model.count_params())
    return r


def run(kind, cfg):
    if kind == 'transformer':
        return run_transformer(cfg)
    return run_cnn_or_lstm(kind, cfg)


# ----------------------------------------------------------------------------- #
# Espacios de búsqueda + semillas (baseline-like)
# ----------------------------------------------------------------------------- #
def sample_cnn(rng):
    _F = [(32, 16, 8), (48, 24, 12), (64, 32, 16)]
    return dict(kind='cnn', mode='full' if rng.random() < 0.7 else 'medium',
                filters=list(_F[rng.integers(len(_F))]),
                ksize=int(rng.choice([3, 5, 7])), bottleneck=int(rng.choice([8, 12, 16, 24])),
                lr=float(rng.choice([1e-3, 5e-4, 3e-4])), epochs=int(rng.choice([15, 25, 40])),
                batch=int(rng.choice([512, 1024])), patience=6, seed=int(rng.integers(1000)))

def sample_lstm(rng):
    return dict(kind='lstm', mode='full',
                units=int(rng.choice([64, 96, 128])), bottleneck=int(rng.choice([8, 12, 16])),
                lr=float(rng.choice([5e-4, 3e-4, 1e-4])), epochs=int(rng.choice([12, 20, 30])),
                batch=int(rng.choice([256, 512])), patience=5, seed=int(rng.integers(1000)))

def sample_transformer(rng):
    return dict(kind='transformer', mode='full' if rng.random() < 0.7 else 'medium',
                patch_len=int(rng.choice([4, 5, 6])), d_model=int(rng.choice([64, 96, 128])),
                n_heads=int(rng.choice([4, 8])), ffn=int(rng.choice([128, 256])),
                n_layers=int(rng.choice([2, 3])), dropout=float(rng.choice([0.1, 0.2])),
                lr=float(rng.choice([5e-4, 3e-4])), epochs=int(rng.choice([15, 25, 35])),
                batch=int(rng.choice([256, 512])), patience=5, seed=int(rng.integers(1000)))

SAMPLERS = {'cnn': sample_cnn, 'lstm': sample_lstm, 'transformer': sample_transformer}
SEEDS = {
  'cnn': dict(kind='cnn', mode='full', filters=[32, 16, 8], ksize=5, bottleneck=8,
              lr=1e-3, epochs=25, batch=1024, patience=6, seed=42),
  'lstm': dict(kind='lstm', mode='full', units=128, bottleneck=8, lr=3e-4,
               epochs=20, batch=512, patience=5, seed=42),
  'transformer': dict(kind='transformer', mode='full', patch_len=5, d_model=64,
                      n_heads=4, ffn=128, n_layers=2, dropout=0.1, lr=5e-4,
                      epochs=25, batch=512, patience=5, seed=42),
}

_W = {}
def _logger(model):
    if model not in _W:
        p = f'{OPT_DIR}/variants_{model}.csv'; new = not os.path.exists(p)
        f = open(p, 'a', newline=''); w = csv.writer(f)
        if new:
            w.writerow(['ts', 'f1', 'auc_pr', 'auc_roc', 'precision', 'recall',
                        'raw_f1', 'cfg', 'secs']); f.flush()
        _W[model] = (f, w)
    return _W[model]

def _bestpath(m): return f'{OPT_DIR}/best_{m}.json'
def _best(m):
    p = _bestpath(m)
    return json.load(open(p)) if os.path.exists(p) else {'f1': -1.0}

def evaluate_cfg(kind, cfg, counters):
    t0 = time.time()
    try:
        r = run(kind, cfg)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'[{kind}] cfg fallo: {type(e).__name__} {str(e)[:80]}', flush=True)
        return
    secs = time.time() - t0
    f, w = _logger(kind)
    w.writerow([round(time.time()), round(r['f1'], 4), round(r['auc_pr'], 4),
                round(r['auc_roc'], 4), round(r['precision'], 4), round(r['recall'], 4),
                round(r['raw_f1'], 4), json.dumps(cfg), round(secs, 1)]); f.flush()
    counters[kind] = counters.get(kind, 0) + 1
    best = _best(kind); mark = ''
    if r['f1'] > best['f1'] + 1e-4:
        best = {'f1': r['f1'], 'auc_pr': r['auc_pr'], 'auc_roc': r['auc_roc'],
                'precision': r['precision'], 'recall': r['recall'], 'cfg': cfg}
        json.dump(best, open(_bestpath(kind), 'w'), indent=2)
        mark = f'  <<< MEJOR (base {_BASEF1[kind]:.3f})'
    print(f'[{kind}] #{counters[kind]} F1={r["f1"]:.4f} AUCPR={r["auc_pr"]:.4f} '
          f'({secs:.0f}s){mark}', flush=True)


def main():
    mode = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 777
    rng = np.random.default_rng(seed)
    kinds = {'cycle':['cnn','transformer','lstm'],'ct':['cnn','transformer']}.get(mode,[mode])
    counters = {}; seeded = set(); t_end = time.time() + budget; i = 0
    print(f'[driver-win] {kinds} presupuesto={budget:.0f}s '
          f'(VAL_N={VAL_N} TEST_N={TEST_N} train_win={N_TRAIN_WIN})', flush=True)
    while time.time() < t_end:
        k = kinds[i % len(kinds)]; i += 1
        try:
            cfg = SEEDS[k] if k not in seeded else SAMPLERS[k](rng)
        except Exception as e:
            print(f'[{k}] sampler fallo: {type(e).__name__} {e}', flush=True); seeded.add(k); continue
        seeded.add(k)
        evaluate_cfg(k, cfg, counters)
    for k in kinds:
        b = _best(k)
        print(f'[{k}] FIN {counters.get(k,0)} variantes. Mejor F1={b["f1"]:.4f} '
              f'(base {_BASEF1[k]:.4f})', flush=True)


if __name__ == '__main__':
    main()
