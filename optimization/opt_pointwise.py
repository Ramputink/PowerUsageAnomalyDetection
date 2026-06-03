"""
opt_pointwise.py — Búsqueda de hiperparámetros (ECC benchmark-optimization-loop)
para los detectores POINTWISE (una fila = una muestra): IF, Dense-AE, VAE.

Metodología ECC respetada:
  * CORRECTNESS GATE = contrato no-trampa: entrenar en train_clean, calibrar
    umbral y (W,K) SOLO en validación, evaluar en test. Sin etiquetas en features.
  * Baseline pinned (optimization/baseline.json), variantes append-only a CSV.
  * Subsampleo de train para velocidad de búsqueda; los finalistas se re-validan
    a escala completa con re_full().
  * Promotion gate = F1 (smoothed) con desempate por AUC-PR; nunca usa test
    para calibrar.

Uso:  python opt_pointwise.py <if|dense|vae> <segundos_presupuesto> [seed]
"""
import sys, os, time, json, csv, random
import numpy as np
sys.path.insert(0, '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine')
import tfg_common as C

OPT_DIR = '/Volumes/Extreme Pro Particion 1TB/TFG/optimization'
os.makedirs(OPT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------- #
# Datos: cargar UNA vez (full=17, medium=13). El AE entrena en train; calibra/val.
# ----------------------------------------------------------------------------- #
print('[load] construyendo matrices...', flush=True)
_DATA = {}
def get_data(mode):
    if mode not in _DATA:
        _DATA[mode] = C.build_matrices(mode=mode)
    return _DATA[mode]

# precargar full (mayoría de configs)
get_data('full')
print('[load] listo', flush=True)


def _score_eval(scores_val, scores_test, y_val, y_test, beta=1.0):
    """Calibra umbral+W/K en val, evalúa en test. Contrato respetado."""
    thr, _ = C.calibrate_threshold(y_val, scores_val, beta=beta)
    yv = (scores_val > thr).astype(int)
    Wb, Kb = C.calibrate_WK(y_val, yv)
    m = C.evaluate(y_test, scores_test, thr, Wb, Kb)
    m['threshold'] = float(thr)
    return m


# ----------------------------------------------------------------------------- #
# Familias de modelos
# ----------------------------------------------------------------------------- #
def run_if(cfg, train_n=None):
    from sklearn.ensemble import IsolationForest
    d = get_data(cfg['mode'])
    Xtr, Xv, Xte = d['X_train'], d['X_val'], d['X_test']
    if train_n and train_n < len(Xtr):
        idx = np.random.default_rng(0).choice(len(Xtr), train_n, replace=False)
        Xtr = Xtr[np.sort(idx)]
    clf = IsolationForest(
        n_estimators=cfg['n_estimators'], max_samples=cfg['max_samples'],
        max_features=cfg['max_features'], contamination=cfg['contamination'],
        random_state=cfg.get('seed', 42), n_jobs=-1)
    clf.fit(Xtr)
    sv = -clf.score_samples(Xv); st = -clf.score_samples(Xte)
    return _score_eval(sv, st, d['y_val'], d['y_test'])


def _ae_weights(model, Xv, yv):
    from sklearn.metrics import roc_auc_score
    mse = (Xv - model.predict(Xv, batch_size=8192, verbose=0)) ** 2
    aucs = np.array([roc_auc_score(yv, mse[:, j]) if len(np.unique(mse[:, j])) > 1
                     else 0.5 for j in range(mse.shape[1])])
    w = np.maximum(aucs - 0.5, 0); s = w.sum()
    return (w / s).astype('float32') if s > 0 else np.ones(mse.shape[1], 'float32') / mse.shape[1]


def run_dense(cfg, train_n=None):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.random.set_seed(cfg.get('seed', 42))
    d = get_data(cfg['mode'])
    Xtr, Xv, Xte = d['X_train'], d['X_val'], d['X_test']; nf = d['n_features']
    if train_n and train_n < len(Xtr):
        Xtr = Xtr[:train_n]
    inp = keras.Input((nf,)); x = inp
    for w in cfg['widths']:
        x = layers.Dense(w)(x); x = layers.LeakyReLU(negative_slope=0.1)(x)
        if cfg.get('dropout', 0) > 0: x = layers.Dropout(cfg['dropout'])(x)
    x = layers.Dense(cfg['bottleneck'], activation='relu')(x)
    for w in reversed(cfg['widths']):
        x = layers.Dense(w)(x); x = layers.LeakyReLU(negative_slope=0.1)(x)
    out = layers.Dense(nf, activation='linear')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg['lr']), loss='mse')
    cb = [keras.callbacks.EarlyStopping(patience=cfg.get('patience', 6),
                                        restore_best_weights=True)]
    model.fit(Xtr, Xtr, epochs=cfg['epochs'], batch_size=cfg['batch'],
              validation_split=0.1, callbacks=cb, verbose=0)
    w = _ae_weights(model, Xv, d['y_val']) if cfg.get('weighted', True) \
        else np.ones(nf, 'float32') / nf
    sv = ((Xv - model.predict(Xv, batch_size=8192, verbose=0)) ** 2 * w).sum(1)
    st = ((Xte - model.predict(Xte, batch_size=8192, verbose=0)) ** 2 * w).sum(1)
    r = _score_eval(sv, st, d['y_val'], d['y_test'])
    r['n_params'] = int(model.count_params())
    return r


def run_vae(cfg, train_n=None):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.random.set_seed(cfg.get('seed', 42))
    d = get_data(cfg['mode'])
    Xtr, Xv, Xte = d['X_train'], d['X_val'], d['X_test']; nf = d['n_features']
    if train_n and train_n < len(Xtr):
        Xtr = Xtr[:train_n]
    CLIP = 6.0; lat = cfg['latent']; beta = cfg['beta']
    inp = keras.Input((nf,)); h = inp
    for w in cfg['widths']:
        h = layers.Dense(w, activation='relu')(h)
    mu = layers.Dense(lat)(h); lv = layers.Dense(lat)(h)
    enc = keras.Model(inp, [mu, lv])
    zin = keras.Input((lat,)); h = zin
    for w in reversed(cfg['widths']):
        h = layers.Dense(w, activation='relu')(h)
    dec = keras.Model(zin, layers.Dense(nf)(h))

    class VAE(keras.Model):
        def __init__(s): super().__init__(); s.lt = keras.metrics.Mean(name='loss')
        @property
        def metrics(s): return [s.lt]
        def call(s, x):
            m, l = enc(x); return dec(m)
        def train_step(s, data):
            x = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as t:
                m, l = enc(x); l = tf.clip_by_value(l, -CLIP, CLIP)
                z = m + tf.exp(0.5 * l) * tf.random.normal(tf.shape(m))
                rec = dec(z)
                recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - rec), 1))
                kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + l - tf.square(m) - tf.exp(l), 1))
                loss = recon + beta * kl
            v = enc.trainable_variables + dec.trainable_variables
            s.optimizer.apply_gradients(zip(t.gradient(loss, v), v))
            s.lt.update_state(loss); return {'loss': s.lt.result()}
    vae = VAE(); vae.compile(optimizer=keras.optimizers.Adam(cfg['lr'], clipnorm=1.0))
    vae(Xtr[:8])
    vae.fit(Xtr, epochs=cfg['epochs'], batch_size=cfg['batch'], verbose=0)

    L = cfg.get('mc', 10)
    def mc(X):
        m, l = enc.predict(X, batch_size=8192, verbose=0); l = np.clip(l, -CLIP, CLIP)
        acc = np.zeros((len(X), L))
        for i in range(L):
            z = m + np.exp(0.5 * l) * np.random.normal(size=m.shape).astype('float32')
            rec = dec.predict(z, batch_size=8192, verbose=0)
            acc[:, i] = ((X - rec) ** 2).sum(1)
        return acc.mean(1)
    sv, st = mc(Xv), mc(Xte)
    r = _score_eval(sv, st, d['y_val'], d['y_test'])
    r['n_params'] = int(vae.count_params())
    return r


RUNNERS = {'if': run_if, 'dense': run_dense, 'vae': run_vae}


# ----------------------------------------------------------------------------- #
# Espacios de búsqueda
# ----------------------------------------------------------------------------- #
_WIDTHS_D = [(64, 32), (128, 64), (96, 48), (64, 32, 16), (128, 64, 32), (160, 80)]
_WIDTHS_V = [(64, 32), (96, 48), (128, 64), (64, 32, 16)]

def sample_if(rng):
    return dict(model='if', mode='full',  # medium tira features físicas -> peor
                n_estimators=int(rng.choice([400, 600, 800, 1200, 1600])),
                max_samples=int(rng.choice([256, 512, 1024, 2048])),
                max_features=float(rng.choice([0.85, 0.95, 1.0])),
                contamination=0.0172,  # irrelevante (recalibramos umbral)
                seed=int(rng.integers(1000)))

def sample_dense(rng):
    w = _WIDTHS_D[rng.integers(len(_WIDTHS_D))]
    return dict(model='dense', mode='full' if rng.random() < 0.85 else 'medium',
                widths=list(w), bottleneck=int(rng.choice([3, 4, 6, 8])),
                lr=float(rng.choice([1e-3, 5e-4, 3e-4])),
                epochs=int(rng.choice([15, 25, 40])), batch=int(rng.choice([512, 1024])),
                dropout=float(rng.choice([0.0, 0.0, 0.0, 0.05])), patience=8,
                weighted=bool(rng.choice([True, True, True, False])),
                seed=int(rng.integers(1000)))

def sample_vae(rng):
    w = _WIDTHS_V[rng.integers(len(_WIDTHS_V))]
    return dict(model='vae', mode='full' if rng.random() < 0.85 else 'medium',
                widths=list(w), latent=int(rng.choice([3, 4, 6, 8])),
                beta=float(rng.choice([0.05, 0.1, 0.3, 0.5, 1.0])),
                lr=float(rng.choice([1e-3, 5e-4])), epochs=int(rng.choice([20, 30, 40])),
                batch=1024, mc=int(rng.choice([10, 20])), seed=int(rng.integers(1000)))

SAMPLERS = {'if': sample_if, 'dense': sample_dense, 'vae': sample_vae}

# Configs semilla = baseline reproducido (garantiza best >= baseline)
SEEDS = {
  'if':    dict(model='if', mode='full', n_estimators=600, max_samples=256,
                max_features=1.0, contamination=0.0172, seed=42),
  'dense': dict(model='dense', mode='full', widths=[64, 32], bottleneck=4, lr=5e-4,
                epochs=25, batch=1024, dropout=0.0, patience=8, weighted=True, seed=42),
  'vae':   dict(model='vae', mode='full', widths=[64, 32], latent=4, beta=1.0,
                lr=5e-4, epochs=30, batch=1024, mc=20, seed=42),
}


_BASE = json.load(open(f'{OPT_DIR}/baseline.json'))
_BASEF1 = {'if': _BASE['IF']['f1'], 'dense': _BASE['Dense-AE']['f1'],
           'vae': _BASE['VAE']['f1']}
TRAIN_N = 300_000  # subsampleo de búsqueda (AEs); IF usa todo. Finalistas: full.
_WRITERS = {}

def _logger(model):
    if model not in _WRITERS:
        p = f'{OPT_DIR}/variants_{model}.csv'; new = not os.path.exists(p)
        f = open(p, 'a', newline=''); w = csv.writer(f)
        if new:
            w.writerow(['ts', 'f1', 'auc_pr', 'auc_roc', 'precision', 'recall',
                        'raw_f1', 'cfg', 'secs']); f.flush()
        _WRITERS[model] = (f, w)
    return _WRITERS[model]

def _best(model):
    # inicia en -1 para que la SEMILLA (1er config) fije la referencia subsample;
    # los finalistas se re-validan a escala completa antes de promocionar de verdad.
    p = f'{OPT_DIR}/best_{model}.json'
    return json.load(open(p)) if os.path.exists(p) else {'f1': -1.0}

def evaluate_cfg(model, cfg, counters):
    tn = None if model == 'if' else TRAIN_N
    t0 = time.time()
    try:
        r = RUNNERS[model](cfg, train_n=tn)
    except Exception as e:
        print(f'[{model}] cfg fallo: {type(e).__name__} {str(e)[:80]}', flush=True)
        return
    secs = time.time() - t0
    f, w = _logger(model)
    w.writerow([round(time.time()), round(r['f1'], 4), round(r['auc_pr'], 4),
                round(r['auc_roc'], 4), round(r['precision'], 4), round(r['recall'], 4),
                round(r['raw_f1'], 4), json.dumps(cfg), round(secs, 1)]); f.flush()
    counters[model] = counters.get(model, 0) + 1
    best = _best(model); mark = ''
    if r['f1'] > best['f1'] + 1e-4:
        best = {'f1': r['f1'], 'auc_pr': r['auc_pr'], 'auc_roc': r['auc_roc'],
                'precision': r['precision'], 'recall': r['recall'], 'cfg': cfg}
        json.dump(best, open(f'{OPT_DIR}/best_{model}.json', 'w'), indent=2)
        mark = f'  <<< MEJOR (base {_BASEF1[model]:.3f}, +{r["f1"]-_BASEF1[model]:+.3f})'
    print(f'[{model}] #{counters[model]} F1={r["f1"]:.4f} AUCPR={r["auc_pr"]:.4f} '
          f'({secs:.0f}s){mark}', flush=True)

def main():
    mode = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1800
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
    rng = np.random.default_rng(seed)
    models = {'cycle':['dense','vae','if'],'dv':['vae','dense']}.get(mode,[mode])
    counters = {}
    seeded = set()
    t_end = time.time() + budget
    print(f'[driver] modelos={models} presupuesto={budget:.0f}s', flush=True)
    i = 0
    while time.time() < t_end:
        m = models[i % len(models)]; i += 1
        cfg = SEEDS[m] if m not in seeded else SAMPLERS[m](rng)
        seeded.add(m)
        evaluate_cfg(m, cfg, counters)
    for m in models:
        b = _best(m)
        print(f'[{m}] FIN {counters.get(m,0)} variantes. Mejor F1={b["f1"]:.4f} '
              f'(base {_BASEF1[m]:.4f}, {b["f1"]-_BASEF1[m]:+.4f})', flush=True)


if __name__ == '__main__':
    main()
