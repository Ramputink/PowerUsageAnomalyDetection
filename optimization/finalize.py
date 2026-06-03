"""
finalize.py — Re-valida la MEJOR config de un modelo a ESCALA COMPLETA
(train completo, val/test completos, stride de evaluación 1), produciendo las
métricas reales que se comparan con el baseline antes de promover (ECC:
"Rerun baseline vs winner -> confirm delta").

Uso: python finalize.py <if|dense|vae|cnn|transformer|lstm>
Salida: optimization/final_<model>.json  +  línea en OPTIMIZATION_LOG.md
"""
import sys, os, json, time
sys.path.insert(0, '/Volumes/Extreme Pro Particion 1TB/TFG/UCIrvine')
OPT = '/Volumes/Extreme Pro Particion 1TB/TFG/optimization'

model = sys.argv[1]
best = json.load(open(f'{OPT}/best_{model}.json'))
cfg = best['cfg']
base = json.load(open(f'{OPT}/baseline.json'))
BMAP = {'if': 'IF', 'dense': 'Dense-AE', 'vae': 'VAE', 'cnn': 'CNN-AE',
        'transformer': 'Transformer-AE', 'lstm': 'LSTM-AE'}
b = base[BMAP[model]]
print(f'[finalize {model}] best subset F1={best["f1"]:.4f} | cfg={cfg}', flush=True)
t0 = time.time()

if model in ('if', 'dense', 'vae'):
    import opt_pointwise as O
    r = O.RUNNERS[model](cfg, train_n=None)   # train completo, val/test completos
else:
    import opt_window as W
    # escala completa: sin subset, stride 1, todas las ventanas de train
    import numpy as np
    W.VAL_N = 10**9; W.TEST_N = 10**9; W.EVAL_STRIDE = 1; W.N_TRAIN_WIN = 10**9
    W._D.clear()
    r = W.run(model, cfg)

dt = time.time() - t0
out = {'model': model, 'cfg': cfg,
       'f1': round(r['f1'], 4), 'auc_pr': round(r['auc_pr'], 4),
       'auc_roc': round(r['auc_roc'], 4), 'precision': round(r['precision'], 4),
       'recall': round(r['recall'], 4), 'raw_f1': round(r['raw_f1'], 4),
       'temporal_W': r.get('temporal_W'), 'temporal_K': r.get('temporal_K'),
       'baseline_f1': b['f1'], 'baseline_auc_pr': b['auc_pr'],
       'delta_f1': round(r['f1'] - b['f1'], 4),
       'delta_auc_pr': round(r['auc_pr'] - b['auc_pr'], 4),
       'finalize_secs': round(dt, 1)}
json.dump(out, open(f'{OPT}/final_{model}.json', 'w'), indent=2)
verdict = 'MEJORA' if out['delta_f1'] > 0.002 else ('=' if abs(out['delta_f1']) <= 0.002 else 'REGRESION')
line = (f"- **[finalize {model}]** full-scale F1={out['f1']:.4f} (base {b['f1']:.4f}, "
        f"{out['delta_f1']:+.4f}) AUC-PR={out['auc_pr']:.4f} ({out['delta_auc_pr']:+.4f}) "
        f"-> {verdict}  cfg={json.dumps(cfg)}")
with open(f'{OPT}/OPTIMIZATION_LOG.md', 'a') as f:
    f.write('\n' + line + '\n')
print(line, flush=True)
print(f'[finalize {model}] {verdict} en {dt:.0f}s', flush=True)
