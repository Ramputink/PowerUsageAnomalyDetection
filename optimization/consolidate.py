"""
consolidate.py — Agrega baseline vs optimizado de todos los detectores en una
tabla comparativa reproducible (ECC: confirmar delta). Fuente de cada optimizado:
  * final_<model>.json  (window finalizado a escala completa)  -> preferente
  * best_<model>.json   (pointwise; IF ya es full-scale)       -> si no hay final
Escribe optimization/optimized_results.json e imprime la tabla.
"""
import os, json
OPT = '/Volumes/Extreme Pro Particion 1TB/TFG/optimization'
base = json.load(open(f'{OPT}/baseline.json'))
NAMES = {'if': 'IF', 'kmeans_if': 'KMeans+IF', 'dense': 'Dense-AE', 'vae': 'VAE',
         'cnn': 'CNN-AE', 'transformer': 'Transformer-AE', 'lstm': 'LSTM-AE'}
FULLSCALE = {'if'}   # pointwise que ya evalúa full val/test

def opt_metrics(m):
    fp = f'{OPT}/final_{m}.json'
    if os.path.exists(fp):
        d = json.load(open(fp)); return d, 'final'
    bp = f'{OPT}/best_{m}.json'
    if os.path.exists(bp) and m in FULLSCALE:
        d = json.load(open(bp)); return d, 'best(full)'
    return None, None

rows = []
for m, disp in NAMES.items():
    if disp not in base:
        continue
    b = base[disp]
    o, src = opt_metrics(m)
    if o is None:
        rows.append({'model': disp, 'base_f1': b['f1'], 'base_aucpr': b['auc_pr'],
                     'opt_f1': None, 'opt_aucpr': None, 'd_f1': None, 'd_aucpr': None,
                     'src': 'pendiente'})
        continue
    rows.append({'model': disp, 'base_f1': round(b['f1'], 4),
                 'base_aucpr': round(b['auc_pr'], 4),
                 'opt_f1': round(o['f1'], 4), 'opt_aucpr': round(o['auc_pr'], 4),
                 'd_f1': round(o['f1'] - b['f1'], 4),
                 'd_aucpr': round(o['auc_pr'] - b['auc_pr'], 4), 'src': src,
                 'cfg': o.get('cfg')})

json.dump(rows, open(f'{OPT}/optimized_results.json', 'w'), indent=2)
print(f"{'Modelo':<16}{'F1 base':>9}{'F1 opt':>9}{'ΔF1':>9}{'AUCPR base':>12}{'AUCPR opt':>11}{'Δ':>9}  src")
for r in rows:
    if r['opt_f1'] is None:
        print(f"{r['model']:<16}{r['base_f1']:>9.3f}{'—':>9}{'—':>9}{r['base_aucpr']:>12.3f}{'—':>11}{'—':>9}  {r['src']}")
    else:
        print(f"{r['model']:<16}{r['base_f1']:>9.3f}{r['opt_f1']:>9.3f}{r['d_f1']:>+9.3f}"
              f"{r['base_aucpr']:>12.3f}{r['opt_aucpr']:>11.3f}{r['d_aucpr']:>+9.3f}  {r['src']}")
print('\n-> optimization/optimized_results.json')
