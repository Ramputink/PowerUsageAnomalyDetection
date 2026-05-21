# Optimization Log — Banco de detectores FDI (TFG)

Bitácora append-only de iteraciones para maximizar F1 (smoothed, test) en los
8 detectores del banco, bajo el CONTRATO CONGELADO del `/goal`.

**Objetivo:** ≥ 3 notebooks con `metrics_*.json["f1"] >= 0.975` (verificados por
re-ejecución completa) **o** declarar TECHO HONESTO con evidencia.

Fecha de inicio: 2026-05-21

---

## Iter 0 — Baseline (estado pre-optimización)

F1 smoothed sobre `test_with_attacks` leído directamente de `data/metrics_*.json`:

| # | Notebook | Modelo | F1 (smoothed) | Raw F1 | Precision | Recall | AUC-PR | W | K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 03 | Isolation Forest | IF + voting | **0.7433** | 0.7359 | 0.7348 | 0.7520 | 0.7586 | 3 | 2 |
| 03b | KMeans+IF | K=3 per-cluster IF | **0.7684** | 0.7563 | 0.7688 | 0.7680 | 0.7644 | 5 | 3 |
| 04 | Dense-AE | DAE v3 weighted MSE | **0.8157** | 0.7810 | 0.8273 | 0.8045 | 0.8553 | 3 | 3 |
| 05 | LSTM-AE | LSTM-AE weighted MSE | **0.6514** | 0.6244 | 0.5055 | 0.9155 | 0.8968 | 21 | 21 |
| 07 | CNN-AE | Conv1D weighted MSE | **0.2743** | 0.2744 | 0.1664 | 0.7795 | 0.2231 | 3 | 2 |
| 08 | Transformer-AE | PatchTST-lite | **0.5872** | 0.5765 | 0.5233 | 0.6688 | 0.5756 | 21 | 21 |
| 09 | LightGBM | Supervisado (yardstick) | **0.8961** | 0.8964 | 0.9424 | 0.8541 | 0.9302 | 3 | 2 |
| 10 | Stacking | Meta-LightGBM[if,cnn,transformer] | **0.8428** | 0.8422 | 0.8319 | 0.8539 | 0.9204 | 3 | 2 |

### Observaciones del baseline

- **NB07 CNN-AE colapsado:** `feature_weights[gap_diff1] = 0.9413` (94 % del peso
  en una sola feature). AUC-ROC = 0.557 (apenas mejor que azar). Hay un problema
  estructural — no es solo recalibración. Probable bug en el cálculo del score
  ponderado o en el escalado de features.
- **NB05 LSTM-AE muy mal calibrado:** W=21, K=21 (consenso unánime en ventana de
  21 min) y aun así smoothed precision = 0.51 (raw 0.46). El threshold elegido
  produce raw_recall=0.95 → el modelo identifica casi todo el test como anomalía.
  Re-grid de (W,K,threshold) con criterio F1 (no F2) debería subirlo.
- **NB08 Transformer-AE en W=21,K=21:** mismo patrón que LSTM, precision baja.
  La detección por paciente (raw recall 0.70) es la que falla, no el smoothing.
- **NB09 LightGBM (0.8961)** es el candidato más realista a 0.975. Gap 0.079.
- **Gap medio de los 8 al objetivo: 0.260** — sin retroceder a palancas
  prohibidas, llegar a 3 detectores ≥ 0.975 es ambicioso pero no imposible si
  NB09/NB10 cooperan y un AE despunta tras recalibración.

### Snapshot git

```
master 2026-05-21 pre-iter 0 baseline
```

---

## Plantilla de entrada por iteración

```
## Iter <n> — <YYYY-MM-DD HH:MM> — <NB nombre> — Palanca <X: descripción corta>

- **F1 antes:** 0.xxxx
- **F1 después:** 0.xxxx  (Δ = ±0.xxxx)
- **Cambio aplicado:** <descripción concreta, qué celdas, qué hiperparámetros>
- **Re-ejecución completa:** nbconvert --execute --inplace OK / FAIL
- **CHECKLIST anti-trampa:** [todos los puntos OK] / [punto X falló → REVERT]
- **Veredicto:** ACEPTADO (commit hash) / REVERTIDO / INVÁLIDO
- **Análisis (1 frase):** <por qué subió/bajó/falló>
```
