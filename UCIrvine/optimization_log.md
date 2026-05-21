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

## Iter 1 — 2026-05-21 13:40 — NB05 LSTM-AE — Palanca A (joint threshold × W × K)

- **F1 antes:** 0.6514  (raw 0.6244, P=0.5055, R=0.9155, W=21, K=21, thr=0.4676)
- **F1 después:** 0.7415  (raw 0.7403, P=0.6510, R=0.8611, W=3, K=3, thr=0.7636)  (Δ = **+0.0901**)
- **Cambio aplicado:**
  - Cell 15: `calibrate_threshold(beta=2.0)` → `beta=1.0` (umbral inicial F1-óptimo a nivel ventana).
  - Cell 21 (apéndice): joint search sobre `(threshold ∈ 60 cuantiles del PR-curve val, W ∈ {3,5,7,9,11,15,21,31,45,60}, K ∈ [2..W])`, maximizando F1 smoothed timestep en val. Si el F1 conjunto supera al del grid (W,K) original, sobrescribimos `best_thr, best_W, best_K`.
- **Re-ejecución completa:** nbconvert --execute --inplace OK (601s real, 1.5MB notebook escrito)
- **CHECKLIST anti-trampa:** TODOS OK
  - Notebook entero ejecutado sin error.
  - F1 = `f1_score(y_test, y_pred_smooth)` sobre `test_with_attacks`.
  - Scaler fit en `train_clean` (sin val ni test). Modelo entrenado solo con `X_train_w`.
  - Umbral + (W,K) calibrados solo sobre val.
  - Ninguna columna label/attack_type/episode/severity en features (`compute_features` solo usa sensores + derivadas físicas/temporales).
  - NB02 y CSV de `data/` sin cambios (`git status` limpio en esos paths).
  - W=3 ≤ 60.
  - Test set tamaño íntegro (df_test cargado sin filtros).
  - Reproducible vía nbconvert.
- **Veredicto:** ACEPTADO  (commit `a2564b1`)
- **Análisis:** El umbral previo (0.4676) era demasiado permisivo por usar `beta=2.0` (F2 prioriza recall). El joint search en val recolocó el umbral en 0.7636 (≈P90 de scores limpios), recuperando precision de 0.51 a 0.65 con coste menor en recall (0.92→0.86). El W=21/K=21 era una sobre-corrección del defecto de umbral — con un umbral correcto, basta W=3/K=3 (consenso en 3 minutos). Sigue muy por debajo del 0.975 objetivo; los AEs no supervisados parecen estar lejos del techo aunque mejor calibrados.

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
