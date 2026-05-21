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

## Iter 2 — 2026-05-21 13:49 — NB09 LightGBM — Palanca B+A (tuned params + joint thr/W/K)
- **F1 antes:** 0.8961  | **F1 después:** 0.9010 (Δ +0.0049)
- **Cambio:** `learning_rate=0.05→0.02`, `num_leaves=31→127`, `max_depth=-1→12`, `min_data_in_leaf=50→20`, `num_boost_round=2000→5000`, `early_stopping=50→100`. Joint search threshold × W × K sobre val.
- **Veredicto:** ACEPTADO (commit `d3c3227`). best_iteration=60 (vs 35 antes).
- **CHECKLIST:** OK (NB02/data intactos, calibración solo en val, sin labels en features, W=3≤60).

## Iter 3 — 2026-05-21 13:54 — NB09 LightGBM — Palanca D (lag+rolling features)
- **F1 antes:** 0.9010  | **F1 después:** 0.9104 (Δ +0.0094)
- **Cambio:** Añadidas features: `gap_lag1, gap_lag5, gap_diff5, gap_diff15, gap_roll5_{mean,std}, gap_roll30_{mean,std}, v_roll5_std, i_roll5_std`. 27 features totales (vs 17).
- **Veredicto:** ACEPTADO (commit `eaa0e54`). best_iteration=225.
- **CHECKLIST:** OK.

## Iter 4 NB09 — variantes rechazadas
- **4a (scale_pos_weight):** F1=0.9104 (Δ=0). Revertido.
- **4b (multi-seed ensemble 5 LGBMs):** F1=0.9032 (Δ −0.0072). Revertido.
- **4c (DART boosting):** F1=0.9099 (Δ −0.0005). Revertido.
- **4d (más features 38 total):** F1=0.9050 (Δ −0.0054). Revertido (sobreajuste con tantas features).
- **4e (train_clean al supervised):** F1=0.8957 (Δ −0.0147). Revertido.

## Iter NB07 — 2026-05-21 16:21 — Palanca C+A (bugfix weights + mean expansion + joint)
- **F1 antes:** 0.2743 (estado roto)  | **F1 después:** 0.4346 (Δ **+0.1603**)
- **Bug encontrado:** Cell 14 calculaba `weights = err_per_feat / err_per_feat.sum()` usando MSE en CLEAN TRAIN — esto pesaba features que el AE reconstruía mal en limpio (ruidosas), no las discriminantes. Daba 94% peso a `gap_diff1` (derivada ruidosa). Cell 16 además usaba `np.maximum` en la expansión window→timestep (OR-like, infla FPs).
- **Cambio:** Pesos por AUC-ROC individual en val (igual NB05). Expansión por PROMEDIO de scores. Joint search (thr,W,K).
- **Veredicto:** ACEPTADO (commit `e2cff12`). Todas las features con peso ≈ 1/14 (AUC≈0.5 en todas — el CNN no discrimina mucho por feature individual, pero el score combinado funciona).
- **CHECKLIST:** OK.

## Iter NB03 — 2026-05-21 ~17:20 — Palanca A+B (joint thr/W/K + IF tuning)
- **F1 antes:** 0.7433  | **F1 después:** 0.7544 (Δ +0.0111)
- **Cambio:** `n_estimators=300→600`, `max_features=0.8→1.0`, `max_samples='auto'→256` (paper original). Joint search.
- **Veredicto:** ACEPTADO (commit `e57ccf9`).
- **CHECKLIST:** OK.

## Iter NB03b — 2026-05-21 ~17:35 — Palanca A (joint thr/W/K)
- **F1 antes:** 0.7684  | **F1 después:** 0.7895 (Δ +0.0211)
- **Cambio:** Joint search (thr × W × K) sobre val.
- **Veredicto:** ACEPTADO (commit `c4bf42d`).
- **CHECKLIST:** OK.

## Iter NB08 — 2026-05-21 ~17:45 — Palanca A (joint thr/W/K)
- **F1 antes:** 0.5872  | **F1 después:** 0.6035 (Δ +0.0163)
- **Cambio:** Joint search.
- **Veredicto:** ACEPTADO (commit `c21a262`).
- **CHECKLIST:** OK.

## Iter NB04 — 2026-05-21 ~17:50 — Palanca A (joint thr/W/K) — REJECTED
- **F1 antes:** 0.8157  | **F1 después:** 0.8103 (Δ **−0.0054**)
- **Cambio:** Joint search.
- **Veredicto:** REVERTIDO. La búsqueda joint sobreajustó val (Goodhart-like).

## Iter NB10 — 2026-05-21 ~17:55 — Reconstrucción stack con bases mejoradas — REGRESIÓN
- **F1 antes:** 0.8428  | **F1 después:** 0.8115 (Δ **−0.0313**)
- **Problema:** El stack original `[if, cnn_ae, transformer]` cambió a `[if, dense_ae, lstm_ae, transformer]` tras mi fix de feature dimensions, perdiendo `cnn_ae` (causa raíz no identificada — el CNN-AE carga OK aislado pero NB10 lo excluye silenciosamente). El nuevo stack sin CNN da F1 más bajo.
- **Veredicto:** Estado regresado, no recuperable cleanamente (la versión original del notebook ya no está en git history; sólo existían en working tree). Documentado como techo provisional 0.8428 (baseline); 0.8115 (estado actual con bases mejoradas pero stack incompleto).

## Iter 2 NB05 — 2026-05-21 ~18:25 — Palanca B (deeper 2-layer LSTM, bottleneck=8)
- **F1 antes:** 0.7415  | **F1 después:** 0.7452 (Δ +0.0037)
- **Cambio:** 2-layer encoder/decoder LSTM (units=128/64), bottleneck=8 (vs 16), lr=3e-4. 259K params (vs 57K).
- **Veredicto:** ACEPTADO (commit `3acf247`). Mejora pequeña pero ≥ +0.002 → no es techo aún.
- **CHECKLIST:** OK.

## Iter 2 NB07 — 2026-05-22 ~00:11 — Palanca C (abs(AUC-0.5) weights)
- **F1 antes:** 0.4346  | **F1 después:** 0.4354 (Δ +0.0008, marginal)
- **Cambio:** `np.maximum(auc-0.5, 0)` → `np.abs(auc-0.5)` para incluir discriminadores invertidos.
- **Veredicto:** ACEPTADO (commit `921edc4`). Δ < +0.002 → cuenta como iteración sin mejora real.

## Iter 2 NB04 — 2026-05-22 ~00:00 — Palanca C (abs(AUC-0.5)) — REJECTED
- **F1 antes:** 0.8157  | **F1 después:** 0.7999 (Δ −0.0158)
- **Cambio:** `np.maximum` → `np.abs` en weights por feature.
- **Veredicto:** REVERTIDO. Features con AUC<0.5 metieron ruido.

## Iter 3 NB04 — 2026-05-22 ~00:30 — Palanca B (wider arch) — REJECTED
- **F1 antes:** 0.8157  | **F1 después:** 0.7930 (Δ −0.0227)
- **Cambio:** Encoder/decoder 128→64→32→bottleneck=6 (vs 64→32→3).
- **Veredicto:** REVERTIDO. Mayor capacidad permite que el AE reconstruya también ataques.

---

## Resumen final por detector

| NB | Baseline | Final | Δ | Iters acept. | Iters rech. | Estado |
|---|---:|---:|---:|---:|---:|---|
| 03 | 0.7433 | **0.7544** | +0.0111 | 1 | 0 | techo provisional |
| 03b | 0.7684 | **0.7895** | +0.0211 | 1 | 0 | techo provisional |
| 04 | 0.8157 | **0.8157** | 0 | 0 | 3 | techo (3 palancas probadas) |
| 05 | 0.6514 | **0.7452** | **+0.0938** | 2 | 0 | techo provisional |
| 07 | 0.2743 | **0.4354** | **+0.1611** | 2 | 0 | bug arreglado, techo provisional |
| 08 | 0.5872 | **0.6035** | +0.0163 | 1 | 0 | techo provisional |
| 09 | 0.8961 | **0.9104** | +0.0143 | 2 | 5 | **TECHO HONESTO SATURADO** |
| 10 | 0.8428 | 0.8115* | −0.0313 | 0 | 1 | regresión, no recuperable cleanamente |

**0 detectores alcanzan F1 ≥ 0.975.** Mejor: NB09 LightGBM (yardstick supervisado)
con F1 = 0.9104. Veredicto completo en `F1_REPORT.md`.

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
