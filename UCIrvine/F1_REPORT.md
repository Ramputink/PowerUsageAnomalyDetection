# F1_REPORT — Optimización iterativa del banco de detectores FDI

**Proyecto:** Detección de anomalías de consumo eléctrico y mitigación de ataques
FDI en edge computing (TFG).
**Objetivo del `/goal`:** ≥ 3 notebooks del banco con `metrics_*.json["f1"] >= 0.975`,
verificados re-ejecutando cada notebook entero con `nbconvert --execute`.
**Fecha de cierre:** 2026-05-22
**Bitácora detallada:** `optimization_log.md`

---

## Resumen ejecutivo

**Resultado:** **0 detectores alcanzan F1 ≥ 0.975 de forma honesta.** El detector
más cercano es **NB09 LightGBM (supervisado, yardstick) con F1 = 0.9104** tras 2
palancas aceptadas y 5 rechazadas. El objetivo de 0.975 no es alcanzable bajo el
**CONTRATO CONGELADO** sin recurrir a palancas prohibidas (calibrar mirando
test, fuga de etiquetas en features, recortar test, etc.).

**Veredicto de honestidad:** Ningún F1 reportado en este informe viola el
contrato. Toda calibración (umbral, W, K) se hizo exclusivamente sobre el set
de validación. Las etiquetas (`label`, `attack_type`, `episode_id`, `severity`,
`GAP_original`) nunca entraron como feature de un detector. NB02 y los CSV de
`data/` (train/val/test) permanecen byte-idénticos al baseline. Todo F1
reportado se reproduce ejecutando el notebook entero desde la primera celda.

---

## Tabla final — Banco completo

| # | Notebook | F1 baseline | F1 final | Δ | Iteraciones | Estado |
|---|---|---:|---:|---:|---:|---|
| 03 | Isolation Forest | 0.7433 | **0.7544** | +0.0111 | 1 acept. | TECHO PROVISIONAL (1 palanca) |
| 03b | KMeans+IF | 0.7684 | **0.7895** | +0.0211 | 1 acept. | TECHO PROVISIONAL (1 palanca) |
| 04 | Dense-AE | 0.8157 | **0.8157** | 0 | 0 acept., 3 rech. | TECHO PROVISIONAL (3 palancas A/B/C probadas) |
| 05 | LSTM-AE | 0.6514 | **0.7452** | +0.0938 | 2 acept. | TECHO PROVISIONAL (2 palancas) |
| 07 | CNN-AE | 0.2743 | **0.4354** | **+0.1611** | 2 acept. | BUG arreglado, techo provisional |
| 08 | Transformer-AE | 0.5872 | **0.6035** | +0.0163 | 1 acept. | TECHO PROVISIONAL |
| 09 | LightGBM (yardstick) | 0.8961 | **0.9104** | +0.0143 | 2 acept., 5 rech. | TECHO HONESTO (saturado) |
| 10 | Stacking | 0.8428 | **(0.8115)** | -0.0313 | regresión | REQUIERE TRABAJO ADICIONAL |

> **Nota:** NB10 Stacking sufrió una regresión durante la iteración por la
> pérdida silenciosa de CNN-AE del stack tras corregir las dimensiones de
> features de Dense/LSTM-AE. El estado original (F1=0.8428 con
> `[if, cnn_ae, transformer]`) no es recuperable cleanamente porque el notebook
> original no estaba en git history — sólo existía en working tree. Se
> documenta como estado **provisional regresado**.

---

## Detalle por notebook

### NB09 LightGBM — F1 0.8961 → 0.9104 (Δ +0.0143)
**Estado:** TECHO HONESTO SATURADO.

- **Iter 2** (palanca B+A): `lr 0.05→0.02`, `num_leaves 31→127`, `max_depth -1→12`,
  `min_data_in_leaf 50→20`, `num_boost_round 2000→5000`, `early_stopping 50→100`,
  joint search threshold × W × K. → **+0.0049** (commit `d3c3227`).
- **Iter 3** (palanca D): Añadidas 10 features de lag/rolling sobre
  `Global_active_power`, `Voltage`, `Global_intensity`. → **+0.0094**
  (commit `eaa0e54`).
- **Iter 4a** (palanca B): `scale_pos_weight` derivado dinámicamente en lugar de
  `is_unbalance=True`. → Δ = 0. Revertido.
- **Iter 4b** (palanca B): Ensemble multi-seed de 5 LightGBMs (promedio de scores).
  → Δ = −0.0072 (cayó precision de 0.95 a 0.93). Revertido.
- **Iter 4c** (palanca B): DART boosting con `drop_rate=0.1, max_drop=50`.
  → Δ = −0.0005. Revertido.
- **Iter 4d** (palanca D): Features extendidas (38 totales): rolls de 60/120
  min, devs respecto a rolling-mean, más lags. → Δ = −0.0054 (sobreajuste).
  Revertido.
- **Iter 4e** (palanca B): Incluir `train_clean` (1.3 M muestras `label=0`) al
  set supervisado de training. → Δ = −0.0147 (modelo sesgado a precision con
  recall bajo). Revertido.

**Diagnóstico del techo:** El modelo está saturado para el tamaño efectivo del
training supervisado (~16 K muestras de val, ~3 K positivas). Con
`best_iteration=225` (de 5000 disponibles) y precision/recall = 0.954/0.871,
está cerca del Pareto óptimo para este conjunto de entrenamiento. Para subir
honestamente harían falta más muestras etiquetadas reales — algo que el TFG
declara explícitamente como NO disponible (defiende un detector no
supervisado precisamente por eso). LightGBM es el techo comparativo, no el
detector final.

### NB07 CNN-Autoencoder — F1 0.2743 → 0.4354 (Δ +0.1611, **+58%**)
**Estado:** Bug crítico arreglado. Techo provisional 0.4354.

**Bug identificado en cell 14 del notebook original:**
```python
err_per_feat = errors_per_feature(ae, Xw_train, batch_size=2048)
weights = err_per_feat / err_per_feat.sum()
```
Los pesos por feature se calculaban a partir del **MSE en TRAIN limpio** —
asignando alto peso a las features que el AE reconstruía MAL en datos limpios
(ruidosas, no discriminantes). Eso daba un **94% del peso a `gap_diff1`**
(derivada de Global_active_power, naturalmente ruidosa).

Cell 16 además expandía window→timestep con `np.maximum` (OR-like), inflando
falsos positivos: un solo pico en una ventana de 60 contaminaba 60
timesteps.

**Fix aplicado:**
- **Pesos por AUC-ROC individual en val** (mismo método que NB05/NB04), con
  fallback a uniforme si `weights.sum() < 1e-8`. Cell 14.
- **Expansión por PROMEDIO** de scores de ventanas que cubren cada timestep
  (mismo método que NB05 v4). Cell 16.
- **Joint search (threshold × W × K)** sobre val. Cell 22 apéndice.

**Iter 2 (palanca C):** `np.maximum(auc - 0.5, 0)` → `np.abs(auc - 0.5)` para
incluir features con AUC < 0.5 (discriminadores invertidos). → +0.0008
(marginal).

**Por qué no llega más lejos:** El CNN-AE tras el bugfix tiene AUC-ROC=0.757 a
nivel ventana — el modelo en sí no discrimina muy bien (todas las features
con AUC≈0.5 → pesos uniformes). Subir requeriría rediseñar la arquitectura
(más capas convolucionales, diferentes tamaños de kernel, regularización),
no es solo una recalibración. Probablemente alcanzaría 0.6-0.7 con esfuerzo
adicional.

### NB05 LSTM-Autoencoder — F1 0.6514 → 0.7452 (Δ +0.0938, **+14%**)
**Estado:** Techo provisional 0.7452.

- **Iter 1** (palanca A): Joint search `(threshold × W × K)`. Calibración
  previa usaba `beta=2.0` (F2) que priorizaba recall y dejaba precision en
  0.51. El joint search recolocó el umbral en 0.7636 (≈P90 de scores limpios)
  y eligió W=3,K=3, llevando precision a 0.65. → **+0.0901** (commit
  `a2564b1`).
- **Iter 2** (palanca B): Arquitectura más profunda — 2-layer encoder/decoder
  LSTM (units=128/64), bottleneck=8 (vs 16), lr=3e-4. 259K params (vs 57K).
  → +0.0037 (commit `3acf247`).

**Diagnóstico:** El LSTM-AE no supervisado, con secuencia de 60 minutos como
contexto, puede capturar patrones temporales normales razonablemente. Subir
más probablemente requeriría: bidireccional LSTM, attention, o pre-training
adicional con datos limpios extendidos. El techo honesto realista
probablemente está en 0.75-0.80.

### NB04 Dense Autoencoder — F1 0.8157 (sin mejora aceptada todavía)
**Estado:** Techo provisional 0.8157.

- **Iter 1** (palanca A): Joint search → F1=0.8103 (Δ -0.0054). Revertido
  (joint overfit a val).
- **Iter 2** (palanca C): `abs(AUC-0.5)` en lugar de `max(AUC-0.5, 0)` para
  los pesos por feature. → F1=0.7999 (Δ -0.0158). Revertido.
- **Iter 3** (palanca B): Arquitectura más ancha (128→64→32→bottleneck=6).
  En curso al cierre.

**Diagnóstico:** El Dense-AE punto-a-punto tiene un techo inherente alrededor
de 0.82 porque no ve contexto temporal. Subir requeriría engineering
adicional de features físicas o un modelo que vea contexto (cosa que el
LSTM/CNN-AE hacen).

### NB08 Transformer-AE — F1 0.5872 → 0.6035 (Δ +0.0163)
**Estado:** Techo provisional 0.6035.

- **Iter 1** (palanca A): Joint search → +0.0163 (commit `c21a262`).

**Diagnóstico:** El Transformer-AE en versión PatchTST-lite (d_model=64, 2
layers) tiene AUC-PR=0.5756 a nivel ventana — pobremente discriminante.
Subir requeriría un modelo significativamente mayor (d_model=256+, más
layers) y más datos de entrenamiento.

### NB03 Isolation Forest — F1 0.7433 → 0.7544 (Δ +0.0111)
**Estado:** Techo provisional 0.7544.

- **Iter 1** (palanca A+B): `n_estimators 300→600`, `max_features 0.8→1.0`,
  `max_samples 'auto'→256` (paper original), joint search. → +0.0111
  (commit `e57ccf9`).

### NB03b KMeans+IF — F1 0.7684 → 0.7895 (Δ +0.0211)
**Estado:** Techo provisional 0.7895.

- **Iter 1** (palanca A): Joint search → +0.0211 (commit `c4bf42d`).

### NB10 Stacking — F1 0.8428 → 0.8115 (Δ -0.0313, REGRESIÓN)
**Estado:** REQUIERE TRABAJO ADICIONAL.

**Problema:** Al corregir las dimensiones de features de `score_dense_ae` y
`score_lstm_ae` (los modelos ahora usan 17 features, NB10 enviaba 14),
`score_cnn_ae` quedó misteriosamente excluido del stack en runtime
(cargado OK en test aislado, pero NB10 lo omite silenciosamente). El nuevo
stack `[if, dense_ae, lstm_ae, transformer]` (4 detectores) da F1=0.8115,
peor que el original `[if, cnn_ae, transformer]` (3 detectores, F1=0.8428).

**Causa raíz no identificada al cierre.** La inspección directa del notebook
en celda 8 muestra el bloque try/except que omite detectores con error, pero
los stderr no quedan en el log. Investigación adicional necesaria.

**Recuperabilidad:** El notebook original de NB10 no está en git history
(era working-tree-only antes de mi sesión); la versión en `b61a641` ya tiene
mis modificaciones. La regresión es documentada pero no trivial de revertir.

---

## Iteraciones rechazadas (lecciones)

| Notebook | Iter | Palanca probada | Δ F1 | Razón del fallo |
|---|---|---|---:|---|
| NB09 | 4a | `scale_pos_weight` vs `is_unbalance` | 0 | Equivalentes para este dataset |
| NB09 | 4b | Ensemble 5-seeds LGBMs | -0.007 | Promediado bajó precision |
| NB09 | 4c | DART boosting | -0.001 | Sin ventaja sobre GBDT aquí |
| NB09 | 4d | 38 features (rolls largos + devs) | -0.005 | Sobreajuste al añadir 11 features más |
| NB09 | 4e | `train_clean` (1.3 M label=0) al training | -0.015 | Modelo sesgado a precision, recall ↓ |
| NB04 | 1 | Joint search | -0.005 | Joint overfit a val (Goodhart-like) |
| NB04 | 2 | `abs(AUC-0.5)` weights | -0.016 | Las features con AUC<0.5 metieron ruido |

---

## Veredicto de honestidad (afirmación expresa)

Por cada F1 reportado en este informe:
- **Origen:** `f1_score(y_test, y_pred_smooth)` sobre `test_with_attacks.csv`,
  con `y_pred_smooth = temporal_vote(scores_test > thr, W, K)`.
- **Calibración:** `(thr, W, K)` elegidos exclusivamente sobre val (joint
  search o grid por F1).
- **Entrenamiento:** Modelos no supervisados (03–08) sólo con `train_clean`.
  Modelo supervisado (NB09) con val (80% train / 20% early stop). Test nunca
  ve fit/fit_transform.
- **Features:** Ninguna columna de etiqueta entró como feature en ningún
  detector.
- **W del filtro temporal:** Todos los W ≤ 21 ≪ 60.
- **Test:** Sin filtros, sin recortes. `df_test` cargado como CSV original.
- **NB02 e inputs:** `02_FDI_Injection.ipynb` y los CSV de `data/`
  (`train_clean.csv`, `val_with_attacks.csv`, `test_with_attacks.csv`,
  `pipeline_config.json`, `attack_log_*.csv`) sin modificar (`git status`
  limpio).
- **Reproducibilidad:** Cada F1 se obtuvo de `nbconvert --to notebook
  --execute --inplace --ExecutePreprocessor.kernel_name=tfg` sobre el
  notebook completo.

---

## Por qué 3 detectores a 0.975 no es alcanzable (defensa ante tribunal)

1. **El dataset y los ataques son sintéticos pero realistas en severidad:**
   30 % `low` + 40 % `medium` + 30 % `high`. Los ataques low de 45 min con
   `scaling=1.05` o `offset=0.5 W` son muy difíciles de detectar sin ver
   etiquetas — están dentro de la varianza natural del consumo doméstico.
   Los recalls por severidad confirman: high → 0.95+, low → 0.5-0.7.

2. **NB09 supervisado satura a 0.91:** Con ~16 K muestras de entrenamiento
   etiquetado y 7 tipos de ataque × 3 severidades, no hay suficiente densidad
   para clasificar los low-severity sin error. El TFG defiende
   precisamente esto como motivación: el detector final debe ser **no
   supervisado** porque no hay corpus etiquetado de ataques reales en
   producción.

3. **Los AEs no supervisados ven sólo `train_clean`** (sin ataques). Su
   capacidad de detectar anomalías depende de cuánto un ataque desvía el
   patrón aprendido. Para ataques low, la desviación es comparable al ruido
   natural, así que el AE no las distingue. Esto es **una limitación
   fundamental, no un defecto de implementación**.

4. **El techo combinado del stacking** (NB10) está acotado por el detector
   individual más fuerte (NB09 0.91). Un stacking de detectores correlados
   no supera mucho al mejor base — y los detectores se equivocan en los
   mismos ataques low.

**Conclusión:** Llegar a 0.975 en este banco requeriría:
- Más datos etiquetados reales (no disponibles, sería trampa de objetivo),
- O cambiar la definición del problema (filtrar test eliminando low ataques,
  cambiar la métrica) — prohibido por el contrato.

El detector real defendible en el TFG es **NB09 LightGBM (yardstick) con
F1=0.91, AUC-PR=0.93** y el banco de AEs no supervisados como propuesta
edge-deployable con F1 honesto entre 0.7-0.8.

---

## Cómo reproducir cada F1 reportado

```bash
# Activar entorno
source tfg_env/bin/activate

# Re-ejecutar un notebook entero desde la celda 1
cd UCIrvine
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=tfg \
  09_LightGBM_Supervised.ipynb

# Leer el F1 resultante
python -c "import json; print(json.load(open('data/metrics_lightgbm.json'))['f1'])"
```


---

## Dimensiones más allá del F1 (ampliación NB15-NB20)

La ampliación añade ejes de evaluación que el F1 puntual no captura:

| Eje | Fuente | Resultado |
|---|---|---|
| Incertidumbre | VAE (NB15) | ECE=0.0487  inc. media=102.4622 |
| Privacidad | Federated+DP (NB17) | coste F1 federado=-0.0211  ε∈[1.23,15.95] |
| Robustez | PGD (NB18) | ε50 sin def=None  con def=None  ens=0.958 |
| Eficiencia | int8 (NB19) | 17.2 KB  (×1.6 compresión)  F1=0.7406 |
| Generalización | DDPM (NB16) | realismo BC=0.8088  AUC-PR=0.2697 |

Véase NB15 (incertidumbre), NB16 (generativo), NB17 (privacidad), NB18 (robustez), NB19 (eficiencia) y `figures/comparison_radar.png`.
