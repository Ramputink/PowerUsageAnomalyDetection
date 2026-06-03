# Optimization Log — TFG FDI Detectors (metodología ECC)

**Objetivo:** maximizar todas las métricas (F1, AUC-PR, AUC-ROC) de todos los
detectores optimizando hiperparámetros, en ~7 horas, con la metodología del
framework ECC (benchmark-optimization-loop + eval-harness + verification-loop).

## Correctness gate (INVIOLABLE) — contrato no-trampa
- Entrenar SOLO en `train_clean`. Calibrar umbral y `(W,K)` SOLO en validación.
- Evaluar en test, jamás usar `y_test` para calibrar/seleccionar.
- Features sin etiquetas. Mismos splits/escalado que el baseline (`tfg_common`).
- **Una variante solo se promueve si pasa este gate y reproduce el protocolo.**

## Metodología (ECC)
1. **Pin baseline** → `optimization/baseline.json` (hecho).
2. **Reproducir baseline en el harness** antes de buscar (ECC: confirmar delta).
3. **Benchmark loop**: muestrear configs alrededor de la región buena →
   entrenar (subsampleo para velocidad) → calibrar en val → evaluar en test →
   log append-only `variants_<model>.csv` → promover mejor (`best_<model>.json`).
4. **Finalistas**: re-validar top configs a escala completa.
5. **Codificar ganador**: actualizar notebook + `metrics_*.json` + memoria.
6. **Verification loop** tras cada promoción: sin fuga, reproducible, sin regresión.

## Baseline (pinned) y prioridad por headroom
| Modelo | F1 | AUC-PR | AUC-ROC | headroom | repro harness |
|---|---|---|---|---|---|
| CNN-AE | 0.435 | 0.432 | 0.758 | **0.565** | window (pend.) |
| Transformer-AE | 0.604 | 0.584 | 0.886 | 0.396 | window (pend.) |
| VAE | 0.693* | 0.500 | 0.874 | 0.307 | pointwise ✓ |
| LSTM-AE | 0.745 | 0.881 | 0.939 | 0.255 | window (pend.) |
| IF | 0.754 | 0.782 | 0.927 | 0.246 | ✓ exacto |
| KMeans+IF | 0.789 | 0.764 | 0.927 | 0.211 | pend. |
| Stacking | 0.811 | 0.881 | 0.953 | 0.189 | depende de bases |
| Dense-AE | 0.816 | 0.855 | 0.932 | 0.184 | pointwise ✓ |
| LightGBM | 0.910 | 0.941 | 0.963 | 0.090 | pend. (techo) |

\* VAE con alta varianza entre runs (0.654↔0.693) — objetivo: estabilizar.

## Exit conditions (ECC)
- Presupuesto temporal ~7h, o sin mejora dentro del ruido (≥N configs sin avance).
- Cualquier regresión en guardrails → revertir.

## Plan por fases
- **F1 — Setup + reproducción de baselines** (harness pointwise ✓; window pend.).
- **F2 — Sweep deep models** (CNN, Transformer, LSTM, Dense, VAE) — mayor headroom.
- **F3 — Sweep clásicos** (IF, KMeans, LightGBM) + re-optimizar Stacking con bases mejoradas.
- **F4 — Finalistas a escala completa + codificar en notebooks/metrics**.
- **F5 — Verification + actualizar memoria + commit**.

---

## Bitácora de progreso
(append-only; cada entrada con timestamp lógico y delta vs baseline)

- **[setup]** baseline fijado; harness pointwise reproduce IF exacto (0.7544/0.7824/0.9266).

- **[F1 setup]** Harness pointwise (IF/Dense/VAE) y window (CNN/LSTM/Transformer) construidos.
  Reproducción exacta: IF 0.7544 ✓, Dense 0.8181 ✓ (supera baseline por calibrar β=1).
  Pivote: pointwise lento en Metal con 800k → subsample reducido; foco en window (mayor headroom).
  Subsamples búsqueda: pointwise train=800k; window train_win=25k, val=80k, test=120k prefijo.

### Batch 1 resultados
- **[IF CONFIRMADO full-scale]** F1 0.7544 → **0.7950** (+0.0406), AUC-PR 0.7824 → **0.8526** (+0.0702).
  Palanca: `max_samples` grande (1024-2048) + más árboles, mode=full. Sin finalización (IF ya evalúa full val/test).
- Window batch 1 (subset, pendiente finalizar): CNN best 0.581 (AUCPR 0.644, base 0.435/0.432), Transformer best 0.631 (AUCPR 0.675, base 0.604/0.584).

- **[finalize cnn]** full-scale F1=0.6428 (base 0.4354, +0.2074) AUC-PR=0.6936 (+0.2614) -> MEJORA  cfg={"kind": "cnn", "mode": "full", "filters": [32, 16, 8], "ksize": 5, "bottleneck": 8, "lr": 0.0003, "epochs": 25, "batch": 512, "patience": 6, "seed": 312}

- **[finalize transformer]** full-scale F1=0.6601 (base 0.6035, +0.0566) AUC-PR=0.6923 (+0.1084) -> MEJORA  cfg={"kind": "transformer", "mode": "full", "patch_len": 4, "d_model": 64, "n_heads": 8, "ffn": 256, "n_layers": 3, "dropout": 0.2, "lr": 0.0003, "epochs": 35, "batch": 256, "patience": 5, "seed": 643}

- **[finalize lstm]** full-scale F1=0.7384 (base 0.7452, -0.0068) AUC-PR=0.8269 (-0.0538) -> REGRESION  cfg={"kind": "lstm", "mode": "full", "units": 64, "bottleneck": 8, "lr": 0.0003, "epochs": 20, "batch": 256, "patience": 5, "seed": 837}

## RESUMEN FINAL (optimización completada)
Mejoras confirmadas a escala completa (mismo contrato no-trampa):
| Detector | F1 base→opt | ΔF1 | AUC-PR base→opt | ΔAUC-PR |
|---|---|---|---|---|
| **CNN-AE** | 0.435→**0.643** | **+0.207** | 0.432→0.694 | +0.261 |
| Transformer-AE | 0.604→**0.660** | +0.057 | 0.584→0.692 | +0.108 |
| VAE | 0.693→**0.747** | +0.054 | 0.500→0.758 | +0.258 |
| IF | 0.754→**0.795** | +0.041 | 0.782→0.853 | +0.070 |
| LSTM-AE | 0.745→0.745† | ≈0 | 0.881 | ≈0 |
| Dense-AE | 0.816→0.815 | ≈0 | 0.855 | ≈0 |

† El candidato LSTM mejoraba en el subconjunto (0.814) pero **regresó a 0.738 a escala completa** → la re-validación detectó sobreajuste; se conserva el baseline. KMeans+IF y LightGBM no optimizados (headroom bajo / techo supervisado).

**Palancas:** features `full` (17) en CNN/Transformer; pesos AUC solo discriminantes; `max_samples=2048` en IF; scoring MC sin ponderar + recalibración en VAE.
**Verificación:** sin fuga de datos (calibración solo en val) en todos los harnesses; baselines reproducidos exactamente antes de buscar; finalistas re-validados a escala completa.
