# Datasets

Este proyecto usa tres datasets de consumo eléctrico. **Los ficheros crudos NO se versionan en Git** (son muy grandes y/o están disponibles públicamente en su fuente oficial). En su lugar se versiona un **subconjunto procesado y comprimido** suficiente para reproducir todos los notebooks, y aquí se documenta cómo obtener/regenerar el resto.

| Dataset | Uso en el TFG | Crudo | En el repo |
|---|---|---|---|
| **UCI Household Power Consumption** | Entrenamiento + evaluación (principal) | ~130 MB (externo) | Splits procesados `.csv.gz` |
| **UK-DALE** | Validación cross-domain (NB13) | ~112 GB (externo) | `ukdale_multihouse_1min.csv.gz` |
| **REFIT (UKREFIT)** | Descartado (no usado) | ~6 GB (externo) | — |

---

## 1. UCI Individual Household Electric Power Consumption (principal)

Vivienda en Sceaux (Francia), 2006–2010, resolución 1 minuto. Es el dataset sobre el que se **entrena** el digital de detección y se **inyectan** los 7 ataques FDI.

- **Fuente oficial:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **Crudo (no versionado):** `household_power_consumption.txt`. Descárgalo del enlace y colócalo en `UCIrvine/`.
- **Regenerar desde el crudo:** ejecutar `UCIrvine/01_HouseholdAnalisis.ipynb` (limpieza → `uci_clean_full.csv`) y luego `UCIrvine/02_FDI_Injection.ipynb` (split temporal + inyección de ataques).
- **Procesado versionado (atajo, en `UCIrvine/data/`):**
  - `train_clean.csv.gz` — train limpio (1,30 M filas)
  - `val_with_attacks.csv.gz` — validación con ataques (144 k filas)
  - `test_with_attacks.csv.gz` — test con ataques (605 k filas)

  Para usarlos directamente: `pd.read_csv('UCIrvine/data/train_clean.csv.gz')` (pandas descomprime solo).

## 2. UK-DALE (validación cross-domain, NB13)

5 hogares del Reino Unido (Kelly & Knottenbelt, 2015). Se usa solo el **agregado (`mains`, channel_1)** de cada casa para evaluar si el detector entrenado en UCI **generaliza** a otros hogares. El crudo (~112 GB) incluye datos a 16 kHz y canales de electrodomésticos que **no** se necesitan para detección sobre consumo agregado.

- **Fuente oficial:** https://jack-kelly.com/data/  (UK-DALE, UKERC EDC)
- **Crudo (no versionado):** carpeta `UKRC/ukdale/house_{1..5}/channel_1.dat`.
- **Procesado versionado (en `UCIrvine/data/`):** `ukdale_multihouse_1min.csv.gz` — `mains` de las 5 casas resampleado a 1 minuto en kW (~16 MB, 3 M filas). Columnas: `datetime`, `Global_active_power` (kW), `house`.
- **Regenerar:** `python UCIrvine/prepare_ukdale.py` (lee el crudo y reescribe el `.csv.gz`).
- **Limitación conocida:** UK-DALE no mide Voltage ni Intensidad → la feature física `VI_residual` es 0 en este dataset (ver discusión en NB13).

## 3. REFIT / UKREFIT (descartado)

20 viviendas del Reino Unido. **No se usa en el análisis final** (sin la estructura de señales que requiere el check físico `P=V·I`). Se documenta solo por trazabilidad.

- **Fuente oficial:** https://www.refitsmarthomes.org/datasets/  (University of Strathclyde)
- No versionado (en `.gitignore`).
