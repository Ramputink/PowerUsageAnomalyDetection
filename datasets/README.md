# Datasets

This project uses three electricity-consumption datasets. **The raw files are NOT versioned in Git** (they are too large and/or are publicly available from their official source). Instead, a **processed and compressed subset** large enough to reproduce all the notebooks is versioned, and this file documents how to obtain/regenerate the rest.

| Dataset | Use in the thesis | Raw | In the repo |
|---|---|---|---|
| **UCI Household Power Consumption** | Training + evaluation (main) | ~130 MB (external) | Processed splits `.csv.gz` |
| **UK-DALE** | Cross-domain validation (NB13) | ~112 GB (external) | `ukdale_multihouse_1min.csv.gz` |
| **REFIT (UKREFIT)** | Discarded (not used) | ~6 GB (external) | — |

---

## 1. UCI Individual Household Electric Power Consumption (main)

A household in Sceaux (France), 2006–2010, 1-minute resolution. This is the dataset on which the detection pipeline is **trained** and onto which the 7 FDI attacks are **injected**.

- **Official source:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **Raw (not versioned):** `household_power_consumption.txt`. Download it from the link and place it in `UCIrvine/`.
- **Regenerate from the raw file:** run `01_HouseholdAnalisis.ipynb` (cleaning → `uci_clean_full.csv`) and then `02_FDI_Injection.ipynb` (temporal split + attack injection).
- **Versioned processed shortcut (in `UCIrvine/data/`):**
  - `train_clean.csv.gz` — clean training set (1.30 M rows)
  - `val_with_attacks.csv.gz` — validation set with attacks (144 k rows)
  - `test_with_attacks.csv.gz` — test set with attacks (605 k rows)

  To use them directly: `pd.read_csv('UCIrvine/data/train_clean.csv.gz')` (pandas decompresses automatically).

## 2. UK-DALE (cross-domain validation, NB13)

5 households in the United Kingdom (Kelly & Knottenbelt, 2015). Only the **aggregate signal (`mains`, channel_1)** of each house is used, to evaluate whether the detector trained on UCI **generalizes** to other households. The raw data (~112 GB) includes 16 kHz signals and per-appliance channels that are **not** needed for detection on aggregate consumption.

- **Official source:** https://jack-kelly.com/data/  (UK-DALE, UKERC EDC)
- **Raw (not versioned):** folder `UKRC/ukdale/house_{1..5}/channel_1.dat`.
- **Versioned processed (in `UCIrvine/data/`):** `ukdale_multihouse_1min.csv.gz` — the `mains` signal of the 5 houses resampled to 1 minute in kW (~16 MB, 3 M rows). Columns: `datetime`, `Global_active_power` (kW), `house`.
- **Regenerate:** `python datasets/prepare_ukdale.py` (reads the raw data and rewrites the `.csv.gz`).
- **Known limitation:** UK-DALE does not measure Voltage or Intensity → the physical feature `VI_residual` is 0 on this dataset (see discussion in NB13).

## 3. REFIT / UKREFIT (discarded)

20 dwellings in the United Kingdom. **Not used in the final analysis** (it lacks the signal structure required by the physical check `P=V·I`). Documented here only for traceability.

- **Official source:** https://www.refitsmarthomes.org/datasets/  (University of Strathclyde)
- Not versioned (in `.gitignore`).
