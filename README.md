# LUMEN-XAI

**Explainable machine learning for early on-treatment risk assessment in metastatic castration-resistant prostate cancer (mCRPC) treated with [¹⁷⁷Lu]Lu-PSMA-617, benchmarked against the Bellmunt Risk Score.**

Companion code for:

> Klein CR, Büttner T, Kuhlgatz V, Theis M, Sprinkart AM, Ritter M, Schmidt-Wolf IGH, Essler B, Gaertner FC, Brossart P, Essler M, Marinova M, Krausewitz P. **Explainable machine learning identifies early on-treatment biomarkers that complement simplified risk stratification in mCRPC undergoing [¹⁷⁷Lu]Lu-PSMA-617 therapy.** *Manuscript submitted.*
<!-- After publication: replace "Manuscript submitted." with journal, year, volume, pages and DOI. -->

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20%7C%20pandas%20%7C%20matplotlib-lightgrey)
<!-- After archiving on Zenodo, add the DOI badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

---

## Overview

Pretreatment scores such as the Bellmunt Risk Score (BRS) are applied once, before therapy starts. Patients receiving [¹⁷⁷Lu]Lu-PSMA-617, however, have routine laboratory tests before every cycle. This project asks whether laboratory information available by the **second treatment cycle** improves prognostic assessment beyond pretreatment risk stratification, and which measurements carry that signal.

Three levels of prognostic information are compared on identical cross-validation folds:

1. the raw and CRP-enhanced **Bellmunt Risk Score**,
2. machine-learning models restricted to **pretreatment** variables (25 features),
3. machine-learning models that additionally use **laboratory data through cycle 2** (51 features).

To prevent later treatment duration or survival from leaking into the predictors, the analysis uses a **cycle-2 landmark design**: only patients alive and on treatment at cycle 2 are analysed, and every longitudinal predictor is computed exclusively from measurements up to and including cycle 2.

All algorithms (decision tree, Random Forest, Gradient Boosting, L2-regularised logistic regression, Kaplan–Meier estimator, Shapley sampler) are implemented **from first principles in NumPy and pandas**, without machine-learning libraries, so that every step from data to reported number can be inspected.

## Pipeline

```
 Cohort export (CSV, not included)
          │
          ▼
 Cohort definition ───── start date and positive follow-up required
          │               landmark: patients alive and on treatment at cycle 2
          ▼
 Endpoints ───────────── 1-year OS (primary), 2-year OS, 1-year radiographic PFS
          │               censored before horizon without event → excluded
          ▼
 Bellmunt Risk Score ─── original (0–3), CRP-enhanced (0–4), modified
          │
          ▼
 Feature engineering ─── clinical, metastatic burden, CT body composition, PSMA PET
          │               8 laboratory parameters: baseline / mean / last / slope (cycles 1–2)
          │               PSA nadir; features with > 70 % missingness dropped
          │               BRS flags and scores removed from model inputs
          ▼
 Model comparison ────── RF · GB · LR · majority baseline, stratified 5-fold CV
          │               full (51) vs. baseline-only (25) feature set vs. raw / enhanced BRS
          │               identical folds, tie-corrected AUC
          ▼
 Explainability ──────── permutation Shapley sampling on the Random Forest
                          150 patients per endpoint, 256 permutations, convergence check
```

## Repository structure

| File | Purpose |
|---|---|
| `lu_psma_xai_pipeline.py` | Core module: data loading, landmark cohort, censoring-aware endpoints, BRS variants, feature engineering, classifier implementations, Kaplan–Meier estimation. Run directly, it produces the Kaplan–Meier figures. |
| `landmark_analysis.py` | Primary analysis: full vs. baseline-only models vs. BRS on identical folds, Shapley attribution with convergence check, endpoint accounting. Imports cohort, feature and model code from the core module, so that logic exists in one place only. |
| `baseline_characteristics.py` | Baseline characteristics (Table 1) of the landmark cohort or the full treated cohort. |
| `data/README.md` | Expected input format and column dictionary. **No patient data are included.** |
| `requirements.txt` | Python dependencies (NumPy, pandas, Matplotlib). |
| `CITATION.cff` | Citation metadata (GitHub shows a *Cite this repository* button). |
| `LICENSE` | MIT licence. |

## Reproducing the article

| Article item | Script | Output file(s) |
|---|---|---|
| Table 1, landmark cohort (N = 350) | `baseline_characteristics.py` (`LANDMARK_CYCLE = 2`) | `baseline_characteristics_cycle2.csv` |
| Table 1, full treated cohort (N = 405) | `baseline_characteristics.py` (`LANDMARK_CYCLE = 1`) | `baseline_characteristics_cycle1.csv` |
| Table 2, Supplementary Table S5 | `landmark_analysis.py` | `performance_full_vs_baseline.csv` |
| Table 3 | `landmark_analysis.py` | `brs_baseline_auc.csv`, `performance_full_vs_baseline.csv` |
| Fig. 1, Supplementary Fig. S1 | `lu_psma_xai_pipeline.py` | `KM_OS.png`, `KM_PFS.png` |
| Fig. 2, Supplementary Figs. S2–S4 | `landmark_analysis.py` | `SHAP_*.png`, `shap_*.csv` |
| Supplementary Table S1 | `landmark_analysis.py` | `feature_inventory.csv` |
| Supplementary Tables S2–S4 | `baseline_characteristics.py`, `landmark_analysis.py` | `baseline_characteristics_cycle*.csv`, `endpoint_counts.csv` |
| Supplementary Table S6 | `landmark_analysis.py` | `shap_convergence.csv` |

`lu_psma_xai_pipeline.py` also runs a preliminary cross-validation and Shapley analysis (40 patients, AUC without tie correction). Those outputs (`performance_summary.csv`, `brs_baseline_auc.csv` and `SHAP_*.png` in `results/`) are **not** the results reported in the article; the reported results are written by `landmark_analysis.py` to `results/landmark_analysis/`.

## Methods at a glance

| Component | Specification |
|---|---|
| Design | Retrospective single-centre cohort (LUMEN, University Hospital Bonn, 2014–2023); cycle-2 landmark |
| Endpoints | 1-year OS (primary), 2-year OS (secondary), 1-year radiographic PFS (exploratory); binary, at fixed horizons from treatment start |
| Censoring | Patients censored before the horizon without an event are **excluded**, not coded event-free |
| Features | 51 (full landmark set) and 25 (baseline-only); BRS components and composites excluded from model inputs |
| Missing data | Mean imputation with training-fold statistics only |
| Validation | Stratified 5-fold cross-validation, seed 42, identical folds for all models and the BRS |
| Random Forest | 80 trees, max depth 6, min. 4 samples per split, ⌊√p⌋ features per split, bootstrap, Gini impurity |
| Gradient Boosting | 80 regression trees, max depth 3, learning rate 0.1, binomial log-loss |
| Logistic regression | Training-fold standardisation, L2 penalty λ = 0.01, batch gradient descent (500 iterations, learning rate 0.05) |
| Baseline | Majority-class predictor |
| Comparator | Raw and enhanced BRS used directly as integer risk rankings |
| Metrics | AUC with mid-rank tie correction; accuracy, sensitivity, specificity, PPV, NPV at threshold 0.5; mean ± SD across folds |
| Hyperparameters | Fixed *a priori*; no tuning |
| Explainability | Permutation Shapley sampling on the Random Forest refitted on all evaluable patients; reference vector = feature means; 256 permutations; up to 150 patients per endpoint; importance = mean absolute per-patient value, normalised to 100 %; convergence by Spearman correlation of rankings from 128 vs. 256 permutations |

## Installation

```bash
git clone https://github.com/ChristianRKlein/LUMEN-XAI.git
cd LUMEN-XAI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Place the cohort export as `data/lu_psma_data.csv` (the folder is git-ignored; see [`data/README.md`](data/README.md)), then run from the repository root:

```bash
python baseline_characteristics.py   # Table 1 (seconds)
python lu_psma_xai_pipeline.py       # Kaplan–Meier figures (plus preliminary analysis)
python landmark_analysis.py          # Tables 2–3, Fig. 2, Supplementary Tables S1, S3–S6
```

Outputs are written to `results/` and `results/landmark_analysis/`. `landmark_analysis.py` records every parameter in `run_settings.json` and the full console log in `run_summary.txt`. Because tree evaluation is pure Python, a complete run takes roughly one to a few hours on a standard laptop, most of it Shapley sampling.

## Reproducibility

- All random processes (fold assignment, bootstrap samples, feature subsampling, Shapley permutations and patient sampling) are seeded.
- No machine-learning libraries are used, so results do not depend on scikit-learn or similar package versions.
- The original analysis was run under Python 3.7. The code was additionally tested with Python 3.13, NumPy 2.5, pandas 3.0 and Matplotlib 3.11 on synthetic data.

## Data availability

Patient-level data of the LUMEN cohort are **not** part of this repository. De-identified data are available from the corresponding author upon reasonable request, subject to institutional approval and a data use agreement. The expected input format is documented in [`data/README.md`](data/README.md).

## Ethics and registration

Ethics Committee of the Medical Faculty, University of Bonn (vote 2024-19-BO); German Clinical Trials Register DRKS00035087. Reporting follows the TRIPOD+AI guideline.

## Limitations

- Internal cross-validation in a single centre only; no external validation. The models are **not intended for clinical decision-making**.
- Results apply to patients alive and on treatment at cycle 2.
- Hyperparameters were fixed *a priori*; performance reflects one configuration per algorithm.
- The BRS liver component is approximated by the presence of at least one visceral metastatic site, because liver involvement is not recorded separately in the source database.
- Shapley values describe the Random Forest models, not causal effects.

## Citation

If you use this code, please cite the article above and this repository ([`CITATION.cff`](CITATION.cff)).

## Related work

- Bellmunt J, *et al.* Prognostic factors in patients with advanced transitional cell carcinoma of the urothelial tract experiencing treatment failure with platinum-containing regimens. *J Clin Oncol.* 2010;28:1850–5.
- Büttner T, *et al.* Bellmunt risk score enables survival prediction in men with mCRPC undergoing PSMA-targeted radioligand therapy (LUMEN). *Eur J Nucl Med Mol Imaging.* 2026;53:931–939.
- Nowak S, *et al.* End-to-end automated body composition analyses with integrated quality control for opportunistic assessment of sarcopenia in CT. *Eur Radiol.* 2022;32:3142–3151.

## Contact

Christian R. Klein, MD — Clinic of Internal Medicine III, University Hospital Bonn, Germany
christian.klein@ukbonn.de

## Licence

Released under the [MIT Licence](LICENSE).
