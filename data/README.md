# Data

**This folder intentionally contains no patient data.**

The LUMEN cohort data are protected health information and cannot be shared publicly. De-identified data are available from the corresponding author upon reasonable request, subject to approval by the University Hospital Bonn and a data use agreement.

Place the export here as `lu_psma_data.csv` (or change `DATA_PATH` in the scripts). Every file in this folder except this README is excluded from version control by `.gitignore`.

## Expected input

One CSV file with one row per patient. Encoding (UTF-8, Latin-1, CP1252) and separator (`;`, `,`, tab) are detected automatically. Dates are parsed day-first (e.g. `31.12.2021`); numbers may use a decimal comma and leading `>`/`<` qualifiers (e.g. `> 10`), which are stripped. Column names are those of the original German-language database export.

### Treatment and outcomes

| Column | Content |
|---|---|
| `z1` … `z13` | Date of [¹⁷⁷Lu]Lu-PSMA-617 cycle 1 … 13 (`z1` = time origin; `z2` must be present for the landmark cohort) |
| `verstorben` | Death (1 = yes) |
| `toddat` | Date of death |
| `Radiographic_Progression_y_n` | Radiographic progression (1 = yes) |
| `Progression_Date_or_last_imaging_without_Progression` | Date of progression or of last imaging without progression |
| `gebdat` | Date of birth (used only to derive age at start of therapy) |

### Baseline clinical, imaging and body-composition features

| Column | Feature name | Content |
|---|---|---|
| `ECOG_first`, `ECOG last` | `ecog_baseline`, `ecog_last` | ECOG performance status at cycle 1 and at the landmark |
| `bmi` | `bmi` | Body-mass index |
| `Gleason_Score` | `gleason` | Gleason score |
| `Radikale_OP`, `CTX`, `NHT`, `ADT` | `prior_op`, `prior_ctx`, `prior_nht`, `prior_adt` | Prior radical prostatectomy, chemotherapy, novel hormonal therapy, androgen deprivation |
| `z1_Met_Total_Anzahl` | `n_total_mets` | Total number of metastases on baseline PSMA PET |
| `z1_viszeral_Anzahl` | `n_visceral_mets` | Number of visceral metastases |
| `z1_Met_Viszeral_1` | `has_liver_mets` | Number of involved visceral sites (0–3); > 0 is used as proxy for the BRS liver component |
| `z1_Auß_WS_Becken_Anzahl` | `n_outside_sp` | Metastases outside spine and pelvis (not among the 51 features of the published analysis: removed by the > 70 % missingness filter or absent from the export) |
| `z1_met1_ES-avg`, `z1_met1_ES-max` | `pet_avg`, `pet_max` | PSMA PET uptake of the index lesion |
| `totalMuscleExclFat_AreaInMM_0` | `muscle_area` | Skeletal muscle area at L3/L4 (CT) |
| `FattyMuscleFraction_0` | `fatty_muscle` | Fatty muscle fraction |
| `visceralFat_AreaInMM_0` | `visceral_fat` | Visceral adipose tissue area |
| `totalMuscleExclFat_meanHU_0` | `muscle_hu` | Mean skeletal muscle attenuation (HU) |

### Laboratory values per cycle

Columns are named `<prefix><cycle>`, e.g. `hb1`, `hb2`. Only cycles 1 and 2 enter the models (landmark window); for each parameter the baseline, mean, last value and slope are derived, plus the PSA nadir.

| Prefix | Parameter |
|---|---|
| `hb` | Hemoglobin (g/dL) |
| `crp` | C-reactive protein (mg/L) |
| `alp` | Alkaline phosphatase |
| `ldh` | Lactate dehydrogenase |
| `psa` | Prostate-specific antigen (ng/mL) |
| `albu` | Albumin |
| `leukos` | Leukocytes |
| `gfr` | Estimated glomerular filtration rate |
