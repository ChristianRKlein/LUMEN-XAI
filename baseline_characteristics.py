# -*- coding: utf-8 -*-
"""
baseline_characteristics.py
===========================

Baseline characteristics (Table 1). Runs in seconds (no model fitting).

With LANDMARK_CYCLE = 2 (default) the script describes the cycle-2 landmark
cohort (right-hand column of Table 1, N = 350). With LANDMARK_CYCLE = 1 it
describes the full treated cohort with evaluable follow-up (left-hand
column, N = 405), because every included patient has received cycle 1.

The table follows the structure of Table 1 in Buettner et al. (Eur J Nucl
Med Mol Imaging 2026) as far as the available columns allow, so that both
cohorts can be compared side by side. Rows that cannot be derived from this
export (e.g. the breakdown of bone metastases into axial-only vs. including
non-axial sites) are omitted rather than approximated.

The cohort definition (valid start date, positive follow-up, landmark
restriction) is identical to `load_and_prepare_cohort()` in
lu_psma_xai_pipeline.py.

Usage
-----
Set DATA_PATH and OUTPUT_DIR below, then run

    python baseline_characteristics.py

Author : Christian R. Klein, University Hospital Bonn (christian.klein@ukbonn.de)
License: MIT (see LICENSE)
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = os.path.join("data", "lu_psma_data.csv")
OUTPUT_DIR = "results"
LANDMARK_CYCLE = 2   # 2 = landmark cohort; 1 = full treated cohort
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS (same conventions as lu_psma_xai_pipeline.py)
# ============================================================

def to_num(series):
    """Convert to numeric; strips leading '>' / '<' and German decimal commas."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.replace({"na": "", "NA": "", "nan": "", "none": "", "None": "", "NaT": ""})
    s = s.str.replace(r"^[><]\s*", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def to_date(series):
    """Convert to datetime (day-first); sentinel dates starting with '3000' -> missing."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*00:00:00\s*UTC\s*$", "", regex=True)
    s = s.str.replace(r"\s*UTC\s*$", "", regex=True)
    s = s.where(~s.str.startswith("3000"), other=np.nan)
    s = s.replace({"nan": np.nan, "NaT": np.nan, "na": np.nan, "": np.nan})
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def load_csv(path):
    """Load a CSV, auto-detecting encoding and separator."""
    for enc in ["utf-8-sig", "latin-1", "cp1252", "utf-8"]:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
                    print(f"  Encoding={enc}, sep='{sep}'")
                    return df
            except Exception:
                continue
    raise ValueError("Could not load file.")


def iqr_str(s: pd.Series) -> str:
    """Format as 'median [Q1-Q3]'."""
    s = s.dropna()
    if len(s) == 0:
        return "NA"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\nLoading: {DATA_PATH}")
    df = load_csv(DATA_PATH)

    # --- Dates and follow-up (identical to lu_psma_xai_pipeline.py) ---
    cycle_date_cols = ["z" + str(i) for i in range(1, 14)]
    for c in cycle_date_cols + ["toddat", "gebdat"]:
        if c in df.columns:
            df[c] = to_date(df[c])
    df["start"] = df["z1"] if "z1" in df.columns else pd.Series(pd.NaT, index=df.index)

    def clean_ev(s):
        """Map an event column to {0, 1}; anything other than 1 (incl. missing) -> 0."""
        x = pd.to_numeric(s.astype(str).str.strip().str.lower()
                           .replace({"na": "0", "nan": "0", "none": "0", "": "0"}),
                           errors="coerce").fillna(0)
        return (x == 1).astype(int)

    df["os_event"] = clean_ev(df.get("verstorben", pd.Series(0, index=df.index)))
    tcols = [c for c in cycle_date_cols if c in df.columns]
    if tcols:
        dmat = df[tcols].copy()
        for c in dmat.columns:
            dmat.loc[dmat[c].dt.year > 2100, c] = pd.NaT
        df["last_date"] = dmat.max(axis=1)
    else:
        df["last_date"] = df["start"]
    death = df.get("toddat", pd.Series(pd.NaT, index=df.index))
    end_os = death.where(df["os_event"] == 1, df["last_date"])
    df["os_months"] = (end_os - df["start"]).dt.days / 30.44

    df = df[df["start"].notna()].copy()
    df = df[~((df["os_months"] <= 0) & df["os_months"].notna())].copy()

    # --- Landmark restriction (LANDMARK_CYCLE = 2 matches lu_psma_xai_pipeline.py) ---
    if f"z{LANDMARK_CYCLE}" in df.columns:
        n_before = len(df)
        df = df[df[f"z{LANDMARK_CYCLE}"].notna()].copy()
        print(f"Landmark cohort (reached cycle {LANDMARK_CYCLE}): {len(df)} "
              f"(excluded {n_before - len(df)})")
    print(f"N = {len(df)}")

    if "gebdat" in df.columns:
        df["age"] = (df["start"] - df["gebdat"]).dt.days / 365.25

    # ECOG performance status at baseline
    ecog = to_num(df.get("ECOG_first", pd.Series(np.nan, index=df.index)))

    # Metastatic burden
    n_total_mets = to_num(df.get("z1_Met_Total_Anzahl", pd.Series(np.nan, index=df.index)))
    n_visceral_mets = to_num(df.get("z1_viszeral_Anzahl", pd.Series(np.nan, index=df.index)))
    # `z1_Met_Viszeral_1` counts involved visceral sites (0-3); organ-level
    # (liver-specific) data are not available in this export. It is reported
    # as "Visceral metastasis (>= 1 site)" and is the same proxy used for the
    # BRS liver component in lu_psma_xai_pipeline.py. Its prevalence
    # (~27 %) agrees with the previously published "visceral (overall)"
    # row (25.6 %) rather than the liver-only row (17.6 %).
    has_liver = (to_num(df.get("z1_Met_Viszeral_1", pd.Series(0, index=df.index))).fillna(0) > 0)

    def met_bucket(n):
        """Categorise the total metastasis count.

        The export records high counts as '> 5' / '> 10'; to_num() strips
        the '>' qualifier, so 10.0 means "more than 10" and 5.0 "more than 5".
        Hence n >= 10 -> '>10' and 5 <= n < 10 -> '5-10'.
        """
        if pd.isna(n):
            return "Unknown"
        if n < 5:
            return "<5"
        if n < 10:
            return "5-10"
        return ">10"

    met_buckets = n_total_mets.apply(met_bucket)

    # Laboratory values at baseline (= first cycle)
    def baseline_lab(prefix, nmax):
        cols = [f"{prefix}{i}" for i in range(1, nmax + 1) if f"{prefix}{i}" in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index)
        return df[cols].apply(to_num).iloc[:, 0]

    psa_baseline = baseline_lab("psa", 12)
    hb_baseline = baseline_lab("hb", 13)
    crp_baseline = baseline_lab("crp", 12)

    # Original Bellmunt Risk Score (cross-check against the published distribution)
    liver_mets_f = has_liver.astype(float)
    crp1 = crp_baseline
    brs_ecog = (ecog > 0).astype(float)
    brs_hb = (hb_baseline < 10).astype(float)
    brs_liver = liver_mets_f
    bellmunt_score = brs_ecog + brs_hb + brs_liver

    # --- Assemble Table 1 ---
    rows = []
    rows.append(["Age (years)", "", ""])
    rows.append(["  Mean (SD)", f"{df['age'].mean():.1f} ({df['age'].std():.1f})", ""])
    rows.append(["  Median [IQR]", iqr_str(df["age"]), ""])
    rows.append(["ECOG PS", "", ""])
    for v in sorted(ecog.dropna().unique()):
        n = (ecog == v).sum()
        rows.append([f"  {int(v)}", f"{n} ({n/len(df)*100:.1f}%)", ""])
    rows.append(["Metastatic count", "", ""])
    for cat in ["<5", "5-10", ">10", "Unknown"]:
        n = (met_buckets == cat).sum()
        rows.append([f"  {cat}", f"{n} ({n/len(df)*100:.1f}%)", ""])
    rows.append(["Visceral metastasis (≥1 site)", f"{int(has_liver.sum())} ({has_liver.mean()*100:.1f}%)", ""])
    rows.append(["Visceral metastases (count, median [IQR])", iqr_str(n_visceral_mets), ""])
    rows.append(["PSA (ng/mL)", "", ""])
    rows.append(["  Mean (SD)", f"{psa_baseline.mean():.0f} ({psa_baseline.std():.0f})", ""])
    rows.append(["  Median [IQR]", iqr_str(psa_baseline), ""])
    rows.append(["Hemoglobin (g/dL)", "", ""])
    rows.append(["  Mean (SD)", f"{hb_baseline.mean():.1f} ({hb_baseline.std():.1f})", ""])
    rows.append(["  Median [IQR]", iqr_str(hb_baseline), ""])
    rows.append(["CRP (mg/L)", "", ""])
    rows.append(["  Mean (SD)", f"{crp_baseline.mean():.1f} ({crp_baseline.std():.1f})", ""])
    rows.append(["  Median [IQR]", iqr_str(crp_baseline), ""])
    rows.append(["Bellmunt Risk Score", "", ""])
    for v in sorted(bellmunt_score.dropna().unique()):
        n = (bellmunt_score == v).sum()
        rows.append([f"  {int(v)}", f"{n} ({n/len(df)*100:.1f}%)", ""])

    out = pd.DataFrame(rows, columns=["Parameter", "Overall (N={})".format(len(df)), ""])
    print("\n" + out.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, f"baseline_characteristics_cycle{LANDMARK_CYCLE}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
