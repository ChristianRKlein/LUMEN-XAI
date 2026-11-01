#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
landmark_analysis.py
====================

Primary analysis of the article: model comparison across information levels
and Shapley-value attribution in the cycle-2 landmark cohort.

On identical, endpoint-specific stratified cross-validation folds, the script
evaluates for each binary endpoint (1-year OS, 2-year OS, 1-year PFS):

  (A) the FULL landmark feature set   - baseline + early on-treatment
                                        information through cycle 2
  (B) the BASELINE-ONLY feature set   - pretreatment variables only
  (C) the raw and enhanced Bellmunt Risk Score, used directly as integer
      risk rankings without model fitting

and additionally computes

  (D) permutation-based Shapley values for the full-feature Random Forest in
      up to N_SHAP_PATIENTS patients per endpoint, with a convergence check
      comparing rankings from N_PERMUTATIONS/2 and N_PERMUTATIONS permutations
  (E) endpoint accounting: landmark cohort size, patients excluded because
      they were censored before the horizon, evaluable N and events

Outputs (written to OUT_DIR) correspond to the article as follows:

  performance_full_vs_baseline.csv  Table 2, Supplementary Table S5
  brs_baseline_auc.csv              Table 3 (raw and enhanced BRS)
  endpoint_counts.csv               Supplementary Tables S3 and S4
  shap_convergence.csv              Supplementary Table S6
  shap_<endpoint>.csv, SHAP_*.png   Fig. 2, Supplementary Figs. S2-S4
  feature_inventory.csv             Supplementary Table S1
  run_settings.json, run_summary.txt  all parameters and the full run log

Design
------
* Cohort definition and feature engineering are NOT re-implemented here:
  `load_and_prepare_cohort()` and `build_feature_matrix()` are imported from
  lu_psma_xai_pipeline.py, as are the Random Forest, Gradient Boosting and
  logistic regression classes, so this logic exists in exactly one place.
* Every column whose name begins with "bellmunt" or "brs_" (BRS component
  flags and composite scores) is removed from both feature sets, because the
  BRS is the comparator (no circularity).
* The baseline-only set additionally drops every mean, last-value and slope
  feature, the PSA nadir and the cycle-2 ECOG (`ecog_last`).
* AUC uses mid-ranks for ties (Mann-Whitney U); a constant predictor scores
  exactly 0.500, and the integer-valued BRS is ranked correctly.
* Shapley values are signed per patient; global importance is the mean
  absolute per-patient value, normalised to 100 % per endpoint.

Usage
-----
Place the data at DATA_PATH (default data/lu_psma_data.csv), then run

    python landmark_analysis.py

Requires only numpy, pandas and matplotlib; compatible with Python >= 3.7.

Author : Christian R. Klein, University Hospital Bonn (christian.klein@ukbonn.de)
License: MIT (see LICENSE)
"""

from __future__ import print_function

import os
import sys
import json
import time
import importlib.util

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# CONFIGURATION
# =========================================================================

# Core pipeline (cohort definition, features, classifiers), the cohort
# export, and the output folder. Relative paths are resolved from the
# current working directory, so run the script from the repository root.
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "lu_psma_xai_pipeline.py")
DATA_PATH     = os.path.join("data", "lu_psma_data.csv")
OUT_DIR       = os.path.join("results", "landmark_analysis")

N_SHAP_PATIENTS = 150     # patients per endpoint for Shapley attribution
N_PERMUTATIONS  = 256     # permutation runs per patient
N_FOLDS         = 5
RANDOM_SEED     = 42
TOP_N_FEATURES  = 15      # features shown per SHAP panel

# Classifier classes taken from the pipeline module (instantiated with the
# default hyperparameters defined there; see README, "Methods at a glance").
MODEL_CLASS_OVERRIDE = {
    "Random Forest":       "RandomForestClassifier",
    "Gradient Boosting":   "GradientBoostingClassifier",
    "Logistic Regression": "LogisticRegressionClassifier",
}

# Endpoint columns, in the order they should appear in the output tables.
ENDPOINTS = [
    ("1-Year OS",  "os_1yr",  "death within 12 months"),
    ("2-Year OS",  "os_2yr",  "death within 24 months"),
    ("1-Year PFS", "pfs_1yr", "radiographic progression within 12 months"),
]

BRS_COLUMNS = [
    ("Bellmunt Risk Score (raw, 0-3)", "bellmunt_score"),
    ("Enhanced BRS (raw, 0-4)",        "bellmunt_enhanced"),
]

# Any feature whose name starts with one of these prefixes is a Bellmunt Risk
# Score component flag or composite score. They are excluded from both
# machine-learning feature sets, because the BRS is the comparator.
BRS_FEATURE_PREFIXES = ("bellmunt", "brs_")

# Features highlighted in red in the Shapley figures (related to BRS components)
BELLMUNT_RELATED = set([
    "ecog_baseline", "ecog_last", "hb_baseline", "has_liver_mets", "crp_baseline",
])

COL_BLUE = "#2b6688"
COL_RED  = "#c0392b"


# =========================================================================
# 0.  Small utilities
# =========================================================================

_LOG_LINES = []


def log(msg=""):
    """Print a line and keep it for run_summary.txt."""
    print(msg)
    sys.stdout.flush()
    _LOG_LINES.append(str(msg))


def rule(title=""):
    log("")
    log("=" * 74)
    if title:
        log(title)
        log("=" * 74)


def ensure_out_dir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    log("Output directory: %s" % OUT_DIR)


# =========================================================================
# 1.  Import the existing pipeline and reuse its cohort / feature logic
# =========================================================================

def import_pipeline():
    """Import lu_psma_xai_pipeline.py as a module without running its main()."""
    if not os.path.isfile(PIPELINE_PATH):
        raise SystemExit(
            "\nPIPELINE_PATH does not point to a file:\n  %s\n"
            "Please correct PIPELINE_PATH at the top of this script." % PIPELINE_PATH)

    spec = importlib.util.spec_from_file_location("lupsma_pipeline", PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Prevent the pipeline's own main() from executing on import
    mod.__name__ = "lupsma_pipeline"
    sys.modules["lupsma_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def diagnostic_dump(mod):
    """List the pipeline module's classes, functions and constants (for troubleshooting)."""
    log("")
    log("-" * 74)
    log("DIAGNOSTIC DUMP  (contents of the pipeline module)")
    log("-" * 74)
    funcs, classes, consts = [], [], []
    for name in sorted(dir(mod)):
        if name.startswith("__"):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type):
            methods = [m for m in ("fit", "predict", "predict_proba") if hasattr(obj, m)]
            classes.append("%s   methods=%s" % (name, methods))
        elif callable(obj):
            funcs.append(name)
        elif isinstance(obj, (int, float, str, bool)):
            consts.append("%s = %r" % (name, obj))
    log("CLASSES:");   [log("   " + c) for c in classes]
    log("FUNCTIONS:"); log("   " + ", ".join(funcs))
    log("CONSTANTS:"); [log("   " + c) for c in consts]
    log("-" * 74)


def resolve_models(mod):
    """Return the RF / GB / LR classes from the pipeline module.

    Uses MODEL_CLASS_OVERRIDE when set; otherwise matches class names by keyword.
    """
    candidates = []
    for name in dir(mod):
        if name.startswith("__"):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "fit"):
            if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
                candidates.append((name, obj))

    keywords = {
        "Random Forest":       ("forest", "randomforest", "rf"),
        "Gradient Boosting":   ("boost", "gradientboost", "gb"),
        "Logistic Regression": ("logistic", "logreg", "lr"),
    }

    resolved = {}
    for label, keys in keywords.items():
        override = MODEL_CLASS_OVERRIDE.get(label)
        if override:
            if not hasattr(mod, override):
                diagnostic_dump(mod)
                raise SystemExit("MODEL_CLASS_OVERRIDE['%s'] = '%s' not found in the "
                                 "pipeline module." % (label, override))
            resolved[label] = getattr(mod, override)
            continue
        hit = None
        for name, obj in candidates:
            low = name.lower()
            if any(k in low for k in keys):
                hit = obj
                break
        if hit is None:
            diagnostic_dump(mod)
            raise SystemExit(
                "\nCould not auto-detect the class for '%s'.\n"
                "Fill in MODEL_CLASS_OVERRIDE at the top of this script using one of the\n"
                "class names listed in the diagnostic dump above, then re-run." % label)
        resolved[label] = hit

    log("Resolved model classes:")
    for label in ("Random Forest", "Gradient Boosting", "Logistic Regression"):
        log("   %-22s -> %s" % (label, resolved[label].__name__))
    return resolved


def call_flexibly(fn, *args):
    """Call fn with as many leading args as its signature accepts."""
    for n in range(len(args), -1, -1):
        try:
            return fn(*args[:n])
        except TypeError as exc:
            if "argument" not in str(exc):
                raise
    raise


# =========================================================================
# 2.  Metrics  (tie-corrected AUC)
# =========================================================================

def auc_score(y_true, y_score):
    """Rank-based AUC with mid-ranks for ties (Mann-Whitney U statistic / n_pos*n_neg)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        mid = 0.5 * ((i + 1) + (j + 1))           # 1-based mid-rank
        ranks[order[i:j + 1]] = mid
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def classification_metrics(y_true, y_prob, threshold=0.5):
    """AUC (threshold-free) and accuracy, sensitivity, specificity, PPV, NPV at `threshold`."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    def safe(num, den):
        return num / den if den > 0 else 0.0

    return {
        "AUC":         auc_score(y_true, y_prob),
        "Accuracy":    safe(tp + tn, tp + tn + fp + fn),
        "Sensitivity": safe(tp, tp + fn),
        "Specificity": safe(tn, tn + fp),
        "PPV":         safe(tp, tp + fp),
        "NPV":         safe(tn, tn + fn),
    }


def fmt_mean_sd(values):
    """Format per-fold values as 'mean+/-SD' (population SD), ignoring NaN."""
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return "n/a"
    return "%.3f\u00b1%.3f" % (values.mean(), values.std())


# =========================================================================
# 3.  Cross-validation
# =========================================================================

def stratified_folds(y, n_folds, seed):
    """Deterministic stratified fold assignment (returns a fold index per patient).

    Within each class, patients are shuffled with `seed` and dealt round-robin
    to the folds, so every fold has (nearly) the same event prevalence.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    assignment = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        assignment[idx] = np.arange(len(idx)) % n_folds
    return assignment


def mean_impute(train_X, test_X):
    """Fill missing values with training-partition means (no test-to-train leakage)."""
    means = np.nanmean(train_X, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    tr = np.where(np.isnan(train_X), means, train_X)
    te = np.where(np.isnan(test_X), means, test_X)
    return tr, te, means


def get_probabilities(model, X):
    """Predicted probability of the event class."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim == 2:
            return p[:, -1] if p.shape[1] > 1 else p[:, 0]
        return p
    return np.asarray(model.predict(X), dtype=float)


def run_cv(X, y, fold_assignment, model_classes):
    """Cross-validate RF, GB and LR, plus a majority-class (Mode) baseline.

    Returns {model_label: {metric: [per-fold values]}}. The Mode baseline
    predicts the training-fold majority class for every test patient.
    """
    labels = ["Random Forest", "Gradient Boosting", "Logistic Regression", "Mode (majority-class)"]
    out = dict((lab, dict((m, []) for m in
                ("AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV")))
               for lab in labels)

    for fold in range(N_FOLDS):
        te_mask = fold_assignment == fold
        tr_mask = ~te_mask
        Xtr, Xte, _ = mean_impute(X[tr_mask], X[te_mask])
        ytr, yte = y[tr_mask], y[te_mask]

        for lab in labels[:3]:
            model = model_classes[lab]()
            model.fit(Xtr, ytr)
            prob = get_probabilities(model, Xte)
            for k, v in classification_metrics(yte, prob).items():
                out[lab][k].append(v)

        majority = 1 if np.mean(ytr) >= 0.5 else 0
        const = np.full(len(yte), float(majority))
        for k, v in classification_metrics(yte, const, threshold=0.5).items():
            out["Mode (majority-class)"][k].append(v)

    return out


def run_cv_score_as_classifier(score_values, y, fold_assignment):
    """Use a raw integer score directly as the risk ranking (no model fitting)."""
    aucs = []
    for fold in range(N_FOLDS):
        te = fold_assignment == fold
        s = score_values[te]
        yy = y[te]
        keep = ~np.isnan(s)
        aucs.append(auc_score(yy[keep], s[keep]) if keep.sum() > 0 else float("nan"))
    return aucs


# =========================================================================
# 4.  Permutation-based Shapley sampling
# =========================================================================

def shapley_values(model, X_background_mean, X_patients, n_permutations, seed):
    """
    Classic permutation sampling of Shapley values.

    For each patient and each random permutation of the features, features are
    switched on one at a time, starting from the background (reference) vector,
    here the feature means of the evaluable cohort.
    The marginal contribution of feature j is the change in predicted
    probability caused by switching j on. Averaging over permutations gives the
    signed Shapley value phi_j for that patient.

    Returns (phi_full, phi_half):
        phi_full  - (n_patients, n_features), averaged over all permutations
        phi_half  - same, averaged over the first half (convergence check)
    """
    rng = np.random.RandomState(seed)
    n_patients, n_features = X_patients.shape
    half = max(1, n_permutations // 2)

    phi_full = np.zeros((n_patients, n_features), dtype=float)
    phi_half = np.zeros((n_patients, n_features), dtype=float)

    t0 = time.time()
    for i in range(n_patients):
        x = X_patients[i]
        perms = np.array([rng.permutation(n_features) for _ in range(n_permutations)])

        # position of each feature within its permutation
        ranks = np.empty((n_permutations, n_features), dtype=int)
        ranks[np.arange(n_permutations)[:, None], perms] = np.arange(n_features)[None, :]

        # state k of a permutation has the first k features of that permutation
        # switched from the background vector to the patient's own values
        k = np.arange(n_features + 1)[None, :, None]          # (1, F+1, 1)
        mask = ranks[:, None, :] < k                          # (P, F+1, F)
        states = X_background_mean[None, None, :] + mask * (x - X_background_mean)[None, None, :]

        flat = states.reshape(-1, n_features)
        preds = get_probabilities(model, flat).reshape(n_permutations, n_features + 1)
        deltas = preds[:, 1:] - preds[:, :-1]          # (n_permutations, n_features)

        contrib = np.zeros((n_permutations, n_features), dtype=float)
        rows = np.arange(n_permutations)[:, None]
        contrib[rows, perms] = deltas

        phi_full[i] = contrib.mean(axis=0)
        phi_half[i] = contrib[:half].mean(axis=0)

        if (i + 1) % 10 == 0 or i == n_patients - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_patients - i - 1)
            log("      patient %3d/%d   elapsed %5.1fs   eta %5.1fs"
                % (i + 1, n_patients, elapsed, eta))

    return phi_full, phi_half


def global_importance(phi):
    """Mean absolute attribution per feature, normalised to sum to 100%."""
    imp = np.mean(np.abs(phi), axis=0)
    total = imp.sum()
    return imp / total * 100.0 if total > 0 else imp


def spearman(a, b):
    """Spearman rank correlation (used for the permutation convergence check)."""
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# =========================================================================
# 5.  Figures
# =========================================================================

def plot_shap_panel(ax, names, values, title):
    """Horizontal bar chart of the TOP_N_FEATURES most important features."""
    order = np.argsort(values)[::-1][:TOP_N_FEATURES]
    names = [names[i] for i in order]
    values = values[order]
    ypos = np.arange(len(names))[::-1]
    colors = [COL_RED if n in BELLMUNT_RELATED else COL_BLUE for n in names]

    ax.barh(ypos, values, color=colors, height=0.72)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean |Shapley| (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    span = values.max() if len(values) else 1.0
    ax.set_xlim(0, span * 1.30)
    for y, v in zip(ypos, values):
        ax.text(v + span * 0.03, y, "%.1f%%" % v, va="center", fontsize=9)


def save_shap_figures(shap_results):
    """Write one Shapley figure per endpoint and a combined panel figure."""
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=COL_RED, label="Bellmunt/ECOG/Hb-related"),
              Patch(facecolor=COL_BLUE, label="Other features")]

    for label, res in shap_results.items():
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        plot_shap_panel(ax, res["features"], res["importance"],
                        "%s\n(Random Forest)" % label)
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, "SHAP_%s.png" % label.replace(" ", "_"))
        fig.savefig(path, dpi=200)
        plt.close(fig)
        log("   wrote %s" % path)

    fig, axes = plt.subplots(1, len(shap_results), figsize=(6.2 * len(shap_results), 6.6))
    if len(shap_results) == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, shap_results.items()):
        plot_shap_panel(ax, res["features"], res["importance"],
                        "%s\n(Random Forest)" % label)
    fig.suptitle("Shapley Feature Importance \u2014 Lu-PSMA Landmark Cohort",
                 fontsize=15, fontweight="bold")
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = os.path.join(OUT_DIR, "SHAP_combined.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log("   wrote %s" % path)


# =========================================================================
# 6.  Main
# =========================================================================

def main():
    ensure_out_dir()

    rule("STEP 1  -  import pipeline and rebuild the landmark cohort")
    mod = import_pipeline()
    if hasattr(mod, "DATA_PATH"):
        mod.DATA_PATH = DATA_PATH
        log("Overrode pipeline DATA_PATH with: %s" % DATA_PATH)
    if hasattr(mod, "LANDMARK_CYCLE"):
        log("Pipeline LANDMARK_CYCLE = %s" % mod.LANDMARK_CYCLE)
    else:
        log("WARNING: pipeline has no LANDMARK_CYCLE constant -- is this the "
            "landmark version of the file?")

    if not hasattr(mod, "load_and_prepare_cohort"):
        diagnostic_dump(mod)
        raise SystemExit("load_and_prepare_cohort() not found in the pipeline file.")
    if not hasattr(mod, "build_feature_matrix"):
        diagnostic_dump(mod)
        raise SystemExit("build_feature_matrix() not found in the pipeline file.")

    df = call_flexibly(mod.load_and_prepare_cohort)
    log("Landmark cohort: %d patients" % len(df))

    feats = call_flexibly(mod.build_feature_matrix, df)
    if not isinstance(feats, pd.DataFrame):
        feats = pd.DataFrame(feats)
    log("Feature matrix: %d patients x %d features" % feats.shape)

    model_classes = resolve_models(mod)

    # ---- feature configurations -----------------------------------------
    returned = [c for c in feats.columns]
    brs_derived = [c for c in returned
                   if c.lower().startswith(BRS_FEATURE_PREFIXES)]
    all_features = [c for c in returned if c not in set(brs_derived)]

    dynamic_suffixes = ("_mean", "_last", "_slope")
    baseline_only = [c for c in all_features
                     if not c.endswith(dynamic_suffixes)
                     and c != "psa_nadir" and c != "ecog_last"]

    rule("STEP 2  -  feature configurations")
    log("build_feature_matrix() returned : %d columns" % len(returned))
    log("Removed as Bellmunt-derived     : %d  ->  %s"
        % (len(brs_derived), ", ".join(sorted(brs_derived))))
    if not brs_derived:
        log("WARNING: no Bellmunt-derived columns were found. If your pipeline "
            "names them differently, extend BRS_FEATURE_PREFIXES at the top of "
            "this script, otherwise the ML models can see the score they are "
            "being benchmarked against.")
    log("")
    log("(A) FULL landmark set          : %d features" % len(all_features))
    log("(B) BASELINE-ONLY set          : %d features" % len(baseline_only))
    log("")
    log("Baseline-only features:")
    log("   " + ", ".join(sorted(baseline_only)))
    log("")
    log("Dropped for the baseline-only model (early on-treatment information):")
    log("   " + ", ".join(sorted(set(all_features) - set(baseline_only))))

    base_set = set(baseline_only)
    full_set = set(all_features)
    pd.DataFrame({
        "feature": sorted(returned),
        "excluded_as_bellmunt_derived": [1 if f in set(brs_derived) else 0
                                         for f in sorted(returned)],
        "in_full_model": [1 if f in full_set else 0 for f in sorted(returned)],
        "in_baseline_only_model": [1 if f in base_set else 0 for f in sorted(returned)],
    }).to_csv(os.path.join(OUT_DIR, "feature_inventory.csv"), index=False)

    X_full_df = feats[all_features]
    X_base_df = feats[baseline_only]

    # ---- endpoint accounting + CV ---------------------------------------
    rule("STEP 3  -  endpoint accounting and cross-validated performance")

    perf_rows = []
    brs_rows = []
    count_rows = []
    shap_results = {}
    convergence_rows = []

    for label, col, definition in ENDPOINTS:
        if col not in df.columns:
            log("SKIPPING %s -- column '%s' not present in the cohort frame." % (label, col))
            continue

        y_raw = pd.to_numeric(df[col], errors="coerce").values
        evaluable = ~np.isnan(y_raw)
        n_eval = int(evaluable.sum())
        n_excluded = int(len(df) - n_eval)
        y = y_raw[evaluable].astype(int)
        n_events = int(y.sum())

        log("")
        log("%s  (%s)" % (label, definition))
        log("   landmark cohort            : %d" % len(df))
        log("   excluded (censored < horizon): %d" % n_excluded)
        log("   evaluable                  : %d" % n_eval)
        log("   events                     : %d  (%.1f%%)"
            % (n_events, 100.0 * n_events / n_eval))
        log("   event-free                 : %d  (%.1f%%)"
            % (n_eval - n_events, 100.0 * (n_eval - n_events) / n_eval))

        count_rows.append({
            "Endpoint": label,
            "Definition": definition,
            "Landmark cohort": len(df),
            "Excluded (censored before horizon)": n_excluded,
            "N evaluable": n_eval,
            "N events": n_events,
            "Event prevalence": "%.1f%%" % (100.0 * n_events / n_eval),
            "N event-free": n_eval - n_events,
            "Event-free proportion": "%.1f%%" % (100.0 * (n_eval - n_events) / n_eval),
        })

        folds = stratified_folds(y, N_FOLDS, RANDOM_SEED)

        for cfg_name, Xdf in (("Full landmark set", X_full_df),
                              ("Baseline-only set", X_base_df)):
            X = Xdf.values[evaluable].astype(float)
            log("   -- %s (%d features)" % (cfg_name, X.shape[1]))
            res = run_cv(X, y, folds, model_classes)
            for model_label, metrics in res.items():
                row = {"Endpoint": label, "Feature set": cfg_name, "Model": model_label}
                for metric in ("AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"):
                    row[metric] = fmt_mean_sd(metrics[metric])
                perf_rows.append(row)
                log("        %-22s AUC %s" % (model_label, row["AUC"]))

        for score_label, score_col in BRS_COLUMNS:
            if score_col not in df.columns:
                log("        (score column '%s' not found -- skipped)" % score_col)
                continue
            s = pd.to_numeric(df[score_col], errors="coerce").values[evaluable]
            aucs = run_cv_score_as_classifier(s, y, folds)
            brs_rows.append({"Endpoint": label, "Score": score_label,
                             "AUC (5-fold CV)": fmt_mean_sd(aucs)})
            log("        %-32s AUC %s" % (score_label, fmt_mean_sd(aucs)))

        # ---- Shapley on the full-feature RF, fitted on the whole cohort ---
        log("   -- Shapley attribution (%d patients, %d permutations)"
            % (N_SHAP_PATIENTS, N_PERMUTATIONS))
        X = X_full_df.values[evaluable].astype(float)
        X_imp, _, means = mean_impute(X, X)
        rf = model_classes["Random Forest"]()
        rf.fit(X_imp, y)

        rng = np.random.RandomState(RANDOM_SEED)
        n_sample = min(N_SHAP_PATIENTS, X_imp.shape[0])
        sample_idx = rng.choice(X_imp.shape[0], size=n_sample, replace=False)

        phi_full, phi_half = shapley_values(
            rf, means, X_imp[sample_idx], N_PERMUTATIONS, RANDOM_SEED)

        imp_full = global_importance(phi_full)
        imp_half = global_importance(phi_half)
        rho = spearman(imp_full, imp_half)
        convergence_rows.append({
            "Endpoint": label,
            "N patients": n_sample,
            "Permutations (full)": N_PERMUTATIONS,
            "Permutations (half)": N_PERMUTATIONS // 2,
            "Spearman rho of global ranking": "%.3f" % rho,
        })
        log("      convergence check: Spearman rho (%d vs %d permutations) = %.3f"
            % (N_PERMUTATIONS // 2, N_PERMUTATIONS, rho))

        shap_results[label] = {"features": list(X_full_df.columns),
                               "importance": imp_full}

        shap_df = pd.DataFrame({
            "feature": list(X_full_df.columns),
            "mean_abs_shapley_pct": imp_full,
            "mean_abs_shapley_pct_half_permutations": imp_half,
            "mean_signed_shapley": phi_full.mean(axis=0),
        }).sort_values("mean_abs_shapley_pct", ascending=False)
        shap_path = os.path.join(OUT_DIR, "shap_%s.csv" % label.replace(" ", "_"))
        shap_df.to_csv(shap_path, index=False)
        log("      wrote %s" % shap_path)

    # ---- write everything ------------------------------------------------
    rule("STEP 4  -  writing outputs")

    pd.DataFrame(count_rows).to_csv(
        os.path.join(OUT_DIR, "endpoint_counts.csv"), index=False)
    log("   wrote endpoint_counts.csv")

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(os.path.join(OUT_DIR, "performance_full_vs_baseline.csv"), index=False)
    log("   wrote performance_full_vs_baseline.csv")

    pd.DataFrame(brs_rows).to_csv(
        os.path.join(OUT_DIR, "brs_baseline_auc.csv"), index=False)
    log("   wrote brs_baseline_auc.csv")

    pd.DataFrame(convergence_rows).to_csv(
        os.path.join(OUT_DIR, "shap_convergence.csv"), index=False)
    log("   wrote shap_convergence.csv")

    if shap_results:
        save_shap_figures(shap_results)

    with open(os.path.join(OUT_DIR, "run_settings.json"), "w") as fh:
        json.dump({
            "pipeline_path": PIPELINE_PATH,
            "data_path": DATA_PATH,
            "n_shap_patients": N_SHAP_PATIENTS,
            "n_permutations": N_PERMUTATIONS,
            "n_folds": N_FOLDS,
            "random_seed": RANDOM_SEED,
            "landmark_cycle": getattr(mod, "LANDMARK_CYCLE", None),
            "n_columns_returned_by_pipeline": len(returned),
            "n_features_excluded_as_bellmunt_derived": len(brs_derived),
            "features_excluded_as_bellmunt_derived": sorted(brs_derived),
            "n_features_full": len(all_features),
            "n_features_baseline_only": len(baseline_only),
            "auc_tie_handling": "mid-ranks (a constant predictor scores exactly 0.500)",
        }, fh, indent=2)
    log("   wrote run_settings.json")

    rule("SUMMARY  -  AUC, full vs. baseline-only vs. Bellmunt")
    if len(perf):
        auc_table = perf.pivot_table(index=["Endpoint", "Model"], columns="Feature set",
                                     values="AUC", aggfunc="first")
        log(auc_table.to_string())
    log("")
    if brs_rows:
        log(pd.DataFrame(brs_rows).to_string(index=False))

    with open(os.path.join(OUT_DIR, "run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG_LINES))
    log("")
    log("   wrote run_summary.txt")
    log("")
    log("Done. All outputs are in:")
    log("   %s" % OUT_DIR)


if __name__ == "__main__":
    main()
