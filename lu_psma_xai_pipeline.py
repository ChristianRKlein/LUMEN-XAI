# -*- coding: utf-8 -*-
"""
lu_psma_xai_pipeline.py
=======================

Explainable machine learning for outcome prediction in men with metastatic
castration-resistant prostate cancer (mCRPC) undergoing [177Lu]Lu-PSMA-617
radioligand therapy: a landmark analysis benchmarked against the Bellmunt
Risk Score (BRS).

Pipeline
--------
  1. Data loading and cleaning (dates, numeric fields, event definitions).
  2. Construction of the original, modified and enhanced Bellmunt Risk Score
     [Bellmunt et al., J Clin Oncol 2010; Buettner et al., EJNMMI 2026].
  3. Kaplan-Meier estimation of OS / PFS stratified by BRS variant.
  4. Landmark feature engineering: static baseline features plus early
     longitudinal laboratory trajectories (cycles 1..LANDMARK_CYCLE).
  5. Classification of three binary endpoints (1-year OS, 2-year OS,
     1-year radiographic PFS) with Random Forest, Gradient Boosting,
     L2-regularised logistic regression and a majority-class baseline,
     evaluated by stratified K-fold cross-validation with fixed
     hyperparameters (single-level, not nested).
  6. Fold-matched benchmark of the raw and enhanced BRS, used directly as
     ranking scores, against the trained models.
  7. Global feature importance via permutation-based Shapley-value sampling.

Relation to the article
-----------------------
This module is the single source of the cohort definition, the endpoint
definitions, the feature engineering and the classifier implementations;
landmark_analysis.py imports them from here. Running this script directly
produces the Kaplan-Meier figures of the article (Fig. 1, Supplementary
Fig. S1; files KM_OS.png and KM_PFS.png).

The cross-validation, BRS benchmark and Shapley steps in `main()` (steps
5-7) are the preliminary analysis (trapezoidal AUC without tie correction;
Shapley attribution in 40 patients, absent features masked by a placeholder
value). All model performance figures, BRS comparisons and Shapley results
reported in the article (Tables 2-3, Fig. 2, Supplementary Tables S1, S3-S6,
Supplementary Figs. S2-S4) are produced by landmark_analysis.py, which uses
tie-corrected AUC, separate full and baseline-only feature sets, 150
patients per endpoint and a mean-value reference for Shapley sampling.

Design decisions
----------------
  * Landmark design. Only patients who reached treatment cycle
    LANDMARK_CYCLE are analysed, and every longitudinal feature is computed
    exclusively from cycles 1..LANDMARK_CYCLE. Using the full treatment
    course would let information recorded after the outcome horizon leak
    into the predictors (e.g. a PSA nadir reached at cycle 8 implies
    survival to cycle 8).
  * Censoring-aware endpoints. A patient censored before the respective
    horizon without an event has an unknown status at that horizon and is
    excluded from that endpoint, rather than coded as event-free.
  * No circularity. BRS component flags and composite scores are removed
    from the machine-learning inputs, because the BRS is the comparator.
  * No leakage in cross-validation. Mean imputation and standardisation
    use training-fold statistics only.

Implementation
--------------
All model classes (decision tree, Random Forest, Gradient Boosting,
logistic regression, Kaplan-Meier estimator, Shapley sampler) are
implemented from first principles using only NumPy and pandas, without
external machine-learning libraries, for full methodological
transparency. The corresponding mathematical description is given in the
Supplementary Methods of the accompanying article.

Usage
-----
Set DATA_PATH and OUTPUT_DIR in the CONFIGURATION block below, then run

    python lu_psma_xai_pipeline.py

Patient-level data are not part of this repository; see data/README.md.

Author : Christian R. Klein, University Hospital Bonn (christian.klein@ukbonn.de)
License: MIT (see LICENSE)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Trapezoidal integration: `np.trapz` was renamed to `np.trapezoid` in
# NumPy 2.0 and later removed. Both compute the identical quantity.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# ============================================================
# CONFIGURATION
# ============================================================
# DATA_PATH must point to a CSV export of the LUMEN cohort with the column
# names referenced in `load_and_prepare_cohort()` and
# `build_feature_matrix()` (documented in data/README.md). Patient-level
# data are not part of this repository.
DATA_PATH = os.path.join("data", "lu_psma_data.csv")
OUTPUT_DIR = "results"

RANDOM_SEED = 42        # global seed for reproducibility
N_FOLDS = 5             # number of cross-validation folds
SHAP_RUNS = 256         # permutation runs per patient for Shapley sampling
LANDMARK_CYCLE = 2      # landmark: patients must have reached this cycle, and
                        # longitudinal features use cycles 1..LANDMARK_CYCLE only
SHAP_PLACEHOLDER = -999.0  # value used to mask an absent feature during
                           # Shapley sampling (Supplementary Methods, Sect. 3.2)

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GENERIC DATA-CLEANING HELPERS
# ============================================================

def to_numeric_robust(series: pd.Series) -> pd.Series:
    """Convert a messy string column to numeric.

    Handles values such as '>70', '<0.6', comma decimal separators, and
    common missing-value tokens ('na', 'NA', 'none', ...). Leading '>' or
    '<' qualifiers are stripped, so '>10' becomes 10.0.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.replace({"na": "", "NA": "", "nan": "", "none": "", "None": "", "NaT": ""})
    s = s.str.replace(r"^[><]\s*", "", regex=True)   # strip leading '>' / '<'
    s = s.str.replace(",", ".", regex=False)          # German decimal comma
    return pd.to_numeric(s, errors="coerce")


def to_date_robust(series: pd.Series) -> pd.Series:
    """Convert a messy string column to datetime.

    Handles a trailing 'UTC' / '00:00:00 UTC' suffix, a data-entry sentinel
    value starting with '3000' (treated as missing), and day-first date
    formats common in German clinical databases.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*00:00:00\s*UTC\s*$", "", regex=True)
    s = s.str.replace(r"\s*UTC\s*$", "", regex=True)
    s = s.where(~s.str.startswith("3000"), other=np.nan)
    s = s.replace({"nan": np.nan, "NaT": np.nan, "na": np.nan, "": np.nan})
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def load_csv_autodetect(path: str) -> pd.DataFrame:
    """Load a CSV file, auto-detecting encoding and separator.

    Tries common encodings (utf-8-sig, latin-1, cp1252, utf-8) and
    separators (';', ',', '\\t') until a plausible table (>5 columns) is
    found. Also strips BOM artefacts from column names.
    """
    for enc in ["utf-8-sig", "latin-1", "cp1252", "utf-8"]:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    df.columns = [
                        c.replace("﻿", "").strip() for c in df.columns
                    ]
                    print(f"  Loaded with encoding={enc}, separator='{sep}'")
                    return df
            except Exception:
                continue
    raise ValueError(f"Could not load {path} with any tried encoding/separator combination.")


def impute_train_mean(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute missing values, using ONLY training-fold statistics.

    The per-feature mean is estimated exclusively from `X_train` and then
    applied to fill missing entries in both `X_train` and `X_test`, so no
    information from the test fold enters the training data.
    """
    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)  # feature entirely missing in this fold -> 0
    A, B = X_train.copy(), X_test.copy()
    for j in range(X_train.shape[1]):
        A[np.isnan(A[:, j]), j] = means[j]
        B[np.isnan(B[:, j]), j] = means[j]
    return A, B


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity 2p(1-p) of a binary (or [0,1]-valued) target vector."""
    p = np.mean(y)
    return 2 * p * (1 - p) if len(y) > 0 else 0.0


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC (trapezoidal integration of the empirical ROC curve).

    Caveat: if `y_score` is constant across all instances (as for the
    majority-class baseline), the value is determined by the tie-breaking
    order of the sort rather than by discriminative signal and must not be
    read as chance-level (0.5) performance (Supplementary Methods, Sect. 2).
    """
    order = np.argsort(-y_score)
    y = y_true[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    tpr.append(1.0)
    fpr.append(1.0)
    return abs(float(_trapezoid(tpr, fpr)))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUC, accuracy, sensitivity, specificity, PPV and NPV."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {
        "auc": auc_score(y_true, y_score),
        "acc": np.mean(y_true == y_pred),
        "sens": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "spec": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


# ============================================================
# FROM-SCRATCH CLASSIFIERS
# ============================================================

class _TreeNode:
    """Node of a binary decision tree (internal split or leaf)."""
    __slots__ = ["feature", "threshold", "left", "right", "value"]

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class DecisionTree:
    """Minimal CART-style tree with Gini-impurity splits.

    Leaves store the mean target value, so the same class serves as a
    classification tree (Random Forest) and as a regression tree on
    residuals (Gradient Boosting).
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 4,
                 n_features_per_split: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_per_split = n_features_per_split
        self.rng = rng or np.random.RandomState(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode()
        node.value = float(np.mean(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return node

        # Random feature subset per split (all features if n_features_per_split is None)
        n_features = X.shape[1]
        k = self.n_features_per_split or n_features
        candidate_features = self.rng.choice(n_features, min(k, n_features), replace=False)

        # Exhaustive search over mid-points between consecutive unique values
        best_gain, best_feature, best_threshold = -1.0, None, None
        parent_impurity = gini_impurity(y)
        for f in candidate_features:
            values = np.unique(X[:, f])
            if len(values) < 2:
                continue
            for t in (values[:-1] + values[1:]) / 2:
                mask = X[:, f] <= t
                if mask.sum() < 2 or (~mask).sum() < 2:
                    continue
                gain = parent_impurity - (
                    mask.sum() / len(y) * gini_impurity(y[mask])
                    + (~mask).sum() / len(y) * gini_impurity(y[~mask])
                )
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, f, t

        if best_feature is None:
            return node

        node.feature, node.threshold = best_feature, best_threshold
        left_mask = X[:, best_feature] <= best_threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> float:
        if node.feature is None:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict_proba_1d(self, X: np.ndarray) -> np.ndarray:
        """Leaf value for each row (class-1 frequency, or residual mean in boosting)."""
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForestClassifier:
    """Bootstrap-aggregated ensemble of `DecisionTree`s.

    Each tree is grown on a bootstrap sample and considers floor(sqrt(p))
    randomly chosen features per split. The forest probability is the mean
    of the trees' leaf class-1 frequencies.
    """

    def __init__(self, n_estimators: int = 80, max_depth: int = 6, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.RandomState(self.random_state)
        k = max(1, int(np.sqrt(X.shape[1])))
        n = len(X)
        self.trees_ = []
        for _ in range(self.n_estimators):
            bootstrap_idx = rng.choice(n, n, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                n_features_per_split=k,
                rng=np.random.RandomState(rng.randint(0, 99_999)),
            )
            tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.mean([tree.predict_proba_1d(X) for tree in self.trees_], axis=0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostingClassifier:
    """Gradient boosting for the binomial log-loss with shallow regression trees.

    Starting from the logit of the training prevalence, each tree is fitted
    to the negative gradient y - sigmoid(F) and added with a fixed learning
    rate; probabilities are sigmoid(F).
    """

    def __init__(self, n_estimators: int = 80, max_depth: int = 3, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        rng = np.random.RandomState(self.random_state)
        p0 = np.clip(np.mean(y), 0.01, 0.99)
        self.base_score_ = np.log(p0 / (1 - p0))  # logit of the base rate
        F = np.full(len(y), self.base_score_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y - sigmoid(F)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=4,
                                 rng=np.random.RandomState(rng.randint(0, 99_999)))
            tree.fit(X, residual)
            F += self.learning_rate * tree.predict_proba_1d(X)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict_proba_1d(X)
        p1 = sigmoid(F)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LogisticRegressionClassifier:
    """L2-regularised logistic regression fitted by batch gradient descent.

    Features are standardised to zero mean and unit variance with
    statistics computed exclusively on the data passed to `fit()` (the
    training fold) and re-used unchanged in `predict_proba()`. With a fixed
    regularisation strength, standardisation is required for gradient
    descent to converge on features of very different scale (e.g. PSA vs.
    binary flags); see Supplementary Methods, Sect. 4.
    """

    def __init__(self, learning_rate: float = 0.05, n_iter: int = 500, l2: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)  # avoid division by zero
        X_std = (X - self.mean_) / self.std_

        self.coef_ = np.zeros(X_std.shape[1])
        self.intercept_ = 0.0
        n = len(X_std)
        for _ in range(self.n_iter):
            p = sigmoid(X_std @ self.coef_ + self.intercept_)
            error = p - y
            grad_coef = X_std.T @ error / n + self.l2 * self.coef_
            grad_intercept = error.mean()
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_std = (X - self.mean_) / self.std_
        p1 = sigmoid(X_std @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MajorityClassBaseline:
    """Predicts the training-fold majority class and prevalence for every instance.

    Because the predicted probability is constant, its AUC is not
    interpretable as chance-level performance (see `auc_score()`).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassBaseline":
        self.majority_class_ = int(np.bincount(y.astype(int)).argmax())
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.full(X.shape[0], self.prevalence_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.majority_class_, dtype=int)


# ============================================================
# KAPLAN-MEIER ESTIMATION
# ============================================================

def kaplan_meier_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product-limit (Kaplan-Meier) survival curve for a single group."""
    df = pd.DataFrame({"t": time, "e": event}).dropna().sort_values("t").reset_index(drop=True)
    survival = 1.0
    n_at_risk = len(df)
    times = [0.0]
    survival_probs = [1.0]
    for t, group in df.groupby("t"):
        n_events = group["e"].sum()
        if n_at_risk > 0 and n_events > 0:
            survival *= 1 - n_events / n_at_risk
        n_at_risk -= len(group)
        times.append(float(t))
        survival_probs.append(survival)
    return np.array(times), np.array(survival_probs)


def plot_km_panel(df: pd.DataFrame, time_col: str, event_col: str, score_col: str, title: str, ax) -> None:
    """Plot Kaplan-Meier curves for each level of `score_col` on a matplotlib axis."""
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad", "#2c3e50"]
    sub = df[[time_col, event_col, score_col]].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    n_plotted = 0
    for i, score in enumerate(sorted(sub[score_col].unique())):
        mask = sub[score_col] == score
        if mask.sum() < 3:
            continue
        t, s = kaplan_meier_curve(sub.loc[mask, time_col].values, sub.loc[mask, event_col].values)
        crossing = np.where(s <= 0.5)[0]
        median = f"{t[crossing[0]]:.1f}" if len(crossing) > 0 else ">max"
        ax.step(t, s, where="post", color=colors[i % len(colors)], lw=2.0,
                label=f"Score {int(score)} (n={mask.sum()}, med={median}m)")
        n_plotted += 1

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if n_plotted > 0:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No valid time data", transform=ax.transAxes,
                 ha="center", va="center", color="red", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)


# ============================================================
# SHAPLEY-VALUE FEATURE ATTRIBUTION
# ============================================================

def shapley_values_single_patient(model, x: np.ndarray, n_runs: int = SHAP_RUNS,
                                   placeholder: float = SHAP_PLACEHOLDER) -> np.ndarray:
    """Estimate per-feature attributions for one patient by permutation sampling.

    For each of `n_runs` random feature orderings, features are revealed one
    at a time (absent features are set to `placeholder`), and the absolute
    change in predicted class-1 probability caused by revealing each feature
    is accumulated. The result is the mean absolute marginal contribution
    per feature.

    Used by the preliminary analysis in `main()` only. The Shapley results
    reported in the article come from `shapley_values()` in
    landmark_analysis.py (signed values relative to the cohort-mean
    reference, then mean absolute value across patients).
    """
    n_features = len(x)
    contributions = np.zeros(n_features)
    for _ in range(n_runs):
        perm = np.random.permutation(n_features)
        x_with = np.full(n_features, placeholder)
        x_without = np.full(n_features, placeholder)
        for pos, feature_idx in enumerate(perm):
            x_with[:] = placeholder
            x_without[:] = placeholder
            for p in perm[: pos + 1]:
                x_with[p] = x[p]
            for p in perm[:pos]:
                x_without[p] = x[p]
            p_with = model.predict_proba(x_with.reshape(1, -1))[0, 1]
            p_without = model.predict_proba(x_without.reshape(1, -1))[0, 1]
            contributions[feature_idx] += abs(p_with - p_without)
    return contributions / n_runs


def shapley_matrix(model, X: np.ndarray, feature_names: list[str],
                    n_patients: int = 40, n_runs: int = SHAP_RUNS) -> np.ndarray:
    """Return a (patients x features) attribution matrix for a random patient sample."""
    idx = np.random.choice(len(X), min(n_patients, len(X)), replace=False)
    matrix = np.zeros((len(idx), len(feature_names)))
    for i, row_idx in enumerate(idx):
        matrix[i] = shapley_values_single_patient(model, X[row_idx], n_runs=n_runs)
        if (i + 1) % 10 == 0:
            print(f"    Shapley sampling: {i + 1}/{len(idx)} patients done")
    return matrix


# Features highlighted in the Shapley figures as related to BRS components
BRS_RELATED_FEATURES = {
    "ecog_baseline", "hb_baseline", "crp_baseline", "has_liver_mets",
    "brs_ecog", "brs_hb", "brs_liver", "brs_crp",
}


def plot_shap_importance(matrix: np.ndarray, feature_names: list[str], endpoint_name: str,
                          model_name: str, top_n: int = 15) -> None:
    """Bar plot of mean absolute attribution (normalised to 100 %), BRS-related features highlighted."""
    mean_abs = np.abs(matrix).mean(axis=0)
    total = mean_abs.sum()
    pct = mean_abs / total * 100 if total > 0 else mean_abs
    top_idx = np.argsort(pct)[-top_n:]
    names = [feature_names[i] for i in top_idx]
    values = pct[top_idx]
    colors = ["#C0392B" if n in BRS_RELATED_FEATURES else "#2C5F8A" for n in names]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, height=0.65)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |Shapley| Contribution (%)", fontsize=10)
    ax.set_title(f"{endpoint_name}\n({model_name})", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.3)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", ls="--", alpha=0.4)
    legend_handles = [
        mpatches.Patch(color="#C0392B", label="Bellmunt/ECOG/Hb-related"),
        mpatches.Patch(color="#2C5F8A", label="Other features"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"SHAP_{endpoint_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# CROSS-VALIDATION (single-level, fixed hyperparameters)
# ============================================================

def stratified_folds(y: np.ndarray, k: int = N_FOLDS, seed: int = RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split indices into k folds that preserve class prevalence.

    Returns a list of (train_idx, test_idx) tuples. Because the partition
    depends only on `y` and `seed`, the ML models and the BRS benchmark are
    evaluated on identical folds (Supplementary Methods, Sect. 4.1).
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    folds = []
    for i in range(k):
        test_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate(
            [np.concatenate([pos_folds[j] for j in range(k) if j != i]),
             np.concatenate([neg_folds[j] for j in range(k) if j != i])]
        )
        folds.append((train_idx, test_idx))
    return folds


def run_cross_validation(X: np.ndarray, y: np.ndarray, endpoint_name: str, n_folds: int = N_FOLDS) -> list[dict]:
    """Cross-validated evaluation of RF / GB / LR / majority baseline for one endpoint.

    Hyperparameters are fixed a priori and identical across folds and
    endpoints. This is a single-level K-fold cross-validation estimating
    the performance of one fixed configuration per algorithm; there is no
    inner-loop tuning (Supplementary Methods, Sect. 4.3). Metrics are
    reported as mean +/- SD across folds.
    """
    model_names = ["Random Forest", "Gradient Boosting", "Logistic Regression", "Mode"]
    results = {m: {k: [] for k in ["auc", "acc", "sens", "spec", "ppv", "npv"]} for m in model_names}
    folds = stratified_folds(y, k=n_folds, seed=RANDOM_SEED)

    for fold_i, (train_idx, test_idx) in enumerate(folds, 1):
        X_train, X_test = impute_train_mean(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        models = [
            RandomForestClassifier(n_estimators=80, max_depth=6, random_state=RANDOM_SEED + fold_i * 10),
            GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED + fold_i * 10),
            LogisticRegressionClassifier(learning_rate=0.05, n_iter=500, l2=0.01),
            MajorityClassBaseline(),
        ]
        for name, model in zip(model_names, models):
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            metrics = classification_metrics(y_test, y_pred, y_score)
            for k, v in metrics.items():
                results[name][k].append(v)
        print(f"  Fold {fold_i}/{n_folds} done")

    rows = []
    print(f"\n{'=' * 55}\n{endpoint_name}\n{'=' * 55}")
    print(f"{'Model':<25} {'AUC':>12} {'Acc':>12} {'Sens':>10} {'Spec':>10}")
    for name in model_names:
        v = results[name]
        mean_auc, sd_auc = np.mean(v["auc"]), np.std(v["auc"])
        mean_acc, sd_acc = np.mean(v["acc"]), np.std(v["acc"])
        mean_sens, sd_sens = np.mean(v["sens"]), np.std(v["sens"])
        mean_spec, sd_spec = np.mean(v["spec"]), np.std(v["spec"])
        mean_ppv, sd_ppv = np.mean(v["ppv"]), np.std(v["ppv"])
        mean_npv, sd_npv = np.mean(v["npv"]), np.std(v["npv"])
        print(f"  {name:<23} {mean_auc:.3f}±{sd_auc:.3f}  {mean_acc:.3f}±{sd_acc:.3f}"
              f"  {mean_sens:.3f}±{sd_sens:.3f}  {mean_spec:.3f}±{sd_spec:.3f}")
        rows.append({
            "Outcome": endpoint_name, "Model": name,
            "AUC": f"{mean_auc:.3f}±{sd_auc:.3f}",
            "Accuracy": f"{mean_acc:.3f}±{sd_acc:.3f}",
            "Sensitivity": f"{mean_sens:.3f}±{sd_sens:.3f}",
            "Specificity": f"{mean_spec:.3f}±{sd_spec:.3f}",
            "PPV": f"{mean_ppv:.3f}±{sd_ppv:.3f}",
            "NPV": f"{mean_npv:.3f}±{sd_npv:.3f}",
        })
    return rows


def compare_brs_baseline(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[dict]:
    """Benchmark the raw and enhanced Bellmunt Risk Score directly as classifiers.

    The integer score is used as-is as a ranking variable (no model
    fitting) and evaluated by AUC on stratified folds generated with the
    same seed as for the ML models, giving a fold-matched comparison.
    """
    print("\n--- Benchmark: raw Bellmunt Score as a simple classifier ---")
    endpoints = [("1-Year OS", "os_1yr"), ("2-Year OS", "os_2yr"), ("1-Year PFS", "pfs_1yr")]
    score_cols = ["bellmunt_score", "bellmunt_enhanced"]
    rows = []
    for endpoint_name, endpoint_col in endpoints:
        if endpoint_col not in df.columns:
            continue
        y_full = df[endpoint_col].values.astype(float)  # float: NaN marks patients excluded by censoring
        for score_col in score_cols:
            if score_col not in df.columns:
                continue
            s_full = df[score_col].values.astype(float)
            valid = ~np.isnan(s_full) & ~np.isnan(y_full)
            y_valid, s_valid = y_full[valid].astype(int), s_full[valid]
            if y_valid.sum() < 8:
                continue
            folds = stratified_folds(y_valid, k=n_folds, seed=RANDOM_SEED)
            fold_aucs = [auc_score(y_valid[test_idx], s_valid[test_idx]) for _, test_idx in folds]
            mean_auc, sd_auc = np.mean(fold_aucs), np.std(fold_aucs)
            print(f"  {endpoint_name:<12} {score_col:<20} AUC = {mean_auc:.3f}±{sd_auc:.3f}")
            rows.append({"Outcome": endpoint_name, "Score": score_col, "AUC": f"{mean_auc:.3f}±{sd_auc:.3f}"})
    if rows:
        out_path = os.path.join(OUTPUT_DIR, "brs_baseline_auc.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    return rows


# ============================================================
# DATA LOADING, EVENT DEFINITION AND BRS CONSTRUCTION
# ============================================================

def load_and_prepare_cohort() -> pd.DataFrame:
    """Load the CSV export, define the landmark cohort, derive OS/PFS and the BRS variants.

    Returns one row per landmark-eligible patient with, among others,
    `os_months`, `os_event`, `pfs_months`, `pfs_event`, the BRS variants
    and the binary endpoints `os_1yr`, `os_2yr`, `pfs_1yr` (NaN = excluded
    because censored before the horizon).
    """
    print(f"\nLoading: {DATA_PATH}")
    df = load_csv_autodetect(DATA_PATH)
    print(f"Raw shape: {df.shape[0]} patients x {df.shape[1]} columns")

    # --- Cycle dates (z1..z13) and other clinically relevant dates ---
    cycle_date_cols = [f"z{i}" for i in range(1, 14)]
    other_date_cols = ["toddat", "Progression_Date_or_last_imaging_without_Progression", "gebdat", "ED"]
    for c in cycle_date_cols + other_date_cols:
        if c in df.columns:
            df[c] = to_date_robust(df[c])

    # Time origin: date of the first [177Lu]Lu-PSMA-617 cycle
    df["start"] = df["z1"] if "z1" in df.columns else pd.Series(pd.NaT, index=df.index)

    # Last documented cycle date, used as the censoring date for OS
    cycle_cols_present = [c for c in cycle_date_cols if c in df.columns]
    if cycle_cols_present:
        date_matrix = df[cycle_cols_present].copy()
        for c in date_matrix.columns:
            date_matrix.loc[date_matrix[c].dt.year > 2100, c] = pd.NaT  # sentinel-date guard
        df["last_date"] = date_matrix.max(axis=1)
    else:
        df["last_date"] = df["start"]

    # --- Overall survival (OS) ---
    def _binary_event(s: pd.Series) -> pd.Series:
        """Map an event column to {0, 1}; anything other than 1 (incl. missing) -> 0."""
        x = pd.to_numeric(
            s.astype(str).str.strip().str.lower().replace({"na": "0", "nan": "0", "none": "0", "": "0"}),
            errors="coerce",
        ).fillna(0)
        return (x == 1).astype(int)

    df["os_event"] = _binary_event(df.get("verstorben", pd.Series(0, index=df.index)))
    death_date = df.get("toddat", pd.Series(pd.NaT, index=df.index))
    end_date_os = death_date.where(df["os_event"] == 1, df["last_date"])
    df["os_months"] = (end_date_os - df["start"]).dt.days / 30.44

    # --- Radiographic progression-free survival (PFS) ---
    df["pfs_event"] = _binary_event(df.get("Radiographic_Progression_y_n", pd.Series(0, index=df.index)))
    progression_date = df.get("Progression_Date_or_last_imaging_without_Progression", pd.Series(pd.NaT, index=df.index))
    df["pfs_months"] = (progression_date - df["start"]).dt.days / 30.44

    # --- Cohort filtering: valid start date and positive follow-up ---
    df = df[df["start"].notna()].copy()
    df = df[~((df["os_months"] <= 0) & df["os_months"].notna())].copy()

    # --- Landmark restriction ---
    # Only patients who reached cycle LANDMARK_CYCLE (alive and on treatment
    # at the landmark) are analysed. This matches the landmark-restricted
    # feature window in build_feature_matrix(), so that no predictor uses
    # information recorded after the outcome horizon. The resulting cohort
    # is more selected than the full treated population, which must be
    # considered when interpreting the results.
    if f"z{LANDMARK_CYCLE}" in df.columns:
        n_before = len(df)
        df = df[df[f"z{LANDMARK_CYCLE}"].notna()].copy()
        print(f"Patients reaching cycle {LANDMARK_CYCLE} (landmark cohort): "
              f"{len(df)} (excluded {n_before - len(df)} who did not reach the landmark)")

    print(f"Patients with evaluable OS data: {df['os_months'].notna().sum()}")
    print(f"OS events: {df['os_event'].sum()}  |  PFS events: {df['pfs_event'].sum()}")

    if "gebdat" in df.columns:
        df["age"] = (df["start"] - df["gebdat"]).dt.days / 365.25

    # --- Bellmunt Risk Score (original, modified, enhanced) ---
    # Components: ECOG > 0, hemoglobin < 10 g/dL, liver metastasis; the
    # enhanced variant adds CRP > 30 mg/L, the modified variant uses the
    # ECOG grade (0-3) instead of a binary flag.
    # The liver component is derived from `z1_Met_Viszeral_1`, which counts
    # involved visceral sites (0-3) rather than flagging liver involvement
    # specifically; organ-level site data are not available in this export.
    # It therefore encodes ">= 1 visceral site" and acts as a proxy for the
    # liver component (see manuscript Methods and Limitations).
    ecog = to_numeric_robust(df.get("ECOG_first", pd.Series(np.nan, index=df.index)))
    hb_baseline = to_numeric_robust(df.get("hb1", pd.Series(np.nan, index=df.index)))
    liver_mets = (to_numeric_robust(df.get("z1_Met_Viszeral_1", pd.Series(0, index=df.index))).fillna(0) > 0).astype(float)
    crp_baseline = to_numeric_robust(df.get("crp1", pd.Series(np.nan, index=df.index)))

    df["brs_ecog"] = (ecog > 0).astype(float)
    df["brs_hb"] = (hb_baseline < 10).astype(float)
    df["brs_liver"] = liver_mets
    df["brs_crp"] = (crp_baseline > 30).astype(float)
    df["bellmunt_score"] = df["brs_ecog"] + df["brs_hb"] + df["brs_liver"]
    df["bellmunt_enhanced"] = df["bellmunt_score"] + df["brs_crp"]
    df["bellmunt_modified"] = ecog.clip(0, 3) + df["brs_hb"] + df["brs_liver"]

    print("\nBellmunt Score distribution:")
    print(df["bellmunt_score"].value_counts().sort_index().to_string())

    # --- Binary endpoints with censoring-aware exclusion ---
    # 1   = event within the horizon
    # 0   = follow-up reaches the horizon without an event
    # NaN = censored before the horizon without an event; the status at the
    #       horizon is unknown, so the patient is excluded from this endpoint
    #       instead of being coded as event-free.
    def _binary_endpoint_with_exclusion(time_months: pd.Series, event: pd.Series, horizon: float) -> pd.Series:
        out = pd.Series(np.nan, index=time_months.index)
        event_in_window = (event == 1) & (time_months <= horizon)
        survived_window = (time_months >= horizon)
        out[event_in_window] = 1
        out[survived_window & ~event_in_window] = 0
        return out  # remaining NaN = censored before horizon -> excluded

    df["os_1yr"] = _binary_endpoint_with_exclusion(df["os_months"], df["os_event"], 12)
    df["os_2yr"] = _binary_endpoint_with_exclusion(df["os_months"], df["os_event"], 24)
    df["pfs_1yr"] = _binary_endpoint_with_exclusion(df["pfs_months"], df["pfs_event"], 12)
    for ep in ["os_1yr", "os_2yr", "pfs_1yr"]:
        n_eval = df[ep].notna().sum()
        n_event = (df[ep] == 1).sum()
        print(f"  {ep}: {n_eval} evaluable (excluded {len(df) - n_eval} censored before horizon), "
              f"{n_event} events ({n_event / n_eval * 100:.1f}% of evaluable)")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the static + landmark-restricted longitudinal feature matrix.

    Static features: clinical, imaging (PSMA PET), CT body composition and
    BRS columns (the latter are removed again before model training in
    `main()`). Longitudinal features: baseline, mean, last value and slope
    of eight laboratory parameters over cycles 1..LANDMARK_CYCLE, plus PSA
    nadir. Features with > 70 % missingness are dropped.
    """
    feats = {}

    # (feature name, source column in the export)
    static_features = [
        ("ecog_baseline", "ECOG_first"), ("ecog_last", "ECOG last"),
        ("bmi", "bmi"), ("gleason", "Gleason_Score"),
        ("prior_op", "Radikale_OP"), ("prior_ctx", "CTX"),
        ("prior_nht", "NHT"), ("prior_adt", "ADT"),
        ("n_total_mets", "z1_Met_Total_Anzahl"),
        ("n_visceral_mets", "z1_viszeral_Anzahl"),
        # Count of involved visceral sites (0-3), used as proxy for the BRS
        # liver component; see load_and_prepare_cohort().
        ("has_liver_mets", "z1_Met_Viszeral_1"),
        ("n_outside_sp", "z1_Auß_WS_Becken_Anzahl"),
        # CT body composition at L3/L4 (Nowak et al., Eur Radiol 2022)
        ("muscle_area", "totalMuscleExclFat_AreaInMM_0"),
        ("fatty_muscle", "FattyMuscleFraction_0"),
        ("visceral_fat", "visceralFat_AreaInMM_0"),
        ("muscle_hu", "totalMuscleExclFat_meanHU_0"),
        # PSMA PET uptake of the index lesion
        ("pet_avg", "z1_met1_ES-avg"), ("pet_max", "z1_met1_ES-max"),
    ]
    for feature_name, col_name in static_features:
        if col_name in df.columns:
            feats[feature_name] = to_numeric_robust(df[col_name])

    if "age" in df.columns:
        feats["age"] = df["age"]

    for c in ["brs_ecog", "brs_hb", "brs_liver", "brs_crp",
              "bellmunt_score", "bellmunt_enhanced", "bellmunt_modified"]:
        if c in df.columns:
            feats[c] = df[c]

    # Longitudinal laboratory trajectories (baseline / mean / last / slope),
    # restricted to cycles 1..LANDMARK_CYCLE irrespective of how many further
    # cycles a patient received (landmark design; see module docstring).
    # (feature prefix, column prefix in the export; columns are <prefix><cycle>)
    lab_specs = [
        ("hb", "hb"), ("crp", "crp"), ("alp", "alp"),
        ("ldh", "ldh"), ("psa", "psa"), ("albu", "albu"),
        ("leuko", "leukos"), ("gfr", "gfr"),
    ]
    for feature_prefix, col_prefix in lab_specs:
        cols = [f"{col_prefix}{i}" for i in range(1, LANDMARK_CYCLE + 1) if f"{col_prefix}{i}" in df.columns]
        if not cols:
            continue
        lab_df = df[cols].apply(to_numeric_robust)
        feats[f"{feature_prefix}_baseline"] = lab_df.iloc[:, 0]
        feats[f"{feature_prefix}_mean"] = lab_df.mean(axis=1)
        feats[f"{feature_prefix}_last"] = lab_df.apply(
            lambda r: r.dropna().iloc[-1] if len(r.dropna()) > 0 else np.nan, axis=1
        )

        def _slope(row: pd.Series) -> float:
            """Least-squares slope per cycle; requires >= 2 measurements."""
            values = row.dropna().values
            if len(values) < 2:
                return np.nan
            return float(np.polyfit(np.arange(len(values)), values, 1)[0])

        feats[f"{feature_prefix}_slope"] = lab_df.apply(_slope, axis=1)
        if feature_prefix == "psa":
            feats["psa_nadir"] = lab_df.min(axis=1)

    feature_df = pd.DataFrame(feats, index=df.index)
    print(f"\nFeature matrix: {feature_df.shape[0]} x {feature_df.shape[1]}")
    missing_fraction = feature_df.isnull().mean()
    to_drop = missing_fraction[missing_fraction > 0.7].index.tolist()
    if to_drop:
        print(f"Dropping features with >70% missingness: {to_drop}")
        feature_df = feature_df.drop(columns=to_drop)
    return feature_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Lu-PSMA Explainable-ML Pipeline")
    print("=" * 60)

    df = load_and_prepare_cohort()
    feature_df = build_feature_matrix(df)

    common_idx = df.index.intersection(feature_df.index)
    df = df.loc[common_idx].reset_index(drop=True)
    feature_df = feature_df.loc[common_idx].reset_index(drop=True)

    # --- Benchmark: Bellmunt Risk Score as a direct classifier ---
    compare_brs_baseline(df)

    # --- Kaplan-Meier curves by BRS variant (descriptive) ---
    print("\n--- Kaplan-Meier plots ---")
    for endpoint, time_col, event_col in [("OS", "os_months", "os_event"), ("PFS", "pfs_months", "pfs_event")]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("white")
        for ax, score_col, panel_title in zip(
            axes,
            ["bellmunt_score", "bellmunt_modified", "bellmunt_enhanced"],
            ["Original BRS (0-3)", "Modified BRS", "Enhanced BRS (+CRP)"],
        ):
            plot_km_panel(df, time_col, event_col, score_col, panel_title, ax)
        plt.suptitle(f"Kaplan-Meier — {endpoint} by Bellmunt Risk Score", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"KM_{endpoint}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")

    # --- Cross-validated classification + Shapley attribution ---
    # BRS components and composites are excluded from the model inputs,
    # because the BRS is the comparator (avoids circularity).
    brs_derived_cols = [c for c in [
        "bellmunt_score", "bellmunt_enhanced", "bellmunt_modified",
        "brs_ecog", "brs_hb", "brs_liver", "brs_crp",
    ] if c in feature_df.columns]
    ml_feature_df = feature_df.drop(columns=brs_derived_cols)
    feature_names = ml_feature_df.columns.tolist()
    X_all = ml_feature_df.values.astype(float)

    all_rows = []
    shap_results = {}
    endpoints = [("1-Year OS", "os_1yr"), ("2-Year OS", "os_2yr"), ("1-Year PFS", "pfs_1yr")]

    print("\n--- Cross-Validation ---")
    for endpoint_name, endpoint_col in endpoints:
        if endpoint_col not in df.columns:
            continue
        y_raw = df[endpoint_col].values.astype(float)  # float: NaN marks patients excluded by censoring
        valid = (~np.isnan(X_all).all(axis=1)) & (~np.isnan(y_raw))
        X_valid, y_valid = X_all[valid], y_raw[valid].astype(int)
        if y_valid.sum() < 8:
            print(f"  Skipping {endpoint_name}: too few events ({y_valid.sum()})")
            continue

        print(f"\nEndpoint: {endpoint_name} | N={len(y_valid)} | events={y_valid.sum()}")
        all_rows.extend(run_cross_validation(X_valid, y_valid, endpoint_name))

        # Global attribution: Random Forest refitted on all evaluable patients
        X_imputed, _ = impute_train_mean(X_valid, X_valid)
        rf_final = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=RANDOM_SEED)
        rf_final.fit(X_imputed, y_valid)

        n_shap_patients = min(40, len(X_imputed))
        print(f"  Computing Shapley values ({n_shap_patients} patients)...")
        shap_matrix = shapley_matrix(rf_final, X_imputed, feature_names, n_patients=n_shap_patients)
        shap_results[endpoint_name] = (shap_matrix, feature_names)
        plot_shap_importance(shap_matrix, feature_names, endpoint_name, "Random Forest")

    if all_rows:
        out_path = os.path.join(OUTPUT_DIR, "performance_summary.csv")
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    # --- Combined Shapley figure across endpoints ---
    if shap_results:
        n_panels = len(shap_results)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8))
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        for ax, (endpoint_name, (matrix, names)) in zip(axes, shap_results.items()):
            mean_abs = np.abs(matrix).mean(axis=0)
            total = mean_abs.sum()
            pct = mean_abs / total * 100 if total > 0 else mean_abs
            top_idx = np.argsort(pct)[-15:]
            top_names = [names[i] for i in top_idx]
            top_values = pct[top_idx]
            colors = ["#C0392B" if n in BRS_RELATED_FEATURES else "#2C5F8A" for n in top_names]
            ax.set_facecolor("white")
            y_pos = np.arange(len(top_names))
            bars = ax.barh(y_pos, top_values, color=colors, height=0.65)
            for bar, v in zip(bars, top_values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%", va="center", ha="left", fontsize=7.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names, fontsize=8.5)
            ax.set_xlabel("Mean |Shapley| (%)", fontsize=9)
            ax.set_title(f"{endpoint_name}\n(Random Forest)", fontsize=10, fontweight="bold")
            ax.set_xlim(0, max(top_values) * 1.3)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.tick_params(left=False)
            ax.grid(axis="x", ls="--", alpha=0.4)
        legend_handles = [
            mpatches.Patch(color="#C0392B", label="Bellmunt/ECOG/Hb-related"),
            mpatches.Patch(color="#2C5F8A", label="Other features"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
        plt.suptitle("Shapley Feature Importance — Lu-PSMA Cohort", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = os.path.join(OUTPUT_DIR, "SHAP_combined.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")

    print("\n" + "=" * 60 + f"\nDone. Results in: {OUTPUT_DIR}\n" + "=" * 60)


if __name__ == "__main__":
    main()
