"""Model zoo, train/test data, and benchmarking for the denial-prediction task.

Default evaluation is temporal: train on a 2024 sample, test on a 2025 sample. This is
stricter than a random split and closer to how a model would actually be used.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from hmda.data import approved_vs_denied, load_hmda
from hmda.features import HMDAFeatures, make_pipeline, split_xy

# Extra columns kept alongside the features for fairness slicing (not model inputs).
AUDIT_COLS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "state_code"]


def load_split(train_frac=0.10, test_frac=0.05, train_year=2024, test_year=2025,
               include_protected=False):
    """Approved-vs-denied train/test frames: (X_train, y_train, X_test, y_test, audit_test)."""
    cols = list(dict.fromkeys(HMDAFeatures(include_protected=True).input_cols() + AUDIT_COLS))

    def load(year, frac):
        df = approved_vs_denied(load_hmda(years=[year], actions=[1, 2, 3], columns=cols, sample=frac))
        X, y = split_xy(df, include_protected)
        return X, y, df[AUDIT_COLS]

    X_tr, y_tr, _ = load(train_year, train_frac)
    X_te, y_te, audit = load(test_year, test_frac)
    return X_tr, y_tr, X_te, y_te, audit


def model_zoo(n_jobs: int = -1, seed: int = 0) -> dict:
    """name -> (estimator, preprocessing kind). Reasonable defaults, not tuned."""
    return {
        "Baseline (always approve)": (DummyClassifier(strategy="prior"), "tree"),
        "Logistic regression": (LogisticRegression(max_iter=1000, C=1.0), "linear"),
        "Random forest": (RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, max_features="sqrt", max_samples=0.3,
            n_jobs=n_jobs, random_state=seed), "tree"),
        "HistGradientBoosting": (HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=seed), "tree"),
        "LightGBM": (LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=n_jobs,
            random_state=seed, verbose=-1), "tree"),
        "XGBoost": (XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=8, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs,
            random_state=seed, eval_metric="logloss"), "tree"),
        "CatBoost": (CatBoostClassifier(
            iterations=800, learning_rate=0.08, depth=8, l2_leaf_reg=3, thread_count=n_jobs,
            random_seed=seed, verbose=0, allow_writing_files=False), "tree"),
    }


def default_torch_device() -> str:
    """'mps' on Apple Silicon, 'cuda' on NVIDIA, else 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tabular_dl_zoo(device: str | None = None, seed: int = 0, tabm_epochs: int = 32) -> dict:
    """Deep-learning models designed for tabular data (via pytabkit), on the linear preprocessing.

    * RealMLP: an MLP with tabular-specific tricks (robust scaling, numeric embeddings, tuned
      defaults); Holzmüller et al., "Better by Default", NeurIPS 2024. Library defaults.
    * TabM: a parameter-efficient ensemble of MLPs; Gorishniy et al., ICLR 2025. Epochs capped
      at `tabm_epochs` to bound training time.
    Both are much slower than GBMs, so the notebook trains them on a smaller sample.
    On an M-series Mac, device="mps" is ~4x faster than CPU.
    """
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier  # imports torch; keep lazy
    device = device or default_torch_device()
    return {
        "RealMLP (tabular DL)": (RealMLP_TD_Classifier(device=device, random_state=seed, verbosity=0), "linear"),
        "TabM (tabular DL)": (TabM_D_Classifier(device=device, random_state=seed, n_epochs=tabm_epochs,
                                                verbosity=0), "linear"),
    }


def score(y_true, p, threshold: float = 0.5) -> dict:
    """Probability metrics plus thresholded classification metrics."""
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "log_loss": log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, p),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def benchmark(models: dict, X_train, y_train, X_test, y_test, include_protected=False, drop=(),
              verbose=True):
    """Fit each pipeline, score on the test set. Returns (results DataFrame, fitted pipelines)."""
    rows, fitted = [], {}
    for name, (est, kind) in models.items():
        pipe = make_pipeline(est, kind, include_protected, drop)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = pipe.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t0
        rows.append({"model": name, **score(y_test, p), "fit_s": fit_s, "predict_s": pred_s})
        fitted[name] = pipe
        if verbose:
            r = rows[-1]
            print(f"{name:28s} AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  "
                  f"F1 {r['f1']:.4f}  fit {fit_s:6.1f}s")
    return pd.DataFrame(rows).set_index("model"), fitted


def group_metrics(y_true, p, groups: pd.Series, threshold: float = 0.5, min_n: int = 500) -> pd.DataFrame:
    """Per-group actual vs predicted denial rates, AUC, false positive/negative rates."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p), "g": np.asarray(groups)})
    df["pred"] = (df["p"] >= threshold).astype(int)
    rows = []
    for g, d in df.groupby("g"):
        if len(d) < min_n:
            continue
        neg, pos = d[d.y == 0], d[d.y == 1]
        rows.append({
            "group": g, "n": len(d),
            "actual_denial_rate": d.y.mean(),
            "mean_predicted_prob": d.p.mean(),
            "flagged_rate": d.pred.mean(),
            "roc_auc": roc_auc_score(d.y, d.p) if d.y.nunique() == 2 else np.nan,
            "false_positive_rate": neg.pred.mean() if len(neg) else np.nan,
            "false_negative_rate": 1 - pos.pred.mean() if len(pos) else np.nan,
        })
    return pd.DataFrame(rows).set_index("group").sort_values("n", ascending=False)
