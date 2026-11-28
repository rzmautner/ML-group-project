"""Train the tabular deep-learning models in a clean process.

PyTorch can deadlock on macOS when it runs in the same process as LightGBM / XGBoost / CatBoost,
because each ships its own OpenMP runtime. The model-selection notebook therefore runs this
module as a subprocess and reads the results it writes:

    python -m hmda.tabular_dl --rows 200000

Uses the same train/test split and the same row subsample (seed 0) as the notebook, and writes
    reports/tabular_dl_results.csv          metrics per model
    reports/tabular_dl_test_probs.parquet   predicted probabilities on the test set
"""

from __future__ import annotations

import argparse
import warnings

# torch must be imported before lightgbm / xgboost / catboost (pulled in by hmda.models):
# loading their OpenMP runtimes first makes torch segfault or hang on macOS.
import torch  # noqa: F401  isort:skip

import numpy as np
import pandas as pd

from hmda.data import REPO_ROOT
from hmda.models import benchmark, default_torch_device, load_split, tabular_dl_zoo

REPORTS = REPO_ROOT / "reports"


def subsample_index(n_total: int, n_rows: int, seed: int = 0) -> np.ndarray:
    """Row positions of the training subsample, shared with the notebook."""
    return np.random.default_rng(seed).choice(n_total, n_rows, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--train-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    X_train, y_train, X_test, y_test, _ = load_split(args.train_frac, args.test_frac)
    idx = subsample_index(len(X_train), args.rows)
    X_small, y_small = X_train.iloc[idx], y_train[idx]
    device = args.device or default_torch_device()
    print(f"training on {len(X_small):,} rows, testing on {len(X_test):,}, device={device}", flush=True)

    results, fitted = benchmark(tabular_dl_zoo(device=device), X_small, y_small, X_test, y_test)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "tabular_dl_results.csv")
    probs = pd.DataFrame({name: pipe.predict_proba(X_test)[:, 1] for name, pipe in fitted.items()})
    probs.to_parquet(REPORTS / "tabular_dl_test_probs.parquet")
    print("wrote", REPORTS / "tabular_dl_results.csv", flush=True)


if __name__ == "__main__":
    main()
