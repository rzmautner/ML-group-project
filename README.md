# ML Group Project

## Setup

### 1. Install Python 3.12

This project uses **Python 3.12**. Check whether you already have it:

```bash
python3.12 --version
```

If that prints `Python 3.12.x`, skip to step 2. Otherwise, install it:

**macOS**

- Recommended: download the macOS installer for the latest 3.12.x release from [python.org/downloads](https://www.python.org/downloads/) and run it.
- Or with Homebrew: `brew install python@3.12`

**Windows**

- Download the Windows installer for the latest 3.12.x release from [python.org/downloads](https://www.python.org/downloads/).
- When you run the installer, check **"Add python.exe to PATH"**.
- Afterwards, `py -3.12 --version` should work.

**Linux (Debian/Ubuntu)**

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

If your distro doesn't package 3.12, use the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) (Ubuntu) or [pyenv](https://github.com/pyenv/pyenv).

### 2. Clone the repo

```bash
git clone https://github.com/rzmautner/ML-group-project.git
cd ML-group-project
```

### 3. Create and activate a virtual environment

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

The `.venv/` folder is in `.gitignore`. Everyone creates their own; don't commit it.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs this repo's `src/hmda` package in editable mode, so notebooks can `from hmda.data import load_hmda` from any folder, and edits to the code take effect without reinstalling.

### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name ml-final-project --display-name "Python (ml-final-project)"
```

Then open a notebook (`jupyter lab` or VS Code) and choose the **Python (ml-final-project)** kernel.

### 6. Get the data

We use the public [HMDA](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset) loan-level (LAR) snapshot files for 2024 and 2025, nationwide. They're about 5 GB each, so they're **not** in git. Everyone downloads them from FFIEC:

```bash
python -m hmda.data
```

This:

1. Downloads `2024_public_lar_csv.csv` and `2025_public_lar_csv.csv` into `data/raw/`. If you already have them anywhere under `data/`, it skips this. You need about 11 GB free while it unzips.
2. Builds `data/processed/hmda_2024_2025.parquet`: all states, action codes 1–5 (~22.5M rows, ~940 MB). Every column is stored as a string so HMDA codes like `Exempt` survive. A `split_key` row hash is added for reproducible sampling.

The build takes about 30 seconds with DuckDB. Re-run with `--rebuild` if the settings in `src/hmda/data.py` change. Once the Parquet file exists you can delete `data/raw/` to free space.

In notebooks:

```python
from hmda.data import connect, load_hmda, approved_vs_denied

df = load_hmda(states=["VA", "NC"])              # one or more states, all rows
df = load_hmda(sample=0.05)                      # reproducible 5% nationwide sample
main = approved_vs_denied(df)                    # actions 1–3, 0/1 `denied` target

con = connect()                                  # DuckDB, view `hmda` = full dataset
con.sql("SELECT state_code, avg((action_taken = '3')::INT) FROM hmda GROUP BY 1").df()
```

Nationwide with all columns is ~22M rows x 100 columns, too big for pandas on most laptops. Use `sample=`, `columns=`, or DuckDB aggregates.

## Notebooks

Run them in order. Each one is saved with its outputs, so you can read the results without re-running.

| Notebook | What it covers | Runtime (M-series Mac) |
|---|---|---|
| `notebooks/01_eda.ipynb` | Nationwide EDA: geography, loan/borrower/credit features, lenders, leakage, protected attributes, feature importance (permutation + SHAP) | ~4 min |
| `notebooks/02_feature_engineering.ipynb` | Column groups, engineered-feature checks, univariate signal (raw vs engineered), encoders, feature-group ablations | ~5 min |
| `notebooks/03_model_selection.ipynb` | Benchmark of 7 models (train 2024 → test 2025), tabular deep learning (RealMLP, TabM), curves and calibration, threshold, training-size curve, fairness audit | ~45 min |

Result tables are written to `reports/`.

## Code

* `src/hmda/data.py`: download, build, and load the data
* `src/hmda/features.py`: `HMDAFeatures` (parsing and feature engineering), `make_preprocessor` / `make_pipeline` (sklearn pipelines), and the column lists for leakage and protected attributes
* `src/hmda/models.py`: model zoo (incl. CatBoost), `tabular_dl_zoo` (RealMLP / TabM via `pytabkit`), `load_split` (train 2024 / test 2025), `benchmark`, `group_metrics` (fairness slices)
* `src/hmda/tabular_dl.py`: trains RealMLP / TabM in a separate process (`python -m hmda.tabular_dl --rows 200000`). On macOS, PyTorch crashes or hangs if LightGBM/XGBoost/CatBoost load first in the same process.
* `src/hmda/plotting.py`: shared chart style

```python
from lightgbm import LGBMClassifier
from hmda.features import make_pipeline
from hmda.models import load_split

X_train, y_train, X_test, y_test, audit = load_split(train_frac=0.10, test_frac=0.05)
pipe = make_pipeline(LGBMClassifier(), kind="tree").fit(X_train, y_train)
```

Run the unit tests (no data needed) with `pytest`.

## Adding packages

If you install a new package, update the requirements file and commit it:

```bash
pip install <package>
pip freeze --exclude-editable > requirements.txt
```

(`--exclude-editable` keeps the local `hmda` package, which has a path specific to your machine, out of the file.)

## Project layout

```
data/raw/         Raw FFIEC CSVs (gitignored)
data/processed/   hmda_2024_2025.parquet (gitignored)
notebooks/        01_eda, 02_feature_engineering, 03_model_selection
reports/          Result tables (CSV) written by the notebooks
src/hmda/         data.py, features.py, models.py, plotting.py
tests/            Unit tests for the feature pipeline
pyproject.toml    Makes src/hmda installable with `pip install -e .`
requirements.txt  Pinned Python dependencies
```
