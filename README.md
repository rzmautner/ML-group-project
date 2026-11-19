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

We use the public [HMDA](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset) loan-level (LAR) snapshot files for 2024 and 2025. They're about 5 GB each, so they're **not** in git. Everyone downloads them from FFIEC:

```bash
python -m hmda.data
```

This:

1. Downloads `2024_public_lar_csv.csv` and `2025_public_lar_csv.csv` into `data/raw/`. If you already have them anywhere under `data/`, it skips this. You need about 11 GB free while it unzips.
2. Filters to **VA and NC**, action codes **1–5**, and writes `data/processed/va_nc_2024_2025.parquet` (all columns, stored as strings).

It takes a few minutes the first time. Re-run with `--rebuild` if the filter settings in `src/hmda/data.py` change. Once the Parquet file exists you can delete `data/raw/` to free space.

In notebooks:

```python
from hmda.data import load_hmda, approved_vs_denied

df = load_hmda()               # numeric columns coerced, `outcome` label added
main = approved_vs_denied(df)  # actions 1–3 only, 0/1 `denied` target
```

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
data/processed/   Filtered Parquet subset (gitignored)
notebooks/        Jupyter notebooks (EDA.ipynb, scratch.ipynb)
src/hmda/data.py  Download, filter, and load the data
pyproject.toml    Makes src/hmda installable with `pip install -e .`
requirements.txt  Pinned Python dependencies
```
