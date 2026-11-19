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
```

### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name ml-final-project --display-name "Python (ml-final-project)"
```

Then open a notebook (`jupyter lab` or VS Code) and choose the **Python (ml-final-project)** kernel.

## Adding packages

If you install a new package, update the requirements file and commit it:

```bash
pip install <package>
pip freeze > requirements.txt
```

## Project layout

```
notebooks/        Jupyter notebooks (scratch.ipynb for quick experiments)
requirements.txt  Pinned Python dependencies
```
