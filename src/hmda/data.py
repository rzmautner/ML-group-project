"""HMDA loan-level data: download, build, and load.

Data layout (everything under data/ is gitignored):

    data/raw/{year}_public_lar_csv.csv            raw FFIEC national files (~5 GB each)
    data/processed/hmda_2024_2025.parquet         all states, action codes 1-5 (~22M rows)

Run once from the repo root to download and build:

    python -m hmda.data

The Parquet file keeps every column as a string so HMDA codes like "Exempt" survive
("" and "NA" become null). It also adds `split_key`, a hash of the whole row, used for
reproducible sampling and train/test splits.

Loading into pandas:

    from hmda.data import load_hmda, approved_vs_denied
    df = load_hmda(states=["VA", "NC"])            # one or more states, all rows
    df = load_hmda(sample=0.05)                    # 5% nationwide sample
    main = approved_vs_denied(df)                  # actions 1-3 with a 0/1 `denied` target

Nationwide aggregates are much faster in DuckDB than in pandas:

    from hmda.data import connect
    con = connect()                                # has a view called `hmda`
    con.sql("SELECT state_code, avg(action_taken = '3') FROM hmda GROUP BY 1").df()
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_PATH = PROCESSED_DIR / "hmda_2024_2025.parquet"

YEARS = (2024, 2025)
# 1 originated, 2 approved not accepted, 3 denied, 4 withdrawn, 5 incomplete.
# Drops 6 (purchased loan), 7/8 (preapproval requests).
KEEP_ACTIONS = (1, 2, 3, 4, 5)

FFIEC_URL = "https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/{year}_public_lar_csv.zip"

# Read as strings, then coerced to numbers by clean() ("Exempt" etc. become NaN).
NUMERIC_COLS = [
    "loan_amount", "income", "property_value", "combined_loan_to_value_ratio",
    "loan_term", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    "tract_population", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units", "action_taken", "activity_year",
]

ACTION_LABELS = {1: "approved", 2: "approved", 3: "denied", 4: "withdrawn", 5: "incomplete"}

SPLIT_MOD = 1_000_000  # sampling resolution: split_key % SPLIT_MOD


# --------------------------------------------------------------------------- raw files

def raw_csv_name(year: int) -> str:
    return f"{year}_public_lar_csv.csv"


def find_raw_csv(year: int) -> Path | None:
    """Find a raw file anywhere under data/ (handles data/, data/raw/, data/2025/, ...)."""
    canonical = RAW_DIR / raw_csv_name(year)
    if canonical.exists():
        return canonical
    matches = sorted(DATA_DIR.rglob(raw_csv_name(year))) if DATA_DIR.exists() else []
    return matches[0] if matches else None


def download_raw(year: int) -> Path:
    """Download and unzip one year's national LAR file from FFIEC into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / raw_csv_name(year)
    zip_path = RAW_DIR / f"{year}_public_lar_csv.zip.part"
    url = FFIEC_URL.format(year=year)

    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(8 * 1024 * 1024):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()

    print(f"unzipping to {dest.relative_to(REPO_ROOT)}")
    part = dest.with_suffix(".csv.part")
    with zipfile.ZipFile(zip_path) as zf:
        member = next(m for m in zf.namelist() if m.endswith(".csv"))
        with zf.open(member) as src, open(part, "wb") as out:
            shutil.copyfileobj(src, out, 16 * 1024 * 1024)
    part.rename(dest)
    zip_path.unlink()
    return dest


def get_raw_csv(year: int, download: bool = True) -> Path:
    path = find_raw_csv(year)
    if path is not None:
        return path
    if not download:
        raise FileNotFoundError(
            f"{raw_csv_name(year)} not found under {DATA_DIR}. Run `python -m hmda.data` to download it."
        )
    return download_raw(year)


# --------------------------------------------------------------------------- build

def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def build_dataset(years=YEARS, actions=KEEP_ACTIONS, download: bool = True) -> Path:
    """Combine the raw CSVs (all states, chosen actions) into one Parquet file."""
    csvs = [str(get_raw_csv(y, download=download)) for y in years]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATASET_PATH.with_suffix(".parquet.part")

    con = duckdb.connect()
    src = f"read_csv([{_sql_list(csvs)}], header=true, all_varchar=true, nullstr=['', 'NA'])"
    columns = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    row_hash = "hash(" + ", ".join(f'"{c}"' for c in columns) + ")"

    print(f"building {DATASET_PATH.relative_to(REPO_ROOT)} from {len(csvs)} file(s) ...")
    con.sql(f"""
        COPY (
            SELECT *, {row_hash} AS split_key
            FROM {src}
            WHERE action_taken IN ({_sql_list(actions)})
        ) TO '{tmp_path}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)
    """)
    tmp_path.rename(DATASET_PATH)

    counts = con.sql(f"""
        SELECT activity_year, count(*) AS n FROM '{DATASET_PATH}' GROUP BY 1 ORDER BY 1
    """).fetchall()
    for year, n in counts:
        print(f"  {year}: {n:,} rows")
    return DATASET_PATH


def ensure_dataset() -> Path:
    if not DATASET_PATH.exists():
        build_dataset()
    return DATASET_PATH


# --------------------------------------------------------------------------- load

def connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a view `hmda` over the full dataset (all columns as strings)."""
    con = duckdb.connect()
    con.sql("SET enable_progress_bar = false")
    if threads:
        con.sql(f"SET threads={threads}")
    con.sql(f"CREATE VIEW hmda AS SELECT * FROM '{ensure_dataset()}'")
    return con


def sample_filter(frac: float, offset: float = 0.0) -> str:
    """SQL condition selecting a reproducible `frac` of rows (by split_key).

    Non-overlapping samples: sample_filter(0.1) and sample_filter(0.1, offset=0.1).
    """
    lo, hi = int(offset * SPLIT_MOD), int((offset + frac) * SPLIT_MOD)
    return f"(split_key % {SPLIT_MOD}) >= {lo} AND (split_key % {SPLIT_MOD}) < {hi}"


def load_hmda(
    states=None,
    years=None,
    actions=None,
    columns=None,
    sample: float | None = None,
    sample_offset: float = 0.0,
    where: str | None = None,
    clean_types: bool = True,
) -> pd.DataFrame:
    """Load rows from the dataset into pandas.

    states/years/actions: filters (None = all). columns: subset of columns (None = all).
    sample: reproducible fraction of rows, e.g. 0.05. Nationwide with all columns is
    ~22M rows x 100 columns, so pass `sample` or `columns` unless you have lots of RAM.
    """
    conds = []
    if states is not None:
        conds.append(f"state_code IN ({_sql_list(states)})")
    if years is not None:
        conds.append(f"activity_year IN ({_sql_list(years)})")
    if actions is not None:
        conds.append(f"action_taken IN ({_sql_list(actions)})")
    if sample is not None:
        conds.append(sample_filter(sample, sample_offset))
    if where:
        conds.append(f"({where})")

    if columns is None:
        select = "* EXCLUDE (split_key)"
    else:
        needed = list(dict.fromkeys([*columns, "action_taken", "activity_year"]))
        select = ", ".join(f'"{c}"' for c in needed)
    sql = f"SELECT {select} FROM hmda" + (f" WHERE {' AND '.join(conds)}" if conds else "")

    con = connect()
    df = con.sql(sql).to_arrow_table().to_pandas()
    con.close()
    return clean(df) if clean_types else df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and add an `outcome` label."""
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outcome"] = df["action_taken"].map(ACTION_LABELS)
    return df


def approved_vs_denied(df: pd.DataFrame) -> pd.DataFrame:
    """Main modeling task: approved (actions 1, 2) vs denied (3), with a 0/1 `denied` target."""
    main = df[df["action_taken"].isin([1, 2, 3])].copy()
    main["denied"] = (main["action_taken"] == 3).astype(int)
    return main


# --------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Download HMDA data and build the dataset.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if it exists")
    args = parser.parse_args()

    for year in YEARS:
        print(f"{year}: {get_raw_csv(year).relative_to(REPO_ROOT)}")
    if args.rebuild or not DATASET_PATH.exists():
        build_dataset()
    else:
        print(f"{DATASET_PATH.relative_to(REPO_ROOT)} already exists (use --rebuild to redo it)")


if __name__ == "__main__":
    main()
