"""HMDA loan-level data: download, filter, and load.

Data layout (everything under data/ is gitignored):

    data/raw/{year}_public_lar_csv.csv        raw FFIEC national files (~5 GB each)
    data/processed/va_nc_2024_2025.parquet    filtered subset the notebooks use

Run once from the repo root to download and build everything:

    python -m hmda.data

In a notebook:

    from hmda.data import load_hmda, approved_vs_denied
    df = load_hmda()                 # all kept actions, numeric columns cleaned
    main = approved_vs_denied(df)    # approved vs denied, with a 0/1 `denied` column
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

YEARS = (2024, 2025)
STATES = ("VA", "NC")
# 1 originated, 2 approved not accepted, 3 denied, 4 withdrawn, 5 incomplete.
# Drops 6 (purchased loan), 7/8 (preapproval requests).
KEEP_ACTIONS = (1, 2, 3, 4, 5)

FFIEC_URL = "https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/{year}_public_lar_csv.zip"

# Read as strings, then coerced to numbers ("Exempt", "NA", etc. become NaN).
NUMERIC_COLS = [
    "loan_amount", "income", "property_value", "combined_loan_to_value_ratio",
    "loan_term", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    "tract_population", "action_taken", "activity_year",
]

ACTION_LABELS = {1: "approved", 2: "approved", 3: "denied", 4: "withdrawn", 5: "incomplete"}


def raw_csv_name(year: int) -> str:
    return f"{year}_public_lar_csv.csv"


def subset_path(states=STATES, years=YEARS) -> Path:
    name = "_".join([s.lower() for s in states] + [str(y) for y in years])
    return PROCESSED_DIR / f"{name}.parquet"


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
    with zipfile.ZipFile(zip_path) as zf:
        member = next(m for m in zf.namelist() if m.endswith(".csv"))
        with zf.open(member) as src, open(dest.with_suffix(".csv.part"), "wb") as out:
            shutil.copyfileobj(src, out, 16 * 1024 * 1024)
    dest.with_suffix(".csv.part").rename(dest)
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


def build_subset(states=STATES, years=YEARS, actions=KEEP_ACTIONS, download: bool = True) -> Path:
    """Stream the raw CSVs, keep the chosen states/actions, and write one Parquet file.

    All columns are kept as strings so codes like "Exempt" survive for missingness analysis.
    """
    out_path = subset_path(states, years)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.part")

    state_set = pa.array(list(states))
    action_set = pa.array([str(a) for a in actions])
    writer = None
    try:
        for year in years:
            csv_path = get_raw_csv(year, download=download)
            with open(csv_path) as f:
                columns = f.readline().strip().split(",")
            reader = pacsv.open_csv(
                csv_path,
                read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
                # strings_can_be_null: "", "NA", "N/A", ... become null, matching pandas' read_csv
                convert_options=pacsv.ConvertOptions(
                    column_types={c: pa.string() for c in columns}, strings_can_be_null=True
                ),
            )
            kept = 0
            print(f"filtering {csv_path.relative_to(REPO_ROOT)} ...")
            for batch in reader:
                mask = pc.and_(
                    pc.is_in(batch["state_code"], value_set=state_set),
                    pc.is_in(batch["action_taken"], value_set=action_set),
                )
                table = pa.Table.from_batches([batch.filter(pc.fill_null(mask, False))])
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                writer.write_table(table)
                kept += table.num_rows
            print(f"  kept {kept:,} rows")
    finally:
        if writer is not None:
            writer.close()

    tmp_path.rename(out_path)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return out_path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and add an `outcome` label."""
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outcome"] = df["action_taken"].map(ACTION_LABELS)
    return df


def load_hmda(states=STATES, years=YEARS, rebuild: bool = False) -> pd.DataFrame:
    """Load the filtered subset (building it from the raw files if needed), cleaned."""
    path = subset_path(states, years)
    if rebuild or not path.exists():
        build_subset(states, years)
    return clean(pd.read_parquet(path))


def approved_vs_denied(df: pd.DataFrame) -> pd.DataFrame:
    """Main modeling task: approved (actions 1, 2) vs denied (3), with a 0/1 `denied` target."""
    main = df[df["action_taken"].isin([1, 2, 3])].copy()
    main["denied"] = (main["action_taken"] == 3).astype(int)
    return main


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HMDA data and build the filtered subset.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the subset even if it exists")
    args = parser.parse_args()

    for year in YEARS:
        print(f"{year}: {get_raw_csv(year).relative_to(REPO_ROOT)}")
    path = subset_path()
    if args.rebuild or not path.exists():
        build_subset()
    else:
        print(f"{path.relative_to(REPO_ROOT)} already exists (use --rebuild to redo it)")


if __name__ == "__main__":
    main()
