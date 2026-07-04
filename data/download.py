"""Download and verify the WANDS dataset (MIT licensed, Wayfair).

Fetches product.csv / query.csv / label.csv from the official GitHub repo
into ./data and verifies row counts against the published figures, so any
upstream change to the dataset is caught loudly instead of silently
shifting benchmark numbers.

Usage:
    python -m data.download            # downloads into ./data
    python -m data.download --dir /tmp/wands
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"

# (filename, expected_data_rows) — counts exclude the header row.
EXPECTED = {
    "product.csv": 42_994,
    "query.csv": 480,
    "label.csv": 233_448,
}


def download(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED:
        dest = data_dir / filename
        if dest.exists() and not force:
            print(f"[skip] {dest} already exists (use --force to re-download)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — fixed https URL
        print(f"[ok  ] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def verify(data_dir: Path) -> bool:
    """Check row counts and basic schema. Returns True if all files pass."""
    ok = True
    for filename, expected_rows in EXPECTED.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[FAIL] {filename}: missing")
            ok = False
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = sum(1 for _ in reader)
        if len(header) < 2:
            print(f"[FAIL] {filename}: not tab-separated? header={header[:1]}")
            ok = False
        elif rows != expected_rows:
            print(f"[FAIL] {filename}: {rows:,} rows, expected {expected_rows:,}")
            ok = False
        else:
            print(f"[ok  ] {filename}: {rows:,} rows, {len(header)} columns")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data", type=Path, help="target directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    download(args.dir, force=args.force)
    if not verify(args.dir):
        sys.exit(1)
    print("\nWANDS dataset ready. Note: label.csv contains 1,467 duplicate "
          "(query, product) pairs (14 with conflicting labels); the loader in "
          "evals/golden_set.py deduplicates last-write-wins -> 231,873 unique judgments.")


if __name__ == "__main__":
    main()
