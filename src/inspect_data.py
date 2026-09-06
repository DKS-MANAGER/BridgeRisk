"""
Inspect raw NBI data files for Maine, Hawaii, and Delaware (2021–2025).

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACT_DIR = BASE_DIR / "data" / "raw" / "extracted"
REPORTS_DIR = BASE_DIR / "reports"

STATES = ["ME", "HI", "DE"]
YEARS = [2021, 2022, 2023, 2024, 2025]

CONDITION_COLS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
]
ID_COLS = ["STATE_CODE_001", "STRUCTURE_NUMBER_008"]


def load_raw_file(file_path):
    """Load delimited or fixed-width NBI text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    if "," in first_line:
        return pd.read_csv(file_path, dtype=str, low_memory=False)
    elif "\t" in first_line:
        return pd.read_csv(file_path, sep="\t", dtype=str, low_memory=False)
    else:
        # Fallback for whitespace/fixed-width
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        header = lines[0].strip().split()
        data = [line.strip().split() for line in lines[1:]]
        return pd.DataFrame(data, columns=header)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print("=" * 70)
    print("NBI MULTI-STATE DATA INSPECTION (2021–2025)")
    print("=" * 70)

    found_any = False
    for state in STATES:
        for year in YEARS:
            yy = str(year)[-2:]
            possible_names = [
                f"{state}{yy}.txt",
                f"{state.lower()}{yy}.txt",
                f"{state}{yy}.TXT",
            ]
            file_path = None
            for fname in possible_names:
                p = EXTRACT_DIR / str(year) / fname
                if p.exists():
                    file_path = p
                    break

            if file_path is None:
                continue

            found_any = True
            df = load_raw_file(file_path)
            bridge_id_count = 0
            if all(c in df.columns for c in ID_COLS):
                bridge_ids = df[ID_COLS[0]].astype(str).str.strip() + "_" + df[ID_COLS[1]].astype(str).str.strip()
                bridge_id_count = bridge_ids.nunique()

            poor_count = 0
            if "DECK_COND_058" in df.columns:
                ratings = pd.to_numeric(df["DECK_COND_058"].str.strip(), errors="coerce")
                poor_count = (ratings <= 4).sum()

            summary_rows.append({
                "state": state,
                "year": year,
                "records": len(df),
                "unique_bridges": bridge_id_count,
                "poor_deck_count": poor_count,
                "columns": len(df.columns),
            })
            print(f"[{state} {year}] Records: {len(df):,d} | Unique Bridges: {bridge_id_count:,d} | Decks <= 4 (Poor): {poor_count:,d}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_csv = REPORTS_DIR / "yearly_counts.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"\nSaved inventory summary to: {out_csv}")
    elif not found_any:
        print("\nNote: Raw extracted files not found in data/raw/extracted/. Using preprocessed data.")

    # Also inspect processed datasets if available
    train_p = BASE_DIR / "data" / "processed" / "train_2021_2024.parquet"
    test_p = BASE_DIR / "data" / "processed" / "test_2024_2025.parquet"

    if train_p.exists() and test_p.exists():
        train_df = pd.read_parquet(train_p)
        test_df = pd.read_parquet(test_p)
        print("\n" + "=" * 70)
        print("PREPROCESSED DATASETS FOR MODELING")
        print("=" * 70)
        print(f"Training Set (2021–2024 transitions): {len(train_df):,d} records across {train_df['state'].nunique() if 'state' in train_df.columns else 1} states")
        if "target_deck_poor_next_year" in train_df.columns:
            pos_train = train_df["target_deck_poor_next_year"].sum()
            print(f"  Poor deck next year: {pos_train:,d} ({pos_train / len(train_df) * 100:.2f}%)")
        print(f"Testing Set (2024–2025 out-of-time): {len(test_df):,d} bridges across {test_df['state'].nunique() if 'state' in test_df.columns else 1} states")
        if "target_deck_poor_next_year" in test_df.columns:
            pos_test = test_df["target_deck_poor_next_year"].sum()
            print(f"  Poor deck next year: {pos_test:,d} ({pos_test / len(test_df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
