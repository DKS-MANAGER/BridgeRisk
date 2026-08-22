"""
Inspect raw NBI data files and generate inspection reports.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, "data", "raw", "extracted")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

YEARS = [2023, 2024, 2025]
FILE_MAP = {
    2023: "ME23.txt",
    2024: "ME24.txt",
    2025: "ME25.txt",
}

CONDITION_COLS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
]
ID_COLS = ["STATE_CODE_001", "STRUCTURE_NUMBER_008"]


def detect_delimiter(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if "," in first_line:
        return ","
    if "\t" in first_line:
        return "\t"
    return None


def load_year(year):
    filename = FILE_MAP[year]
    path = os.path.join(EXTRACT_DIR, str(year), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    delimiter = detect_delimiter(path)
    if delimiter == ",":
        df = pd.read_csv(path, dtype=str, low_memory=False)
    elif delimiter == "\t":
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        header = lines[0].strip().split()
        data = [line.strip().split() for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)

    return df, delimiter


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("NBI DATA INSPECTION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    yearly_counts = []
    matching_counts = []

    dfs = {}
    for year in YEARS:
        df, delimiter = load_year(year)
        dfs[year] = df

        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"YEAR: {year}")
        report_lines.append(f"{'='*70}")
        report_lines.append(f"File: {FILE_MAP[year]}")
        report_lines.append(f"Delimiter: {delimiter if delimiter else 'Fixed-width'}")
        report_lines.append(f"Rows: {len(df)}")
        report_lines.append(f"Columns: {len(df.columns)}")

        report_lines.append("\nFirst 5 rows (transposed for readability):")
        for col in df.columns[:10]:
            report_lines.append(f"  {col}: {list(df[col].head())}")

        report_lines.append("\nColumn names:")
        for col in df.columns:
            report_lines.append(f"  {col}")

        report_lines.append("\nMissing values (top 20):")
        missing = df.isnull().sum().sort_values(ascending=False).head(20)
        for col, count in missing.items():
            pct = count / len(df) * 100
            report_lines.append(f"  {col}: {count} ({pct:.1f}%)")

        report_lines.append("\nData types (first 20):")
        for col in list(df.dtypes.index)[:20]:
            report_lines.append(f"  {col}: {df[col].dtype}")

        report_lines.append("\nCondition rating fields - unique values:")
        for col in CONDITION_COLS:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                report_lines.append(
                    f"  {col}: {sorted(unique_vals)[:20]} (total: {len(unique_vals)})"
                )
            else:
                report_lines.append(f"  {col}: NOT FOUND")

        report_lines.append("\nDuplicate records check:")
        if all(c in df.columns for c in ID_COLS):
            df["bridge_id"] = (
                df[ID_COLS[0]].astype(str).str.strip()
                + "_"
                + df[ID_COLS[1]].astype(str).str.strip()
            )
            dup_count = df["bridge_id"].duplicated().sum()
            report_lines.append("  Duplicate bridge IDs: " + str(dup_count))
        else:
            report_lines.append("  ID columns not found, cannot check duplicates")

        yearly_counts.append({"year": year, "records": len(df)})

    report_lines.append(f"\n{'='*70}")
    report_lines.append("CROSS-YEAR MATCHING")
    report_lines.append(f"{'='*70}")

    if (
        all(c in dfs[2023].columns for c in ID_COLS)
        and all(c in dfs[2024].columns for c in ID_COLS)
        and all(c in dfs[2025].columns for c in ID_COLS)
    ):
        dfs[2023]["bridge_id"] = (
            dfs[2023][ID_COLS[0]].astype(str).str.strip()
            + "_"
            + dfs[2023][ID_COLS[1]].astype(str).str.strip()
        )
        dfs[2024]["bridge_id"] = (
            dfs[2024][ID_COLS[0]].astype(str).str.strip()
            + "_"
            + dfs[2024][ID_COLS[1]].astype(str).str.strip()
        )
        dfs[2025]["bridge_id"] = (
            dfs[2025][ID_COLS[0]].astype(str).str.strip()
            + "_"
            + dfs[2025][ID_COLS[1]].astype(str).str.strip()
        )

        ids_2023 = set(dfs[2023]["bridge_id"])
        ids_2024 = set(dfs[2024]["bridge_id"])
        ids_2025 = set(dfs[2025]["bridge_id"])

        match_23_24 = len(ids_2023 & ids_2024)
        match_23_25 = len(ids_2023 & ids_2025)
        match_24_25 = len(ids_2024 & ids_2025)
        match_all = len(ids_2023 & ids_2024 & ids_2025)

        report_lines.append(f"Bridges in 2023: {len(ids_2023)}")
        report_lines.append(f"Bridges in 2024: {len(ids_2024)}")
        report_lines.append(f"Bridges in 2025: {len(ids_2025)}")
        report_lines.append(f"Bridges in 2023 AND 2024: {match_23_24}")
        report_lines.append(f"Bridges in 2023 AND 2025: {match_23_25}")
        report_lines.append(f"Bridges in 2024 AND 2025: {match_24_25}")
        report_lines.append(f"Bridges in ALL THREE YEARS: {match_all}")

        matching_counts.append({"comparison": "2023-2024", "matched": match_23_24})
        matching_counts.append({"comparison": "2023-2025", "matched": match_23_25})
        matching_counts.append({"comparison": "2024-2025", "matched": match_24_25})
        matching_counts.append({"comparison": "all_three", "matched": match_all})
    else:
        report_lines.append(
            "Cannot perform matching - ID columns not found in all years."
        )

    report_text = "\n".join(report_lines)
    with open(
        os.path.join(REPORTS_DIR, "data_inspection.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)
    print(report_text)

    yearly_df = pd.DataFrame(yearly_counts)
    yearly_df.to_csv(os.path.join(REPORTS_DIR, "yearly_counts.csv"), index=False)

    if matching_counts:
        matching_df = pd.DataFrame(matching_counts)
        matching_df.to_csv(
            os.path.join(REPORTS_DIR, "matching_counts.csv"), index=False
        )

    print("\nInspection complete. Results saved to reports/")


if __name__ == "__main__":
    main()
