"""
Clean, normalize, and prepare NBI data for modeling.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, "data", "raw", "extracted")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

YEARS = [2023, 2024, 2025]
FILE_MAP = {2023: "ME23.txt", 2024: "ME24.txt", 2025: "ME25.txt"}

CONDITION_COLS = ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062"]
NUMERIC_COLS = [
    "YEAR_BUILT_027", "ADT_029", "PERCENT_ADT_TRUCK_109",
    "MAIN_UNIT_SPANS_045", "MAX_SPAN_LEN_MT_048", "STRUCTURE_LEN_MT_049",
    "ROADWAY_WIDTH_MT_051", "DECK_WIDTH_MT_052", "LEFT_CURB_MT_050A", "RIGHT_CURB_MT_050B",
    "HORR_CLR_MT_047", "VERT_CLR_OVER_MT_053", "VERT_CLR_UND_054B", "LAT_UND_MT_055B",
    "LEFT_LAT_UND_MT_056", "OPERATING_RATING_064", "INVENTORY_RATING_066",
    "BRIDGE_IMP_COST_094", "ROADWAY_IMP_COST_095", "TOTAL_IMP_COST_096",
    "FUTURE_ADT_114", "MIN_NAV_CLR_MT_116", "DECK_AREA",
    "TRAFFIC_LANES_ON_028A", "TRAFFIC_LANES_UND_028B",
    "APPR_WIDTH_MT_032", "KILOPOINT_011",
]

CATEGORICAL_COLS = [
    "STRUCTURE_KIND_043A", "STRUCTURE_TYPE_043B", "SCOUR_CRITICAL_113",
    "WATERWAY_EVAL_071", "FUNCTIONAL_CLASS_026", "HIGHWAY_SYSTEM_104",
    "OPEN_CLOSED_POSTED_041", "SERVICE_ON_042A", "SERVICE_UND_042B",
    "DECK_STRUCTURE_TYPE_107", "SURFACE_TYPE_108A", "MEMBRANE_TYPE_108B", "DECK_PROTECTION_108C",
]

MISSING_CODES = ["N", "n", "NA", "na", "", " "]


def load_year(year):
    path = os.path.join(EXTRACT_DIR, str(year), FILE_MAP[year])
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df


def clean_structure_number(val):
    if pd.isna(val):
        return val
    return str(val).strip()


def safe_numeric(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val in MISSING_CODES:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def prepare_year(df, year):
    df = df.copy()
    original_count = len(df)
    
    df["STATE_CODE_001"] = df["STATE_CODE_001"].astype(str).str.strip()
    df["STRUCTURE_NUMBER_008"] = df["STRUCTURE_NUMBER_008"].apply(clean_structure_number)
    df["bridge_id"] = df["STATE_CODE_001"] + "_" + df["STRUCTURE_NUMBER_008"]
    
    for col in CONDITION_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if str(x).strip().upper() in ["N", ""] else x)
    
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(safe_numeric)
    
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if str(x).strip().upper() in MISSING_CODES else str(x).strip())
    
    df["bridge_age"] = year - pd.to_numeric(df["YEAR_BUILT_027"], errors="coerce")
    df["inspection_year"] = year
    
    dup_count = df["bridge_id"].duplicated().sum()
    df = df.drop_duplicates(subset=["bridge_id"], keep="first")
    after_dedup = len(df)
    
    return df, original_count, after_dedup, dup_count


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DATA PREPARATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    dfs = {}
    for year in YEARS:
        df = load_year(year)
        df, orig, after, dups = prepare_year(df, year)
        dfs[year] = df
        
        report_lines.append(f"\nYear {year}:")
        report_lines.append(f"  Original records: {orig}")
        report_lines.append(f"  After deduplication: {after}")
        report_lines.append(f"  Duplicates removed: {dups}")
        report_lines.append(f"  Missing condition values:")
        for col in CONDITION_COLS:
            if col in df.columns:
                missing = df[col].isna().sum()
                report_lines.append(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    ids_2023 = set(dfs[2023]["bridge_id"])
    ids_2024 = set(dfs[2024]["bridge_id"])
    ids_2025 = set(dfs[2025]["bridge_id"])
    
    common_ids = ids_2023 & ids_2024 & ids_2025
    
    report_lines.append(f"\nCross-year matching:")
    report_lines.append(f"  Bridges in all three years: {len(common_ids)}")
    
    train_df = dfs[2023][dfs[2023]["bridge_id"].isin(common_ids)].copy()
    target_2024 = dfs[2024][["bridge_id", "DECK_COND_058"]].copy()
    target_2024.columns = ["bridge_id", "target_deck_cond_2024"]
    
    test_df = dfs[2024][dfs[2024]["bridge_id"].isin(common_ids)].copy()
    target_2025 = dfs[2025][["bridge_id", "DECK_COND_058"]].copy()
    target_2025.columns = ["bridge_id", "target_deck_cond_2025"]
    
    train_df = train_df.merge(target_2024, on="bridge_id", how="inner")
    test_df = test_df.merge(target_2025, on="bridge_id", how="inner")
    
    train_df["target_deck_poor_next_year"] = train_df["target_deck_cond_2024"].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip().isdigit() and int(str(x).strip()) <= 4 else 0
    )
    test_df["target_deck_poor_next_year"] = test_df["target_deck_cond_2025"].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip().isdigit() and int(str(x).strip()) <= 4 else 0
    )
    
    report_lines.append(f"\nTraining set (2023 -> 2024):")
    report_lines.append(f"  Records: {len(train_df)}")
    report_lines.append(f"  Poor deck next year: {train_df['target_deck_poor_next_year'].sum()} ({train_df['target_deck_poor_next_year'].mean()*100:.1f}%)")
    
    report_lines.append(f"\nTesting set (2024 -> 2025):")
    report_lines.append(f"  Records: {len(test_df)}")
    report_lines.append(f"  Poor deck next year: {test_df['target_deck_poor_next_year'].sum()} ({test_df['target_deck_poor_next_year'].mean()*100:.1f}%)")
    
    train_path = os.path.join(PROCESSED_DIR, "train_2023_2024.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    report_lines.append(f"\nSaved:")
    report_lines.append(f"  Training: {train_path}")
    report_lines.append(f"  Testing: {test_path}")
    
    report_text = "\n".join(report_lines)
    with open(os.path.join(REPORTS_DIR, "data_preparation.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
