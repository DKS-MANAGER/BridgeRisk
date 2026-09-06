"""
Clean, normalize, and prepare NBI data for Maine (ME), Hawaii (HI), and Delaware (DE) (2021–2025).

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, "data", "raw", "extracted")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

STATES = ["ME", "HI", "DE"]
YEARS = [2021, 2022, 2023, 2024, 2025]

CONDITION_COLS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
]
NUMERIC_COLS = [
    "YEAR_BUILT_027",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109",
    "MAIN_UNIT_SPANS_045",
    "MAX_SPAN_LEN_MT_048",
    "STRUCTURE_LEN_MT_049",
    "ROADWAY_WIDTH_MT_051",
    "DECK_WIDTH_MT_052",
    "LEFT_CURB_MT_050A",
    "RIGHT_CURB_MT_050B",
    "HORR_CLR_MT_047",
    "VERT_CLR_OVER_MT_053",
    "VERT_CLR_UND_054B",
    "LAT_UND_MT_055B",
    "LEFT_LAT_UND_MT_056",
    "OPERATING_RATING_064",
    "INVENTORY_RATING_066",
    "BRIDGE_IMP_COST_094",
    "ROADWAY_IMP_COST_095",
    "TOTAL_IMP_COST_096",
    "FUTURE_ADT_114",
    "MIN_NAV_CLR_MT_116",
    "DECK_AREA",
    "TRAFFIC_LANES_ON_028A",
    "TRAFFIC_LANES_UND_028B",
    "APPR_WIDTH_MT_032",
    "KILOPOINT_011",
]

CATEGORICAL_COLS = [
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B",
    "SCOUR_CRITICAL_113",
    "WATERWAY_EVAL_071",
    "FUNCTIONAL_CLASS_026",
    "OWNER_027",
    "LANES_ON_STRUCT",
]


def load_state_year(state, year):
    filename = f"{state}{str(year)[-2:]}.txt"
    filepath = os.path.join(EXTRACT_DIR, str(year), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file for {state} year {year}: {filepath}")

    # FHWA NBI files are typically comma-delimited or fixed-width depending on year
    try:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
    except Exception:
        df = pd.read_csv(filepath, low_memory=False, encoding="latin1")

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    return df


def prepare_state_year(df, state, year):
    # Ensure bridge_id exists
    if "STRUCTURE_NUMBER_008" in df.columns:
        df["bridge_id"] = state + "_" + df["STRUCTURE_NUMBER_008"].astype(str).str.strip()
    elif "bridge_id" not in df.columns:
        raise ValueError(f"Structure number column not found in {state} {year}")

    orig_count = len(df)
    df = df.drop_duplicates(subset=["bridge_id"]).copy()
    after_dedup = len(df)
    dups_removed = orig_count - after_dedup

    # Numeric conversion
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Condition conversion
    for col in CONDITION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering: bridge age
    if "YEAR_BUILT_027" in df.columns:
        df["bridge_age"] = year - df["YEAR_BUILT_027"]
        df.loc[df["bridge_age"] < 0, "bridge_age"] = np.nan

    # Deck area computation if missing
    if "DECK_AREA" not in df.columns and "STRUCTURE_LEN_MT_049" in df.columns and "DECK_WIDTH_MT_052" in df.columns:
        df["DECK_AREA"] = df["STRUCTURE_LEN_MT_049"] * df["DECK_WIDTH_MT_052"]

    return df, orig_count, after_dedup, dups_removed


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_lines = []
    report_lines.append("======================================================================")
    report_lines.append(f"MULTI-STATE DATA PREPARATION REPORT (ME, HI, DE: 2021–2025)")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("======================================================================\n")

    all_state_trains = []
    all_state_tests = []

    for state in STATES:
        report_lines.append(f"\n=================== STATE: {state} ===================")
        dfs = {}
        for year in YEARS:
            try:
                df = load_state_year(state, year)
                df, orig, after, dups = prepare_state_year(df, state, year)
                dfs[year] = df

                report_lines.append(f"\nYear {year}:")
                report_lines.append(f"  Original records: {orig}")
                report_lines.append(f"  After deduplication: {after}")
                report_lines.append(f"  Duplicates removed: {dups}")
            except Exception as e:
                report_lines.append(f"\nYear {year}: Error loading - {e}")

        # Check common IDs across years for training (2021, 2022, 2023, 2024) and test (2024, 2025)
        available_train_years = [y for y in [2021, 2022, 2023, 2024] if y in dfs]
        if 2024 in dfs and 2025 in dfs:
            test_ids = set(dfs[2024]["bridge_id"]) & set(dfs[2025]["bridge_id"])
            test_df = dfs[2024][dfs[2024]["bridge_id"].isin(test_ids)].copy()
            target_2025 = dfs[2025][["bridge_id", "DECK_COND_058"]].copy()
            target_2025.columns = ["bridge_id", "target_deck_cond_next"]
            test_df = test_df.merge(target_2025, on="bridge_id", how="inner")
            test_df["target_deck_poor_next_year"] = test_df["target_deck_cond_next"].apply(
                lambda x: 1 if pd.notna(x) and x <= 4 else 0
            )
            test_df["state"] = state
            all_state_tests.append(test_df)

        # Build training transitions (2021->2022, 2022->2023, 2023->2024)
        transitions = [(2021, 2022), (2022, 2023), (2023, 2024)]
        for y_curr, y_next in transitions:
            if y_curr in dfs and y_next in dfs:
                common_ids = set(dfs[y_curr]["bridge_id"]) & set(dfs[y_next]["bridge_id"])
                t_df = dfs[y_curr][dfs[y_curr]["bridge_id"].isin(common_ids)].copy()
                t_next = dfs[y_next][["bridge_id", "DECK_COND_058"]].copy()
                t_next.columns = ["bridge_id", "target_deck_cond_next"]
                t_df = t_df.merge(t_next, on="bridge_id", how="inner")
                t_df["target_deck_poor_next_year"] = t_df["target_deck_cond_next"].apply(
                    lambda x: 1 if pd.notna(x) and x <= 4 else 0
                )
                t_df["state"] = state
                all_state_trains.append(t_df)

    if all_state_trains:
        final_train_df = pd.concat(all_state_trains, ignore_index=True)
    else:
        final_train_df = pd.DataFrame()

    if all_state_tests:
        final_test_df = pd.concat(all_state_tests, ignore_index=True)
    else:
        final_test_df = pd.DataFrame()

    for df_obj in [final_train_df, final_test_df]:
        if not df_obj.empty:
            for col in df_obj.select_dtypes(include=["object"]).columns:
                df_obj[col] = df_obj[col].astype(str)

    train_path = os.path.join(PROCESSED_DIR, "train_2021_2024.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")

    final_train_df.to_parquet(train_path, index=False)
    final_test_df.to_parquet(test_path, index=False)

    report_lines.append(f"\nSaved Training Set (2021–2024 transitions): {train_path} ({len(final_train_df)} records)")
    report_lines.append(f"Saved Testing Set (2024–2025): {test_path} ({len(final_test_df)} records)")

    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "matching_counts.csv"), "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"train_records,{len(final_train_df)}\n")
        f.write(f"test_records,{len(final_test_df)}\n")


if __name__ == "__main__":
    main()
