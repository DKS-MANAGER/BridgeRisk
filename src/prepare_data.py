"""
Clean, normalize, and prepare nationwide 50-State NBI data (2021–2025) with longitudinal degradation dynamics.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, "data", "raw", "extracted")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY", "PR"
]
YEARS = [2021, 2022, 2023, 2024, 2025]

CONDITION_COLS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
]

NUMERIC_COLS = [
    "YEAR_BUILT_027",
    "YEAR_RECONSTRUCTED_106",
    "ADT_029",
    "PERCENT_ADT_TRUCK_109",
    "MAIN_UNIT_SPANS_045",
    "MAX_SPAN_LEN_MT_048",
    "STRUCTURE_LEN_MT_049",
    "ROADWAY_WIDTH_MT_051",
    "DECK_WIDTH_MT_052",
    "HORR_CLR_MT_047",
    "VERT_CLR_OVER_MT_053",
    "OPERATING_RATING_064",
    "INVENTORY_RATING_066",
    "DECK_AREA",
    "TRAFFIC_LANES_ON_028A",
]

CATEGORICAL_COLS = [
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B",
    "FUNCTIONAL_CLASS_026",
    "HIGHWAY_SYSTEM_104",
    "OPEN_CLOSED_POSTED_041",
    "DECK_STRUCTURE_TYPE_107",
    "SURFACE_TYPE_108A",
    "DECK_PROTECTION_108C",
]


def load_state_year(state, year):
    candidates = [
        os.path.join(EXTRACT_DIR, str(year), f"{state}{str(year)[-2:]}.txt"),
        os.path.join(EXTRACT_DIR, str(year), f"{state}{str(year)[-2:]}.TXT"),
        os.path.join(EXTRACT_DIR, str(year), f"{state.lower()}{str(year)[-2:]}.txt"),
        os.path.join(EXTRACT_DIR, str(year), f"{state.upper()}{str(year)[-2:]}.txt"),
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break
    if filepath is None:
        raise FileNotFoundError(f"Missing file for {state} {year}")

    try:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(filepath, low_memory=False, encoding="latin1", on_bad_lines="skip")

    df.columns = df.columns.str.strip()
    return df


def prepare_state_year(df, state, year):
    if "STRUCTURE_NUMBER_008" in df.columns:
        df["bridge_id"] = state + "_" + df["STRUCTURE_NUMBER_008"].astype(str).str.strip()
    elif "bridge_id" not in df.columns:
        raise ValueError(f"Structure number column not found in {state} {year}")

    df = df.drop_duplicates(subset=["bridge_id"]).copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CONDITION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Structural Age
    if "YEAR_BUILT_027" in df.columns:
        df["bridge_age"] = year - df["YEAR_BUILT_027"]
        df.loc[df["bridge_age"] < 0, "bridge_age"] = np.nan

    # 2. Deck Slenderness Ratio (Span-to-Width)
    if "MAX_SPAN_LEN_MT_048" in df.columns and "DECK_WIDTH_MT_052" in df.columns:
        df["span_to_width_ratio"] = df["MAX_SPAN_LEN_MT_048"] / df["DECK_WIDTH_MT_052"].replace(0, np.nan)

    # 3. Estimated Cumulative Fatigue Cycles (ESAL Proxy)
    if "bridge_age" in df.columns and "ADT_029" in df.columns and "PERCENT_ADT_TRUCK_109" in df.columns:
        truck_vol = df["ADT_029"].fillna(0) * (df["PERCENT_ADT_TRUCK_109"].fillna(0) / 100.0)
        df["est_lifetime_truck_passes"] = df["bridge_age"].fillna(0) * truck_vol * 365.0

    # 4. Deck Area Computation
    if "DECK_AREA" not in df.columns and "STRUCTURE_LEN_MT_049" in df.columns and "DECK_WIDTH_MT_052" in df.columns:
        df["DECK_AREA"] = df["STRUCTURE_LEN_MT_049"] * df["DECK_WIDTH_MT_052"]

    return df


def compute_longitudinal_deltas(t_df, prev_dfs, current_year):
    prev_year_1 = current_year - 1
    if prev_year_1 in prev_dfs:
        p1 = prev_dfs[prev_year_1][["bridge_id", "DECK_COND_058"]].copy()
        p1.columns = ["bridge_id", "deck_cond_prev1"]
        t_df = t_df.merge(p1, on="bridge_id", how="left")
        t_df["delta_deck_1yr"] = t_df["DECK_COND_058"] - t_df["deck_cond_prev1"]
        t_df.drop(columns=["deck_cond_prev1"], inplace=True)
    else:
        t_df["delta_deck_1yr"] = 0.0

    prev_year_2 = current_year - 2
    if prev_year_2 in prev_dfs:
        p2 = prev_dfs[prev_year_2][["bridge_id", "DECK_COND_058"]].copy()
        p2.columns = ["bridge_id", "deck_cond_prev2"]
        t_df = t_df.merge(p2, on="bridge_id", how="left")
        t_df["delta_deck_2yr"] = t_df["DECK_COND_058"] - t_df["deck_cond_prev2"]
        t_df.drop(columns=["deck_cond_prev2"], inplace=True)
    else:
        t_df["delta_deck_2yr"] = 0.0

    return t_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("======================================================================")
    print("NATIONWIDE 50-STATE NBI DATA PREPARATION (2021–2025)")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("======================================================================\n")

    all_state_trains = []
    all_state_tests = []
    processed_count = 0

    for state in ALL_STATES:
        dfs = {}
        for year in YEARS:
            try:
                df = load_state_year(state, year)
                df = prepare_state_year(df, state, year)
                dfs[year] = df
            except Exception as e:
                pass

        if not dfs:
            continue

        processed_count += 1
        print(f"[{processed_count}/{len(ALL_STATES)}] Processed State: {state} (Years: {list(dfs.keys())})")

        # Out-of-time Test Set: 2024 -> 2025
        if 2024 in dfs and 2025 in dfs:
            test_ids = set(dfs[2024]["bridge_id"]) & set(dfs[2025]["bridge_id"])
            test_df = dfs[2024][dfs[2024]["bridge_id"].isin(test_ids)].copy()
            target_2025 = dfs[2025][["bridge_id", "DECK_COND_058"]].copy()
            target_2025.columns = ["bridge_id", "target_deck_cond_next"]
            test_df = test_df.merge(target_2025, on="bridge_id", how="inner")
            test_df["target_deck_poor_next_year"] = test_df["target_deck_cond_next"].apply(
                lambda x: 1 if pd.notna(x) and x <= 4 else 0
            )
            test_df = compute_longitudinal_deltas(test_df, dfs, 2024)
            test_df["state"] = state
            all_state_tests.append(test_df)

        # Training Transitions: (2021->2022), (2022->2023), (2023->2024)
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
                t_df = compute_longitudinal_deltas(t_df, dfs, y_curr)
                t_df["state"] = state
                all_state_trains.append(t_df)

    print("\nMerging all nationwide state records...")
    final_train_df = pd.concat(all_state_trains, ignore_index=True) if all_state_trains else pd.DataFrame()
    final_test_df = pd.concat(all_state_tests, ignore_index=True) if all_state_tests else pd.DataFrame()

    # Normalize object string columns
    for df_obj in [final_train_df, final_test_df]:
        if not df_obj.empty:
            for col in df_obj.columns:
                if df_obj[col].dtype == "object":
                    df_obj[col] = df_obj[col].astype(str)

    train_path = os.path.join(PROCESSED_DIR, "train_2021_2024.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")

    final_train_df.to_parquet(train_path, index=False)
    final_test_df.to_parquet(test_path, index=False)

    print(f"\nSaved Nationwide Training Set: {train_path} ({len(final_train_df):,d} transitions)")
    print(f"Saved Nationwide Test Set (2024->2025): {test_path} ({len(final_test_df):,d} bridges)")
    print(f"Nationwide Train Poor Defect Rate: {final_train_df['target_deck_poor_next_year'].mean()*100:.2f}%")
    print(f"Nationwide Test Poor Defect Rate: {final_test_df['target_deck_poor_next_year'].mean()*100:.2f}%")

    with open(os.path.join(REPORTS_DIR, "matching_counts.csv"), "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"train_records,{len(final_train_df)}\n")
        f.write(f"test_records,{len(final_test_df)}\n")


if __name__ == "__main__":
    main()
