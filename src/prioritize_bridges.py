"""
Rank bridges by maintenance priority using model predictions and engineering multipliers.

Priority Score = Predicted Probability × Condition Multiplier × Traffic Multiplier × Scour Multiplier

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "figures"

# Priority thresholds: Top 10% High priority, next 20% Medium priority, remaining 70% Low
HIGH_PRIORITY_FRACTION = 0.10
MEDIUM_PRIORITY_FRACTION = 0.20


def condition_severity_multiplier(deck_cond):
    """Bridges with deck condition rating <= 4 (Poor) receive highest maintenance weight."""
    if pd.isna(deck_cond):
        return 1.0
    try:
        val = float(str(deck_cond).strip())
    except ValueError:
        return 1.0
    if val <= 4:
        return 3.0
    elif val <= 6:
        return 2.0
    else:
        return 1.0


def traffic_multiplier(adt):
    """Bridges carrying heavy average daily traffic receive higher priority due to consequence of closure."""
    if pd.isna(adt):
        return 1.0
    try:
        val = float(adt)
    except ValueError:
        return 1.0
    if val < 1000:
        return 1.0
    elif val < 10000:
        return 1.5
    else:
        return 2.0


def scour_multiplier(scour):
    """Bridges flagged as scour critical (NBI item 113 in 1, 2, or T) receive double weight."""
    if pd.isna(scour):
        return 1.0
    val = str(scour).strip().upper()
    if val in ["1", "2", "T"]:
        return 2.0
    return 1.0


def calculate_overall_condition(row):
    """Lowest rating among deck, superstructure, and substructure."""
    vals = []
    for col in ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060"]:
        if col in row and pd.notna(row[col]):
            try:
                vals.append(float(str(row[col]).strip()))
            except ValueError:
                pass
    return min(vals) if vals else np.nan


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and test data...")
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_2024_2025.parquet")

    feature_cols = [
        c
        for c in test_df.columns
        if c
        not in [
            "bridge_id",
            "state",
            "target_deck_poor_next_year",
            "target_deck_cond_next",
            "target_deck_cond_2025",
            "STATE_CODE_001",
            "STRUCTURE_NUMBER_008",
        ]
    ]
    X_test = preprocessor.transform(test_df[feature_cols])

    print("Predicting probabilities of deterioration to poor condition...")
    test_df["predicted_probability"] = model.predict_proba(X_test)[:, 1]

    print("Calculating priority scores...")
    test_df["condition_mult"] = test_df["DECK_COND_058"].apply(condition_severity_multiplier)
    test_df["traffic_mult"] = test_df["ADT_029"].apply(traffic_multiplier)
    test_df["scour_mult"] = test_df["SCOUR_CRITICAL_113"].apply(scour_multiplier)

    test_df["priority_score"] = (
        test_df["predicted_probability"]
        * test_df["condition_mult"]
        * test_df["traffic_mult"]
        * test_df["scour_mult"]
    )

    test_df = test_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    n = len(test_df)
    high_thresh = int(np.ceil(n * HIGH_PRIORITY_FRACTION))
    medium_thresh = int(np.ceil(n * (HIGH_PRIORITY_FRACTION + MEDIUM_PRIORITY_FRACTION)))

    test_df["priority_class"] = "Low"
    test_df.loc[: high_thresh - 1, "priority_class"] = "High"
    test_df.loc[high_thresh : medium_thresh - 1, "priority_class"] = "Medium"
    test_df["current_overall_condition"] = test_df.apply(calculate_overall_condition, axis=1)

    output_cols = ["bridge_id"]
    if "state" in test_df.columns:
        output_cols.append("state")

    output_cols.extend([
        "predicted_probability",
        "current_deck_condition",
        "current_superstructure_condition",
        "current_substructure_condition",
        "current_overall_condition",
        "average_daily_traffic",
        "scour_criticality",
        "priority_score",
        "priority_class",
    ])

    rename_map = {
        "DECK_COND_058": "current_deck_condition",
        "SUPERSTRUCTURE_COND_059": "current_superstructure_condition",
        "SUBSTRUCTURE_COND_060": "current_substructure_condition",
        "ADT_029": "average_daily_traffic",
        "SCOUR_CRITICAL_113": "scour_criticality",
    }

    output_df = test_df.rename(columns=rename_map)[output_cols]
    output_path = REPORTS_DIR / "maintenance_priority_2025.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\nPriority class distribution:")
    print(test_df["priority_class"].value_counts())

    print("\nTop 10 bridges by priority score:")
    preview_cols = ["bridge_id"]
    if "state" in test_df.columns:
        preview_cols.append("state")
    preview_cols.extend(["predicted_probability", "priority_score", "priority_class"])
    print(test_df[preview_cols].head(10).to_string(index=False))

    # 1. Distribution of predicted risk by priority class
    plt.figure(figsize=(9, 5))
    plt.hist(
        test_df[test_df["priority_class"] == "High"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="High Priority (Top 10%)",
        color="#d9534f",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Medium"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="Medium Priority (Next 20%)",
        color="#f0ad4e",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Low"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="Low Priority (Remaining 70%)",
        color="#5cb85c",
    )
    plt.xlabel("Predicted Probability of Poor Deck Next Year")
    plt.ylabel("Number of Bridges")
    plt.title("Predicted Risk Distribution by Maintenance Priority Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_risk_distribution.png", dpi=150)
    plt.close()

    # 2. Top 20 bridges horizontal bar chart
    top20 = test_df.head(20)
    plt.figure(figsize=(10, 7))
    labels = top20["bridge_id"] if "state" not in top20.columns else top20["state"] + "_" + top20["bridge_id"]
    plt.barh(range(len(top20)), top20["priority_score"][::-1], color="#2b5c8f")
    plt.yticks(range(len(top20)), labels[::-1])
    plt.xlabel("Maintenance Priority Score")
    plt.title("Top 20 Maintenance Priority Bridges (2025 Out-of-Time Test)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top20_priority.png", dpi=150)
    plt.close()
    print("Visualizations saved to figures/")


if __name__ == "__main__":
    main()
