"""
Rank bridges by maintenance priority using model predictions and configurable multipliers.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "priority_config.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def condition_severity_multiplier(deck_cond):
    if pd.isna(deck_cond):
        return 1.0
    try:
        val = int(float(str(deck_cond).strip()))
    except ValueError:
        return 1.0
    if val <= 4:
        return 3.0
    elif val <= 6:
        return 2.0
    else:
        return 1.0


def traffic_multiplier(adt):
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
    if pd.isna(scour):
        return 1.0
    val = str(scour).strip().upper()
    if val in ["0", "N", ""]:
        return 1.0
    elif val == "U":
        return 1.0
    elif val in ["1", "2", "T"]:
        return 2.0
    else:
        return 1.0


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    config = load_config()
    print("Loaded priority config.")

    print("Loading model and test data...")
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    preprocessor = joblib.load(
        os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    )
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_2024_2025.parquet"))

    feature_cols = [
        c
        for c in test_df.columns
        if c
        not in [
            "bridge_id",
            "target_deck_poor_next_year",
            "target_deck_cond_2025",
            "STATE_CODE_001",
            "STRUCTURE_NUMBER_008",
        ]
    ]
    X_test = preprocessor.transform(test_df[feature_cols])

    print("Predicting probabilities...")
    y_proba = model.predict_proba(X_test)[:, 1]
    test_df["predicted_probability"] = y_proba

    print("Calculating priority scores...")
    test_df["condition_severity_multiplier"] = test_df["DECK_COND_058"].apply(
        condition_severity_multiplier
    )
    test_df["traffic_multiplier"] = test_df["ADT_029"].apply(traffic_multiplier)
    test_df["scour_multiplier"] = test_df["SCOUR_CRITICAL_113"].apply(scour_multiplier)

    test_df["priority_score"] = (
        test_df["predicted_probability"]
        * test_df["condition_severity_multiplier"]
        * test_df["traffic_multiplier"]
        * test_df["scour_multiplier"]
    )

    test_df = test_df.sort_values("priority_score", ascending=False).reset_index(
        drop=True
    )

    n = len(test_df)
    high_thresh = int(np.ceil(n * config["priority_thresholds"]["high"]))
    medium_thresh = int(
        np.ceil(
            n
            * (
                config["priority_thresholds"]["high"]
                + config["priority_thresholds"]["medium"]
            )
        )
    )

    test_df["priority_class"] = "Low"
    test_df.loc[: high_thresh - 1, "priority_class"] = "High"
    test_df.loc[high_thresh : medium_thresh - 1, "priority_class"] = "Medium"

    def overall_condition(row):
        vals = []
        for col in [
            "DECK_COND_058",
            "SUPERSTRUCTURE_COND_059",
            "SUBSTRUCTURE_COND_060",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    vals.append(int(float(str(row[col]).strip())))
                except ValueError:
                    pass
        if not vals:
            return np.nan
        return min(vals)

    test_df["current_overall_condition"] = test_df.apply(overall_condition, axis=1)

    output_cols = [
        "bridge_id",
        "predicted_probability",
        "current_deck_condition",
        "current_superstructure_condition",
        "current_substructure_condition",
        "current_overall_condition",
        "average_daily_traffic",
        "scour_criticality",
        "priority_score",
        "priority_class",
    ]

    rename_map = {
        "DECK_COND_058": "current_deck_condition",
        "SUPERSTRUCTURE_COND_059": "current_superstructure_condition",
        "SUBSTRUCTURE_COND_060": "current_substructure_condition",
        "ADT_029": "average_daily_traffic",
        "SCOUR_CRITICAL_113": "scour_criticality",
    }

    output_df = test_df.rename(columns=rename_map)[output_cols]

    output_path = os.path.join(REPORTS_DIR, "maintenance_priority_2025.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\nPriority class distribution:")
    print(test_df["priority_class"].value_counts())

    print("\nTop 10 bridges by priority score:")
    print(
        test_df[
            ["bridge_id", "predicted_probability", "priority_score", "priority_class"]
        ]
        .head(10)
        .to_string(index=False)
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        test_df[test_df["priority_class"] == "High"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="High Priority",
        color="red",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Medium"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="Medium Priority",
        color="orange",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Low"]["predicted_probability"],
        bins=20,
        alpha=0.7,
        label="Low Priority",
        color="green",
    )
    plt.xlabel("Predicted Probability of Poor Deck Next Year")
    plt.ylabel("Number of Bridges")
    plt.title("Predicted Risk Distribution by Priority Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predicted_risk_distribution.png"), dpi=150)
    plt.close()
    print("Saved: figures/predicted_risk_distribution.png")

    top20 = test_df.head(20)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top20)), top20["priority_score"][::-1])
    plt.yticks(range(len(top20)), top20["bridge_id"][::-1])
    plt.xlabel("Priority Score")
    plt.title("Top 20 Maintenance Priority Bridges (2025)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "top20_priority.png"), dpi=150)
    plt.close()
    print("Saved: figures/top20_priority.png")

    print("\nMaintenance prioritization complete.")


if __name__ == "__main__":
    main()
