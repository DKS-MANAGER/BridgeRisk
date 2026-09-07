"""
Rank bridges by maintenance priority using model predictions and structural consequence factors.

Structural Priority Score = P(Poor Deck Next Year) × Structural Consequence
                          = P(DECK_COND_(t+1) <= 4) × (C_service × C_capacity × C_support)

Where:
- P_poor_next_year: 1-year ML-predicted probability of crossing the poor deck threshold (NBI Item 58 <= 4).
- C_service: Dimensionless structural traffic service demand index = (1 + log10(1 + ADT / 100)) × (1 + Truck_Pct / 100).
- C_capacity: Structural load rating consequence based on Operating Rating (NBI Item 64).
- C_support: Superstructure & Substructure integrity modifier (NBI Items 59 & 60).

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

# Priority classification thresholds: Top 10% High, next 20% Medium, remaining 70% Low
HIGH_PRIORITY_FRACTION = 0.10
MEDIUM_PRIORITY_FRACTION = 0.20


def calculate_structural_consequence(df):
    """
    Compute structural and service consequence of deck deficiency using structural NBI fields.
    
    Components:
    1. C_service: Service interruption & fatigue load demand from ADT and heavy truck percentage.
    2. C_capacity: Load-carrying vulnerability from Operating Rating (Item 64).
    3. C_support: Support integrity modifier from Superstructure (Item 59) and Substructure (Item 60).
    """
    # 1. Traffic demand factor
    adt = pd.to_numeric(df["ADT_029"], errors="coerce").fillna(0).clip(lower=0)
    truck_pct = pd.to_numeric(df["PERCENT_ADT_TRUCK_109"], errors="coerce").fillna(0).clip(0, 100)
    c_service = (1.0 + np.log10(1.0 + adt / 100.0).clip(lower=0)) * (1.0 + (truck_pct / 100.0))

    # 2. Operating Load Rating factor (Item 64)
    opr = pd.to_numeric(df["OPERATING_RATING_064"], errors="coerce").fillna(36.0)
    c_capacity = np.where(opr < 20.0, 1.5, np.where(opr < 30.0, 1.2, 1.0))

    # 3. Superstructure & Substructure condition support (Items 59 & 60)
    def get_min_support(row):
        vals = []
        for col in ["SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060"]:
            if col in row and pd.notna(row[col]):
                try:
                    vals.append(float(str(row[col]).strip()))
                except ValueError:
                    pass
        return min(vals) if vals else 7.0

    min_support = df.apply(get_min_support, axis=1)
    c_support = np.where(min_support <= 4.0, 1.4, np.where(min_support <= 5.0, 1.2, 1.0))

    structural_consequence = c_service * c_capacity * c_support
    return structural_consequence, min_support


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
    test_df["P_poor_next_year"] = model.predict_proba(X_test)[:, 1]

    print("Calculating structural consequence and structural priority scores...")
    test_df["structural_consequence"], test_df["min_support_condition"] = calculate_structural_consequence(test_df)
    test_df["structural_priority_score"] = test_df["P_poor_next_year"] * test_df["structural_consequence"]

    # Rank bridges (1 = highest priority)
    test_df["priority_rank"] = test_df["structural_priority_score"].rank(ascending=False, method="min").astype(int)
    test_df = test_df.sort_values("structural_priority_score", ascending=False).reset_index(drop=True)

    n = len(test_df)
    high_thresh = int(np.ceil(n * HIGH_PRIORITY_FRACTION))
    medium_thresh = int(np.ceil(n * (HIGH_PRIORITY_FRACTION + MEDIUM_PRIORITY_FRACTION)))

    test_df["priority_class"] = "Low"
    test_df.loc[: high_thresh - 1, "priority_class"] = "High"
    test_df.loc[high_thresh : medium_thresh - 1, "priority_class"] = "Medium"

    output_cols = [
        "priority_rank",
        "bridge_id",
        "state",
        "P_poor_next_year",
        "structural_consequence",
        "structural_priority_score",
        "priority_class",
        "current_deck_condition",
        "current_superstructure_condition",
        "current_substructure_condition",
        "operating_rating",
        "average_daily_traffic",
        "truck_percent",
    ]

    rename_map = {
        "DECK_COND_058": "current_deck_condition",
        "SUPERSTRUCTURE_COND_059": "current_superstructure_condition",
        "SUBSTRUCTURE_COND_060": "current_substructure_condition",
        "OPERATING_RATING_064": "operating_rating",
        "ADT_029": "average_daily_traffic",
        "PERCENT_ADT_TRUCK_109": "truck_percent",
    }

    output_df = test_df.rename(columns=rename_map)[[c for c in output_cols if c in test_df.rename(columns=rename_map).columns]]
    output_path = REPORTS_DIR / "maintenance_priority_2025.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\nPriority class distribution:")
    print(test_df["priority_class"].value_counts())

    print("\nTop 10 bridges by Structural Deck Maintenance Priority Score:")
    preview_cols = ["priority_rank", "bridge_id", "state", "P_poor_next_year", "structural_consequence", "structural_priority_score", "priority_class"]
    print(test_df[preview_cols].head(10).to_string(index=False))

    # 1. Distribution of predicted risk by priority class
    plt.figure(figsize=(9, 5))
    plt.hist(
        test_df[test_df["priority_class"] == "High"]["P_poor_next_year"],
        bins=20,
        alpha=0.7,
        label="High Priority (Top 10%)",
        color="#d9534f",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Medium"]["P_poor_next_year"],
        bins=20,
        alpha=0.7,
        label="Medium Priority (Next 20%)",
        color="#f0ad4e",
    )
    plt.hist(
        test_df[test_df["priority_class"] == "Low"]["P_poor_next_year"],
        bins=20,
        alpha=0.7,
        label="Low Priority (Remaining 70%)",
        color="#5cb85c",
    )
    plt.xlabel("Predicted Probability of Poor Deck Next Year P(Poor)")
    plt.ylabel("Number of Bridges")
    plt.title("Structural Risk Distribution by Maintenance Priority Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_risk_distribution.png", dpi=150)
    plt.close()

    # 2. Top 20 bridges horizontal bar chart
    top20 = test_df.head(20)
    plt.figure(figsize=(10, 7))
    labels = top20["bridge_id"] if "state" not in top20.columns else top20["state"] + "_" + top20["bridge_id"]
    plt.barh(range(len(top20)), top20["structural_priority_score"][::-1], color="#2b5c8f")
    plt.yticks(range(len(top20)), labels[::-1])
    plt.xlabel("Structural Deck Maintenance Priority Score")
    plt.title("Top 20 Priority Bridges (2025 Out-of-Time Test)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top20_priority.png", dpi=150)
    plt.close()
    print("Visualizations saved to figures/")


if __name__ == "__main__":
    main()
