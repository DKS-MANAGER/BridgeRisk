"""
Generate SHAP explanations for XGBoost model predictions.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "figures"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models and test data...")
    model_wrapper = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_2024_2025.parquet")

    # If model is CalibratedClassifierCV, extract underlying base estimator
    if hasattr(model_wrapper, "calibrated_classifiers_"):
        base_model = model_wrapper.calibrated_classifiers_[0].estimator
    elif hasattr(model_wrapper, "estimator"):
        base_model = model_wrapper.estimator
    else:
        base_model = model_wrapper

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

    # Sample for fast SHAP computation across nationwide dataset
    sample_df = test_df.sample(n=min(10000, len(test_df)), random_state=42).reset_index(drop=True)
    X_sample = preprocessor.transform(sample_df[feature_cols])
    feature_names = preprocessor.get_feature_names_out()
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    print(f"Computing TreeSHAP values for {len(X_sample_df):,d} sampled nationwide test bridges...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample_df)

    # 1. Global Mean |SHAP| Ranking Bar Plot
    print("Generating global feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance (Nationwide XGBoost)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150)
    plt.close()
    print("Saved: figures/shap_bar.png")

    # 2. Global Beeswarm Summary Plot
    print("Generating SHAP summary beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample_df, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Feature Impact on Poor Deck Prediction)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print("Saved: figures/shap_summary.png")

    # 3. Local explanations: High-risk and Low-risk bridges
    sample_df["predicted_proba"] = model_wrapper.predict_proba(X_sample)[:, 1]
    high_risk_idx = int(sample_df["predicted_proba"].idxmax())
    low_risk_idx = int(sample_df["predicted_proba"].idxmin())

    high_id = sample_df.loc[high_risk_idx, "bridge_id"]
    high_state = sample_df.loc[high_risk_idx, "state"] if "state" in sample_df.columns else ""
    low_id = sample_df.loc[low_risk_idx, "bridge_id"]
    low_state = sample_df.loc[low_risk_idx, "state"] if "state" in sample_df.columns else ""

    print(f"High-risk bridge: {high_state}_{high_id} (prob={sample_df.loc[high_risk_idx, 'predicted_proba']:.3f})")
    print(f"Low-risk bridge: {low_state}_{low_id} (prob={sample_df.loc[low_risk_idx, 'predicted_proba']:.3f})")

    # High-risk force plot
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx, :],
        X_sample_df.iloc[high_risk_idx, :],
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Local Explanation: High-Risk Bridge ({high_state} {high_id})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_high_risk.png", dpi=150)
    plt.close()
    print("Saved: figures/shap_high_risk.png")

    # Low-risk force plot
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[low_risk_idx, :],
        X_sample_df.iloc[low_risk_idx, :],
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Local Explanation: Low-Risk Bridge ({low_state} {low_id})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_low_risk.png", dpi=150)
    plt.close()
    print("Saved: figures/shap_low_risk.png")

    # 4. Feature dependence plots
    print("Generating feature dependence plots...")
    for col in ["num__DECK_COND_058", "num__bridge_age", "num__PERCENT_ADT_TRUCK_109", "num__delta_deck_1yr"]:
        if col in X_sample_df.columns:
            clean_name = col.replace("num__", "")
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(col, shap_values, X_sample_df, show=False)
            plt.title(f"SHAP Dependence: {clean_name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_dependence_{clean_name}.png", dpi=150)
            plt.close()
            print(f"Saved: figures/shap_dependence_{clean_name}.png")

    # 5. Export feature importances
    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)
    print("Saved: reports/shap_feature_importance.csv")
    print("\nSHAP explanation complete.")


if __name__ == "__main__":
    main()
