import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading models and data...")
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
    
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")
    test_df = pd.read_parquet(test_path)
    
    feature_cols = [c for c in test_df.columns if c not in ["bridge_id", "target_deck_poor_next_year", "target_deck_cond_2025", "STATE_CODE_001", "STRUCTURE_NUMBER_008"]]
    X_test = preprocessor.transform(test_df[feature_cols])
    
    feature_names = preprocessor.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    
    print("Generating global feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_bar.png"), dpi=150)
    plt.close()
    print(f"Saved: figures/shap_bar.png")
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("SHAP Summary Plot (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: figures/shap_summary.png")
    
    print("Selecting high-risk and low-risk bridges...")
    y_proba = model.predict_proba(X_test)[:, 1]
    test_df["predicted_proba"] = y_proba
    
    high_risk_idx = test_df["predicted_proba"].idxmax()
    low_risk_idx = test_df["predicted_proba"].idxmin()
    
    print(f"High-risk bridge ID: {test_df.loc[high_risk_idx, 'bridge_id']} (prob={test_df.loc[high_risk_idx, 'predicted_proba']:.3f})")
    print(f"Low-risk bridge ID: {test_df.loc[low_risk_idx, 'bridge_id']} (prob={test_df.loc[low_risk_idx, 'predicted_proba']:.3f})")
    
    print("Generating high-risk bridge explanation...")
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx, :],
        X_test_df.iloc[high_risk_idx, :],
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Explanation: High-Risk Bridge {test_df.loc[high_risk_idx, 'bridge_id']}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_high_risk.png"), dpi=150)
    plt.close()
    print(f"Saved: figures/shap_high_risk.png")
    
    print("Generating low-risk bridge explanation...")
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[low_risk_idx, :],
        X_test_df.iloc[low_risk_idx, :],
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Explanation: Low-Risk Bridge {test_df.loc[low_risk_idx, 'bridge_id']}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_low_risk.png"), dpi=150)
    plt.close()
    print(f"Saved: figures/shap_low_risk.png")
    
    print("Generating feature-dependence plots...")
    for col in ["bridge_age", "DECK_COND_058"]:
        if col in X_test_df.columns:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(col, shap_values, X_test_df, show=False)
            plt.title(f"SHAP Dependence: {col}")
            plt.tight_layout()
            safe_col = col.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(FIGURES_DIR, f"shap_dependence_{safe_col}.png"), dpi=150)
            plt.close()
            print(f"Saved: figures/shap_dependence_{safe_col}.png")
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    
    importance_df.to_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv"), index=False)
    print(f"Saved: reports/shap_feature_importance.csv")
    
    print("\nSHAP explanation complete.")


if __name__ == "__main__":
    main()
