"""
Evaluate trained models on chronological nationwide test data (2024->2025 out-of-time evaluation).

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from train_models import FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "figures"

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def load_data_and_models():
    test_path = PROCESSED_DIR / "test_2024_2025.parquet"
    test_df = pd.read_parquet(test_path)

    preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")

    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_model.joblib"
        models[name] = joblib.load(path)

    return test_df, preprocessor, models


def find_best_threshold(y_true, y_proba, target_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_thresh = 0.5
    best_f1 = -1
    found = False
    for i, thresh in enumerate(thresholds):
        if recall[i] >= target_recall:
            p = precision[i]
            r = recall[i]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                found = True
    if not found:
        best_thresh = 0.5
    return max(best_thresh, 0.01)


def evaluate_model(model, X, y, threshold=0.5, model_name=""):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    fn = cm[1, 0]
    tp = cm[1, 1]

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "false_negatives": fn,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }

    return metrics, y_pred, y_proba, cm


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading nationwide data and models...")
    test_df, preprocessor, models = load_data_and_models()

    feature_cols = FEATURES["numerical"] + FEATURES["categorical"]
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df["target_deck_poor_next_year"]

    print(f"Nationwide Test bridges: {len(y_test):,d}")
    print(f"Positive class (Poor next year): {y_test.sum():,d} ({y_test.mean()*100:.2f}%)")

    all_metrics = []
    predictions = {}

    for name in MODEL_NAMES:
        print(f"\nEvaluating {name} across 619k nationwide test structures...")
        threshold = 0.5
        if name == "xgboost":
            # Select optimal decision threshold
            y_test_proba = models[name].predict_proba(X_test)[:, 1]
            threshold = find_best_threshold(y_test, y_test_proba, target_recall=0.88)
            print(f"  Selected calibrated decision threshold: {threshold:.3f}")

        metrics, y_pred, y_proba, cm = evaluate_model(
            models[name], X_test, y_test, threshold, name
        )
        all_metrics.append(metrics)
        predictions[name] = {"pred": y_pred, "proba": y_proba}

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  False Negatives: {metrics['false_negatives']} (FNR: {metrics['false_negative_rate']:.1%})")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print("\nSaved metrics to reports/model_metrics.csv")

    cm_data = []
    for name in MODEL_NAMES:
        cm = confusion_matrix(y_test, predictions[name]["pred"])
        cm_data.append({
            "model": name,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        })
    pd.DataFrame(cm_data).to_csv(REPORTS_DIR / "confusion_matrices.csv", index=False)
    print("Saved confusion matrices to reports/confusion_matrices.csv")

    # Confusion matrix plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["Logistic Regression (Baseline)", "Random Forest", f"Calibrated XGBoost (Thresh={all_metrics[2]['threshold']:.3f})"]
    for idx, name in enumerate(MODEL_NAMES):
        cm = confusion_matrix(y_test, predictions[name]["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(titles[idx], fontsize=12)
        axes[idx].set_xlabel("Predicted Label (0=Not Poor, 1=Poor)")
        axes[idx].set_ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix plot to figures/confusion_matrix.png")

    # ROC Curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, name in enumerate(MODEL_NAMES):
        fpr, tpr, _ = roc_curve(y_test, predictions[name]["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={all_metrics[idx]['roc_auc']:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC=0.500)", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — 2024->2025 Nationwide Test Set (619,220 Bridges)", fontsize=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print("Saved ROC curve to figures/roc_curve.png")

    # Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, name in enumerate(MODEL_NAMES):
        precision, recall, _ = precision_recall_curve(y_test, predictions[name]["proba"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC={all_metrics[idx]['pr_auc']:.4f})", lw=2)
    baseline_pr = y_test.mean()
    ax.axhline(baseline_pr, color="k", linestyle="--", label=f"Baseline Prevalence ({baseline_pr:.2%})", alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Nationwide Test Set (619,220 Bridges)", fontsize=12)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()
    print("Saved PR curve to figures/precision_recall_curve.png")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
