import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def load_data_and_models():
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")
    test_df = pd.read_parquet(test_path)
    
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
    
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
        models[name] = joblib.load(path)
    
    return test_df, preprocessor, models


def find_best_threshold(y_true, y_proba, target_recall=0.80):
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
        "false_negatives": confusion_matrix(y, y_pred)[1, 0],
        "false_negative_rate": confusion_matrix(y, y_pred)[1, 0] / (confusion_matrix(y, y_pred)[1, 0] + confusion_matrix(y, y_pred)[1, 1]) if (confusion_matrix(y, y_pred)[1, 0] + confusion_matrix(y, y_pred)[1, 1]) > 0 else 0,
    }
    
    return metrics, y_pred, y_proba, confusion_matrix(y, y_pred)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading data and models...")
    test_df, preprocessor, models = load_data_and_models()
    
    feature_cols = [c for c in test_df.columns if c not in ["bridge_id", "target_deck_poor_next_year", "target_deck_cond_2025", "STATE_CODE_001", "STRUCTURE_NUMBER_008"]]
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df["target_deck_poor_next_year"]
    
    print(f"Test samples: {len(y_test)}")
    print(f"Positive class: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    all_metrics = []
    predictions = {}
    
    for name in MODEL_NAMES:
        print(f"\nEvaluating {name}...")
        
        threshold = 0.5
        if name == "xgboost":
            train_path = os.path.join(PROCESSED_DIR, "train_2023_2024.parquet")
            train_df = pd.read_parquet(train_path)
            X_train = preprocessor.transform(train_df[feature_cols])
            y_train = train_df["target_deck_poor_next_year"]
            y_train_proba = models[name].predict_proba(X_train)[:, 1]
            threshold = find_best_threshold(y_train, y_train_proba, target_recall=0.80)
            print(f"  Selected recall-oriented threshold: {threshold:.3f}")
        
        metrics, y_pred, y_proba, cm = evaluate_model(models[name], X_test, y_test, threshold, name)
        all_metrics.append(metrics)
        predictions[name] = {"pred": y_pred, "proba": y_proba}
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"  False negatives: {metrics['false_negatives']}")
        print(f"  False negative rate: {metrics['false_negative_rate']:.3f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"), index=False)
    print(f"\nSaved metrics to reports/model_metrics.csv")
    
    pred_df = test_df[["bridge_id"]].copy()
    for name in MODEL_NAMES:
        pred_df[f"{name}_pred"] = predictions[name]["pred"]
        pred_df[f"{name}_proba"] = predictions[name]["proba"]
    pred_df["true_label"] = y_test.values
    pred_df.to_csv(os.path.join(REPORTS_DIR, "predictions_2025.csv"), index=False)
    print(f"Saved predictions to reports/predictions_2025.csv")
    
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
    pd.DataFrame(cm_data).to_csv(os.path.join(REPORTS_DIR, "confusion_matrices.csv"), index=False)
    print(f"Saved confusion matrices to reports/confusion_matrices.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, name in enumerate(MODEL_NAMES):
        cm = confusion_matrix(y_test, predictions[name]["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
        axes[idx].set_title(f"{name}\nThreshold={all_metrics[idx]['threshold']:.2f}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to figures/confusion_matrix.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_test, predictions[name]["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={all_metrics[MODEL_NAMES.index(name)]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (2024 -> 2025 Test Data)")
    ax.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"Saved ROC curve to figures/roc_curve.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(y_test, predictions[name]["proba"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC={all_metrics[MODEL_NAMES.index(name)]['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (2024 -> 2025 Test Data)")
    ax.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print(f"Saved PR curve to figures/precision_recall_curve.png")
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
