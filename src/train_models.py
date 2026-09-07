"""
Train Logistic Regression, Random Forest, and Calibrated XGBoost models on nationwide NBI data.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURES = {
    "numerical": [
        "bridge_age",
        "span_to_width_ratio",
        "est_lifetime_truck_passes",
        "delta_deck_1yr",
        "delta_deck_2yr",
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
        "DECK_COND_058",
        "SUPERSTRUCTURE_COND_059",
        "SUBSTRUCTURE_COND_060",
    ],
    "categorical": [
        "STRUCTURE_KIND_043A",
        "STRUCTURE_TYPE_043B",
        "FUNCTIONAL_CLASS_026",
        "HIGHWAY_SYSTEM_104",
        "OPEN_CLOSED_POSTED_041",
        "DECK_STRUCTURE_TYPE_107",
        "SURFACE_TYPE_108A",
        "DECK_PROTECTION_108C",
    ],
}


def load_data(sample_train=None):
    train_path = os.path.join(PROCESSED_DIR, "train_2021_2024.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if sample_train and len(train_df) > sample_train:
        # Stratified sample to train efficiently while preserving positive rate
        pos = train_df[train_df["target_deck_poor_next_year"] == 1]
        neg = train_df[train_df["target_deck_poor_next_year"] == 0].sample(n=sample_train - len(pos), random_state=42)
        train_df = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return train_df, test_df


def build_preprocessor():
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, FEATURES["numerical"]),
            ("cat", categorical_transformer, FEATURES["categorical"]),
        ]
    )

    return preprocessor


def get_model_params(y):
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    params = {
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "gamma": 0.5,
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": -1,
            "tree_method": "hist",
        },
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    }

    return params, n_pos, n_neg


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading nationwide data (1.8M transitions)...")
    train_df, test_df = load_data()

    X_train = train_df[FEATURES["numerical"] + FEATURES["categorical"]]
    y_train = train_df["target_deck_poor_next_year"]

    print(f"Nationwide Training transitions: {len(X_train):,d}")
    print(f"Nationwide Testing bridges: {len(test_df):,d}")
    print(f"Positive poor class in train: {y_train.sum():,d} ({y_train.mean()*100:.2f}%)")

    print("\nBuilding preprocessor...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    params, n_pos, n_neg = get_model_params(y_train)
    print(f"Scale pos weight: {params['xgboost']['scale_pos_weight']:.2f}")

    models = {}

    print("\n1. Training Logistic Regression Baseline...")
    lr = LogisticRegression(**params["logistic_regression"])
    lr.fit(X_train_processed, y_train)
    models["logistic_regression"] = lr

    print("2. Training Random Forest (Hist/Fast)...")
    rf = RandomForestClassifier(**params["random_forest"])
    rf.fit(X_train_processed, y_train)
    models["random_forest"] = rf

    print("3. Training Optimized XGBoost Classifier...")
    xgb_base = XGBClassifier(**params["xgboost"])
    xgb_base.fit(X_train_processed, y_train)

    print("4. Calibrating XGBoost Probabilities (Isotonic Regression)...")
    # In modern scikit-learn, use 3-fold cross-validation calibration
    calibrated_xgb = CalibratedClassifierCV(estimator=XGBClassifier(**params["xgboost"]), method="isotonic", cv=3)
    calibrated_xgb.fit(X_train_processed, y_train)
    models["xgboost"] = calibrated_xgb

    print("\nSaving trained models & pipeline...")
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    preprocessor_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Saved: {preprocessor_path}")

    params_path = os.path.join(MODELS_DIR, "model_parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {params_path}")

    print("\nNationwide training complete.")


if __name__ == "__main__":
    main()
