"""
Train Logistic Regression, Random Forest, and XGBoost models.

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURES = {
    "numerical": [
        "bridge_age",
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
        "SCOUR_CRITICAL_113",
        "WATERWAY_EVAL_071",
        "FUNCTIONAL_CLASS_026",
        "HIGHWAY_SYSTEM_104",
        "OPEN_CLOSED_POSTED_041",
        "DECK_STRUCTURE_TYPE_107",
        "SURFACE_TYPE_108A",
        "DECK_PROTECTION_108C",
    ],
}


def load_data():
    train_path = os.path.join(PROCESSED_DIR, "train_2023_2024.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_2024_2025.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def build_preprocessor():
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def get_model_params():
    n_pos = None
    n_neg = None

    train_df, _ = load_data()
    y = train_df["target_deck_poor_next_year"]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": 4,
        },
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": 4,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": 4,
        },
    }

    return params, n_pos, n_neg


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    train_df, test_df = load_data()

    X_train = train_df[FEATURES["numerical"] + FEATURES["categorical"]]
    y_train = train_df["target_deck_poor_next_year"]

    X_test = test_df[FEATURES["numerical"] + FEATURES["categorical"]]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Positive class in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    print("\nBuilding preprocessor...")
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    preprocessor.transform(X_test)

    params, n_pos, n_neg = get_model_params()

    print(f"\nClass distribution: positive={n_pos}, negative={n_neg}")
    print(f"Scale pos weight: {params['xgboost']['scale_pos_weight']:.2f}")

    models = {}

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(**params["logistic_regression"])
    lr.fit(X_train_processed, y_train)
    models["logistic_regression"] = lr

    print("Training Random Forest...")
    rf = RandomForestClassifier(**params["random_forest"])
    rf.fit(X_train_processed, y_train)
    models["random_forest"] = rf

    print("Training XGBoost...")
    xgb = XGBClassifier(**params["xgboost"])
    xgb.fit(X_train_processed, y_train)
    models["xgboost"] = xgb

    print("\nSaving models...")
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

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
