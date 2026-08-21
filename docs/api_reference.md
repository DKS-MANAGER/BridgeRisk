# API, CLI & Usage Reference

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## Script Interface

All scripts are executed via Python CLI with no required arguments. Optional flags are documented below.

### `src/download_data.py`

Download FHWA NBI data for Maine (2023–2025).

```bash
python src/download_data.py
```

**Behavior:**
- Downloads 2023 and 2024 from FHWA ZIP archives
- Downloads 2025 as delimited text
- Saves to `data/raw/downloads/` and `data/raw/extracted/`
- Prints SHA256 checksums
- Skips existing files (no overwrite)

### `src/inspect_data.py`

Inspect raw data files and generate reports.

```bash
python src/inspect_data.py
```

**Output:**
- `reports/data_inspection.txt` — Full inspection log
- `reports/yearly_counts.csv` — Row counts per year
- `reports/matching_counts.csv` — Bridge ID matching statistics

### `src/prepare_data.py`

Clean data and create train/test splits.

```bash
python src/prepare_data.py
```

**Output:**
- `data/processed/train_2023_2024.parquet`
- `data/processed/test_2024_2025.parquet`
- `reports/data_preparation.txt`

### `src/train_models.py`

Train Logistic Regression, Random Forest, and XGBoost.

```bash
python src/train_models.py
```

**Output:**
- `models/logistic_regression_model.joblib`
- `models/random_forest_model.joblib`
- `models/xgboost_model.joblib`
- `models/preprocessing_pipeline.joblib`
- `models/model_parameters.json`

### `src/evaluate_models.py`

Evaluate models on test data and generate figures.

```bash
python src/evaluate_models.py
```

**Output:**
- `reports/model_metrics.csv`
- `reports/predictions_2025.csv`
- `reports/confusion_matrices.csv`
- `figures/roc_curve.png`
- `figures/precision_recall_curve.png`
- `figures/confusion_matrix.png`

### `src/explain_model.py`

Generate SHAP explanations for XGBoost.

```bash
python src/explain_model.py
```

**Output:**
- `figures/shap_bar.png`
- `figures/shap_summary.png`
- `figures/shap_high_risk.png`
- `figures/shap_low_risk.png`
- `reports/shap_feature_importance.csv`

### `src/prioritize_bridges.py`

Rank bridges by maintenance priority.

```bash
python src/prioritize_bridges.py
```

**Output:**
- `reports/maintenance_priority_2025.csv`
- `figures/predicted_risk_distribution.png`
- `figures/top20_priority.png`

## Configuration Reference

### `configs/priority_config.yaml`

```yaml
condition_severity_multiplier:
  poor: 3.0      # Deck condition ≤ 4
  fair: 2.0      # Deck condition 5–6
  good: 1.0      # Deck condition 7–9

traffic_multiplier:
  low: 1.0       # ADT < 1,000
  medium: 1.5    # ADT 1,000–10,000
  high: 2.0      # ADT > 10,000

scour_multiplier:
  unknown: 1.0   # Missing or unknown
  not_critical: 1.0  # Scour rating 0, N, U
  critical: 2.0  # Scour rating 1, 2, T

priority_thresholds:
  high: 0.10     # Top 10% = High priority
  medium: 0.20   # Next 20% = Medium priority
```

## Error Taxonomy & Troubleshooting

| Error Code | Symptom | Root Cause | Fix |
|-----------|---------|------------|-----|
| `E_DOWNLOAD_404` | HTTP 404 on data URL | FHWA URL changed or file moved | Verify URLs in `docs/data_selection.md`; update `src/download_data.py` |
| `E_DOWNLOAD_TIMEOUT` | Connection timeout | Network issues or FHWA server down | Retry with exponential backoff; check FHWA status page |
| `E_PARSE_MISSING_COL` | Column not found during parsing | NBI format changed across years | Inspect `data/raw/extracted/` headers; update column mapping |
| `E_PARSE_BRIDGE_ID` | Bridge ID mismatch | Leading/trailing spaces or encoding issues | Ensure `STRUCTURE_NUMBER_008` is stripped; check for non-ASCII chars |
| `E_TRAIN_NAN` | NaN values in training matrix | Missing values not imputed | Verify `SimpleImputer` in preprocessing pipeline |
| `E_TRAIN_CONVERGE` | Logistic Regression failed to converge | Max iterations too low or data not scaled | Increase `max_iter` to 2000; add `StandardScaler` |
| `E_EVAL_SHAPE` | Feature count mismatch | Preprocessor fitted on different columns | Ensure train/test feature alignment; refit preprocessor |
| `E_SHAP_MEMORY` | MemoryError during SHAP | Test set too large for TreeExplainer | Use `shap.sample()` to subset; reduce `n_samples` |
| `E_PRIORITY_KEY` | KeyError in priority scoring | Missing configuration key | Validate `configs/priority_config.yaml` against schema |

## Data Schema Contracts

### Training/Testing Parquet Schema

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `bridge_id` | string | No | `STATE_CODE_001 + "_" + STRUCTURE_NUMBER_008` |
| `STATE_CODE_001` | string | No | State code (23) |
| `STRUCTURE_NUMBER_008` | string | No | Bridge structure number |
| `DECK_COND_058` | string | Yes | Deck condition (0–9 or N) |
| `SUPERSTRUCTURE_COND_059` | string | Yes | Superstructure condition |
| `SUBSTRUCTURE_COND_060` | string | Yes | Substructure condition |
| `YEAR_BUILT_027` | float | Yes | Year built |
| `bridge_age` | float | No | `inspection_year - YEAR_BUILT_027` |
| `inspection_year` | int | No | 2023, 2024, or 2025 |
| `target_deck_poor_next_year` | int | No | 1 if next-year deck ≤ 4, else 0 |

## Programmatic Usage (Python API)

```python
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("models/xgboost_model.joblib")
preprocessor = joblib.load("models/preprocessing_pipeline.joblib")

# Load test data
test_df = pd.read_parquet("data/processed/test_2024_2025.parquet")

# Prepare features
feature_cols = [c for c in test_df.columns if c not in [
    "bridge_id", "target_deck_poor_next_year", "target_deck_cond_2025",
    "STATE_CODE_001", "STRUCTURE_NUMBER_008"
]]
X_test = preprocessor.transform(test_df[feature_cols])

# Predict probabilities
proba = model.predict_proba(X_test)[:, 1]
test_df["predicted_proba"] = proba

# Rank by predicted risk
ranked = test_df[["bridge_id", "predicted_proba"]].sort_values("predicted_proba", ascending=False)
print(ranked.head(10))
```

## Advanced Configuration

### Custom State Selection

Modify `src/download_data.py`:

```python
STATE = "VT"  # Vermont
STATE_CODE = "50"
FILES = {
    2023: {"url": f"{BASE_URL}/2023/50VT.txt", "filename": "50VT_2023.txt"},
    2024: {"url": f"{BASE_URL}/2024/50VT.txt", "filename": "50VT_2024.txt"},
    2025: {"url": f"{BASE_URL}/2025/delimited/50VT.txt", "filename": "50VT_2025.txt"},
}
```

### Custom Feature Set

Modify `FEATURES` dict in `src/train_models.py`:

```python
FEATURES = {
    "numerical": ["bridge_age", "ADT_029", "DECK_AREA"],
    "categorical": ["STRUCTURE_KIND_043A", "SCOUR_CRITICAL_113"],
}
```

### Custom XGBoost Hyperparameters

Modify `src/train_models.py` `get_model_params()`:

```python
"xgboost": {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": scale_pos_weight,
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (missing file, bad input) |
| 2 | Download failure (HTTP error, timeout) |
| 3 | Data validation failure (schema mismatch) |
| 4 | Model training failure |
| 5 | Evaluation failure |
