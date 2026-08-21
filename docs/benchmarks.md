# Feature Matrix & Performance Benchmarks

## Core Functional Modules

| Module | Entry Point | Input | Output | Runtime | Dependencies |
|--------|-------------|-------|--------|---------|--------------|
| Data Download | `src/download_data.py` | FHWA URLs | ZIP/TXT files | ~2–5 min | `requests` |
| Data Inspection | `src/inspect_data.py` | Raw text files | TXT/CSV reports | ~10 sec | `pandas` |
| Data Preparation | `src/prepare_data.py` | Raw text files | Parquet train/test | ~30 sec | `pandas`, `pyarrow` |
| Model Training | `src/train_models.py` | Parquet files | Joblib models | ~2–5 min | `scikit-learn`, `xgboost` |
| Model Evaluation | `src/evaluate_models.py` | Models + test data | Metrics + figures | ~1 min | `matplotlib`, `seaborn` |
| SHAP Explanation | `src/explain_model.py` | Models + test data | PNG + CSV | ~5–10 min | `shap`, `matplotlib` |
| Maintenance Ranking | `src/prioritize_bridges.py` | Models + test data | CSV + PNG | ~1 min | `pandas`, `matplotlib` |

## Model Performance Benchmarks (2024→2025 Test Data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Accuracy | False Negatives | FNR |
|-------|----------|-----------|--------|----|---------|--------|-------------------|-----------------|-----|
| Majority Baseline | 0.908 | 0.000 | 0.000 | 0.000 | 0.500 | 0.092 | 0.500 | 230 | 1.000 |
| Logistic Regression | 0.886 | 0.442 | 0.930 | 0.599 | 0.963 | 0.742 | 0.907 | 16 | 0.070 |
| Random Forest | 0.978 | 0.854 | 0.917 | 0.885 | 0.980 | 0.846 | 0.954 | 19 | 0.083 |
| XGBoost (threshold=0.846) | 0.981 | 0.882 | 0.913 | 0.897 | 0.983 | 0.854 | 0.960 | 20 | 0.087 |

**Test set:** 2,509 bridges  
**Positive class:** 230 bridges (9.2%)  
**Threshold selection:** Recall-oriented (≥80% recall on training data, highest F1 among candidates)

## Resource Footprint

| Resource | Estimate | Notes |
|----------|----------|-------|
| Disk (raw data) | ~100 MB | ZIP files for 3 years |
| Disk (processed) | ~2 MB | Parquet train/test splits |
| Disk (models) | ~50 MB | Joblib serialized objects |
| RAM (training) | ~2 GB | XGBoost + Random Forest |
| RAM (inference) | ~500 MB | Single model prediction |
| CPU (training) | ~5 min | 4-core modern CPU |
| CPU (inference) | ~1 sec | Full test set (2,509 bridges) |

## Feature Importance (SHAP Mean Absolute)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|---------------|----------------|
| 1 | `DECK_COND_058` | Highest | Current deck condition is strongest predictor |
| 2 | `bridge_age` | High | Older bridges deteriorate faster |
| 3 | `SUPERSTRUCTURE_COND_059` | High | Superstructure health correlates with deck |
| 4 | `SUBSTRUCTURE_COND_060` | Medium | Substructure issues indicate overall decline |
| 5 | `ADT_029` | Medium | Higher traffic accelerates wear |
| 6 | `PERCENT_ADT_TRUCK_109` | Medium | Truck traffic causes more damage |
| 7 | `STRUCTURE_LEN_MT_049` | Low-Medium | Longer bridges may have more exposure |
| 8 | `SCOUR_CRITICAL_113` | Low | Scour risk contributes to deterioration |
| 9 | `DECK_WIDTH_MT_052` | Low | Wider decks may have different stress patterns |
| 10 | `MAIN_UNIT_SPANS_045` | Low | More spans = more potential failure points |

*Full ranking: `reports/shap_feature_importance.csv`*

## Comparison with Industry Alternatives

| Criterion | This Project | Generic AutoML | Deep Learning (LSTM/CNN) | Commercial Bridge Mgmt |
|-----------|-------------|----------------|---------------------------|------------------------|
| **Primary model** | XGBoost (gradient boosting) | Varies | Neural networks | Proprietary |
| **Interpretability** | SHAP explanations | Limited | Black box | Limited |
| **Data requirement** | ~2,500 bridges | ~10,000+ | ~50,000+ | Varies |
| **Training time** | ~5 min | ~30 min | ~2–4 hrs | N/A |
| **Inference latency** | <1 sec | ~1 sec | ~500 ms | N/A |
| **Dependency count** | 10 | 15–25 | 20–40 | N/A |
| **Hardware** | CPU only | CPU + GPU | GPU required | Server |
| **Deployment** | Script / CLI | API service | API service | Web portal |
| **License** | MIT / Open | Varies | Proprietary | Proprietary |
| **Explainability** | First-class | Afterthought | None | Limited |

## Maintenance Priority Distribution (2025 Test Set)

| Priority Class | Count | Percentage | Threshold |
|----------------|-------|------------|-----------|
| High | 251 | 10.0% | Top 10% |
| Medium | 502 | 20.0% | Next 20% |
| Low | 1,756 | 70.0% | Remaining |

*Note: These are resource-allocation categories, not failure thresholds.*

## Data Quality Metrics

| Metric | 2023 | 2024 | 2025 |
|--------|-------|-------|-------|
| Total records | 2,521 | 2,518 | 2,542 |
| Duplicate bridge IDs | 0 | 0 | 0 |
| Missing deck condition | 413 (16.4%) | 413 (16.4%) | 419 (16.5%) |
| Missing superstructure | 407 (16.1%) | 407 (16.2%) | 414 (16.3%) |
| Missing substructure | 407 (16.1%) | 407 (16.2%) | 414 (16.3%) |
| Matched across all years | 2,509 | 2,509 | 2,509 |

## Target Class Balance

| Split | Positive (Poor) | Negative (Not Poor) | Positive Rate |
|-------|----------------|---------------------|---------------|
| Training (2023→2024) | 239 | 2,270 | 9.5% |
| Testing (2024→2025) | 230 | 2,279 | 9.2% |
