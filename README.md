---
title: "BridgeRisk: Explainable XGBoost for One-Year-Ahead Bridge Condition Prediction"
description: "FHWA NBI bridge deck deterioration prediction and maintenance prioritization using XGBoost + SHAP. Chronological train/test split (2023→2024→2025), Maine state data."
tags: [civil-engineering, bridge-management, infrastructure, nbi-data, xgboost, shap, maintenance-prioritization, fhwa]
---

<div align="center">

# Divyansh Kumar Singh (DKS)

### CFD & Hydraulics · Sediment Transport · Water Resources · Stochastic Hydrology · Bridge Infrastructure ML

📍 IIT Kanpur, Kanpur, UP, India &nbsp;|&nbsp; 📧 divyansh179@gmail.com

[![GitHub followers](https://img.shields.io/github/followers/DKS-MANAGER?style=social)](https://github.com/DKS-MANAGER)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=social&logo=linkedin)](https://www.linkedin.com/in/divyansh-kumar-singh-92bb621b6)

</div>

---

# Explainable XGBoost for One-Year-Ahead Bridge Condition Prediction and Maintenance Prioritization

## Civil-Engineering Problem
Bridge decks deteriorate over time due to traffic, weather, and age. Inspectors rate deck condition on a 0–9 scale, but predicting which bridges will become poor *next year* helps agencies prioritize limited maintenance funds.

## Research Question
Using National Bridge Inventory (NBI) inspection data from 2023, 2024, and 2025 for one U.S. state, can we predict whether a bridge deck will be in poor condition (rating ≤ 4) in year *t*+1 using only information available in year *t*?

## Data Source
- **Official FHWA NBI ASCII files:** https://www.fhwa.dot.gov/bridge/nbi.cfm
- **2023 data:** https://www.fhwa.dot.gov/bridge/nbi/ascii2023.cfm
- **2024 data:** https://www.fhwa.dot.gov/bridge/nbi/ascii2024.cfm
- **2025 data:** https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm
- **Format reference:** https://www.fhwa.dot.gov/bridge/nbi/format.cfm
- **Condition-rating reference:** https://www.fhwa.dot.gov/bridge/britab.cfm

## Selected State
**Maine (State Code: 23)**

## Number of Records Per Year
| Year | Records |
|------|---------|
| 2023 | 2,521 |
| 2024 | 2,518 |
| 2025 | 2,542 |

Bridges matched across all three years: **2,509**

## Data Cleaning Process
1. Downloaded official FHWA delimited files for Maine.
2. Standardized column names to FHWA field names (e.g., `DECK_COND_058`).
3. Stripped whitespace from bridge identifiers.
4. Converted NBI missing-value codes (`N`, blank) to `NaN`.
5. Removed duplicate bridge IDs (none found).
6. Matched bridges across years using `STATE_CODE_001 + STRUCTURE_NUMBER_008`.

## Feature List
- `bridge_age` = inspection_year - year_built
- `current_deck_condition` (Item 58)
- `current_superstructure_condition` (Item 59)
- `current_substructure_condition` (Item 60)
- `average_daily_traffic` (ADT_029)
- `average_daily_truck_traffic` (PERCENT_ADT_TRUCK_109)
- `bridge_length` (STRUCTURE_LEN_MT_049)
- `maximum_span_length` (MAX_SPAN_LEN_MT_048)
- `deck_width` (DECK_WIDTH_MT_052)
- `number_of_spans` (MAIN_UNIT_SPANS_045)
- `structure_type` (STRUCTURE_TYPE_043B)
- `structure_material` (STRUCTURE_KIND_043A)
- `scour_criticality` (SCOUR_CRITICAL_113)
- `waterway condition` (WATERWAY_EVAL_071)
- `inspection_year`

## Target Definition
`target_deck_poor_next_year = 1` if the bridge deck condition rating in year *t*+1 is **≤ 4** (Poor).  
`target_deck_poor_next_year = 0` if the rating is **> 4** (Fair or Good).

This follows the FHWA Poor condition threshold used in national bridge performance measures.

## Training/Testing Split
- **Training:** Features from 2023 → Target from 2024 (2,509 bridges)
- **Testing:** Features from 2024 → Target from 2025 (2,509 bridges)

No random row splitting is used for the main experiment.

## Models Used
1. **Majority-class baseline** (predicts the most common class)
2. **Logistic Regression** (simple linear model)
3. **Random Forest** (ensemble of decision trees)
4. **XGBoost Classifier** (gradient-boosted trees, main model)

## Model Performance (2024→2025 Test Data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | False Negatives | FNR |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Majority Baseline | 0.908 | 0.000 | 0.000 | 0.000 | 0.500 | 0.092 | 230 | 100.0% |
| Logistic Regression | 0.886 | 0.442 | 0.930 | 0.599 | 0.963 | 0.742 | 16 | 7.0% |
| Random Forest | 0.978 | 0.854 | 0.917 | 0.885 | 0.980 | 0.846 | 19 | 8.3% |
| **XGBoost** (threshold=0.846) | **0.981** | **0.882** | **0.913** | **0.897** | **0.983** | **0.854** | **20** | **8.7%** |

> **Note:** XGBoost threshold was selected to prioritize recall (≥80% on training data). False Negative Rate (FNR) is the critical safety metric — it measures bridges the model missed that became poor in 2025.

## Key Visualizations

### ROC Curves
![ROC Curve](figures/roc_curve.png)

### Precision-Recall Curves
![Precision-Recall Curve](figures/precision_recall_curve.png)

### Confusion Matrix (XGBoost)
![Confusion Matrix](figures/confusion_matrix.png)

### SHAP Feature Importance
![SHAP Feature Importance](figures/shap_bar.png)

### SHAP Summary Plot
![SHAP Summary](figures/shap_summary.png)

### SHAP High-Risk Bridge Explanation
![SHAP High-Risk](figures/shap_high_risk.png)

### SHAP Low-Risk Bridge Explanation
![SHAP Low-Risk](figures/shap_low_risk.png)

### Predicted Risk Distribution by Priority Class
![Predicted Risk Distribution](figures/predicted_risk_distribution.png)

### Top 20 Maintenance Priority Bridges (2025)
![Top 20 Priority Bridges](figures/top20_priority.png)

## Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** Of bridges predicted poor, how many actually became poor?
- **Recall:** Of bridges that became poor, how many did we catch?
- **F1-score:** Balance of precision and recall
- **ROC-AUC:** Ability to discriminate poor vs. non-poor bridges
- **PR-AUC:** Performance on the imbalanced positive class
- **Balanced accuracy:** Accuracy adjusted for class imbalance
- **False-negative count & rate:** Bridges missed by the model (critical for safety)

## SHAP Explanation
SHAP (SHapley Additive exPlanations) shows how each feature pushed the model's prediction up or down for individual bridges. It does **not** prove physical causation; it explains the model's learned pattern.

## Maintenance-Priority Method
Bridges are ranked using a transparent score:

```
priority_score = predicted_probability × condition_severity_multiplier × traffic_multiplier × scour_multiplier
```

Multipliers are defined in `configs/priority_config.yaml`. Priority classes are resource-allocation categories, not failure thresholds:
- **High:** Top 10%
- **Medium:** Next 20%
- **Low:** Remaining 70%

## Limitations
See `docs/limitations.md`.

## Future Work: Production Readiness

This project is a **research prototype** designed for academic demonstration. The following steps would be required to transition it to a production system used by a state DOT or bridge management agency:

### 1. Data Pipeline & Automation
- **Automated data ingestion:** Scheduled downloads of annual NBI data with schema validation and anomaly detection (e.g., sudden drops in bridge counts, new condition codes).
- **Data quality monitoring:** Dashboards tracking missing-value rates, duplicate records, and ID-matching success across years.
- **Versioned data lake:** Store raw and processed data in a versioned format (e.g., Delta Lake, Iceberg) with data contracts and schema evolution tracking.

### 2. Model Robustness & Validation
- **Temporal cross-validation:** Implement rolling-origin CV (train on 2023→test 2024, train on 2023+2024→test 2025) to estimate out-of-time performance stability.
- **Uncertainty quantification:** Add bootstrap confidence intervals, prediction intervals, or Bayesian neural network alternatives so decision-makers understand risk margins.
- **Probability calibration:** Apply Platt scaling or isotonic regression and report calibration error (ECE, MCE) to ensure predicted probabilities match empirical frequencies.
- **Spatial validation:** Evaluate performance by county, route type, and climate zone to detect geographic bias.

### 3. Model Architecture Enhancements
- **Multi-state training:** Extend beyond Maine to 5–10 diverse states (different climates, traffic volumes, bridge inventories) to improve generalization.
- **Temporal models:** Incorporate LSTM or Transformer architectures to capture multi-year deterioration sequences, not just single-year snapshots.
- **Physics-informed constraints:** Enforce physical bounds on deterioration rates (e.g., a bridge cannot drop from condition 9 to ≤4 in one year without catastrophic event flags).
- **Ensemble methods:** Combine XGBoost with survival analysis (Cox proportional hazards, random survival forests) to predict time-to-poor-condition rather than binary one-year-ahead classification.

### 4. Feature Engineering & Data Enrichment
- **Environmental data:** Integrate freeze-thaw cycle counts, precipitation, salinity (coastal corrosion), and temperature extremes from NOAA or PRISM datasets.
- **Maintenance history:** Link to state DOT work-order databases to distinguish "natural deterioration" from "post-maintenance jumps" in condition.
- **Traffic loading spectra:** Replace scalar ADT with WIM (Weigh-In-Motion) data for actual load spectra and fatigue accumulation.
- **Material-specific models:** Train separate models for concrete, steel, and timber bridges, as deterioration mechanisms differ fundamentally.

### 5. Explainability & Civil Engineering Validation
- **Domain expert validation:** Present SHAP explanations to certified bridge inspectors and ask whether the model's "reasons" align with engineering judgment.
- **Counterfactual analysis:** Generate "what-if" scenarios (e.g., "If this bridge were 10 years younger, its predicted probability drops from 78% to 32%").
- **Failure mode attribution:** Link SHAP features to specific deterioration mechanisms (corrosion, fatigue, delamination, scour) rather than generic NBI codes.

### 6. Maintenance Decision Support
- **Cost-benefit optimization:** Replace heuristic priority multipliers with a constrained optimization model that maximizes "years of service restored per dollar" under budget constraints.
- **Remaining service life (RSL) integration:** Calibrate model outputs against existing RSL estimates from state BMS software.
- **Multi-criteria decision analysis:** Incorporate agency priorities (political, economic, social) via Analytic Hierarchy Process (AHP) or PROMETHEE methods.
- **Interactive dashboard:** Deploy a web interface (Streamlit, Plotly Dash) allowing inspectors to query bridges, adjust thresholds, and simulate budget scenarios.

### 7. Deployment & Governance
- **CI/CD for ML:** Automated retraining pipelines triggered when new inspection data arrives (annual or bi-annual).
- **Model monitoring:** Track prediction drift, feature drift, and performance degradation over time. Alert when model accuracy drops below acceptable thresholds.
- **Human-in-the-loop:** Design workflow where model high-priority flags are reviewed by inspectors before entering the official maintenance backlog.
- **Audit trail:** Log every prediction, feature value, and SHAP explanation for regulatory compliance and liability protection.

### 8. Regulatory & Standards Compliance
- **FHWA compliance:** Align with 23 CFR 490 Subpart D performance measures and emerging SNBI (Specifications for the NBI) data standards.
- **State BMS integration:** Export results in formats compatible with existing Bridge Management Systems (e.g., AASHTOWare BrM, Pontis).
- **Documentation standards:** Produce a formal Engineering Analysis Report following state DOT documentation guidelines.

## Current Project Status
This prototype demonstrates feasibility and methodology. It is **not** intended for direct maintenance decision-making without the validation, calibration, and governance steps outlined above.

## How to Run the Project From the Beginning

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate bridge_condition_xgboost

# 2. Download data
python src/download_data.py

# 3. Inspect data
python src/inspect_data.py

# 4. Prepare data
python src/prepare_data.py

# 5. Train models
python src/train_models.py

# 6. Evaluate models
python src/evaluate_models.py

# 7. Explain model (requires shap)
python src/explain_model.py

# 8. Prioritize bridges
python src/prioritize_bridges.py
```

## Git Repository
Initialized with Git by [DKS-MANAGER](https://github.com/DKS-MANAGER). Use GitHub Desktop or command line to push to a remote repository.

**Author:** [DKS-MANAGER](https://github.com/DKS-MANAGER) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
