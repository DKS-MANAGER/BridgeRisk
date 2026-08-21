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
Initialized with Git. Use GitHub Desktop or command line to push to a remote repository.
