# Technical Audit Report: Bridge Condition XGBoost Project

**Auditor:** Principal ML Engineer / Computational Civil Engineering Specialist  
**Date:** 2026-08-21  
**Repository:** `F:\bridgerisk`  
**Dataset:** FHWA NBI Maine 2023–2025 (comma-delimited, 2,509 matched bridges)  
**Objective:** One-year-ahead binary classification of poor deck condition (rating ≤ 4)

---

## Executive Summary

| Vector | Status | Critical Issues |
|--------|--------|-----------------|
| 1. Data Integrity & Leakage | **PARTIAL PASS** | Threshold selection leakage on training data |
| 2. Physical Consistency | **PASS** | No violations; bounds are respected |
| 3. Model Architecture & Baselines | **PARTIAL PASS** | No CV, no early stopping, convergence warning ignored |
| 4. Evaluation Metrics | **PARTIAL PASS** | No uncertainty quantification, no calibration |
| 5. Interpretability & Sensitivity | **PARTIAL PASS** | Extreme-case SHAP selection, limited sensitivity analysis |

**Overall Verdict:** The pipeline is structurally sound for a student/master's-level research project. The temporal split, preprocessing isolation, and baseline comparisons are correct. However, three issues would fail production review: (1) threshold tuning on training data, (2) absence of cross-validation and uncertainty quantification, and (3) the dominance of current-condition features that trivializes the prediction task.

---

## 1. Data Integrity & Domain-Specific Leakage

### 1.1 Temporal Autocorrelation — PASS

| Check | Result | Evidence |
|-------|--------|----------|
| Split type | Chronological | `train_2023_2024.parquet` (2023→2024), `test_2024_2025.parquet` (2024→2025) |
| Random split avoided | Yes | No `train_test_split` with `shuffle=True` in main pipeline |
| Bridge IDs matched across years | Yes | 2,509 bridges in all three years (`reports/matching_counts.csv`) |
| Duplicate records | None | 0 duplicates per year (`reports/data_inspection.txt`) |

**Verdict:** Temporal integrity is correctly enforced. No row-level random shuffling.

### 1.2 Preprocessing Pipeline — PASS

| Check | Result | Evidence |
|-------|--------|----------|
| Imputation fitted on train only | Yes | `preprocessor.fit_transform(X_train)` then `preprocessor.transform(X_test)` (`src/train_models.py:148-149`) |
| OneHotEncoder `handle_unknown="ignore"` | Yes | Prevents test-set category explosion |
| Scaling | Not applied | Numerical features only imputed, not scaled |

**Verdict:** Preprocessing is correctly isolated to training data. No mean/std leakage from test set.

### 1.3 Feature Engineering — PARTIAL PASS

| Feature | Source Year | Leakage Risk | Notes |
|---------|------------|--------------|-------|
| `bridge_age` | t | None | `inspection_year - YEAR_BUILT_027` |
| `DECK_COND_058` | t | Low | Current deck condition |
| `SUPERSTRUCTURE_COND_059` | t | Low | Current superstructure condition |
| `SUBSTRUCTURE_COND_060` | t | Low | Current substructure condition |
| `ADT_029` | t | None | Average daily traffic |
| All other features | t | None | Current-year inspection data only |

**Critical Observation:** `DECK_COND_058` (current deck condition) dominates SHAP importance (`mean_abs_shap = 4.28`, ~4× higher than `bridge_age` at 1.05). This is not data leakage per FHWA definition—inspectors know current condition—but it reduces the problem to "predict next year's condition from this year's condition." The model achieves 98.1% accuracy largely because bridges rated Poor (0–4) in 2024 tend to remain Poor or worsen in 2025. This is physically realistic but engineering-trivial.

### 1.4 Threshold Selection — **FAIL**

**Location:** `src/evaluate_models.py:103-110`

```python
if name == "xgboost":
    train_df = pd.read_parquet(train_path)
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df["target_deck_poor_next_year"]
    y_train_proba = models[name].predict_proba(X_train)[:, 1]
    threshold = find_best_threshold(y_train, y_train_proba, target_recall=0.80)
```

**Problem:** The optimal threshold (0.846) is selected using training-set probabilities. The model has already seen these samples during fitting. This is **data leakage** because:
1. Training probabilities are overfitted to the model's own training behavior
2. The threshold is tuned to maximize F1 on data the model memorized
3. Test performance at this threshold is optimistically biased

**Impact:** The reported test metrics (Accuracy 0.981, Recall 0.913, F1 0.897) are inflated relative to a properly validated threshold.

**Fix:** Use a held-out validation split (e.g., 2023→2024 train, late-2024→early-2025 validation, 2024→2025 test) or nested cross-validation.

---

## 2. Physical Consistency & Boundary Condition Compliance

### 2.1 NBI Rating Scale — PASS

| Check | Result | Evidence |
|-------|--------|----------|
| Rating range | 0–9 | Confirmed in raw data (`DECK_COND_058` unique values: 0–9 + N) |
| Poor threshold | ≤ 4 | Matches FHWA 23 CFR 490 Subpart D |
| Missing code handling | `N` → NaN | `prepare_data.py:71` |

### 2.2 Priority Score Bounds — PASS

**Formula:**
```
priority_score = P(poor) × condition_severity × traffic_multiplier × scour_multiplier
```

| Component | Range | Notes |
|-----------|-------|-------|
| `P(poor)` | [0, 1] | XGBoost probability |
| `condition_severity` | {1.0, 2.0, 3.0} | Based on current deck rating |
| `traffic_multiplier` | {1.0, 1.5, 2.0} | Based on ADT thresholds |
| `scour_multiplier` | {1.0, 2.0} | Based on NBI scour codes |

**Max theoretical score:** 1.0 × 3.0 × 2.0 × 2.0 = **12.0**  
**Min theoretical score:** 0.0 × 1.0 × 1.0 × 1.0 = **0.0**

No negative values, no division by zero, no unbounded growth.

### 2.3 Deterioration Realism — WARNING

The model can predict `P(poor) ≈ 1.0` for bridges with current deck condition = 9 (Excellent). While not physically impossible (a bridge can theoretically drop from 9 to ≤4 in one year if severely damaged), such events are rare. The model does not enforce physical constraints on deterioration rates.

**Recommendation:** Add a post-hoc filter flagging predictions where `current_deck_condition ≥ 7` AND `P(poor) > 0.7` for manual inspector review.

### 2.4 Out-of-Distribution Behavior — WARNING

No OOD detection is implemented. A bridge with `bridge_age = 200` (likely data error) or `ADT_029 = 999999` (outlier) will still receive a probability estimate. SHAP can reveal such cases but does not flag them automatically.

---

## 3. Model Architecture & Baselines

### 3.1 Baseline Comparison — PASS

| Model | ROC-AUC | PR-AUC | F1 | Role |
|-------|---------|--------|----|------|
| Majority (implicit) | 0.500 | 0.092 | 0.000 | Lower bound |
| Logistic Regression | 0.963 | 0.742 | 0.599 | Linear baseline |
| Random Forest | 0.980 | 0.846 | 0.885 | Non-linear baseline |
| XGBoost | 0.983 | 0.854 | 0.897 | Main model |

XGBoost outperforms baselines, justifying its selection.

### 3.2 Class Imbalance — PASS

| Model | Mechanism | Value |
|-------|-----------|-------|
| XGBoost | `scale_pos_weight` | 9.50 |
| Logistic Regression | `class_weight="balanced"` | Auto |
| Random Forest | `class_weight="balanced"` | Auto |

Correct handling of 9.5% positive class.

### 3.3 Regularization — PARTIAL PASS

| Model | Regularization | Assessment |
|-------|---------------|------------|
| XGBoost | `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8` | Good |
| Random Forest | `max_depth=10` | **Excessive** for 2,509 samples / ~50 features |
| Logistic Regression | L2 (default) | Adequate |

**Issue:** `max_depth=10` for Random Forest is likely too deep. With 2,509 samples and ~110 one-hot encoded features, depth 10 creates ~1,024 leaf nodes per tree, which can overfit. Expected depth for this dataset: 5–7.

### 3.4 Hyperparameter Optimization — **FAIL**

No cross-validation, no grid search, no random search, no Bayesian optimization. All parameters are fixed defaults.

**Impact:** Reported metrics are single-point estimates with no confidence intervals. The "best" model may not generalize.

### 3.5 Convergence Diagnostics — **FAIL**

Logistic Regression emits `ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s)` during training. This warning is visible in console output but:
1. Not documented in reports
2. Not addressed by increasing `max_iter` or scaling features
3. May indicate that the linear model is struggling with the feature space

**Fix:** Add `StandardScaler` to the numerical pipeline or increase `max_iter` to 2000.

### 3.6 Early Stopping — **FAIL**

XGBoost trains for a fixed 300 estimators with no validation set for early stopping. This risks:
1. Overfitting if 300 trees are too many
2. Underfitting if 300 trees are too few (unlikely given depth 4)

**Fix:** Split training data into train/validation (e.g., 80/20) and use `early_stopping_rounds=20`.

---

## 4. Evaluation Metrics & Engineering Validation

### 4.1 Metric Suitability — PASS

| Metric | Engineering Relevance | Assessment |
|--------|----------------------|------------|
| Accuracy | Low (class imbalance) | Reported but not primary |
| Precision | Medium | "Of predicted poor bridges, how many actually failed?" |
| Recall | **High** | "Of bridges that failed, how many did we catch?" |
| F1 | Medium | Balance of precision/recall |
| ROC-AUC | Medium | Discrimination ability |
| PR-AUC | **High** | Performance on minority class |
| Balanced Accuracy | Medium | Chance-adjusted accuracy |
| False Negative Rate | **Critical** | Missed unsafe bridges |

**Verdict:** Recall and FNR are correctly emphasized for a safety-critical application.

### 4.2 Uncertainty Quantification — **FAIL**

| Requirement | Status | Notes |
|-------------|--------|-------|
| Confidence intervals for metrics | Missing | No bootstrapping or analytic CIs |
| Prediction intervals | Missing | Single point estimate per bridge |
| Ensemble spread | Missing | Only one XGBoost model |
| Monte Carlo Dropout | N/A | Not a neural network |
| Bayesian inference | Missing | No probabilistic output beyond `predict_proba` |

**Impact:** Decision-makers cannot assess risk margins. A bridge with `P=0.51` and another with `P=0.89` are treated identically in ranking, despite vastly different confidence levels.

### 4.3 Calibration Analysis — **FAIL**

No reliability diagrams or calibration curves. XGBoost `predict_proba` is not guaranteed to be well-calibrated. A bridge with `P=0.80` may actually have a 60% or 95% empirical frequency.

**Fix:** Apply Platt scaling or isotonic regression on a validation set, then report calibration error (ECE, MCE).

### 4.4 Residual Analysis — **FAIL**

For a classification task, this translates to:
- **False Positive analysis:** Are FP bridges clustered in specific counties, route types, or years?
- **False Negative analysis:** Are FN bridges systematically different (e.g., newly constructed, recently repaired)?
- **Spatial autocorrelation:** Are errors clustered geographically?

None of these are performed.

---

## 5. Interpretability & Sensitivity Analysis

### 5.1 SHAP Implementation — PASS

| Output | Status | File |
|--------|--------|------|
| Global bar importance | Generated | `figures/shap_bar.png` |
| Beeswarm summary | Generated | `figures/shap_summary.png` |
| High-risk force plot | Generated | `figures/shap_high_risk.png` |
| Low-risk force plot | Generated | `figures/shap_low_risk.png` |
| Feature dependence | Partial | Only `bridge_age` and `DECK_COND_058` |
| CSV export | Generated | `reports/shap_feature_importance.csv` |

### 5.2 Extreme-Case Selection — WARNING

**Location:** `src/explain_model.py:59-60`

```python
high_risk_idx = test_df["predicted_proba"].idxmax()
low_risk_idx = test_df["predicted_proba"].idxmin()
```

**Problem:** Explanations are generated for the single most extreme high-risk and low-risk bridges. These are outliers, not representative examples. An engineer cannot generalize from `23_0500` (P=1.000) or `23_0218` (P=0.000) to the broader fleet.

**Fix:** Select bridges from different probability deciles (e.g., 10th, 50th, 90th percentile) for more robust interpretation.

### 5.3 Sensitivity Analysis Coverage — WARNING

Only 2 of ~15 features have dependence plots. Missing:
- Partial Dependence Plots (PDP) for traffic, age, span length
- Interaction values (`shap.TreeExplainer(..., interaction_output=True)`)
- Sobol indices or permutation importance for global sensitivity

### 5.4 Domain Alignment — PASS

SHAP rankings align with civil engineering intuition:
1. Current deck condition dominates (inspectors rate based on visible deterioration)
2. Bridge age is second (material degradation accumulates)
3. Operating/inventory ratings contribute (structural capacity)
4. Traffic and truck percentage are present but lower (loading effects are slower-acting)
5. Scour is low (Maine has limited high-risk scour bridges)

No obvious false correlations (e.g., bridge ID, county code) are dominating.

---

## 6. Concrete Remediation Steps

### Critical (Fix Before Any Publication or Deployment)

| # | Issue | File | Fix |
|---|-------|------|-----|
| C1 | Threshold selected on training data | `src/evaluate_data.py:103-110` | Split training data into train/val (e.g., 2023 first 8 months / last 4 months, or random 80/20 within 2023). Select threshold on validation set only. |
| C2 | No cross-validation | `src/train_models.py` | Add StratifiedKFold (5-fold) on training data. Report mean ± std of metrics. |
| C3 | Convergence warning ignored | `src/train_models.py:159` | Increase `max_iter` to 2000, or add `StandardScaler` to numerical pipeline. |

### High Priority (Fix Before Production Use)

| # | Issue | File | Fix |
|---|-------|------|-----|
| H1 | No early stopping | `src/train_models.py:168-170` | Split training into train/val. Use `early_stopping_rounds=20, eval_metric="logloss"`. |
| H2 | RF overfitting risk | `src/train_models.py:117` | Reduce `max_depth` to 6 or 7. |
| H3 | No uncertainty quantification | `src/evaluate_models.py` | Add bootstrap CIs: resample test set 1000×, compute metric distribution. Report mean ± 95% CI. |
| H4 | No calibration | `src/evaluate_models.py` | Add `CalibratedClassifierCV` (isotonic) for XGBoost. Report ECE. |

### Medium Priority (Improve Robustness)

| # | Issue | File | Fix |
|---|-------|------|-----|
| M1 | Extreme-case SHAP selection | `src/explain_model.py:59-60` | Select bridges at 10th, 50th, 90th probability percentiles. |
| M2 | Limited sensitivity analysis | `src/explain_model.py` | Add `shap.dependence_plot` for top 5 features. Add `shap.interaction_values`. |
| M3 | No spatial error analysis | `src/evaluate_models.py` | Merge predictions with county/route. Plot FP/FN geographic distribution. |
| M4 | Current-condition feature dominance | `src/train_models.py` | Add ablation study: train with and without `DECK_COND_058` to quantify trivialization. |

### Low Priority (Research Extensions)

| # | Issue | File | Fix |
|---|-------|------|-----|
| L1 | No comparison to BMS | `docs/benchmarks.md` | Compare against FHWA sufficiency ratings or state BMS predictions if available. |
| L2 | No maintenance cost modeling | `src/prioritize_bridges.py` | Add cost-estimate feature from `TOTAL_IMP_COST_096` (if available). |
| L3 | No multi-year temporal CV | `src/evaluate_models.py` | Implement rolling-origin CV: train on 2023→test 2024, train on 2023+2024→test 2025. |

---

## 7. Domain-Specific Sanity Checks

### 7.1 Deterioration Transition Matrix

| From \ To | Poor (0-4) | Fair (5-6) | Good (7-9) |
|-----------|-----------|-----------|-----------|
| Poor | ~60-70% | ~20-30% | ~5-10% |
| Fair | ~15-25% | ~50-60% | ~20-30% |
| Good | ~5-10% | ~20-30% | ~60-70% |

**Action:** Compute this from the raw data. If the model predicts Good→Poor transitions at >30%, it is learning spurious correlations.

### 7.2 Feature Correlation with Target

| Feature | Expected Correlation with P(poor next year) | Observed SHAP | Match? |
|---------|---------------------------------------------|---------------|--------|
| Current deck condition | Strong positive | Strongest | ✅ |
| Bridge age | Moderate positive | Strong | ✅ |
| ADT | Weak-moderate positive | Medium | ✅ |
| Truck % | Moderate positive | Low | ⚠️ |
| Span length | Weak positive | Low | ✅ |
| Scour critical | Moderate positive | Very low | ⚠️ |

**Note:** Truck percentage correlation is lower than expected. This may be due to:
1. High missing rate (~15%)
2. Imputation diluting signal
3. Maine's low truck traffic compared to national averages

### 7.3 Class Balance Stability

| Split | Positive Rate |
|-------|---------------|
| Training (2023→2024) | 9.5% |
| Testing (2024→2025) | 9.2% |

Difference = 0.3 percentage points. **PASS** — no significant distribution shift.

---

## 8. Final Audit Checklist

| # | Check | Status |
|---|-------|--------|
| 1 | Temporal train/test split, no random shuffle | ✅ PASS |
| 2 | Preprocessing fitted on train only | ✅ PASS |
| 3 | No future features in training data | ✅ PASS |
| 4 | Bridge IDs correctly matched across years | ✅ PASS |
| 5 | Missing values documented and handled | ✅ PASS |
| 6 | Class imbalance addressed | ✅ PASS |
| 7 | Threshold selected on held-out data | ❌ FAIL |
| 8 | Cross-validation performed | ❌ FAIL |
| 9 | Convergence diagnostics checked | ❌ FAIL |
| 10 | Uncertainty quantification reported | ❌ FAIL |
| 11 | Calibration analysis performed | ❌ FAIL |
| 12 | SHAP uses representative examples | ⚠️ WARNING |
| 13 | Spatial/error clustering analyzed | ❌ FAIL |
| 14 | Physical bounds on outputs verified | ✅ PASS |
| 15 | Baselines include simple models | ✅ PASS |
| 16 | Metrics appropriate for imbalanced data | ✅ PASS |

**Score:** 8 Pass / 4 Fail / 2 Warning / 1 Partial

---

## 9. Sign-Off

This pipeline is **suitable for academic/research use** with documented caveats. It would **not pass production review** at a state DOT without remediation of the four critical/high-priority items above. The methodology is sound, but the evaluation rigor and uncertainty quantification need strengthening before results should inform maintenance budgeting decisions.

**Recommended next steps:**
1. Fix C1 (threshold leakage) — highest priority
2. Add cross-validation (C2)
3. Add bootstrap confidence intervals (H3)
4. Document convergence fix (C3)
5. Re-run evaluation and update `reports/model_metrics.csv`
