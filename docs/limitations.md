# Limitations

## 1. NBI Condition Data Are Inspection Ratings, Not Direct Structural Measurements
The NBI deck condition is a **subjective professional rating** (0–9), not a direct measurement of material strength, crack width, or load capacity. Two inspectors may rate the same bridge differently. The model learns patterns in these ratings, not in physical reality.

## 2. One-Year Prediction Does Not Prove Causation
The model identifies correlations between 2023 features and 2024 conditions. It does not prove that, for example, "older bridges always deteriorate faster." Unobserved factors (maintenance spending, weather events, inspection standards) may drive both the features and the target.

## 3. State-Specific Data May Not Generalize
Maine has a distinct climate (freeze-thaw cycles, coastal corrosion), traffic volume, and bridge inventory composition. A model trained on Maine bridges may not perform well in Texas, Florida, or urban states with different bridge types and loading patterns.

## 4. Missing Values and Coding Assumptions Affect Results
- ~16% of deck condition values are missing (`N` or blank) in each year.
- Missing values are imputed (median for numerical, most-frequent for categorical), which may dilute true signal.
- Some bridges may have been repaired between inspections but not recorded in the NBI.

## 5. Maintenance Multipliers Are Decision Assumptions
The priority score multipliers (`configs/priority_config.yaml`) were chosen for transparency, not calibrated through engineering optimization. Different agencies would reasonably choose different values based on local policy, risk tolerance, and budget constraints.

## 6. The Model Should Support, Not Replace, Professional Inspection
This tool is intended to:
- Highlight bridges that may need closer review
- Support resource-allocation discussions
- Provide a reproducible baseline for future work

It is **not** intended to:
- Override inspector judgment
- Predict exact failure dates
- Replace physical bridge inspections or structural analysis

## 7. Class Imbalance and Metric Interpretation
Only ~9.5% of bridges become Poor in one year. A model that always predicts "Not Poor" would achieve ~90% accuracy but zero recall. Metrics like PR-AUC and recall are more informative than accuracy for this imbalanced problem, but they still simplify complex engineering trade-offs.
