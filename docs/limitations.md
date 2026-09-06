# Limitations

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## 1. NBI Condition Data Are Inspection Ratings, Not Direct Structural Measurements
The NBI deck condition is a **subjective professional rating** (0–9), not a direct measurement of material strength, crack width, or load capacity. Two inspectors may rate the same bridge differently. The model learns patterns in these ratings, not in physical reality.

## 2. One-Year Prediction Does Not Prove Causation
The model identifies correlations between year $t$ features and year $t+1$ conditions. It does not prove that, for example, "older bridges always deteriorate faster." Unobserved factors (maintenance spending, localized weather events, inspector variance) may drive both features and outcomes.

## 3. Geographic & Regional Coverage
While the model incorporates three diverse states (Maine, Hawaii, Delaware) representing distinct climates (freeze-thaw, marine tropical, mid-Atlantic transit), caution is still required when generalizing to states with arid climates or dramatically larger inventories (e.g., Texas, California).

## 4. Missing Values and Coding Assumptions Affect Results
- Missing condition values (coded as `N` or blank) are treated as `NaN` and imputed (median for numerical, most-frequent for categorical).
- Bridges that received unrecorded maintenance between inspections may introduce label noise.

## 5. Maintenance Multipliers Are Decision Assumptions
The priority score multipliers in `src/prioritize_bridges.py` were chosen for transparency and engineering defensibility, not calibrated through cost-benefit optimization. Different transportation agencies may reasonably choose different multipliers based on local policy, traffic thresholds, and budget constraints.

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
