# Methodology

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## What Bridge Condition Rating Means
The National Bridge Inventory (NBI) assigns each bridge a **deck condition rating** from 0 to 9:
- **9** = Excellent
- **7–8** = Good
- **5–6** = Fair
- **0–4** = Poor

These ratings are assigned by certified bridge inspectors during routine inspections. They reflect the inspector's professional judgment about the deck's structural integrity, surface condition, and safety.

## Why Rating ≤ 4 Is Classified as Poor
The Federal Highway Administration (FHWA) uses the threshold **≤ 4** to define "Poor" condition in national performance reporting (23 CFR 490 Subpart D). Bridges rated Poor may have structural deficiencies, significant deterioration, or safety concerns that require monitoring or repair.

For this project, predicting `rating ≤ 4` means predicting bridges that have crossed from Fair into the Poor category — a meaningful signal for maintenance planners.

## How One-Year-Ahead Prediction Works
We use a **temporal chain**:
1. Year *t* inspection data → features
2. Year *t*+1 deck condition → target

Example:
- **Training:** 2023 features + 2024 target
- **Testing:** 2024 features + 2025 target

This mimics real-world decision making: an engineer in 2023 only knows the 2023 inspection results when deciding which bridges to repair before 2024.

## Why Chronological Splitting Is Used
Bridge condition is **not independent** across years — a bridge's 2024 condition depends on its 2023 condition. Randomly splitting rows (e.g., 80% train, 20% test) would leak future information into training and produce unrealistically optimistic accuracy. Chronological splitting respects time and produces honest performance estimates.

## What Data Leakage Means
Data leakage happens when a model sees information during training that would not be available at prediction time. Examples in bridge prediction:
- Using 2024 traffic data to predict 2024 deck condition
- Using next-year inspection results as an input feature
- Randomly mixing years in train/test splits

We avoid leakage by strictly using only year *t* features to predict year *t*+1.

## Why XGBoost Was Selected
XGBoost (Extreme Gradient Boosting) is a tree-based model that:
- Handles non-linear relationships (e.g., age × traffic interaction)
- Is robust to outliers and missing values
- Provides feature importance for explainability
- Works well on tabular data like NBI

It is more powerful than Logistic Regression but simpler than deep learning, fitting the project's scope.

## Why Logistic Regression and Random Forest Are Included
- **Logistic Regression:** A simple, interpretable baseline. If it performs nearly as well as XGBoost, the problem may be linearly separable and easy to explain.
- **Random Forest:** A non-parametric ensemble that captures interactions without the tuning complexity of gradient boosting. It provides a middle ground between LR and XGBoost.

Comparing all three shows whether the added complexity of XGBoost is justified.

## Why SHAP Is Used
SHAP (SHapley Additive exPlanations) is a game-theoretic method that assigns each feature a contribution value for every prediction. It is preferred over simple feature importance because:
- It explains **individual** predictions, not just global averages
- It is consistent and locally accurate
- It works with any model, including XGBoost

For a civil engineer, SHAP answers: *"Why did the model flag this specific bridge?"*

## Why Priority Score Is Not the Same as Failure Probability
The priority score combines:
1. **Predicted probability** (model output)
2. **Condition severity** (current deck rating)
3. **Traffic exposure** (ADT and truck percentage)
4. **Scour criticality** (foundation risk)

A bridge with a 60% probability but low traffic and no scour risk may have the same score as a bridge with 40% probability but extremely high traffic and critical scour. The score is a **decision tool** that balances deterioration risk with consequence, not a pure physical failure forecast.
