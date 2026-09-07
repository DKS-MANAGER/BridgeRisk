# BridgeRisk

One-year-ahead highway bridge deck deterioration prediction and maintenance prioritization using FHWA National Bridge Inventory (NBI) data, XGBoost, and SHAP.

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

---

## Problem

Highway bridge decks deteriorate over time under thermal cycling, moisture, de-icing salts, and heavy commercial vehicle axle loads. State Departments of Transportation (DOTs) conduct biennial bridge inspections rating deck condition on a 0–9 integer scale (FHWA NBI Item 58). Decks rated **$\le 4$** are officially categorized as **Poor** (structurally deficient, requiring structural rehabilitation, load restrictions, or deck replacement).

Predicting which bridges will deteriorate into Poor condition *one year in advance* allows transportation agencies to shift from reactive emergency repairs to planned, cost-effective preventative maintenance.

---

## Approach

```
NBI Inspection Data (2021–2025)
  ├── Clean & standardize field formats
  ├── Compute engineering features (bridge age, truck volume, load ratings)
  └── Link consecutive inspection cycles
       │
       ▼
Chronological Train/Test Partitioning
  ├── Training Transitions: 2021→2022, 2022→2023, 2023→2024 (13,618 records)
  └── Out-of-Time Test Set: 2024→2025 (4,558 bridges)
       │
       ▼
Machine Learning & Validation
  ├── Majority Baseline, Logistic Regression, Random Forest, XGBoost
  ├── Imbalance handling (class weighting & scale_pos_weight)
  └── Recall-oriented decision threshold selection (minimizing missed defects)
       │
       ▼
Explainability & Decision Support
  ├── SHAP feature attribution (global drivers & individual bridge waterfall plots)
  └── Multi-attribute Maintenance Prioritization (Risk × Severity × Traffic × Scour)
```

No random cross-validation row-splitting is used for evaluation: bridge condition is time-dependent, so random row splits leak future inspection data into training. Instead, we use a strict **out-of-time evaluation** (training on 2021–2024 transitions, testing on 2024–2025).

---

## Data

The dataset is derived from the complete official Federal Highway Administration (FHWA) National Bridge Inventory (NBI) database across **all 50 US States, District of Columbia, and Puerto Rico (2021–2025)**:
- **Freeze-Thaw & Salt Belts:** OH, PA, ME, VT, MI, NY, IL, etc.
- **Marine & Tropical Environments:** HI, FL, CA, PR, LA, TX, etc.
- **High-Density Freight Corridors:** Interstate highway freight networks across the US.

### Inventory Summary
| Dataset Split | Inspection Transitions | Bridge Records | Poor Decks Next Year | Positive Rate |
|:---|:---:|---:|---:|---:|
| **Nationwide Training Set** | 2021→2022, 2022→2023, 2023→2024 | **1,848,747** | 46,583 | 2.52% |
| **Nationwide Test Set (Out-of-Time)** | 2024→2025 (Unseen Future Year) | **619,220** | 14,847 | 2.40% |

---

## Features & Physical Causality

Training features for the XGBoost $P(\text{Poor})$ model are selected based on strict physical causality to superstructure deterioration, mapping FHWA NBI variables to mechanical and chemical degradation mechanisms:

1. **Longitudinal Degradation Dynamics:** 1-year condition velocity (`delta_deck_1yr`), 2-year condition velocity (`delta_deck_2yr`). Captures active degradation trajectories rather than static condition ratings.
2. **Mechanical Fatigue & Cumulative Load:** Estimated lifetime truck passes (`est_lifetime_truck_passes`), Average Daily Traffic (`ADT_029`), Truck Percentage (`PERCENT_ADT_TRUCK_109`). Heavy axle loads (ESALs) drive cyclic flexural wear, rutting, and micro-cracking.
3. **Material Kinetics (Durability):** `bridge_age` (Inspection Year − Year Built), Deck Protection (`DECK_PROTECTION_108C`), Wearing Surface (`SURFACE_TYPE_108A`), Deck Structure Type (`DECK_STRUCTURE_TYPE_107`), Material Kind (`STRUCTURE_KIND_043A`), Design Type (`STRUCTURE_TYPE_043B`). Govern chloride diffusion thresholds, moisture ingress, and rebar corrosion rates.
4. **Geometric Deflection & Slenderness:** Span-to-width ratio (`span_to_width_ratio`), Maximum Span Length (`MAX_SPAN_LEN_MT_048`), Total Length (`STRUCTURE_LEN_MT_049`), Deck Width (`DECK_WIDTH_MT_052`), Roadway Width (`ROADWAY_WIDTH_MT_051`), Number of Spans (`MAIN_UNIT_SPANS_045`), Deck Area. Longer unsupported spans undergo higher dynamic live-load deflections, accelerating transverse deck strain.
5. **Structural Boundary Conditions:** Deck Condition (`DECK_COND_058`), Superstructure Condition (`SUPERSTRUCTURE_COND_059`), Substructure Condition (`SUBSTRUCTURE_COND_060`), Operating Load Rating (`OPERATING_RATING_064`), Inventory Load Rating (`INVENTORY_RATING_066`). Establish existing baseline deficiency, secondary stress transfer from yielding supports, and remaining structural capacity.

> **Exclusion Criteria:** Administrative codes and purely hydraulic sub-surface failure modes (`SCOUR_CRITICAL_113`, `WATERWAY_EVAL_071`) are excluded from training and scoring to prevent spurious ML correlation with upper-level deck durability.

*See [`docs/data_dictionary.md`](docs/data_dictionary.md) for full field definitions and FHWA coding guides.*

---

## Model

We compare three models trained with identical preprocessor pipelines:
1. **Logistic Regression:** Linear baseline with balanced class weights.
2. **Random Forest:** Non-linear ensemble (200 trees, max depth 12, balanced class weights).
3. **Calibrated XGBoost:** Histogram-accelerated Extreme Gradient Boosting with Isotonic Probability Calibration (`cv=3`, max depth 5, learning rate 0.04, `scale_pos_weight` tuned to class imbalance).

---

## Results

Performance evaluated on the **Nationwide 2024→2025 out-of-time test set** (619,220 unseen bridges across all 50 states):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Negatives | FNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9281 | 0.2457 | **0.9661** | 0.3918 | 0.9904 | 0.8714 | **504** | **3.4%** |
| Random Forest | 0.9700 | 0.4404 | 0.9357 | 0.5989 | 0.9898 | 0.8903 | 954 | 6.4% |
| **Calibrated XGBoost** (threshold = 0.596) | **0.9955** | **0.9148** | 0.8959 | **0.9053** | **0.9916** | **0.8965** | 1,546 | 10.4% |

*Source: `reports/model_metrics.csv`*

### Confusion Matrix (Nationwide Test Set: 619,220 Bridges)
![Confusion Matrix](figures/confusion_matrix.png)

- Out of 14,847 bridges that actually deteriorated to Poor condition in 2025 across the US, Calibrated XGBoost correctly identified **13,301 (89.6% recall)**.
- XGBoost achieved **91.48% precision** (only 1,239 false alarms across 619k national bridges), producing an overall accuracy of **99.55%**.

### Discrimination Curves
| ROC Curve (ROC-AUC: 0.9916) | Precision-Recall Curve (PR-AUC: 0.8965) |
|:---:|:---:|
| ![ROC Curve](figures/roc_curve.png) | ![Precision-Recall Curve](figures/precision_recall_curve.png) |

---

## Explainability

Using TreeSHAP across the nationwide inventory, we interpret model predictions globally and locally.

| Mean \|SHAP\| Feature Importance | SHAP Summary Beeswarm Plot |
|:---:|:---:|
| ![SHAP Bar](figures/shap_bar.png) | ![SHAP Summary](figures/shap_summary.png) |

### Key Physical & Operational Drivers
1. **Current Deck Condition (`DECK_COND_058`):** Dominant baseline condition indicator.
2. **Longitudinal Degradation Rate (`delta_deck_1yr`):** Bridges exhibiting negative condition velocity in preceding inspections show elevated risk of crossing the structural deficiency threshold.
3. **Bridge Age & Fatigue Passes:** Cumulative operational years and heavy commercial truck volume accelerate flexural cracking and chloride penetration.
4. **Operating Load Rating (`OPERATING_RATING_064`):** Bridges with restricted permissible load capacity correlate strongly with impending deck deficiency.
5. **Deck Protection System (`DECK_PROTECTION_108C`):** Protective membranes and epoxy-coated rebar significantly retard corrosion kinetics.

---

## Structural Deck Maintenance Prioritization

Raw deterioration probabilities alone do not capture operational fatigue demands or structural capacity limits. In accordance with AASHTO Bridge Management System (BMS) principles, we compute a **Structural Deck Maintenance Priority Score**:

$$\text{Structural Priority Score} = P(\text{Poor Deck Next Year}) \times \text{Structural Consequence}$$

$$\text{Structural Consequence} = C_{\text{service}} \times C_{\text{capacity}} \times C_{\text{support}}$$

Where:
- **$P(\text{Poor Deck Next Year})$:** 1-year ML-predicted conditional probability of deck condition dropping to $\le 4$ (`DECK_COND_058`).
- **$C_{\text{service}}$ (Traffic & Dynamic Fatigue Demand):** Logarithmic vehicle volume scaled by commercial truck percentage:
  $$C_{\text{service}} = \left[1 + \log_{10}\left(1 + \frac{\text{ADT}}{100}\right)\right] \times \left(1 + \frac{\text{Truck \%}}{100}\right)$$
- **$C_{\text{capacity}}$ (Operating Load Capacity Vulnerability):** Bounded multiplier based on Operating Rating (`OPERATING_RATING_064`):
  - $1.5$ for restricted load rating ($< 20\text{ metric tons}$ / load-posted)
  - $1.2$ for marginal capacity ($20\text{–}30\text{ metric tons}$)
  - $1.0$ for standard operating capacity ($\ge 30\text{ metric tons}$)
- **$C_{\text{support}}$ (Superstructure & Substructure Integrity):** Bounded multiplier based on supporting element conditions (`SUPERSTRUCTURE_COND_059` & `SUBSTRUCTURE_COND_060`):
  - $1.4$ if supporting elements are in Poor condition ($\le 4$)
  - $1.2$ if supporting elements are in Fair condition ($5$)
  - $1.0$ if supporting elements are sound ($\ge 6$)

| Structural Risk by Priority Class | Top 20 Priority Bridges (2025) |
|:---:|:---:|
| ![Risk Distribution](figures/predicted_risk_distribution.png) | ![Top 20 Priority](figures/top20_priority.png) |

### Resource Allocation Summary (619,220 Test Bridges)
- **High Priority (Top 10% = 61,922 bridges):** Primary candidate list for immediate in-depth non-destructive evaluation (NDE), ultrasonic pulse velocity (UPV) testing, and capital rehabilitation budgeting.
- **Medium Priority (Next 20% = 123,845 bridges):** Candidate list for preventative maintenance (epoxy deck sealing, crack injection, joint replacement).
- **Low Priority (Remaining 70% = 433,453 bridges):** Standard biennial NBI inspection cycle.

*Full ranked bridge list exported to [`reports/maintenance_priority_2025.csv`](reports/maintenance_priority_2025.csv).*

*Full ranked bridge list exported to [`reports/maintenance_priority_2025.csv`](reports/maintenance_priority_2025.csv).*

---

## Limitations & Engineering Boundary Conditions

1. **Subjective Condition Ratings:** NBI condition ratings (0–9) represent visual inspector assessments rather than continuous physical sensor measurements (such as half-cell potential, concrete resistivity, or ultrasonic pulse velocity).
2. **One-Year Horizon:** The model predicts transitions over a single year; multi-year Markovian continuous degradation curves require longer historical panel tracking.
3. **Unrecorded Interventions:** Bridges that received unrecorded localized maintenance or patch repairs between official inspections may introduce label noise.
4. **Decision Support Tool:** Predictions are intended to optimize preventative maintenance budgets and guide inspection schedules—not to replace certified Professional Engineer (PE) structural ratings or load ratings.

---

## Project Structure

```text
BridgeRisk/
├── README.md                      # Project overview, methodology, and results
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment specification
├── .gitignore                     # Repository git ignore rules
├── data/
│   ├── README.md                  # Data source documentation
│   └── processed/
│       ├── train_2021_2024.parquet # Multi-year training transitions (13,618 records)
│       └── test_2024_2025.parquet  # Out-of-time test set (4,558 bridges)
├── src/
│   ├── download_data.py           # Automated FHWA NBI download utility
│   ├── inspect_data.py            # Inventory inspection and duplicate verification
│   ├── prepare_data.py            # Data cleaning, feature creation, and chronological linking
│   ├── train_models.py            # Trains Baseline, Random Forest, and XGBoost models
│   ├── evaluate_models.py         # Out-of-time evaluation and performance visualization
│   ├── explain_model.py           # SHAP global and local feature explanations
│   └── prioritize_bridges.py      # Multi-attribute maintenance priority ranking
├── notebooks/
│   └── bridge_risk_analysis.ipynb # Interactive exploratory analysis notebook
├── figures/                       # Generated evaluation plots and SHAP charts
└── reports/                       # Metrics CSVs, predictions, and priority rankings
```

---

## Running the Project & Data Acquisition

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/DKS-MANAGER/bridgerisk.git
cd bridgerisk

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download Data & Prepare Panel (Reproducible Workflow)
Large raw NBI ASCII files and processed `.parquet` datasets are excluded from Git version control (`.gitignore`) to keep the repository lightweight. You can download and generate the complete nationwide dataset (2021–2025) with two commands:

```bash
# 1. Download official FHWA NBI data (2021–2025 across all 50 states + DC + PR)
python src/download_data.py

# 2. Clean, compute degradation velocity features, and assemble train/test panels
python src/prepare_data.py
```

### 3. Run Pipeline (Train, Evaluate & Prioritize)
```bash
# Train baseline models, Random Forest, and Calibrated XGBoost
python src/train_models.py

# Evaluate metrics (ROC-AUC, PR-AUC, Confusion Matrix) on 2024->2025 test set
python src/evaluate_models.py

# Compute TreeSHAP feature attribution plots
python src/explain_model.py

# Generate Structural Deck Maintenance Priority rankings
python src/prioritize_bridges.py
```
