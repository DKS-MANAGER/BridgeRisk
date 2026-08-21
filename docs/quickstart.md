# Deterministic Quickstart & Setup

## Prerequisite Matrix

| Dependency | Minimum Version | Recommended | Validation Command |
|------------|----------------|-------------|-------------------|
| Python | 3.10 | 3.11 | `python --version` |
| Conda | 4.12 | Latest | `conda --version` |
| Git | 2.30 | Latest | `git --version` |
| RAM | 4 GB | 8 GB | `free -h` / Task Manager |
| Disk | 500 MB | 1 GB | `df -h` / Explorer |
| OS | Windows 10+, macOS 12+, Linux (Ubuntu 20.04+) | Latest stable | `uname -a` |

## One-Liner Installation

### macOS / Linux
```bash
conda env create -f environment.yml && conda activate bridge_condition_xgboost
```

### Windows (PowerShell)
```powershell
conda env create -f environment.yml; conda activate bridge_condition_xgboost
```

### Docker (if containerized)
```bash
docker build -t bridge-condition-xgboost . && docker run --rm -v ${PWD}:/app bridge-condition-xgboost
```

### Pip-only (no Conda)
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
# FHWA Data Source
FHWA_BASE_URL=https://www.fhwa.dot.gov/bridge/nbi
NBI_STATE_CODE=23
NBI_STATE_ABBR=ME

# Data Paths
RAW_DOWNLOAD_DIR=data/raw/downloads
RAW_EXTRACT_DIR=data/raw/extracted
PROCESSED_DIR=data/processed

# Model Parameters
RANDOM_STATE=42
N_ESTIMATORS=300
MAX_DEPTH=4
LEARNING_RATE=0.05
SUBSAMPLE=0.8
COLSAMPLE_BYTREE=0.8
SCALE_POS_WEIGHT=9.5

# Priority Configuration
CONDITION_SEVERITY_POOR=3.0
CONDITION_SEVERITY_FAIR=2.0
CONDITION_SEVERITY_GOOD=1.0
TRAFFIC_MULTIPLIER_LOW=1.0
TRAFFIC_MULTIPLIER_MEDIUM=1.5
TRAFFIC_MULTIPLIER_HIGH=2.0
SCOUR_MULTIPLIER_CRITICAL=2.0
HIGH_PRIORITY_THRESHOLD=0.10
MEDIUM_PRIORITY_THRESHOLD=0.20

# Execution
N_JOBS=4
```

## Reproducibility Verification

```bash
# Step 1: Download data
python src/download_data.py

# Step 2: Inspect data
python src/inspect_data.py

# Step 3: Prepare data
python src/prepare_data.py

# Step 4: Train models
python src/train_models.py

# Step 5: Evaluate models
python src/evaluate_models.py

# Step 6: Explain model
python src/explain_model.py

# Step 7: Prioritize bridges
python src/prioritize_bridges.py

# Step 8: Validate pipeline
cat reports/pipeline_validation.txt
```

## Hardware Constraints

| Constraint | Minimum | Recommended |
|------------|---------|-------------|
| CPU cores | 2 | 4+ |
| RAM | 4 GB | 8 GB |
| Disk space | 500 MB | 1 GB |
| GPU | Not required | Optional (XGBoost GPU) |
| Network | 10 Mbps | 50 Mbps (for downloads) |

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Conda environment activation: `conda activate bridge_condition_xgboost`
- Paths use backslashes; scripts normalize internally

### macOS / Linux
- Use Bash or Zsh
- Conda environment activation: `conda activate bridge_condition_xgboost`
- Ensure `file` command is available for MIME type detection

### Docker
```dockerfile
FROM continuumio/miniconda3:latest
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml
ENV PATH /opt/conda/envs/bridge_condition_xgboost/bin:$PATH
COPY . /app
WORKDIR /app
CMD ["python", "src/download_data.py"]
```
