# Contributing Guidelines

**Author:** Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur  
**GitHub:** [DKS-MANAGER](https://github.com/DKS-MANAGER)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize civil engineering accuracy over algorithmic novelty

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/bridge_condition_xgboost.git
cd bridge_condition_xgboost
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate bridge_condition_xgboost
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Follow PEP 8 style guide
- Add docstrings to all functions
- Update `docs/data_dictionary.md` if adding features
- Update `docs/benchmarks.md` if changing model parameters

### 5. Run Quality Checks

```bash
# Lint
flake8 src/ --max-line-length=100
black --check src/
isort --check-only --profile black src/

# Reproduce pipeline
python src/download_data.py
python src/inspect_data.py
python src/prepare_data.py
python src/train_models.py
python src/evaluate_models.py
python src/explain_model.py
python src/prioritize_bridges.py
```

### 6. Commit

```bash
git add .
git commit -m "feat: add your feature description"
```

Use conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code restructuring
- `test:` — test additions
- `chore:` — maintenance

### 7. Push & Pull Request

```bash
git push origin feature/your-feature-name
```

Open a PR against `master` branch.

## Pull Request Checklist

- [ ] Code follows PEP 8
- [ ] All scripts run without errors
- [ ] `reports/pipeline_validation.txt` shows all `[PASS]`
- [ ] Documentation updated
- [ ] Benchmarks updated if model changed
- [ ] No large binary files committed

## Issue Reporting

See `.github/ISSUE_TEMPLATE/bug_report.md` and `.github/ISSUE_TEMPLATE/feature_request.md`.

## Review Process

1. Maintainer reviews PR within 7 days
2. CI must pass (lint, test, docker)
3. At least one approval required
4. Squash merge to master

## Development Setup

```bash
# Install in editable mode
pip install -e .

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/
isort --profile black src/
```
