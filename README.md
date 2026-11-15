# AI Model Health Monitor

An end-to-end, local-first MLOps project for customer churn prediction.

The system will progressively implement:
1. Data ingestion, validation and preprocessing
2. Model training and evaluation
3. MLflow experiment tracking and model versioning
4. FastAPI prediction service
5. Docker packaging
6. Data and model monitoring with Evidently
7. Streamlit monitoring dashboard
8. GitHub Actions CI
9. Controlled retraining, model promotion and rollback

> Current status: Phase 2 — model training. Features are added phase by phase after their tests pass.

## Technology stack

Python, pandas, NumPy, scikit-learn, MLflow, Evidently, FastAPI, SQLite, Docker, Streamlit, Pytest and GitHub Actions.

## Project structure

```text
ai-model-health-monitor/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── docs/
├── models/
├── reports/
├── src/
│   ├── data/
│   ├── training/
│   ├── monitoring/
│   ├── serving/
│   └── retraining/
├── dashboard/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or 3.12
- Git
- VS Code (recommended)

Docker will be introduced in the deployment phase.

## Setup (Windows — Git Bash)

```bash
git clone https://github.com/YOUR-USERNAME/ai-model-health-monitor.git
cd ai-model-health-monitor

python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use Windows Command Prompt instead of Git Bash, activate with:
```bat
.venv\Scripts\activate
```

## Configuration

Copy `.env.example` to `.env` and update values only when needed. Never commit `.env`, credentials, private customer data, model binaries or generated datasets.

## Phase 2 — Model training

Download the IBM Telco Customer Churn CSV and save it as `data/raw/telco_customer_churn.csv`. The file is gitignored.

```bash
# 1. Create processed train and test datasets (Phase 1 cleaning, validation and stratified split)
python -m src.training.prepare_data

# 2. Tune, compare and save the models
python -m src.training.train

# 3. Run the tests
python -m pytest
```

Training uses 5-fold stratified cross-validation on the training set only. It compares logistic regression, random forest and gradient boosting. Class imbalance is handled with balanced sample weights. The best model is chosen by mean CV ROC-AUC. The test set is scored once per model for reporting only.

Outputs (gitignored):

- `models/churn_pipeline_v1.joblib`: the full fitted pipeline (cleaning, preprocessing and model)
- `models/churn_pipeline_v1.metadata.json`: input columns, positive class, decision threshold and selection details
- `reports/model_comparison.csv`: one row per model with CV and test metrics
- `reports/evaluation_metrics.json`: all metrics and confusion matrices

Load the artifact for serving. Pass raw rows with the columns listed under `input_columns`:

```python
from src.training.artifacts import load_model_artifact

pipeline, metadata = load_model_artifact("models/churn_pipeline_v1.joblib")
probability_of_churn = pipeline.predict_proba(raw_rows_df)[:, 1]
```

Hyperparameters, seeds, CV folds, scoring and the selection metric are set in the `training:` section of `config/config.yaml`. The artifact was built with scikit-learn 1.8.0, so load it with the same version.

## Development approach

Build one phase at a time. Run relevant tests, inspect outputs and commit only working milestones. See `docs/` for design notes as the project grows.

## Current limitations

Experiment tracking (MLflow), the prediction API, Docker, monitoring and retraining are not implemented yet. The classification threshold is the default 0.5, and it has not been tuned to a business cost.
