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

> Current status: Phase 0 — repository foundation. Features will be added phase by phase after their tests pass.

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

## Development approach

Build one phase at a time. Run relevant tests, inspect outputs and commit only working milestones. See `docs/` for design notes as the project grows.

## Current limitations

The repository is only the initial foundation. It does not yet train a model, expose an API, monitor drift or perform retraining.
