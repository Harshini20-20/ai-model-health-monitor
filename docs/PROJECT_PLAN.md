# AI Model Health Monitor — Project Plan

The implementation follows the approved development roadmap. Each phase is designed with deployment compatibility, reproducibility, testing, and maintainability in mind.

## Development Roadmap

| Phase | Scope | Exit condition |
|---|---|---|
| 0 | Repository foundation | Documented local setup works |
| 1 | Dataset and data engineering | Validation passes; no leakage; shared preprocessing |
| 2 | ML development | Reproducible candidate evaluation |
| 3 | MLflow | Runs and model artifacts are traceable |
| 4 | FastAPI | Valid and invalid requests are tested |
| 5 | Docker | Service runs in a container |
| 6 | Monitoring | Simulated drift is detected and recorded |
| 7 | Streamlit | Dashboard reflects stored monitoring data |
| 8 | GitHub Actions | CI checks run and fail correctly |
| 9 | Retraining and rollback | Candidate is promoted only after quality gates |

## Phase 1 — Dataset and Data Engineering

**Status:** Implementation and verification completed.

### Dataset

- Dataset: IBM Telco Customer Churn
- Use case: Customer churn prediction
- Format: CSV
- Records: 7,043
- Features/columns: 21
- Target column: `Churn`
- Identifier column: `customerID`
- Dataset location: `data/raw/telco_customer_churn.csv`

### Data Quality Findings

- No duplicate rows detected.
- No missing values detected by the initial pandas inspection.
- Eleven blank `TotalCharges` values were identified during numeric conversion.
- These values correspond to customers with zero tenure.
- Missing `TotalCharges` values are retained as NaN for later imputation.
- No missing target values were detected.

### Implemented Components

| Component | File | Responsibility |
|---|---|---|
| Data loading | `src/data/loader.py` | Load configured local CSV datasets |
| Data cleaning | `src/data/cleaning.py` | Trim whitespace and convert numerical columns |
| Data validation | `src/data/validation.py` | Check dataset structure, missing values, target validity, and duplicates |
| Preprocessing | `src/data/preprocessing.py` | Build reusable numerical and categorical preprocessing pipelines |

### Preprocessing Strategy

**Numerical features**
- Missing-value imputation using median.
- Feature scaling using StandardScaler.

**Categorical features**
- Missing-value imputation using most frequent value.
- One-hot encoding.
- Unknown categories handled using `handle_unknown="ignore"`.

**Data splitting**
- Training/test split: 80/20.
- Random state: 42.
- Stratified splitting using the target column.

The identifier and target columns are excluded from model input features.

Preprocessing is fitted only on training data and reused to transform test data to prevent data leakage.

### Testing

- Unit tests cover configuration, loading, cleaning, validation, and preprocessing.
- An end-to-end integration test verifies the complete data pipeline using the actual dataset.
- Total tests passing: **24**.

## Engineering Rules

- Keep raw data and customer information out of Git.
- Use the same fitted preprocessing pipeline at training and inference.
- Record random seeds, dataset provenance and evaluation protocol.
- Treat drift as a signal to investigate, not proof of model failure.
- Never promote a candidate model without explicit acceptance criteria.
- Preserve the last approved model for rollback.
- Keep configuration, application logic, and environment-specific settings separate.
- Test components before integrating them into the deployment workflow.
- Keep secrets and credentials outside source control.
