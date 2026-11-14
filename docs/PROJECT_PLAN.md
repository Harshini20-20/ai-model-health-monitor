# Project plan

The implementation follows the approved development roadmap.

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

## Engineering rules

- Keep raw data and customer information out of Git.
- Use the same fitted preprocessing pipeline at training and inference.
- Record random seeds, dataset provenance and evaluation protocol.
- Treat drift as a signal to investigate, not proof of model failure.
- Never promote a candidate model without explicit acceptance criteria.
- Preserve the last approved model for rollback.
