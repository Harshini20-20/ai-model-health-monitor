"""Candidate classifiers and their hyperparameter search spaces."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_candidate_models(training_config: dict) -> dict[str, dict]:
    """Return {name: {"estimator": ..., "param_grid": ...}} for each configured model.

    Every estimator receives the same random_state, which keeps results reproducible.
    """
    random_state = training_config["random_state"]
    candidates = {}

    for name, settings in training_config["models"].items():
        if name not in MODEL_CLASSES:
            raise ValueError(
                f"Unsupported model '{name}'. Choose from: {sorted(MODEL_CLASSES)}"
            )

        params = settings.get("params") or {}
        estimator = MODEL_CLASSES[name](random_state=random_state, **params)

        candidates[name] = {
            "estimator": estimator,
            "param_grid": settings.get("param_grid") or {},
        }

    return candidates
