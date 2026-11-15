"""Model Registry helpers: versions, aliases and tags.

Lifecycle is expressed with registry aliases rather than the deprecated stage API:
- the candidate alias is set by training and marks the newest trained model;
- other aliases (for example a production alias) are assigned later, once the
  quality gates exist (Phase 9). Nothing in Phase 3 promotes a model automatically.
"""

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


def assign_alias(client: MlflowClient, model_name: str, alias: str, version: int | str) -> None:
    """Point an alias at a model version. Moving an alias is how a version is promoted or retired."""
    client.set_registered_model_alias(model_name, alias, str(version))


def remove_alias(client: MlflowClient, model_name: str, alias: str) -> None:
    """Delete an alias. The model versions themselves are kept."""
    client.delete_registered_model_alias(model_name, alias)


def get_version_by_alias(client: MlflowClient, model_name: str, alias: str) -> ModelVersion | None:
    """Return the version an alias points to, or None when the alias is not set."""
    if alias not in client.get_registered_model(model_name).aliases:
        return None

    return client.get_model_version_by_alias(model_name, alias)


def tag_model_version(
    client: MlflowClient,
    model_name: str,
    version: int | str,
    tags: dict[str, object],
) -> None:
    """Attach string tags to a model version, for example the Phase 2 model version label."""
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, str(value))


def list_model_versions(client: MlflowClient, model_name: str) -> list[dict]:
    """Return every version of a registered model, oldest first, with aliases and tags.

    Aliases come from the registered model's alias map: search results do not
    reliably include them in MLflow 3.x.
    """
    aliases_by_version: dict[int, list[str]] = {}
    for alias, version in client.get_registered_model(model_name).aliases.items():
        aliases_by_version.setdefault(int(version), []).append(alias)

    versions = client.search_model_versions(f"name='{model_name}'")

    return [
        {
            "version": int(version.version),
            "aliases": sorted(aliases_by_version.get(int(version.version), [])),
            "run_id": version.run_id,
            "tags": dict(version.tags),
        }
        for version in sorted(versions, key=lambda item: int(item.version))
    ]
