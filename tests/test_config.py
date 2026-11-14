from src.config import load_config


def test_load_config_has_project_name():
    config = load_config()
    assert config["project"]["name"] == "ai-model-health-monitor"


def test_load_config_has_target_column():
    config = load_config()
    assert config["data"]["target_column"] == "Churn"
