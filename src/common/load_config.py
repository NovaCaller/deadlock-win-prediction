import logging
import tomllib
from pathlib import Path
from typing import Any


def load_model_config(file_path: Path) -> dict[str, Any]:
    required_keys = ["number_of_features", "number_of_hidden_layers", "neurons_per_layer"]
    try:
        config = _load_config_with_required_keys(file_path, required_keys)
    except KeyError as e:
        logging.error(f"error loading model config: {e}")
        raise e
    return config

def _load_config_with_required_keys(file_path: Path, required_keys: list[str]) -> dict[str, Any]:
    assert file_path.exists()
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required key {key} is missing")
    return config