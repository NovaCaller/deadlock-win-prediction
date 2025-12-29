import logging
from pathlib import Path

from src.common.pytorch_setup import ensure_torch
ensure_torch()
# noinspection PyPackageRequirements
import torch

from src.common.load_config import load_model_config
from src.common.predictors import load_fully_connected_model
from src.common.set_up_logging import set_up_logging

LOG_LEVEL = logging.DEBUG
MODEL_PATH = Path("model")

if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    assert MODEL_PATH.exists()
    model_config_path = MODEL_PATH / "model.toml"
    assert model_config_path.exists()
    weights_path = MODEL_PATH / "model_weights.pth"
    assert weights_path.exists()

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load model config
    model_config = load_model_config(MODEL_PATH / "model.toml")
    logging.debug(f"Loaded model config: {model_config}")

    # load model with weights
    model = load_fully_connected_model(weights_path, model_config["number_of_hidden_layers"], model_config["number_of_features"], model_config["neurons_per_layer"])
    model.eval()
    model.to(device)

