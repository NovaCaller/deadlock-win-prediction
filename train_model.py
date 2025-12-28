import logging
from pathlib import Path
from tqdm import tqdm

from src.common.pytorch_setup import ensure_torch

ensure_torch()
# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
from torch import nn

from src.train.dataloaders import get_dataloaders
from src.train.training import training
from src.common.set_up_logging import set_up_logging
from src.train.predictors import get_fully_connected
from src.train.util import test_loop

MODEL_PATH: Path = Path("model")
LOG_LEVEL = logging.INFO
NUMBER_OF_HIDDEN_LAYERS: int = 4
NEURONS_PER_LAYER: int = 448
BATCH_SIZE: int = 32
VALIDATION_PERCENTAGE: float = 0.15
TEST_PERCENTAGE: float = 0.15
LOSS_FUNCTION = nn.BCEWithLogitsLoss()
OPTIMIZER_TYPE: type = torch.optim.Adam
LEARNING_RATE: float = 0.001
NUMBER_OF_EPOCHS: int = 10


if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    assert MODEL_PATH.exists()
    tensor_path = MODEL_PATH / "training_tensor.pt"
    assert tensor_path.exists()

    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load data
    train_loader, val_loader, test_loader, number_of_features = get_dataloaders(tensor_path, BATCH_SIZE, VALIDATION_PERCENTAGE, TEST_PERCENTAGE, device)

    # load model
    model = get_fully_connected(NUMBER_OF_HIDDEN_LAYERS, number_of_features, NEURONS_PER_LAYER)
    model = model.to(device)

    # train model
    optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LEARNING_RATE)
    training(model, train_loader, val_loader, LOSS_FUNCTION, optimizer, NUMBER_OF_EPOCHS)
    logging.info(f"Finished training")

    # final test
    test_loss, test_acc = test_loop(model, test_loader, LOSS_FUNCTION)
    tqdm.write(f"Final Test Loss={test_loss:.4f}, Final Test Acc={test_acc:.4f}")