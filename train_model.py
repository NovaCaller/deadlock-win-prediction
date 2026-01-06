import logging
from pathlib import Path
from tqdm import tqdm

# early torch setup
from src.common.pytorch_setup import ensure_torch
ensure_torch()

# early config setup
from src.common.load_config import load_model_config
MODEL_PATH: Path = Path("model")
assert MODEL_PATH.exists()
assert (MODEL_PATH / "model.toml").exists()
MODEL_CONFIG: dict = load_model_config(MODEL_PATH / "model.toml")
print(f"Loaded model config: {MODEL_CONFIG}")

# early reproducibility setup
from src.common.reproducibility import ensure_reproducibility
ensure_reproducibility(MODEL_CONFIG["seed"])

# continue normally with imports / global vars
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
# noinspection PyPackageRequirements, PyUnresolvedReferences
from torch import nn

from src.train.dataloaders import get_dataloaders
from src.train.training import training
from src.common.set_up_logging import set_up_logging
from src.common.predictors import get_new_fully_connected_model
from src.train.util import test_loop


LOG_LEVEL = logging.DEBUG
BATCH_SIZE: int = 32
VALIDATION_PERCENTAGE: float = 0.15
TEST_PERCENTAGE: float = 0.15
LOSS_FUNCTION = nn.BCEWithLogitsLoss()
OPTIMIZER_TYPE: type = torch.optim.Adam
LEARNING_RATE: float = 0.001
NUMBER_OF_EPOCHS: int = 10


if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    tensor_path = MODEL_PATH / "training_tensor.pt"
    assert tensor_path.exists()

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load data
    train_loader, val_loader, test_loader, number_of_features = get_dataloaders(tensor_path, BATCH_SIZE, VALIDATION_PERCENTAGE, TEST_PERCENTAGE, device, MODEL_CONFIG["seed"])
    assert number_of_features == MODEL_CONFIG["number_of_features"]

    # load model
    model = get_new_fully_connected_model(MODEL_CONFIG["number_of_hidden_layers"], number_of_features, MODEL_CONFIG["neurons_per_layer"])
    model = model.to(device)

    # train model
    optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LEARNING_RATE)
    training_losses, training_accuracies, validation_losses, validation_accuracies, best_epoch = training(model, train_loader, val_loader, LOSS_FUNCTION, optimizer, NUMBER_OF_EPOCHS)
    logging.info(f"Finished training")

    # final test
    test_loss, test_acc = test_loop(model, test_loader, LOSS_FUNCTION)
    tqdm.write(f"Final Test Loss={test_loss:.4f}, Final Test Acc={test_acc:.4f}")

    # save weights
    torch.save(model.state_dict(), MODEL_PATH / "model_weights.pth")
    logging.info("wrote model weights to disk.")