import logging
from pathlib import Path

import pandas as pd
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
# ensure_reproducibility(MODEL_CONFIG["seed"])

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
LEARNING_RATE: float = 0.0001
NUMBER_OF_EPOCHS: int = 10

TRAIN_LOG_FILE_PATH = Path("logs")

def write_logs(train_loss, train_acc, val_loss, val_acc, best_epoch_es):
    TRAIN_LOG_FILE_PATH.mkdir(parents=True, exist_ok=True)
    logs = []
    for epoch_nr in range(len(train_loss)):
        logs.append({"epoch": epoch_nr+1, "train_loss": train_loss[epoch_nr], "train_acc": train_acc[epoch_nr], "val_loss": val_loss[epoch_nr], "val_acc": val_acc[epoch_nr]})
        if epoch_nr+1 == best_epoch_es:
            break

    new_df = pd.DataFrame(logs)

    try:
        existing = pd.read_parquet(TRAIN_LOG_FILE_PATH / "train_log.parquet")
        df = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        df = new_df

    df.to_parquet(TRAIN_LOG_FILE_PATH / "train_log.parquet", index=False)


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

    logging.info(f"Length of training loss: {len(training_losses)}, Length of training accuracy: {len(training_accuracies)}, Length of validation accuracy: {len(validation_accuracies)}, Length of validation loss: {len(validation_losses)}")
    logging.info(f"Finished training")

    write_logs(training_losses, training_accuracies, validation_losses, validation_accuracies, best_epoch)
    # final test
    test_loss, test_acc = test_loop(model, test_loader, LOSS_FUNCTION)
    tqdm.write(f"Final Test Loss={test_loss:.4f}, Final Test Acc={test_acc:.4f}")

    # save weights
    torch.save(model.state_dict(), MODEL_PATH / "model_weights.pth")
    logging.info("wrote model weights to disk.")