import logging
from pathlib import Path
from sklearn.model_selection import ParameterGrid

# early torch setup
from src.common.pytorch_setup import ensure_torch
from src.common.set_up_logging import set_up_logging
from src.train.dataloaders import get_dataloaders

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
ensure_reproducibility(None)

# continue normally with imports / global vars
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
# noinspection PyPackageRequirements, PyUnresolvedReferences
from torch import nn

from src.train.training import training
from src.common.predictors import get_new_fully_connected_model

LOG_LEVEL = logging.DEBUG
BATCH_SIZE: int = 32
VALIDATION_PERCENTAGE: float = 0.15
TEST_PERCENTAGE: float = 0.15
LOSS_FUNCTION = nn.BCEWithLogitsLoss()
OPTIMIZER_TYPE: type = torch.optim.Adam
NUMBER_OF_EPOCHS: int = 200

def grid_search(device):
    neuronal_network_params = {
        'neurons_per_layer': [100, 200, 400, 600, 800],
        'hidden_layer_count': [2, 3, 4, 6, 8, 10],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    }

    # Perform grid search to find the best parameters
    results = []

    for params in ParameterGrid(neuronal_network_params):
        print(f"Testing parameters: {params}")

        # Initialize model, loss function, and optimizer with current parameters
        model = get_new_fully_connected_model(params['hidden_layer_count'], number_of_features,
                                            params['neurons_per_layer'])
        model.to(device)

        optimizer = OPTIMIZER_TYPE(model.parameters(), lr=params['learning_rate'])

        # Train the model
        result_list, _, _, _ = training(model, train_loader, val_loader, LOSS_FUNCTION, optimizer, NUMBER_OF_EPOCHS, use_tqdm=False)
        best_epoch = len(result_list)
        # Validate the model
        val_loss = validate(model, val_loader, LOSS_FUNCTION)

        print(f"Results for parameters: {params}:")
        print(f"Best epoch: {best_epoch}")
        print(f"Validation loss: {val_loss}")

        results.append({
            "params": params,
            "best_epoch": best_epoch,
            "validation_loss": val_loss,
        })
    results.sort(key=lambda x: x["validation_loss"], reverse=False)
    return results


def validate(model, valloader, criterion):
    """
    Evaluate the model on the validation data.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in valloader:
            outputs = model(inputs).squeeze(1)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate loss
    return val_loss / len(valloader)  # Return average validation loss


if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    tensor_path = MODEL_PATH / "training_tensor.pt"
    assert tensor_path.exists()

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load data
    train_loader, val_loader, test_loader, number_of_features = get_dataloaders(torch.load(tensor_path), BATCH_SIZE,
                                                                                VALIDATION_PERCENTAGE, TEST_PERCENTAGE,
                                                                                device, MODEL_CONFIG["seed"])
    assert number_of_features == MODEL_CONFIG["number_of_features"]

    results = grid_search(device)
    print("Grid search results:")
    for rank, r in enumerate(results[:10], start=1):
        print(
            f"{rank:2d}. "
            f"Val Loss: {r['validation_loss']:.6f} | "
            f"Best Epoch: {r['best_epoch']} | "
            f"Params: {r['params']}"
        )
