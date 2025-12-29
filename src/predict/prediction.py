# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
from torch import nn

from src.predict.conversion import game_state_to_tensor
from src.predict.models import GameState


def predict_with_game_state(model: nn.Module, game_state: GameState, normalization_params: dict[str, tuple[float, float]], hero_list: list[str], device: str) -> int:
    tensor: torch.Tensor = game_state_to_tensor(game_state, normalization_params, hero_list)
    tensor = tensor.float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        y = torch.sigmoid(logits)
    y = y.squeeze().item()
    return y