# noinspection PyPackageRequirements
import torch

from src.predict.models import GameState


def game_state_to_tensor(game_state: GameState, normalization_params: dict[str, tuple[float, float]]) -> torch.Tensor:
    raise NotImplementedError