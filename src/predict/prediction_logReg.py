import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.predict.conversion import game_state_to_tensor
from src.predict.models import GameState


def predict_with_game_state_logreg(
    model: LogisticRegression,
    game_state: GameState,
    normalization_params: dict[str, tuple[float, float]],
    hero_list: list[str],
) -> float:
    """
    Returns probability for class 1 (team 1 wins).
    """

    tensor: torch.Tensor = game_state_to_tensor(
        game_state,
        normalization_params,
        hero_list,
    )

    # Torch -> NumPy
    X: np.ndarray = tensor.cpu().numpy()

    # shape safety: (1, num_features)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    proba: float = model.predict_proba(X)[0, 1]
    return proba
