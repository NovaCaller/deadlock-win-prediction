import json
import logging
from pathlib import Path

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
# noinspection PyPackageRequirements
import torch

from src.common.predictors import load_fully_connected_model
from src.common.set_up_logging import set_up_logging
from src.predict.prediction import predict_with_game_state
from src.predict.models import GameState, Team, Player, Objective, Hero
from src.prep.util import get_hero_list

LOG_LEVEL = logging.DEBUG
HEROES_PARQUET: Path = Path("db_dump/heroes.parquet")

if __name__ == "__main__":
    set_up_logging(LOG_LEVEL)
    assert HEROES_PARQUET.exists()
    weights_path = MODEL_PATH / "model_weights.pth"
    assert weights_path.exists()
    normalization_params_path = MODEL_PATH / "normalization_params.json"
    assert normalization_params_path.exists()

    # get device
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device}")

    # load model with weights
    model = load_fully_connected_model(weights_path, MODEL_CONFIG["number_of_hidden_layers"], MODEL_CONFIG["number_of_features"], MODEL_CONFIG["neurons_per_layer"])
    model = model.to(device)

    # load normalization parameters
    with open(normalization_params_path) as f:
        loaded_params: dict[str, list[float]] = json.load(f)
    normalization_params: dict[str, tuple[float, float]] = {
        k: (v[0], v[1])
        for k, v in loaded_params.items()
    }

    # set gamestate
    GAME_STATE = GameState(
        timestamp = 540,
        team0 = Team(players=[
            Player(hero=Hero.HAZE, net_worth=6288, ability_points=6, level=10),  # acc 1118078287
            Player(hero=Hero.WARDEN, net_worth=7100, ability_points=7, level=11),  # acc 1911140385
            Player(hero=Hero.MO_AND_KRILL, net_worth=6278, ability_points=6, level=10),  # acc 57438032
            Player(hero=Hero.PARADOX, net_worth=8845, ability_points=9, level=13),  # acc 1929234247
            Player(hero=Hero.HOLLIDAY, net_worth=7881, ability_points=8, level=12),  # acc 969975089
            Player(hero=Hero.LADY_GEIST, net_worth=9232, ability_points=10, level=14)  # acc 1115620324
        ], lost_objectives={
            Objective.CORE: False,
            Objective.TIER_1_LANE_1: True,
            Objective.TIER_1_LANE_3: False,
            Objective.TIER_1_LANE_4: False,
            Objective.TIER_2_LANE_1: False,
            Objective.TIER_2_LANE_3: False,
            Objective.TIER_2_LANE_4: False,
            Objective.BARRACK_BOSS_LANE_1: False,
            Objective.BARRACK_BOSS_LANE_3: False,
            Objective.BARRACK_BOSS_LANE_4: False,
            Objective.TITAN: False,
            Objective.TITAN_SHIELD_GENERATOR_1: False,
            Objective.TITAN_SHIELD_GENERATOR_2: False
        }),
        team1 = Team(players=[
            Player(hero=Hero.MCGINNIS, net_worth=6853, ability_points=7, level=11),  # acc 1125297246
            Player(hero=Hero.BEBOP, net_worth=7977, ability_points=8, level=12),  # acc 1866661741
            Player(hero=Hero.LASH, net_worth=8266, ability_points=8, level=12),  # acc 123756868
            Player(hero=Hero.INFERNUS, net_worth=6687, ability_points=7, level=11),  # acc 214095947
            Player(hero=Hero.YAMATO, net_worth=7161, ability_points=7, level=11),  # acc 131386009
            Player(hero=Hero.VISCOUS, net_worth=8365, ability_points=8, level=12)  # acc 420181684
        ], lost_objectives={
            Objective.CORE: False,
            Objective.TIER_1_LANE_1: False,
            Objective.TIER_1_LANE_3: True,
            Objective.TIER_1_LANE_4: True,
            Objective.TIER_2_LANE_1: False,
            Objective.TIER_2_LANE_3: False,
            Objective.TIER_2_LANE_4: False,
            Objective.BARRACK_BOSS_LANE_1: False,
            Objective.BARRACK_BOSS_LANE_3: False,
            Objective.BARRACK_BOSS_LANE_4: False,
            Objective.TITAN: False,
            Objective.TITAN_SHIELD_GENERATOR_1: False,
            Objective.TITAN_SHIELD_GENERATOR_2: False
        })
    )  # winner: team 0

    # predict
    logging.info(f"predicting with gamestate:\n{GAME_STATE}")
    prediction = predict_with_game_state(model, GAME_STATE, normalization_params, get_hero_list(HEROES_PARQUET), device)
    print(f"prediction: {prediction:.6f}")