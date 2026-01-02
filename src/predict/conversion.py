import logging

import pandas as pd
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch

from src.predict.models import GameState, Objective
from src.prep.encode_features import encode_player_general_df
from src.prep.join_dataframes import join_dataframes
from src.prep.normalize_non_key_features import normalize_non_key_numeric_features
from src.prep.util import normalize_df


def game_state_to_tensor(game_state: GameState, normalization_params: dict[str, tuple[float, float]],
                         hero_list: list[str]) -> torch.Tensor:
    # convert into dataframes, same state as in prepare_data right before encoding
    info_timestamp_df, player_general_df, player_timestamp_df = _game_state_to_dfs(game_state)

    player_general_df = encode_player_general_df(player_general_df, hero_list)
    logging.debug("done encoding features.")

    info_timestamp_df, player_timestamp_df, _ = normalize_non_key_numeric_features(info_timestamp_df,
                                                                                   player_timestamp_df,
                                                                                   normalization_params)
    logging.info("done normalizing features (except timestamps).")

    merged_df = join_dataframes(info_timestamp_df, player_general_df, player_timestamp_df)
    logging.info(
        f"merged dataframes to single dataframe with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

    merged_df, _ = normalize_df(merged_df, ["timestamp"], normalization_params)
    logging.info("done normalizing timestamps.")
    logging.info("finalized dataframe.")

    tensor = torch.from_numpy(merged_df.values)
    logging.info(
        f"converted to tensor with {tensor.shape[0]} rows and {tensor.shape[1]} columns.")
    return tensor


def _game_state_to_dfs(game_state: GameState) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info_timestamp_df = _game_state_to_info_timestamp_df(game_state)
    player_general_df, player_timestamp_df = _game_state_to_player_dfs(game_state)
    return info_timestamp_df, player_general_df, player_timestamp_df


def _game_state_to_info_timestamp_df(game_state: GameState) -> pd.DataFrame:
    total_gold = _calculate_total_gold(game_state)
    objectives_dict: dict[str, int] = _convert_team_objectives(game_state.team0.lost_objectives,
                                                               game_state.team1.lost_objectives)
    info_timestamp_df = pd.DataFrame([{
        "match_id": 1,
        "timestamp": game_state.timestamp,
        "total_gold": total_gold,
        **objectives_dict
    }])
    return info_timestamp_df


def _calculate_total_gold(game_state: GameState) -> int:
    return sum(player.net_worth for team in [game_state.team0, game_state.team1] for player in team.players)


def _convert_team_objectives(team0_dict: dict[Objective, bool], team1_dict: dict[Objective, bool]) -> dict[str, int]:
    objectives_dict = _flatten_team_objectives(team0_dict, "team0")
    objectives_dict = objectives_dict | _flatten_team_objectives(team1_dict, "team1")
    return objectives_dict


def _flatten_team_objectives(team_dict: dict[Objective, bool], team_prefix: str) -> dict[str, int]:
    return {f"{team_prefix}_{obj.value}": int(destroyed) for obj, destroyed in team_dict.items()}


def _game_state_to_player_dfs(game_state: GameState) -> tuple[pd.DataFrame, pd.DataFrame]:
    general_rows = [
        {
            "match_id": 1,
            "account_id": idx,
            "team": "Team0" if team_idx == 0 else "Team1",
            "hero_name": player.hero.value
        }
        for team_idx, team in enumerate([game_state.team0, game_state.team1])
        for idx, player in enumerate(team.players, start=1 if team_idx == 0 else 7)
    ]

    timestamp_rows = [
        {
            "match_id": 1,
            "account_id": idx,
            "timestamp_s": game_state.timestamp,
            "net_worth": player.net_worth,
            "ability_points": player.ability_points,
            "level": player.level
        }
        for team_idx, team in enumerate([game_state.team0, game_state.team1])
        for idx, player in enumerate(team.players, start=1 if team_idx == 0 else 7)
    ]

    return pd.DataFrame(general_rows), pd.DataFrame(timestamp_rows)
