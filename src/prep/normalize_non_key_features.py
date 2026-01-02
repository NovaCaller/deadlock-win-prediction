import pandas as pd

from src.prep.util import normalize_df


def normalize_non_key_numeric_features(info_timestamp_df: pd.DataFrame, player_timestamp_df: pd.DataFrame, normalization_params: dict[str, tuple[float, float]] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    numeric_cols_info = ["total_gold"]
    numeric_cols_player = ["ability_points", "level", "net_worth_ratio"]

    info_timestamp_df, normalization_params = normalize_df(info_timestamp_df, numeric_cols_info, normalization_params)

    player_timestamp_df["total_gold_match_ts"] = (
        player_timestamp_df.groupby(["match_id", "timestamp_s"])["net_worth"].transform("sum")
    )
    player_timestamp_df["net_worth_ratio"] = (
        player_timestamp_df["net_worth"] / player_timestamp_df["total_gold_match_ts"]
    )
    player_timestamp_df, normalization_params = normalize_df(player_timestamp_df, numeric_cols_player, normalization_params)
    player_timestamp_df = player_timestamp_df.drop(columns=["total_gold_match_ts", "net_worth"])

    return info_timestamp_df, player_timestamp_df, normalization_params