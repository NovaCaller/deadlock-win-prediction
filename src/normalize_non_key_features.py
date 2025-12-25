import pandas as pd

from src.util import normalize_df


def normalize_non_key_features(info_general_df: pd.DataFrame, info_timestamp_df: pd.DataFrame, player_general_df: pd.DataFrame, player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info_timestamp_df, player_timestamp_df = _normalize_non_key_numeric_features(info_timestamp_df, player_timestamp_df)
    return info_general_df, info_timestamp_df, player_general_df, player_timestamp_df


def _normalize_non_key_numeric_features(info_timestamp_df: pd.DataFrame, player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols_info = ["total_gold"]
    numeric_cols_player = ["ability_points", "level", "net_worth_ratio"]

    info_timestamp_df = normalize_df(info_timestamp_df, numeric_cols_info)

    player_timestamp_df["total_gold_match_ts"] = (
        player_timestamp_df.groupby(["match_id", "timestamp_s"])["net_worth"].transform("sum")
    )
    player_timestamp_df["net_worth_ratio"] = (
        player_timestamp_df["net_worth"] / player_timestamp_df["total_gold_match_ts"]
    )
    player_timestamp_df = normalize_df(player_timestamp_df, numeric_cols_player)
    player_timestamp_df = player_timestamp_df.drop(columns=["total_gold_match_ts", "net_worth"])

    return info_timestamp_df, player_timestamp_df