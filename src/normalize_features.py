import pandas as pd


def normalize_features(info_general_df: pd.DataFrame, info_timestamp_df: pd.DataFrame, player_general_df: pd.DataFrame, player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info_timestamp_df, player_timestamp_df = _normalize_numeric_features(info_timestamp_df, player_timestamp_df)
    return info_general_df, info_timestamp_df, player_general_df, player_timestamp_df


def _normalize_numeric_features(info_timestamp_df: pd.DataFrame, player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols_info = ["timestamp", "total_gold"]

    info_timestamp_df[numeric_cols_info] = (
        info_timestamp_df[numeric_cols_info] - info_timestamp_df[numeric_cols_info].mean()
    ) / info_timestamp_df[numeric_cols_info].std(ddof=0)


    player_timestamp_df["total_gold_match_ts"] = (
        player_timestamp_df.groupby(["match_id", "timestamp_s"])["net_worth"].transform("sum")
    )
    player_timestamp_df["net_worth_ratio"] = (
        player_timestamp_df["net_worth"] / player_timestamp_df["total_gold_match_ts"]
    )

    numeric_cols_player = [
        c for c in player_timestamp_df.columns
        if pd.api.types.is_numeric_dtype(player_timestamp_df[c]) and c not in ["match_id", "account_id", "total_gold_match_ts", "net_worth"]
    ]

    player_timestamp_df[numeric_cols_player] = (
        player_timestamp_df[numeric_cols_player] - player_timestamp_df[numeric_cols_player].mean()
    ) / player_timestamp_df[numeric_cols_player].std(ddof=0)
    player_timestamp_df = player_timestamp_df.drop(columns=["total_gold_match_ts", "net_worth"])

    return info_timestamp_df, player_timestamp_df