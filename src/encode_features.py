from pathlib import Path

import duckdb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_features(info_general_df: pd.DataFrame, info_timestamp_df: pd.DataFrame, player_general_df: pd.DataFrame, player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info_general_df, player_general_df = _encode_team(info_general_df, player_general_df)
    player_general_df = _encode_heroes(player_general_df)
    return info_general_df, info_timestamp_df, player_general_df, player_timestamp_df


# noinspection PyUnusedLocal
def _encode_team(info_general_df: pd.DataFrame, player_general_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # case expression is only needed for if you for some reason execute this again, if the variables are
    # already numerically encoded. Since then the check team='Team1' obviously does not work then.
    # If already numerically encoded, just let the value be and do nothing lol.
    encoded_info_general_df = duckdb.sql(f"""
        SELECT * REPLACE (
            CASE
                WHEN typeof(winning_team) = 'VARCHAR'
                THEN (winning_team='Team1')::INT
                ELSE winning_team::INT
            END AS winning_team
        )
        FROM info_general_df
    """).fetchdf()

    encoded_player_general_df = duckdb.sql(f"""
        SELECT * REPLACE (
            CASE
                WHEN typeof(team) = 'VARCHAR'
                THEN (team='Team1')::INT
                ELSE team::INT
            END AS team
        )
        FROM player_general_df
    """).fetchdf()

    return encoded_info_general_df, encoded_player_general_df


def _encode_heroes(player_general_df: pd.DataFrame) -> pd.DataFrame:
    hero_col = "hero_name"
    hero_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ).set_output(transform="pandas")

    hero_ohe_df = hero_ohe.fit_transform(player_general_df[[hero_col]])
    hero_ohe_df.index = player_general_df.index

    df_enc: pd.DataFrame = pd.concat(
        [player_general_df.drop(columns=[hero_col]), hero_ohe_df],
        axis=1
    )

    return df_enc