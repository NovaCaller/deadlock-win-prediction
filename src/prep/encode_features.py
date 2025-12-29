import duckdb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_info_general_df(info_general_df: pd.DataFrame) -> pd.DataFrame:
    return _encode_team(info_general_df, "winning_team")

def encode_player_general_df(player_general_df: pd.DataFrame, hero_list: list[str]) -> pd.DataFrame:
    player_general_df = _encode_team(player_general_df, "team")
    player_general_df = _encode_heroes(player_general_df, hero_list)
    return player_general_df

# noinspection PyUnusedLocal
def _encode_team(df: pd.DataFrame, team_attr_name: str) -> pd.DataFrame:
    # case expression is only needed for if you for some reason execute this again, if the variables are
    # already numerically encoded. Since then the check team='Team1' obviously does not work then.
    # If already numerically encoded, just let the value be and do nothing lol.
    encoded_df = duckdb.sql(f"""
        SELECT * REPLACE (
            CASE
                WHEN typeof({team_attr_name}) = 'VARCHAR'
                THEN ({team_attr_name}='Team1')::INT
                ELSE {team_attr_name}::INT
            END AS {team_attr_name}
        )
        FROM df
    """).fetchdf()
    return encoded_df

def _encode_heroes(player_general_df: pd.DataFrame, hero_list: list[str]) -> pd.DataFrame:
    hero_col = "hero_name"
    hero_ohe = OneHotEncoder(
        categories=[hero_list],
        handle_unknown="error",
        sparse_output=False
    ).set_output(transform="pandas")

    hero_ohe_df = hero_ohe.fit_transform(player_general_df[[hero_col]])
    hero_ohe_df.index = player_general_df.index

    df_enc: pd.DataFrame = pd.concat(
        [player_general_df.drop(columns=[hero_col]), hero_ohe_df],
        axis=1
    )

    return df_enc