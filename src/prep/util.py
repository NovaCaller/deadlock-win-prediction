import pandas as pd


def normalize_df(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df[features] = (
        df[features] - df[features].mean()
    ) / df[features].std(ddof=0)
    return df