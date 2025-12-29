from pathlib import Path

import pandas as pd


def get_hero_list(hero_parquet: Path) -> list[str]:
    df = pd.read_parquet(hero_parquet)
    assert "name" in df.columns

    return df["name"].tolist()