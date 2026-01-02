from pathlib import Path

import duckdb
import pandas as pd


# noinspection PyUnusedLocal
def replace_hero_ids_with_names(player_df: pd.DataFrame, heroes_parquet: Path):
    df = duckdb.sql(f"""
        SELECT
            *
            EXCLUDE (hero_id, id)
            RENAME (heroes.name as hero_name)
        FROM player_df as p
        JOIN read_parquet('{heroes_parquet}') as heroes
        ON p.hero_id = heroes.id;
    """).fetchdf()
    return df