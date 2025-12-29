import logging
from typing import Optional

import duckdb
import pandas as pd


# noinspection PyUnusedLocal
def join_dataframes(info_timestamp_df: pd.DataFrame, player_general_df: pd.DataFrame,
                    player_timestamp_df: pd.DataFrame, info_general_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:

    query = """
    WITH
        numbered_players AS (
            SELECT 
                match_id,
                timestamp_s,
                ROW_NUMBER() OVER (PARTITION BY pt.match_id, pt.timestamp_s, pg.team ORDER BY pt.account_id) as player_num,
                pg.* EXCLUDE (match_id, account_id),
                pt.* EXCLUDE (match_id, account_id, timestamp_s)
            FROM player_timestamp_df as pt
            JOIN player_general_df as pg
                USING (match_id, account_id)
        )
    """

    if info_general_df is not None:
        query += """
    SELECT
        * EXCLUDE (match_id, winning_team, timestamp_s, team, player_num),
        winning_team 
    FROM info_timestamp_df as info_timestamp
    JOIN info_general_df as info_general
        USING(match_id)
        """
    else:
        query += """
    SELECT
        * EXCLUDE (match_id, timestamp_s, team, player_num)
    FROM info_timestamp_df as info_timestamp
        """

    query += """
    INNER JOIN numbered_players p1 ON info_timestamp.match_id = p1.match_id AND info_timestamp.timestamp = p1.timestamp_s AND p1.team = 0 AND p1.player_num = 1
    INNER JOIN numbered_players p2 ON info_timestamp.match_id = p2.match_id AND info_timestamp.timestamp = p2.timestamp_s AND p2.team = 0 AND p2.player_num = 2
    INNER JOIN numbered_players p3 ON info_timestamp.match_id = p3.match_id AND info_timestamp.timestamp = p3.timestamp_s AND p3.team = 0 AND p3.player_num = 3
    INNER JOIN numbered_players p4 ON info_timestamp.match_id = p4.match_id AND info_timestamp.timestamp = p4.timestamp_s AND p4.team = 0 AND p4.player_num = 4
    INNER JOIN numbered_players p5 ON info_timestamp.match_id = p5.match_id AND info_timestamp.timestamp = p5.timestamp_s AND p5.team = 0 AND p5.player_num = 5
    INNER JOIN numbered_players p6 ON info_timestamp.match_id = p6.match_id AND info_timestamp.timestamp = p6.timestamp_s AND p6.team = 0 AND p6.player_num = 6
    INNER JOIN numbered_players p7 ON info_timestamp.match_id = p7.match_id AND info_timestamp.timestamp = p7.timestamp_s AND p7.team = 1 AND p7.player_num = 1
    INNER JOIN numbered_players p8 ON info_timestamp.match_id = p8.match_id AND info_timestamp.timestamp = p8.timestamp_s AND p8.team = 1 AND p8.player_num = 2
    INNER JOIN numbered_players p9 ON info_timestamp.match_id = p9.match_id AND info_timestamp.timestamp = p9.timestamp_s AND p9.team = 1 AND p9.player_num = 3
    INNER JOIN numbered_players p10 ON info_timestamp.match_id = p10.match_id AND info_timestamp.timestamp = p10.timestamp_s AND p10.team = 1 AND p10.player_num = 4
    INNER JOIN numbered_players p11 ON info_timestamp.match_id = p11.match_id AND info_timestamp.timestamp = p11.timestamp_s AND p11.team = 1 AND p11.player_num = 5
    INNER JOIN numbered_players p12 ON info_timestamp.match_id = p12.match_id AND info_timestamp.timestamp = p12.timestamp_s AND p12.team = 1 AND p12.player_num = 6
    ;"""

    rel = duckdb.sql(query)

    df = rel.fetchdf()
    logging.debug(f"number of missing values in merged df: {df.isna().sum().sum()}")
    return df