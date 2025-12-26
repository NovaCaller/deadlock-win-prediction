import logging

import duckdb
import pandas as pd


def split_off_timestamps(match_info_df: pd.DataFrame, match_player_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_general_df, player_timestamp_df = _split_player_stats(match_player_df)
    info_general_df, info_timestamp_df = _split_info_stats(match_info_df, player_timestamp_df)
    return info_general_df, info_timestamp_df, player_general_df, player_timestamp_df


# noinspection PyUnusedLocal
def _split_info_stats(match_info_df: pd.DataFrame, match_player_timestamp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    objective_names = [
        "Core", "Tier1Lane1", "Tier1Lane3", "Tier1Lane4",
        "Tier2Lane1", "Tier2Lane3", "Tier2Lane4",
        "BarrackBossLane1", "BarrackBossLane3", "BarrackBossLane4",
        "Titan", "TitanShieldGenerator1", "TitanShieldGenerator2"
    ]

    case_statements = []
    for team in [0, 1]:
        for obj in objective_names:
            case_statements.append(f"""
                MAX(
                    CASE
                        WHEN obj_team = {team} AND obj_name = '{obj}' AND timestamp >= destroyed_time AND destroyed_time != 0
                        THEN 1
                        ELSE 0
                    END
                ) AS team{team}_{obj}
            """)

    query = f"""
        WITH timestamps AS (
            SELECT match_id, timestamp_s AS timestamp
            FROM match_player_timestamp_df
            GROUP BY match_id, timestamp_s
        ),
        gold AS (
            SELECT match_id, timestamp_s AS gtimestamp, CAST(SUM(net_worth) AS BIGINT) AS total_gold
            FROM match_player_timestamp_df
            GROUP BY match_id, timestamp_s
        ),
        objectives AS (
            SELECT
                match_id,
                unnest("objectives.destroyed_time_s") AS destroyed_time,
                unnest("objectives.team_objective") AS obj_name,
                CASE unnest("objectives.team")
                    WHEN 'Team0' THEN 0
                    WHEN 'Team1' THEN 1
                END AS obj_team
            FROM match_info_df
        )
        SELECT
            t.match_id,
            t.timestamp,
            g.total_gold,
            {', '.join(case_statements)}
        FROM timestamps t
        LEFT JOIN gold g
            ON t.match_id = g.match_id
           AND t.timestamp = g.gtimestamp
        LEFT JOIN objectives o
            ON t.match_id = o.match_id
        GROUP BY t.match_id, t.timestamp, g.total_gold
    """

    match_info_timestamp_df = duckdb.sql(query).fetchdf()
    match_info_general_df = duckdb.sql(f"""
        SELECT
            match_id,
            winning_team,
            duration_s
        FROM match_info_df;
    """).fetchdf()

    return match_info_general_df, match_info_timestamp_df


# noinspection PyUnusedLocal
def _split_player_stats(match_player_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_timestamp_df = duckdb.sql(f"""
        SELECT
            match_id,
            account_id,
            unnest(list_slice("stats.time_stamp_s", 1, length("stats.time_stamp_s") - 1)) AS timestamp_s,
            unnest(list_slice("stats.net_worth", 1, length("stats.net_worth") - 1)) AS net_worth,
            unnest(list_slice("stats.ability_points", 1, length("stats.ability_points") - 1)) AS ability_points,
            unnest(list_slice("stats.level", 1, length("stats.level") - 1)) AS level
        FROM match_player_df;
    """).fetchdf()
    logging.debug(f"unnested player stats to {len(player_timestamp_df)} rows.")

    player_general_df = duckdb.sql(f"""
        SELECT
            match_id,
            account_id,
            team,
            hero_name,
            net_worth,
            ability_points,
            player_level
        FROM match_player_df;
    """).fetchdf()

    del match_player_df
    return player_general_df, player_timestamp_df