from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


def filter_matches(match_metadata_path: Path,
                   match_id_range: range, info_columns: str, start_datetime: datetime, end_datetime: datetime,  # parameters for prefilter_match_info
                   min_rank_badge: int, max_rank_disparity: int,  # parameters for prefilter_match_info (cont.)
                   player_columns: str,  # parameter for prefilter_match_player
                   leaver_time_to_leave_before_match_end_leniency: int, # parameter for prune_matches_with_early_leavers
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:

    match_info_files = [
        str(match_metadata_path / f"match_info_{i}.parquet")
        for i in match_id_range
        if (match_metadata_path / f"match_info_{i}.parquet").exists()
    ]
    match_player_files = [
        str(match_metadata_path / f"match_player_{i}.parquet")
        for i in match_id_range
        if (match_metadata_path / f"match_player_{i}.parquet").exists()
    ]
    if not match_info_files or not match_player_files:
        raise FileNotFoundError(f"no matching parquet files found for the given range ({match_id_range})!")

    match_info_df = _prefilter_match_info(match_info_files, info_columns, start_datetime, end_datetime, min_rank_badge, max_rank_disparity)
    match_player_df = _prefilter_match_player(match_player_files, player_columns, match_info_df)
    match_info_df, match_player_df = _prune_matches_with_missing_player_data(match_info_df, match_player_df)
    match_info_df, match_player_df = _prune_matches_with_early_leavers(match_info_df, match_player_df, leaver_time_to_leave_before_match_end_leniency)
    return match_info_df, match_player_df


def _prefilter_match_info(input_parquet_files: list[str], info_columns: str, start_datetime: datetime, end_datetime: datetime, min_rank_badge: int, max_rank_disparity: int) -> pd.DataFrame:
    df = duckdb.sql(f"""
        SELECT {info_columns}
        FROM read_parquet({input_parquet_files})
        WHERE start_time BETWEEN '{start_datetime}' AND '{end_datetime}'
        AND duration_s >= 480
        AND is_high_skill_range_parties IS FALSE
        AND low_pri_pool IS FALSE
        AND new_player_pool IS FALSE
        AND average_badge_team0 >= {min_rank_badge}
        AND average_badge_team1 >= {min_rank_badge}
        AND
            ABS(
                CAST( --- convert to linear scale: 1 -> 1, 6 -> 6, 11 -> 7, 16 -> 12, 21 -> 13 etc.
                    ((average_badge_team0 // 10 * 6) + (average_badge_team0 % 10))
                    AS BIGINT 
                )
                - CAST(
                    ((average_badge_team1 // 10 * 6) + (average_badge_team1 % 10))
                    AS BIGINT 
                )
            ) <= {max_rank_disparity}
        AND rewards_eligible IS TRUE
        AND not_scored IS FALSE;
    """).fetchdf()
    print(f"matches after prefiltering: {len(df)}")
    return df


# noinspection PyUnusedLocal
def _prefilter_match_player(input_parquet_files: list[str], player_columns: str, match_info_df: pd.DataFrame) -> pd.DataFrame:
    df = duckdb.sql(f"""
        SELECT {player_columns}
        FROM read_parquet({input_parquet_files})
        WHERE match_id IN (SELECT match_id FROM match_info_df);
    """).fetchdf()
    print(f"player rows after prefiltering: {len(df)}")
    return df


# noinspection PyUnusedLocal
def _prune_matches_with_missing_player_data(match_info_df: pd.DataFrame, match_player_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_ids = duckdb.sql(f"""
        WITH valid_ids AS (
            SELECT match_id
            FROM match_player_df
            GROUP BY match_id
            HAVING COUNT(*) = 12
        )
        SELECT * FROM valid_ids;
    """).fetchdf()
    print(f"number of valid ids: {len(valid_ids)}")
    info_df = duckdb.sql(f"""
        SELECT *
        FROM match_info_df
        WHERE match_id IN (SELECT match_id FROM valid_ids);
    """).fetchdf()
    player_df = duckdb.sql(f"""
        SELECT *
        FROM match_player_df
        WHERE match_id IN (SELECT match_id FROM valid_ids);
    """).fetchdf()
    print(f"matches after pruning matches with missing players: {len(info_df)}")
    print(f"player rows after pruning matches with missing players: {len(player_df)}")
    return info_df, player_df


# noinspection PyUnusedLocal
def _prune_matches_with_early_leavers(match_info_df: pd.DataFrame, match_player_df: pd.DataFrame, leaver_time_to_leave_before_match_end_leniency: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    early_leaver_ids = duckdb.sql(f"""
        WITH early_leaver_ids AS (
            SELECT DISTINCT player.match_id
            FROM match_player_df AS player
            JOIN match_info_df AS info
            ON player.match_id = info.match_id
            WHERE player.abandon_match_time_s > 0
            AND info.duration_s - player.abandon_match_time_s > {str(leaver_time_to_leave_before_match_end_leniency)}
        )
        SELECT * FROM early_leaver_ids;
    """).fetchdf()
    print(f"matches with early leavers: {len(early_leaver_ids)}")
    info_df = duckdb.sql(f"""
        SELECT *
        FROM match_info_df
        WHERE match_id NOT IN (SELECT match_id FROM early_leaver_ids);
    """).fetchdf()
    player_df = duckdb.sql(f"""
        SELECT * EXCLUDE (abandon_match_time_s)
        FROM match_player_df
        WHERE match_id NOT IN (SELECT match_id FROM early_leaver_ids);
    """).fetchdf()
    print(f"matches after pruning matches with early leavers: {len(info_df)}")
    print(f"player rows after pruning matches with early leavers: {len(player_df)}")
    return info_df, player_df