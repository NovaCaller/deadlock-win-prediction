from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import duckdb

# start_time between patches (2025-10-02T22:03:05+0200 to 2025-10-25T01:54:51+0200 (+1h on start and -1h on end)
# is_high_skill_range_parties: false
# filter low priority matchmaking matches as they have worse matchmaking: low_pri_pool: false
# new_player_pool: false
# ascendant 1+ (top ~12,5% of matches): average_badge_team0 >= 101 && average_badge_team1 >= 101
# max diff between average_badge_team_x: 2 ?
# filters bot matches and private games: rewards_eligible: true
# filter games with early leavers / afk players and cheaters (?) not_scored: false

RELEVANT_MATCH_INFO_COLUMNS: str = 'match_id, winning_team, duration_s, objectives_mask_team0, objectives_mask_team1, "objectives.destroyed_time_s", "objectives.team_objective", "objectives.team"'
RELEVANT_MATCH_PLAYER_COLUMNS: str = 'match_id, account_id, team, net_worth, hero_id, ability_points, player_level, abandon_match_time_s, "stats.time_stamp_s", "stats.net_worth", "stats.ability_points", "stats.tech_power", "stats.level"'
MATCH_METADATA_PATH: Path = Path("db_dump/match_metadata")
HEROES_PARQUET: Path = Path("db_dump/heroes.parquet")
RELEVANT_MATCH_ID_RANGE: range = range(45, 48) # 43 to 45
OUTPUT_PATH: Path = Path("filtered_data")

START_DATETIME: datetime = datetime(2025, 10, 2, 23, 3, 5, tzinfo=ZoneInfo("Europe/Berlin"))
END_DATETIME: datetime = datetime(2025, 10, 25, 0, 54, 51, tzinfo=ZoneInfo("Europe/Berlin"))
MIN_RANK_BADGE: int = 101
MAX_RANK_DISPARITY: int = 2
LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY: int = 60 # players can leave 90s before match end to not be considered leavers


def filter_matches():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    match_info_output_path = (OUTPUT_PATH / "match_info.parquet").absolute()
    match_player_output_path = (OUTPUT_PATH / "match_player.parquet").absolute()
    match_player_timestamp_path = (OUTPUT_PATH / "match_player_timestamp.parquet").absolute()
    match_player_general_path = (OUTPUT_PATH / "match_player_general.parquet").absolute()

    match_info_files = [
        str(MATCH_METADATA_PATH / f"match_info_{i}.parquet")
        for i in RELEVANT_MATCH_ID_RANGE
        if (MATCH_METADATA_PATH / f"match_info_{i}.parquet").exists()
    ]
    match_player_files = [
        str(MATCH_METADATA_PATH / f"match_player_{i}.parquet")
        for i in RELEVANT_MATCH_ID_RANGE
        if (MATCH_METADATA_PATH / f"match_player_{i}.parquet").exists()
    ]
    if not match_info_files or not match_player_files:
        raise FileNotFoundError(f"no matching parquet files found for the given range ({RELEVANT_MATCH_ID_RANGE})!")

    prefilter_match_info(match_info_files, match_info_output_path)
    prefilter_match_player(match_player_files, match_player_output_path, match_info_output_path)
    prune_matches_with_missing_player_data(match_info_output_path, match_player_output_path)
    prune_matches_with_early_leavers(match_info_output_path, match_player_output_path)
    split_player_stats(match_player_output_path, match_player_timestamp_path, match_player_general_path)
    replace_hero_ids_with_names(match_player_general_path)


def prefilter_match_info(input_parquet_files: list[str], output_parquet_path: Path):
    df = duckdb.sql(f"""
        SELECT {RELEVANT_MATCH_INFO_COLUMNS}
        FROM read_parquet({input_parquet_files})
        WHERE start_time BETWEEN '{START_DATETIME}' AND '{END_DATETIME}'
        AND duration_s >= 480
        AND is_high_skill_range_parties IS FALSE
        AND low_pri_pool IS FALSE
        AND new_player_pool IS FALSE
        AND average_badge_team0 >= {MIN_RANK_BADGE}
        AND average_badge_team1 >= {MIN_RANK_BADGE}
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
            ) <= {MAX_RANK_DISPARITY}
        AND rewards_eligible IS TRUE
        AND not_scored IS FALSE;
    """).fetchdf()
    print(f"matches after prefiltering: {len(df)}")
    df.to_parquet(output_parquet_path)


def prefilter_match_player(input_parquet_files: list[str], output_parquet_path: Path, match_info_parquet: Path) -> None:
    df = duckdb.sql(f"""
        SELECT {RELEVANT_MATCH_PLAYER_COLUMNS}
        FROM read_parquet({input_parquet_files})
        WHERE match_id IN (SELECT match_id FROM read_parquet('{str(match_info_parquet)}'));
    """).fetchdf()
    print(f"player rows after prefiltering: {len(df)}")
    df.to_parquet(output_parquet_path)


def prune_matches_with_missing_player_data(match_info_parquet: Path, match_player_parquet: Path) -> None:
    valid_ids = duckdb.sql(f"""
        WITH valid_ids AS (
            SELECT match_id
            FROM read_parquet('{str(match_player_parquet)}')
            GROUP BY match_id
            HAVING COUNT(*) = 12
        )
        SELECT * FROM valid_ids;
    """).fetchdf()
    print(f"number of valid ids: {len(valid_ids)}")
    info_df = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{str(match_info_parquet)}')
        WHERE match_id IN (SELECT match_id FROM valid_ids);
    """).fetchdf()
    player_df = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{str(match_player_parquet)}')
        WHERE match_id IN (SELECT match_id FROM valid_ids);
    """).fetchdf()
    print(f"matches after pruning matches with missing players: {len(info_df)}")
    print(f"player rows after pruning matches with missing players: {len(player_df)}")
    info_df.to_parquet(match_info_parquet)
    player_df.to_parquet(match_player_parquet)


def prune_matches_with_early_leavers(match_info_parquet: Path, match_player_parquet: Path) -> None:
    early_leaver_ids = duckdb.sql(f"""
        WITH early_leaver_ids AS (
            SELECT DISTINCT player.match_id
            FROM read_parquet('{str(match_player_parquet)}') AS player
            JOIN read_parquet('{str(match_info_parquet)}') AS info
            ON player.match_id = info.match_id
            WHERE player.abandon_match_time_s > 0
            AND info.duration_s - player.abandon_match_time_s > {str(LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY)}
        )
        SELECT * FROM early_leaver_ids;
    """).fetchdf()
    print(f"matches with early leavers: {len(early_leaver_ids)}")
    info_df = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{str(match_info_parquet)}')
        WHERE match_id NOT IN (SELECT match_id FROM early_leaver_ids);
    """).fetchdf()
    player_df = duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{str(match_player_parquet)}')
        WHERE match_id NOT IN (SELECT match_id FROM early_leaver_ids);
    """).fetchdf()
    print(f"matches after pruning matches with early leavers: {len(info_df)}")
    print(f"player rows after pruning matches with early leavers: {len(player_df)}")
    info_df.to_parquet(match_info_parquet)
    player_df.drop("abandon_match_time_s", axis=1).to_parquet(match_player_parquet)


def split_player_stats(match_player_parquet: Path, match_player_timestamp_output: Path, match_player_general_output: Path) -> None:
    player_timestamp_df = duckdb.sql(f"""
        SELECT
            match_id,
            account_id,
            unnest(list_slice("stats.time_stamp_s", 1, length("stats.time_stamp_s") - 1)) AS timestamp_s,
            unnest(list_slice("stats.net_worth", 1, length("stats.net_worth") - 1)) AS net_worth,
            unnest(list_slice("stats.ability_points", 1, length("stats.ability_points") - 1)) AS ability_points,
            unnest(list_slice("stats.tech_power", 1, length("stats.tech_power") - 1)) AS tech_power,
            unnest(list_slice("stats.level", 1, length("stats.level") - 1)) AS level
        FROM read_parquet('{match_player_parquet}');
    """).fetchdf()
    player_timestamp_df.to_parquet(match_player_timestamp_output)
    print(f"unnested player stats to {len(player_timestamp_df)} rows.")

    player_general_df = duckdb.sql(f"""
        SELECT
            match_id,
            account_id,
            team,
            hero_id,
            net_worth,
            ability_points,
            player_level
        FROM read_parquet('{match_player_parquet}');
    """).fetchdf()
    player_general_df.to_parquet(match_player_general_output)
    print(f"moved general player data to '{match_player_general_output.name}'.")

    match_player_parquet.unlink()
    print(f"removed leftover '{match_player_parquet.name}'.")


def replace_hero_ids_with_names(match_player_general_parquet: Path):
    player_general_df = duckdb.sql(f"""
        SELECT
            player_general.match_id,
            player_general.account_id,
            player_general.team,
            heroes.name as hero_name,
            player_general.net_worth,
            player_general.ability_points,
            player_general.player_level
        FROM read_parquet('{match_player_general_parquet}') as player_general
        JOIN read_parquet('{HEROES_PARQUET}') as heroes
        ON player_general.hero_id = heroes.id;
    """).fetchdf()
    player_general_df.to_parquet(match_player_general_parquet)
    print("replaced hero ids with names.")

def generate_objectives_time_series():
    match_info_parquet = OUTPUT_PATH / "match_info.parquet"
    match_player_timestamp_parquet = OUTPUT_PATH / "match_player_timestamp.parquet"
    output_parquet = OUTPUT_PATH / "match_info_timestamp.parquet"

    objective_names = [
        "Core", "Tier1Lane1", "Tier1Lane3", "Tier1Lane4",
        "Tier2Lane1", "Tier2Lane3", "Tier2Lane4",
        "BarrackBossLane1", "BarrackBossLane3", "BarrackBossLane4",
        "Titan", "TitanShieldGenerator1", "TitanShieldGenerator2"
    ]

    # CASE Statements für alle Objectives + Teams
    case_statements = []
    for team in [0, 1]:
        for obj in objective_names:
            case_statements.append(f"""
                MAX(
                    CASE
                        WHEN obj_team = {team} AND obj_name = '{obj}' AND timestamp >= destroyed_time
                        THEN 1
                        ELSE 0
                    END
                ) AS team{team}_{obj}
            """)

    query = f"""
        WITH timestamps AS (
            SELECT match_id, timestamp_s AS timestamp
            FROM read_parquet('{match_player_timestamp_parquet}')
            GROUP BY match_id, timestamp_s  -- <-- unique timestamps pro match
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
            FROM read_parquet('{match_info_parquet}')
        )
        SELECT
            t.match_id,
            t.timestamp,
            {', '.join(case_statements)}
        FROM timestamps t
        LEFT JOIN objectives o
        ON t.match_id = o.match_id
        GROUP BY t.match_id, t.timestamp
        ORDER BY t.match_id, t.timestamp
    """

    duckdb.sql(query).to_parquet(str(output_parquet))
    print(f"Generiert '{output_parquet}' mit allen Objectives korrekt gesetzt pro Match & Timestamp")


if __name__ == "__main__":
    filter_matches()
    print("successfully completed filtering matches.")
    generate_objectives_time_series()
    print("successfully generated objectives time series.")
    exit(0)