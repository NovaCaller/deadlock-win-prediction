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
# TODO: find out how to figure out which games have late leavers, e.g. match id 46186766 does not have indication in match_info_46 but marked as having leavers on

RELEVANT_MATCH_INFO_COLUMNS: str = 'match_id, winning_team, duration_s, objectives_mask_team0, objectives_mask_team1, "objectives.destroyed_time_s", "objectives.team_objective", "objectives.team"'
RELEVANT_MATCH_PLAYER_COLUMNS: str = 'match_id, account_id, team, net_worth, hero_id, ability_points, player_level, abandon_match_time_s, "stats.time_stamp_s", "stats.net_worth", "stats.ability_points", "stats.tech_power", "stats.level"'
MATCH_METADATA_PATH: Path = Path("db_dump/match_metadata")
RELEVANT_MATCH_ID_RANGE: range = range(43, 46) # 43 to 45
OUTPUT_PATH: Path = Path("filtered_data")

START_DATETIME: datetime = datetime(2025, 10, 2, 23, 3, 5, tzinfo=ZoneInfo("Europe/Berlin"))
END_DATETIME: datetime = datetime(2025, 10, 25, 0, 54, 51, tzinfo=ZoneInfo("Europe/Berlin"))
MIN_RANK_BADGE: int = 101
MAX_RANK_DISPARITY: int = 2
LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY: int = 60 # players can leave 90s before match end to not be considered leavers

API_URL = "https://api.deadlock-api.com/v1/matches/metadata"
# parameters not needed still included, since params may need to be added
API_PARAMS = {
        "include_info": "true",
        "include_objectives": "true",
        "include_mid_boss": "true",
        "include_player_info": "true",
        "include_player_items": "true",
        "include_player_stats": "true",
        "include_player_death_details": "true",
        "match_ids": None,
        "min_unix_timestamp": None,
        "max_unix_timestamp": None,
        "min_duration_s": 480,
        "max_duration_s": None,
        "min_average_badge": 101,
        "max_average_badge": None,
        "min_match_id": None,
        "max_match_id": None,
        "is_high_skill_range_parties": "false",
        "is_low_pri_pool": "false",
        "is_new_player_pool": "false",
        "hero_ids": None,
        "order_by": "match_id",
        "order_direction": "asc",
        "limit": 10
    }

def read_player_info():
    df = pd.read_parquet("db_dump/match_metadata/match_player_46.parquet")
    print(df.info)

def filter_player_attributes(player_list, allowed_keys):
    # player_list: list of dicts
    return [
        {k: v for k, v in player.items() if k in allowed_keys}
        for player in player_list
    ]

def read_player_info_via_api():
    import requests

    response = requests.get(API_URL, params=API_PARAMS)

    if response.ok:
        data = response.json()
        print(f"matches: {len(data)}")
        data = data[0]
    else:
        print("Error:", response.status_code, response.text)
        exit(1)

    rows = []

    total_net_worth_team_0 = 0
    total_net_worth_team_1 = 0

    for i in range(len(data["players"])):
        # player is a dictionary ..
        player = data["players"][i]
        account_id = player.get("account_id")
        hero_id = player.get("hero_id")
        #print("Hero ID " + str(hero_id))
        team = player.get("team")

        # encode variable team for later evaluation
        if team == "Team0":
            team = 0
        else:
            team = 1

        level = player.get("player_level")
        # Iterate over snapshots inside stats
        for snapshot in player.get("stats", []):
            row = {
                "account_id": account_id,
                "hero_id": hero_id,
                "team": team,
                "level": level,

                # stat specific information
                "timestamp": snapshot.get("time_stamp_s"),
                "ability_points": snapshot.get("ability_points"),
                "net_worth": snapshot.get("net_worth"),
                "tech_power": snapshot.get("tech_power"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    print(df)
    print(df.columns)

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
        AND abs(CAST(average_badge_team0 AS BIGINT) - CAST(average_badge_team1 AS BIGINT)) <= {MAX_RANK_DISPARITY}
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


if __name__ == "__main__":
    filter_matches()
    exit(0)