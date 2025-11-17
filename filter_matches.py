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

COLUMNS_TO_DROP: list[str] = ["start_time", "match_outcome", "match_mode", "game_mode", "is_high_skill_range_parties", "low_pri_pool", "new_player_pool", "average_badge_team0", "average_badge_team1", "rewards_eligible", "not_scored", "created_at", "game_mode_version"]
MATCH_METADATA_PATH: Path = Path("db_dump/match_metadata")
RELEVANT_MATCH_ID_RANGE: range = range(44, 46) # 44 and 45
OUTPUT_PATH: Path = Path("filtered_data")

START_DATETIME: datetime = datetime(2025, 10, 2, 23, 3, 5, tzinfo=ZoneInfo("Europe/Berlin"))
END_DATETIME: datetime = datetime(2025, 10, 25, 0, 54, 51, tzinfo=ZoneInfo("Europe/Berlin"))
MIN_RANK_BADGE: int = 101
MAX_RANK_DISPARITY: int = 2

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
        "match_ids": "45932614",
        "min_unix_timestamp": None,
        "max_unix_timestamp": None,
        "min_duration_s": None,
        "max_duration_s": None,
        "min_average_badge": None,
        "max_average_badge": None,
        "min_match_id": None,
        "max_match_id": None,
        "is_high_skill_range_parties": None,
        "is_low_pri_pool": None,
        "is_new_player_pool": None,
        "hero_ids": None,
        "order_by": "match_id",
        "order_direction": "desc",
        "limit": 1000
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

def read_match_metadata():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    files_to_load = [
        str(MATCH_METADATA_PATH / f"match_info_{i}.parquet")
        for i in RELEVANT_MATCH_ID_RANGE
        if (MATCH_METADATA_PATH / f"match_info_{i}.parquet").exists()
    ]
    if not files_to_load:
        raise FileNotFoundError("no matching parquet files found for the given range!")

    df = duckdb.sql(f"""
            SELECT *
            FROM read_parquet({files_to_load})
            WHERE start_time BETWEEN '{START_DATETIME}' AND '{END_DATETIME}'
            AND is_high_skill_range_parties IS FALSE
            AND low_pri_pool IS FALSE
            AND new_player_pool IS FALSE
            AND average_badge_team0 >= {MIN_RANK_BADGE}
            AND average_badge_team1 >= {MIN_RANK_BADGE}
            AND abs(CAST(average_badge_team0 AS BIGINT) - CAST(average_badge_team1 AS BIGINT)) <= {MAX_RANK_DISPARITY}
            AND rewards_eligible IS TRUE
            AND not_scored IS FALSE
        """).fetchdf()
    print(df)
    print(df.info())
    df = df.drop(columns=COLUMNS_TO_DROP, axis=1)
    df.to_parquet(OUTPUT_PATH / "match_info.parquet")

if __name__ == "__main__":
    # read_player_info()
    read_player_info_via_api()
    exit(0)