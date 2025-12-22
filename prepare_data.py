from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

from src.encode_features import encode_features
from src.filter_matches import filter_matches
from src.normalize_features import normalize_features
from src.replace_hero_ids_with_names import replace_hero_ids_with_names
from src.split_off_timestamps import split_off_timestamps

# start_time between patches (2025-10-02T22:03:05+0200 to 2025-10-25T01:54:51+0200 (+1h on start and -1h on end)
# is_high_skill_range_parties: false
# filter low priority matchmaking matches as they have worse matchmaking: low_pri_pool: false
# new_player_pool: false
# ascendant 1+ (top ~12,5% of matches): average_badge_team0 >= 101 && average_badge_team1 >= 101
# max diff between average_badge_team_x: 2 ?
# filters bot matches and private games: rewards_eligible: true
# filter games with early leavers / afk players and cheaters (?) not_scored: false

RELEVANT_MATCH_INFO_COLUMNS: str = 'match_id, winning_team, duration_s, "objectives.destroyed_time_s", "objectives.team_objective", "objectives.team"'
RELEVANT_MATCH_PLAYER_COLUMNS: str = 'match_id, account_id, team, net_worth, hero_id, ability_points, player_level, abandon_match_time_s, "stats.time_stamp_s", "stats.net_worth", "stats.ability_points", "stats.level"'
MATCH_METADATA_PATH: Path = Path("db_dump/match_metadata")
HEROES_PARQUET: Path = Path("db_dump/heroes.parquet")
RELEVANT_MATCH_ID_RANGE: range = range(45, 48)  # 45 to 47
OUTPUT_PATH: Path = Path("filtered_data")
PROCESSED_PATH: Path = OUTPUT_PATH / "processed"

START_DATETIME: datetime = datetime(2025, 10, 25, 2, 54, 51, tzinfo=ZoneInfo("Europe/Berlin"))
END_DATETIME: datetime = datetime(2025, 11, 21, 22, 53, 12, tzinfo=ZoneInfo("Europe/Berlin"))
MIN_RANK_BADGE: int = 101
MAX_RANK_DISPARITY: int = 2
LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY: int = 70 # players can leave 70s before match end to not be considered leavers


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    info_df, player_df = filter_matches(MATCH_METADATA_PATH, RELEVANT_MATCH_ID_RANGE, RELEVANT_MATCH_INFO_COLUMNS, START_DATETIME, END_DATETIME, MIN_RANK_BADGE, MAX_RANK_DISPARITY, RELEVANT_MATCH_PLAYER_COLUMNS, LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY)
    print("done filtering matches.")

    player_df = replace_hero_ids_with_names(player_df, HEROES_PARQUET)
    print("done replacing hero ids with names.")

    info_general_df, info_timestamp_df, player_general_df, player_timestamp_df = split_off_timestamps(info_df, player_df)
    print("done splitting data into general and timestamps.")

    info_general_df.to_parquet(OUTPUT_PATH / "match_info_general.parquet")
    info_timestamp_df.to_parquet(OUTPUT_PATH / "match_info_timestamp.parquet")
    player_general_df.to_parquet(OUTPUT_PATH / "match_player_general.parquet")
    player_timestamp_df.to_parquet(OUTPUT_PATH / "match_player_timestamp.parquet")

    info_general_df.drop("duration_s", axis=1, inplace=True)
    player_general_df.drop(["ability_points", "player_level", "net_worth"], axis=1, inplace=True)

    info_general_df, info_timestamp_df, player_general_df, player_timestamp_df = encode_features(info_general_df, info_timestamp_df, player_general_df, player_timestamp_df)
    print("done encoding features.")

    info_general_df, info_timestamp_df, player_general_df, player_timestamp_df = normalize_features(info_general_df, info_timestamp_df, player_general_df, player_timestamp_df)
    print("done normalizing features.")
    print("finalized dataframes.")

    info_general_df.to_parquet(PROCESSED_PATH / "info_general_final.parquet")
    info_timestamp_df.to_parquet(PROCESSED_PATH / "info_timestamp_final.parquet")
    player_general_df.to_parquet(PROCESSED_PATH / "player_general_final.parquet")
    player_timestamp_df.to_parquet(PROCESSED_PATH / "player_timestamp_final.parquet")
