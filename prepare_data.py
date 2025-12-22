from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import duckdb
from pandas import read_parquet
from sklearn.preprocessing import OneHotEncoder

from src.filter_matches import filter_matches

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


def split_player_stats(match_player_parquet: Path, match_player_timestamp_output: Path, match_player_general_output: Path) -> None:
    player_timestamp_df = duckdb.sql(f"""
        SELECT
            match_id,
            account_id,
            unnest(list_slice("stats.time_stamp_s", 1, length("stats.time_stamp_s") - 1)) AS timestamp_s,
            unnest(list_slice("stats.net_worth", 1, length("stats.net_worth") - 1)) AS net_worth,
            unnest(list_slice("stats.ability_points", 1, length("stats.ability_points") - 1)) AS ability_points,
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
            FROM read_parquet('{match_player_timestamp_parquet}')
            GROUP BY match_id, timestamp_s
        ),
        gold AS (
            SELECT match_id, timestamp_s AS gtimestamp, CAST(SUM(net_worth) AS BIGINT) AS total_gold
            FROM read_parquet('{match_player_timestamp_parquet}')
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
            FROM read_parquet('{match_info_parquet}')
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
        ORDER BY t.match_id, t.timestamp
    """

    duckdb.sql(query).to_parquet(str(output_parquet))
    print(f"generated '{output_parquet}'.")

def normalize_features():
    timestamp_path = OUTPUT_PATH / "match_player_timestamp.parquet"
    df_ts = pd.read_parquet(timestamp_path)



    df_ts["total_gold_match_ts"] = (
        df_ts.groupby(["match_id", "timestamp_s"])["net_worth"].transform("sum")
    )
    df_ts["net_worth_ratio"] = (
        df_ts["net_worth"] / df_ts["total_gold_match_ts"]
    )

    numeric_cols_ts = [
        c for c in df_ts.columns
        if pd.api.types.is_numeric_dtype(df_ts[c]) and c not in ["match_id", "account_id", "net_worth_ratio"]
    ]

    df_ts_norm = df_ts.copy()
    df_ts_norm[numeric_cols_ts] = (
        df_ts_norm[numeric_cols_ts] - df_ts_norm[numeric_cols_ts].mean()
    ) / df_ts_norm[numeric_cols_ts].std(ddof=0)

    df_ts_norm = df_ts_norm.drop(columns=["total_gold_match_ts", "net_worth"])

    df_ts_norm.to_parquet(PROCESSED_PATH / "match_player_timestamp_norm.parquet")

    match_info_timestamp_path = OUTPUT_PATH / "match_info_timestamp.parquet"
    df_info_timestamp = pd.read_parquet(match_info_timestamp_path)

    numeric_cols_info = ["timestamp", "total_gold"]

    df_info_timestamp_norm = df_info_timestamp.copy()
    df_info_timestamp_norm[numeric_cols_info] = (
        df_info_timestamp_norm[numeric_cols_info] - df_info_timestamp_norm[numeric_cols_info].mean()
    ) / df_info_timestamp_norm[numeric_cols_info].std(ddof=0)

    df_info_timestamp_norm.to_parquet(PROCESSED_PATH / "match_info_timestamp_norm.parquet")


def normalize_team_attribute(match_player_general_parquet: Path, match_info: Path):

    # case expression is only needed for if you for some reason execute this again, if the variables are
    # already numerically encoded. Since then the check team='Team1' obviously does not work then.
    # If already numerically encoded, just let the value be and do nothing lol.
    match_player_general_df = duckdb.sql(f"""
        SELECT * REPLACE (
            CASE
                WHEN typeof(team) = 'VARCHAR'
                THEN (team='Team1')::INT
                ELSE team::INT
            END AS team
        )
        FROM read_parquet('{match_player_general_parquet}')
    """).fetchdf()

    match_info_df = duckdb.sql(f"""
        SELECT * REPLACE (
            CASE
                WHEN typeof(winning_team) = 'VARCHAR'
                THEN (winning_team='Team1')::INT
                ELSE winning_team::INT
            END AS winning_team
        )
        FROM read_parquet('{match_info}')
    """).fetchdf()

    match_player_general_df.to_parquet(match_player_general_parquet)
    match_info_df.to_parquet(match_info)

def encode_heroes(match_player_general, hero_col="hero_name"):
    match_player_general_df = read_parquet(match_player_general)
    hero_ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ).set_output(transform="pandas")

    hero_ohe_df = hero_ohe.fit_transform(match_player_general_df[[hero_col]])
    hero_ohe_df.index = match_player_general_df.index

    df_enc: pd.DataFrame = pd.concat(
        [match_player_general_df.drop(columns=[hero_col]), hero_ohe_df],
        axis=1
    )

    df_enc.to_parquet(match_player_general)

if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    match_info_output_path = (OUTPUT_PATH / "match_info.parquet").absolute()
    match_player_output_path = (OUTPUT_PATH / "match_player.parquet").absolute()
    match_player_timestamp_output_path = (OUTPUT_PATH / "match_player_timestamp.parquet").absolute()
    match_player_general_output_path = (OUTPUT_PATH / "match_player_general.parquet").absolute()

    info_df, player_df = filter_matches(MATCH_METADATA_PATH, RELEVANT_MATCH_ID_RANGE, RELEVANT_MATCH_INFO_COLUMNS, START_DATETIME, END_DATETIME, MIN_RANK_BADGE, MAX_RANK_DISPARITY, RELEVANT_MATCH_PLAYER_COLUMNS, LEAVER_TIME_TO_LEAVE_BEFORE_MATCH_END_LENIENCY)
    print("successfully completed filtering matches.")
    exit(0)

    split_player_stats(match_player_output_path, match_player_timestamp_output_path, match_player_general_output_path)
    replace_hero_ids_with_names(match_player_general_output_path)
    normalize_team_attribute(match_player_general_output_path, match_info_output_path)
    encode_heroes(match_player_general_output_path)

    generate_objectives_time_series()
    print("successfully generated objectives time series.")
    normalize_features()
    print("successfully normalized features.")
    exit(0)