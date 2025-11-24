from pathlib import Path

import numpy as np
from pandas import read_parquet
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

# optional für prints wo man sich alles vom dataframe anschauen will
#pd.set_option("display.max_rows", None)
#pd.set_option("display.max_columns", None)
#pd.set_option("display.width", None)
#pd.set_option("display.max_colwidth", None)

OUTPUT_PATH: Path = Path("filtered_data")

def compare_wins_and_gold_difference_ts(match_metadata_path, match_player_path, match_player_timestamp_path):
    metadata_columns_to_drop = ["duration_s", "objectives_mask_team0", "objectives_mask_team1", "objectives.destroyed_time_s", "objectives.team_objective", "objectives.team"]
    player_general_columns_to_drop = ["hero_id", "ability_points", "player_level", "net_worth"]
    player_timestamp_columns_to_drop = ["ability_points", "tech_power", "level"]

    metadata_df = pd.DataFrame(read_parquet(match_metadata_path))
    player_general_df = pd.DataFrame(read_parquet(match_player_path))
    player_timestamp_df = pd.DataFrame(read_parquet(match_player_timestamp_path))

    metadata_df = metadata_df.drop(metadata_columns_to_drop, axis=1)
    player_general_df = player_general_df.drop(player_general_columns_to_drop, axis=1)
    player_timestamp_df = player_timestamp_df.drop(player_timestamp_columns_to_drop, axis=1)

    prepared_df = metadata_df.merge(player_general_df, how="inner", on="match_id")
    prepared_df = prepared_df.merge(player_timestamp_df, how="inner", on=["match_id","account_id"])
    prepared_df["net_worth"] = prepared_df["net_worth"].astype(np.int64)

    # count total gold per team in each match & timestamp
    gold_by_team_ts = (
        prepared_df
        .groupby(["match_id", "timestamp_s", "team", "winning_team"])["net_worth"]
        .sum()
        .reset_index(name="total_gold")
    )

    # Pivot to get one column per team
    pivoted = gold_by_team_ts.pivot(index=["match_id", "timestamp_s", "winning_team"],columns="team",values="total_gold")

    # Calculate Gold Difference in pct
    pivoted["gold_diff_pct"] = (
                                       (pivoted["Team0"] - pivoted["Team1"]) /
                                       pivoted[["Team0", "Team1"]].max(axis=1)
                               ) * 100

    # Clean up axis name
    pivoted = pivoted.rename_axis(columns=None)
    gold_diff_df = pivoted.reset_index()

    print(gold_diff_df)

    min_val = gold_diff_df["gold_diff_pct"].min()
    max_val = gold_diff_df["gold_diff_pct"].max()

    print("Min and max values: " + str(min_val) + "  ,  " + str(max_val))
    return gold_diff_df


def calculate_end_gold_difference(match_metadata_path, match_player_path):
    metadata_columns_to_drop = ["duration_s", "objectives_mask_team0", "objectives_mask_team1",
                                "objectives.destroyed_time_s", "objectives.team_objective", "objectives.team"]
    player_general_columns_to_drop = ["hero_id", "ability_points", "player_level"]

    metadata_df = pd.DataFrame(read_parquet(match_metadata_path))
    player_general_df = pd.DataFrame(read_parquet(match_player_path))

    # drop irrelevant columns
    metadata_df = metadata_df.drop(metadata_columns_to_drop, axis=1)
    player_general_df = player_general_df.drop(player_general_columns_to_drop, axis=1)

    prepared_df = metadata_df.merge(player_general_df, how="inner", on="match_id")
    prepared_df["net_worth"] = prepared_df["net_worth"].astype(np.int64) # otherwise it would be u32

    # count total gold per team in each match
    gold_by_team = (
        prepared_df
        .groupby(["match_id", "team", "winning_team"])["net_worth"]
        .sum()
        .reset_index(name="total_gold")
    )

    # Pivot to get one column per team
    pivoted = (gold_by_team.pivot(index=["match_id", "winning_team"], columns="team",
                                    values="total_gold"))

    # Calculate Gold Difference in pct
    pivoted["gold_diff_pct"] = (
                                       (pivoted["Team0"] - pivoted["Team1"]) /
                                       pivoted[["Team0", "Team1"]].max(axis=1)
                               ) * 100

    # Clean up axis name
    pivoted = pivoted.rename_axis(columns=None)

    gold_diff_df = pivoted.reset_index()

    print(gold_diff_df)

    # Theoretisch notwendig wenn wir mehrere Zeilen pro Match hätten für die Golddifferenz jeweils aus der Sicht eines Teams
    # df_unique_team0 = gold_diff_df.drop_duplicates("match_id").copy()
    df_unique_team0 = gold_diff_df.copy()
    # Define: did Team0 win?
    df_unique_team0["result"] = df_unique_team0.apply(
        lambda r: "won" if r["winning_team"] == "Team0" else "lost",
        axis=1
    )

    # Make the boxplot
    df_unique_team0.boxplot(column="gold_diff_pct", by="result")
    plt.xlabel("Result for Team0")
    plt.ylabel("Gold Difference (%)")
    plt.title("Gold Difference (%) by Win/Loss")
    plt.suptitle("")  # remove Pandas default title
    plt.show()

    df_unique_team1 = gold_diff_df.copy()

    # df_unique_team1["gold_diff_pct"] = -df_unique_team1["gold_diff_pct"]

    df_unique_team1["result"] = df_unique_team1.apply(
        lambda r: "won" if r["winning_team"] == "Team1" else "lost",
        axis=1
    )

    # Make the boxplot
    df_unique_team1.boxplot(column="gold_diff_pct", by="result")
    plt.xlabel("Result for Team1")
    plt.ylabel("Gold Difference (%)")
    plt.title("Gold Difference (%) by Win/Loss")
    plt.suptitle("")  # remove Pandas default title
    plt.show()

    return gold_diff_df


def analyze_most_popular_heroes(match_player_path: Path, top_n: int = 20):

    # Data prep
    player_general_columns_to_drop = ["match_id", "account_id", "team", "net_worth", "ability_points", "player_level"]
    player_general_df = pd.read_parquet(match_player_path)
    player_general_df = player_general_df.drop(player_general_columns_to_drop, axis=1)

    # Count picks per hero
    hero_pickrate_df = player_general_df['hero_name'].value_counts().reset_index()
    hero_pickrate_df.columns = ['hero_name', 'pick_count']

    # Bar plot of top 20 heroes
    hero_pickrate_df.head(top_n).plot(
        kind='bar',
        x='hero_name',
        y='pick_count',
        legend=False,
        figsize=(12, 6),
        title=f"Top {top_n} Most Picked Heroes"
    )
    plt.xlabel("Hero")
    plt.ylabel("Pick Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return hero_pickrate_df


# debugging
def get_info_from_db_for_specific_match(db_pq):
    df = duckdb.sql(f"""
    SELECT * FROM read_parquet('{str(db_pq)}')
    WHERE match_id = 44611001
    """).fetchdf()

    return df
if __name__ == "__main__" :
    match_info_output_path = (OUTPUT_PATH / "match_info.parquet").absolute()
    match_player_output_path = (OUTPUT_PATH / "match_player_general.parquet").absolute()
    match_player_ts_output_path = (OUTPUT_PATH / "match_player_timestamp.parquet").absolute()

    calculate_end_gold_difference(match_info_output_path, match_player_output_path)
    # compare_wins_and_gold_difference_ts(match_info_output_path, match_player_output_path, match_player_ts_output_path)
    analyze_most_popular_heroes(match_player_output_path, top_n=30)