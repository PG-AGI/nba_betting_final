## Imports and Global Settings
import os
import json
import sys
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.gridspec as gridspec

here = os.getcwd()
sys.path.append(os.path.join(here, ".."))
sys.path.append('D:\\Projects\\NBA_Betting')
from config import NBA_IMPORTANT_DATES, TEAM_MAP
from src.etl.main_etl import ETLPipeline
sys.path.append('D:\\Projects\\NBA_Betting\\src')
from utils.general_utils import (
    find_season_information,
    determine_season_type,
    add_season_timeframe_info,
)


RDS_ENDPOINT = os.getenv("RDS_ENDPOINT")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")

# Pandas Settings
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.options.display.max_info_columns = 1000
pd.options.display.precision = 5

# Graphing Settings
sns.set_theme()
## Data Loading
### Database Connection
username = "postgres"
password = 123654
endpoint = "localhost"
database = "nba_betting"
port = "5432"

# Create the connection string
connection_string = (
    f"postgresql+psycopg2://{username}:{password}@{endpoint}:{port}/{database}"
)
### Loading Games
start_date = "2010-09-01"

features = [
    "game_id",
    "game_datetime",
    "home_team",
    "away_team",
    "open_line",
    "home_score",
    "away_score",
]
games_query = (
    f"SELECT {', '.join(features)} FROM games WHERE game_datetime >= '{start_date}';"
)

with create_engine(connection_string).connect() as connection:
    games = pd.read_sql_query(games_query, connection, parse_dates=["game_datetime"])
### Loading Features
start_date = "2010-09-01"
start_date_int = int(start_date.replace("-", ""))  # Convert date to YYYYMMDD format

features = ["game_id", "data"]

# Extracting the YYYYMMDD portion of the game_id and comparing it with start_date_int
features_query = f"SELECT {', '.join(features)} FROM all_features_json WHERE CAST(LEFT(game_id, 8) AS INTEGER) >= {start_date_int};"

with create_engine(connection_string).connect() as connection:
    all_features = pd.read_sql_query(features_query, connection)

# Normalize the JSON strings in the 'data' column
expanded_data = pd.json_normalize(all_features["data"])

# Drop the original 'data' column and concatenate the expanded data
all_features = pd.concat([all_features.drop(columns=["data"]), expanded_data], axis=1)
### Loading Other Tables
# start_date = "2021-09-01"
# # NBA Stats Team Stats - Traditional
# features = [
#     "team_name",
#     "to_date",
#     "season",
#     "season_type",
#     "games",
#     "w",
#     "l",
#     "w_pct",
#     "plus_minus",
# ]
# team_nbastats_traditional_query = f"SELECT {', '.join(features)} FROM team_nbastats_general_traditional WHERE to_date >= '{start_date}';"

# # 538 Games and Team Stats
# features = [
#     "date",
#     "season",
#     "season_type",
#     "team1",
#     "team2",
#     "elo1_pre",
#     "elo2_pre",
#     "elo_prob1",
#     "elo_prob2",
#     "elo1_post",
#     "elo2_post",
#     "raptor1_pre",
#     "raptor2_pre",
#     "raptor_prob1",
#     "raptor_prob2",
#     "score1",
#     "score2",
# ]
# team_538_query = f"SELECT {', '.join(features)} FROM team_fivethirtyeight_games WHERE date >= '{start_date}'"

# with create_engine(connection_string).connect() as connection:
#     team_nbastats_traditional = pd.read_sql(
#         team_nbastats_traditional_query, connection, parse_dates=["to_date"]
#     )
#     team_538 = pd.read_sql(team_538_query, connection, parse_dates=["date"])
## Working Tables - **Restart From Here**
# Games
games_df = games.copy()

# Features
all_features_df = all_features.copy()
# # Other Tables
# # NBA Stats Team Traditional
# nbastats_team_traditional_df = team_nbastats_traditional.copy()

# # 538 Games and Team Stats
# team_538_df = team_538.copy()
## Data Cleaning and Preprocessing 
def print_table_info(df):
    # Print a sample of the data
    print("\nSample:")
    print(df.sample(10, random_state=42, ignore_index=True))

    # Print the info()
    print("\nInfo:")
    print(df.info(verbose=True, show_counts=True))
for df in [games_df, all_features_df]:
    print_table_info(df)
### Standardize Team Names
# nbastats_team_traditional_df = ETLPipeline.standardize_team_names(
#     nbastats_team_traditional_df, ["team_name"], TEAM_MAP, print_details=True
# )
# team_538_df = ETLPipeline.standardize_team_names(
#     team_538_df, ["team1", "team2"], TEAM_MAP, print_details=True
# )
### Downcast Data Types
# nbastats_team_traditional_df = ETLPipeline.downcast_data_types(
#     nbastats_team_traditional_df, print_details=True
# )
# team_538_df = ETLPipeline.downcast_data_types(team_538_df, print_details=True)
### Duplicate Records
# nbastats_team_traditional_df = ETLPipeline.check_duplicates(
#     nbastats_team_traditional_df, ["team_name", "to_date", "games"], print_details=True
# )
# team_538_df = ETLPipeline.check_duplicates(
#     team_538_df, ["date", "team1", "team2"], print_details=True
# )
### Pandas Profiling
# games_profile_report = ProfileReport(games_df, title="Games Profiling Report")
# games_profile_report.to_notebook_iframe()
# all_features_profile_report = ProfileReport(
#     all_features_df, title="All Features Profiling Report"
# )
# all_features_profile_report.to_notebook_iframe()
# nbastats_team_traditional_profile_report = ProfileReport(
#     nbastats_team_traditional_df, title="NBAStats Team Traditional Profiling Report"
# )
# nbastats_team_traditional_profile_report.to_notebook_iframe()
# team_538_profile_report = ProfileReport(team_538_df, title="Team 538 Profiling Report")
# team_538_profile_report.to_notebook_iframe()
## Vegas Miss Analysis
games_vegas_analysis_df = games_df.copy()
### Add Season and Date Information
games_vegas_analysis_df = add_season_timeframe_info(games_vegas_analysis_df)
### Optional: Restrict to Regular Season or Postseason Games Only
games_vegas_analysis_df = games_vegas_analysis_df[
    games_vegas_analysis_df["season_type"] == "reg"
]
### Add Targets - Vegas Miss and Vegas Absolute Miss
games_vegas_analysis_df["total_score"] = (
    games_vegas_analysis_df["home_score"] + games_vegas_analysis_df["away_score"]
)
games_vegas_analysis_df["actual_score_diff_hv"] = (
    games_vegas_analysis_df["home_score"] - games_vegas_analysis_df["away_score"]
)
games_vegas_analysis_df["vegas_score_diff_hv"] = -games_vegas_analysis_df["open_line"]
games_vegas_analysis_df["vegas_miss"] = (
    games_vegas_analysis_df["actual_score_diff_hv"]
    - games_vegas_analysis_df["vegas_score_diff_hv"]
)
games_vegas_analysis_df["vegas_miss_zscore"] = (
    games_vegas_analysis_df["vegas_miss"] - games_vegas_analysis_df["vegas_miss"].mean()
) / games_vegas_analysis_df["vegas_miss"].std()
games_vegas_analysis_df["vegas_miss_abs"] = games_vegas_analysis_df["vegas_miss"].abs()
games_vegas_analysis_df["vegas_miss_abs_zscore"] = (
    games_vegas_analysis_df["vegas_miss_abs"]
    - games_vegas_analysis_df["vegas_miss_abs"].mean()
) / games_vegas_analysis_df["vegas_miss_abs"].std()

# A negative vegas_miss means that vegas undervalued the home team and/or overvalued the away team
# A negative vegas_miss also means that the home team outperformed vegas' expectations
### Univariate Analysis for Vegas Miss and Vegas Absolute Miss
def compute_stats(series, print_stats=False, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    series = pd.Series(series)  # Make sure the input is a pandas Series
    stats = {
        "count": len(series),
        "max": series.max(),
        "min": series.min(),
        "range": series.max() - series.min(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode()[0] if not series.mode().empty else None,
        "variance": series.var(),
        "std_dev": series.std(),
        "IQR": series.quantile(0.75) - series.quantile(0.25),
        "skewness": series.skew(),
        "kurtosis": series.kurt(),
    }

    # Add percentiles to stats
    for percentile in percentiles:
        key = f"{int(percentile*100)}th_pct"
        stats[key] = series.quantile(percentile)

    if print_stats:
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")

    return stats
print("VEGAS MISS STATS")
vegas_miss_stats = compute_stats(games_vegas_analysis_df["vegas_miss"], print_stats=True)
print("\nVEGAS ABSOLUTE MISS STATS")
vegas_abs_miss_stats = compute_stats(
    games_vegas_analysis_df["vegas_miss_abs"], print_stats=True
)
def create_univarate_plots(df, target, title):
    # create 1x3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # displot with rug in first subplot
    sns.histplot(df[target], kde=True, ax=ax[0])
    sns.rugplot(df[target], ax=ax[0])
    ax[0].set_title("Density plot with Rug plot")

    # boxplot in second subplot
    sns.boxplot(x=df[target], ax=ax[1])
    ax[1].set_title("Box plot")

    # ECDF in third subplot
    sns.ecdfplot(df[target], ax=ax[2])
    ax[2].set_title("ECDF plot")

    # set title for the whole figure
    fig.suptitle(title, fontsize=20)

    # automatically adjust the subplot layout
    plt.tight_layout()

    # adjust the position of the suptitle to prevent it from overlapping with subplots' titles
    fig.subplots_adjust(top=0.88)

    # display the plot
    plt.show()
create_univarate_plots(games_vegas_analysis_df, "vegas_miss", "Vegas Miss")
create_univarate_plots(games_vegas_analysis_df, "vegas_miss_abs", "Vegas Absolute Miss")
games_vegas_analysis_df.info()
### Main Vegas Miss Graph
def vegas_miss_graph(vegas_miss_abs, season, save=False, image_name=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(
        " Vegas Average Point Spread Error Per Game",
        fontsize=24,
        pad=16,
        fontweight="bold",
    )
    ax.set_xlabel("Season", fontsize=18, labelpad=8, fontweight="bold")
    ax.set_ylabel("Spread Error (Points)", fontsize=18, labelpad=8, fontweight="bold")

    # Create a new DataFrame from the two Series
    df = pd.DataFrame({"vegas_miss_abs": vegas_miss_abs, "season": season})

    # Sort the DataFrame by the 'season' column
    df = df.sort_values("season")

    # Calculate overall mean value
    overall_avg = df["vegas_miss_abs"].mean()

    sns.lineplot(
        x="season", y="vegas_miss_abs", data=df, ax=ax, linewidth=4, errorbar=None
    )

    ax.axhline(overall_avg, color="#C9082A", linestyle="--", linewidth=2)
    ax.text(
        x=df["season"].min(),
        y=overall_avg + 0.05,
        s=f"Overall Average: {overall_avg:.2f}",
        color="#C9082A",
        fontsize=16,
        fontweight="bold",
    )

    # Get the existing x-ticks and labels
    existing_ticks = ax.get_xticks()
    existing_labels = ax.get_xticklabels()

    # Extract the start year from each label and set new x-tick labels
    new_labels = [label.get_text().split("-")[0] for label in existing_labels]
    ax.set_xticklabels(new_labels)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.legend(fontsize=16)

    if save:
        plt.savefig(f"{image_name}.png", dpi=300, bbox_inches="tight")
vegas_miss_graph(
    games_vegas_analysis_df["vegas_miss_abs"],
    games_vegas_analysis_df["season"],
    save=False,
    image_name="vegas_miss_abs",
)
### Secondary Vegas Miss Graphs
# Sort the DataFrame by the 'season' column
games_vegas_analysis_df = games_vegas_analysis_df.sort_values("season")

# Calculate the total average score for each season
games_vegas_analysis_df["avg_score"] = (
    games_vegas_analysis_df["home_score"] + games_vegas_analysis_df["away_score"]
) / 2
avg_score_by_season = games_vegas_analysis_df.groupby("season")["avg_score"].mean()

# Adjust the "vegas_miss_abs" values by the total average score
games_vegas_analysis_df["adjusted_vegas_miss_abs"] = games_vegas_analysis_df.apply(
    lambda row: row["vegas_miss_abs"] / avg_score_by_season[row["season"]], axis=1
)

# Calculate overall average
overall_mean = games_vegas_analysis_df["vegas_miss_abs"].mean()
overall_mean_pct = games_vegas_analysis_df["adjusted_vegas_miss_abs"].mean()

# Create a line plot
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=games_vegas_analysis_df,
    x="season",
    y="adjusted_vegas_miss_abs",
    marker="o",
    linewidth=2,
    markersize=5,
)

# Add horizontal line showing overall average
plt.axhline(overall_mean_pct, color="#C9082A", linestyle="--")

# Annotate the overall mean
overall_mean_text = f"Overall Mean: {overall_mean:.2f} ({overall_mean_pct*100:.2f})%"
plt.annotate(
    overall_mean_text,
    xy=(0.05, 0.05),
    xycoords="axes fraction",
    backgroundcolor="white",
    fontsize=16,
    color="#C9082A",
)

# Annotating the mean for each week of the season
for season in games_vegas_analysis_df["season"].unique():
    season_mean = games_vegas_analysis_df.loc[
        games_vegas_analysis_df["season"] == season, "adjusted_vegas_miss_abs"
    ].mean()
    season_abs_mean = games_vegas_analysis_df.loc[
        games_vegas_analysis_df["season"] == season, "vegas_miss_abs"
    ].mean()
    plt.annotate(
        f"{season}\nMean: {season_abs_mean:.2f}\n({season_mean*100:.2f}%)",
        (season, season_mean),
        textcoords="offset points",
        xytext=(0, -100),
        ha="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

# Set title
plt.title("Vegas Miss by Season (Adjusted for Total Average Score)")
plt.xlabel("Season")
plt.ylabel("Vegas Miss (Percentage of Average Team Score)")
plt.xticks(rotation=45, ha="right")
plt.yticks(
    ticks=plt.yticks()[0], labels=[f"{label*100:.0f}%" for label in plt.yticks()[0]]
)

plt.show()
# Calculate overall average
overall_avg = games_vegas_analysis_df["vegas_miss_abs"].mean()

# Create a line plot
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=games_vegas_analysis_df,
    x="week_of_season",
    y="vegas_miss_abs",
    marker="o",
    linewidth=2,
    markersize=5,
)

# Add horizontal line showing overall average
plt.axhline(overall_avg, color="#C9082A", linestyle="--")

# Annotate the overall mean
overall_mean_text = f"Overall Mean: {overall_avg:.2f}"
plt.annotate(
    overall_mean_text,
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    backgroundcolor="white",
    fontsize=16,
    color="#C9082A",
)

# Annotating the mean for each week of the season
for week in games_vegas_analysis_df["week_of_season"].unique():
    week_mean = games_vegas_analysis_df.loc[
        games_vegas_analysis_df["week_of_season"] == week, "vegas_miss_abs"
    ].mean()
    plt.annotate(
        f"{week_mean:.2f}",
        (week, week_mean),
        textcoords="offset points",
        xytext=(0, 30),
        ha="center",
        fontsize=10,
    )

# Set title
plt.title("Vegas Miss by Week of Season")
plt.xlabel("Week of Season")
plt.ylabel("Vegas Miss (Points)")


plt.show()
### Multivariate Analysis for Vegas Miss and Vegas Absolute Miss
def compare_stats(df, series_name, situations):
    overall_stats = compute_stats(df[series_name])

    result = pd.DataFrame()
    for situation, condition in situations.items():
        subset = df.loc[condition]
        subset_stats = compute_stats(subset[series_name])
        for key, value in subset_stats.items():
            result.at[situation, key] = value
        t_stat, p_value_ttest = ttest_ind(
            df[series_name], subset[series_name], equal_var=False, nan_policy="omit"
        )
        ks_stat, p_value_ks = ks_2samp(
            df[series_name], subset[series_name], alternative="two-sided"
        )
        result.at[situation, "p_value_ttest"] = p_value_ttest
        result.at[situation, "p_value_ks"] = p_value_ks
        result.at[situation, "significant_difference_ttest"] = (
            "Yes" if p_value_ttest < 0.05 else "No"
        )
        result.at[situation, "significant_difference_ks"] = (
            "Yes" if p_value_ks < 0.05 else "No"
        )
        result.at[situation, "difference_in_mean"] = (
            subset[series_name].mean() - df[series_name].mean()
        )

    return result
situations = {
    "Home Team is Favored": games_vegas_analysis_df["vegas_score_diff_hv"] > 0,
    "Away Team is Favored": games_vegas_analysis_df["vegas_score_diff_hv"] < 0,
    "Home team wins ATS": games_vegas_analysis_df["actual_score_diff_hv"]
    > games_vegas_analysis_df["vegas_score_diff_hv"],
    "Away team wins ATS": games_vegas_analysis_df["actual_score_diff_hv"]
    < games_vegas_analysis_df["vegas_score_diff_hv"],
    "Favorite wins ATS": (games_vegas_analysis_df["vegas_score_diff_hv"] > 0)
    & (
        games_vegas_analysis_df["actual_score_diff_hv"]
        > games_vegas_analysis_df["vegas_score_diff_hv"]
    )
    | (games_vegas_analysis_df["vegas_score_diff_hv"] < 0)
    & (
        games_vegas_analysis_df["actual_score_diff_hv"]
        < games_vegas_analysis_df["vegas_score_diff_hv"]
    ),
    "Underdog wins ATS": (games_vegas_analysis_df["vegas_score_diff_hv"] > 0)
    & (
        games_vegas_analysis_df["actual_score_diff_hv"]
        < games_vegas_analysis_df["vegas_score_diff_hv"]
    )
    | (games_vegas_analysis_df["vegas_score_diff_hv"] < 0)
    & (
        games_vegas_analysis_df["actual_score_diff_hv"]
        > games_vegas_analysis_df["vegas_score_diff_hv"]
    ),
}

for i in range(1, int(max(games_vegas_analysis_df["month_of_season"])) + 1):
    situations[f"Month {i} of season"] = games_vegas_analysis_df["month_of_season"] == i

spread_conditions = [
    (games_vegas_analysis_df["vegas_score_diff_hv"].abs() <= 3, "Within 3 of even"),
    (
        (games_vegas_analysis_df["vegas_score_diff_hv"].abs() > 3)
        & (games_vegas_analysis_df["vegas_score_diff_hv"].abs() <= 5),
        "3-5 of even",
    ),
    (
        (games_vegas_analysis_df["vegas_score_diff_hv"].abs() > 5)
        & (games_vegas_analysis_df["vegas_score_diff_hv"].abs() <= 10),
        "5-10 of even",
    ),
    (games_vegas_analysis_df["vegas_score_diff_hv"].abs() > 10, ">10 from even"),
]

for condition, label in spread_conditions:
    situations[label] = condition

result = compare_stats(games_vegas_analysis_df, "vegas_miss_abs", situations)
result
# Compute pairwise correlation of columns
correlation_df = games_vegas_analysis_df.corr(numeric_only=True)

# Select the rows corresponding to the target columns
correlation_df = correlation_df.loc[["vegas_miss", "vegas_miss_abs"]]
correlation_df
# games_vegas_analysis_df_report = ProfileReport(
#     games_vegas_analysis_df, title="Game/Bet Profiling Report"
# )
# games_vegas_analysis_df_report.to_notebook_iframe()
## Target Feature Correlations
target_feature_df = all_features_df.copy()
target_feature_df["open_line_hv"] = -target_feature_df["open_line"]
target_feature_df["actual_score_diff_hv"] = (
    target_feature_df["home_score"] - target_feature_df["away_score"]
)
target_feature_df["REG_TARGET"] = target_feature_df["actual_score_diff_hv"]
target_feature_df["CLS_TARGET"] = (
    target_feature_df["actual_score_diff_hv"] > target_feature_df["open_line_hv"]
)  # bet_on_home boolean
target_feature_df.info(verbose=True, show_counts=True)
# Prep Df

# Drop columns that are not needed
drop_columns = [
    "game_id",
    "home_score",
    "away_score",
    "season",
    "season_type",
    "home_team",
    "away_team",
    "game_datetime",
    "actual_score_diff_hv",
]

target_feature_df = target_feature_df.drop(columns=drop_columns)
target_feature_df.info(verbose=True, show_counts=True)
# Drop NA rows
target_feature_df.dropna(inplace=True)
target_feature_df.info(verbose=True, show_counts=True)
features = [
    col for col in target_feature_df.columns if col not in ["CLS_TARGET", "REG_TARGET"]
]

# Point-biserial correlation for classification
cls_correlations = (
    target_feature_df[features].corrwith(target_feature_df["CLS_TARGET"]).abs()
)

# Pearson correlation for regression
reg_correlations = (
    target_feature_df[features].corrwith(target_feature_df["REG_TARGET"]).abs()
)

# Mutual Information
mi_classif = mutual_info_classif(
    target_feature_df[features], target_feature_df["CLS_TARGET"]
)
mi_reg = mutual_info_regression(
    target_feature_df[features], target_feature_df["REG_TARGET"]
)

# Feature importance using RandomForest
rf_classif = RandomForestClassifier(n_estimators=50, random_state=1)
rf_classif.fit(target_feature_df[features], target_feature_df["CLS_TARGET"])
classif_importance = rf_classif.feature_importances_

rf_reg = RandomForestRegressor(n_estimators=50, random_state=1)
rf_reg.fit(target_feature_df[features], target_feature_df["REG_TARGET"])
reg_importance = rf_reg.feature_importances_

# Combine results into a DataFrame
feature_selection_df = pd.DataFrame(
    {
        "Feature": features,
        "PB Corr CLS": cls_correlations.values,
        "Pearson Corr REG": reg_correlations.values,
        "MI CLS": mi_classif,
        "MI REG": mi_reg,
        "RF FI CLS": classif_importance,
        "RF FI REG": reg_importance,
    }
)

# Round to 2 decimals before ranking
feature_selection_df["PB Corr CLS"] = feature_selection_df["PB Corr CLS"].round(2)
feature_selection_df["Pearson Corr REG"] = feature_selection_df[
    "Pearson Corr REG"
].round(2)
feature_selection_df["MI CLS"] = feature_selection_df["MI CLS"].round(2)
feature_selection_df["MI REG"] = feature_selection_df["MI REG"].round(2)
feature_selection_df["RF FI CLS"] = feature_selection_df["RF FI CLS"].round(2)
feature_selection_df["RF FI REG"] = feature_selection_df["RF FI REG"].round(2)

# Rank features for each metric, using 'max' for tie-breaking
feature_selection_df["Rank PB Corr CLS"] = feature_selection_df["PB Corr CLS"].rank(
    ascending=False, method="min"
)
feature_selection_df["Rank Pearson Corr REG"] = feature_selection_df[
    "Pearson Corr REG"
].rank(ascending=False, method="min")
feature_selection_df["Rank MI CLS"] = feature_selection_df["MI CLS"].rank(
    ascending=False, method="min"
)
feature_selection_df["Rank MI REG"] = feature_selection_df["MI REG"].rank(
    ascending=False, method="min"
)
feature_selection_df["Rank RF FI CLS"] = feature_selection_df["RF FI CLS"].rank(
    ascending=False, method="min"
)
feature_selection_df["Rank RF FI REG"] = feature_selection_df["RF FI REG"].rank(
    ascending=False, method="min"
)

# Calculate average rank
feature_selection_df["Average_Rank"] = (
    feature_selection_df.loc[:, "Rank PB Corr CLS":"Rank RF FI REG"]
    .mean(axis=1)
    .round(2)
)

# Sort by average rank
feature_selection_df.sort_values("Average_Rank", inplace=True)
# Create columns for display: rounded metric value (rank)
for metric in [
    "PB Corr CLS",
    "Pearson Corr REG",
    "MI CLS",
    "MI REG",
    "RF FI CLS",
    "RF FI REG",
]:
    feature_selection_df[f"_{metric}"] = (
        feature_selection_df[metric].astype(str)
        + " ("
        + feature_selection_df[f"Rank {metric}"].astype(int).astype(str)
        + ")"
    )

# Columns to display
columns_to_display = [col for col in feature_selection_df.columns if col[0] == "_"]
columns_to_display = ["Feature", "Average_Rank"] + columns_to_display

# Sort by average rank and display the columns
feature_selection_df.sort_values("Average_Rank", inplace=True)
feature_selection_df[columns_to_display]
top_20_features = feature_selection_df[:20]
top_20_features.Feature.to_list()
## Model Performance Chart
# Re-running the function to include y-tick labels and horizontal grid lines
def plot_model_accuracy_seaborn(
    accuracies, categories, colors, save=False, image_name=None
):
    # Set the Seaborn theme
    sns.set_theme()

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a DataFrame from the data
    df = pd.DataFrame({"Accuracy": accuracies, "Model": categories})

    # Create the bar chart using Seaborn
    sns.barplot(x="Model", y="Accuracy", data=df, palette=colors, ax=ax)

    # Set the chart title and y-axis label
    ax.set_title("Classification Model Accuracy", fontsize=24, pad=16, fontweight="bold")
    ax.set_ylabel("Accuracy %", fontsize=20, labelpad=8, fontweight="bold")

    # Customize the appearance of the bars
    for index, value in enumerate(accuracies):
        ax.text(
            index,
            value - 5,
            str(value),
            color="white",
            ha="center",
            fontsize=22,
            fontweight="bold",
        )

    # Customize axis tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Remove x-axis label
    ax.set_xlabel("")

    # Remove all spines
    sns.despine(left=True, bottom=True, right=True, top=True)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="white")

    plt.tight_layout()

    if save:
        plt.savefig(f"{image_name}.png", dpi=300, bbox_inches="tight")
# Example usage with updated specifications
accuracies = [50.0, 52.4, 52.1, 47.9]
categories = ["Random Guess", "Profitable", "AutoML", "AutoDL"]
colors = ["grey", "green", "#17408B", "#C9082A"]

plot_model_accuracy_seaborn(
    accuracies, categories, colors, save=True, image_name="cls_model_accuracy"
)
