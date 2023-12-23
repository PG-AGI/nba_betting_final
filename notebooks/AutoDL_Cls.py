# AutoDL - Classification
## Imports and Global Settings

import os
import sys
import datetime
import json
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import autokeras as ak
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

here = os.getcwd()
sys.path.append(os.path.join(here, ".."))

from src.utils.modeling_utils import (
    ModelSetup,
    evaluate_reg_model,
    calculate_roi,
    save_model_report,
)

RDS_ENDPOINT = "localhost"
RDS_PASSWORD = 123654

# Pandas Settings
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.options.display.max_info_columns = 200
pd.options.display.precision = 5

## Load Data
username = "postgres"
password = 123654
endpoint = "localhost"
database = "nba_betting"
port = "5432"

# Create the connection string
connection_string = (
    f"postgresql+psycopg2://{username}:{password}@{endpoint}:{port}/{database}"
)
### Games
start_date = "2023-11-12"
start_date_int = int(start_date.replace("-", ""))  # Convert date to YYYYMMDD format

features = [
    "game_id",
    "game_datetime",
    "home_team",
    "away_team",
    "open_line",
    "home_score",
    "away_score",
    "game_completed",
    "odds_last_update",
    "scores_last_update",
]

# Extracting the YYYYMMDD portion of the game_id and comparing it with start_date_int
games_query = f"SELECT {', '.join(features)} FROM games WHERE CAST(LEFT(game_id, 8) AS INTEGER) >= {start_date_int} AND LEFT(game_id, 8) ~ E'^\\\\d+$';"

with create_engine(connection_string).connect() as connection:
    games = pd.read_sql_query(games_query, connection)
### Features
start_date = "2023-11-12"
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
games_features = pd.merge(
    games,
    all_features,
    on="game_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_drop"),
)

"print(games_features)"
# Drop the columns from df2 (with suffix '_drop')
games_features = games_features[
    games_features.columns.drop(list(games_features.filter(regex="_drop")))
]

## Basic Data Overview
df = games_features.copy()
"print(games_features)"
"df.info(verbose=True, show_counts=True)"
"df.head(10)"

## Data Preparation
#### Drop Non-Completed Games and Games with No Line
df = df[df["game_completed"] == True]

columns_with_missing_values = df.columns[df.isnull().any()].tolist()

# Removing columns with None or NaT values
df_cleaned = df.drop(columns=columns_with_missing_values)

df, columns_with_missing_values, df_cleaned
# df = df.dropna()
print(df_cleaned)
## Create Targets
df = ModelSetup.add_targets(df)
### Select Features
# Example of proper data splitting
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(features, test_size=0.2, shuffle=False)


training_seasons = [x for x in range(2019, 2022)]
training_dates, testing_dates = ModelSetup.choose_dates(training_seasons, [2023], "Reg")
# print("Training Dates:")
# print(training_dates)
# print("Testing Dates:")
# print(testing_dates)
# for col in df.columns:
#     print(col)
features_to_use = [
    "open_line",
    "rest_diff_hv",
    "day_of_season",
    "last_5_hv",
    "streak_hv",
    "point_diff_last_5_hv",
    "point_diff_hv",
    "win_pct_hv",
    "pie_percentile_away_all_advanced",
    "home_team_avg_point_diff",
    "net_rating_away_all_advanced",
    "net_rating_home_all_advanced",
    "plus_minus_home_all_traditional",
    "e_net_rating_zscore_away_all_advanced",
    "net_rating_zscore_away_all_advanced",
    "plus_minus_away_all_opponent",
    "away_team_avg_point_diff",
    "plus_minus_away_all_traditional",
    "pie_zscore_away_all_advanced",
    "e_net_rating_away_all_advanced",
    "plus_minus_percentile_away_all_traditional",
    "net_rating_zscore_home_l2w_advanced",
    "e_net_rating_home_all_advanced",
    "w_zscore_away_all_traditional",
    "pie_away_all_advanced",
    "w_pct_zscore_away_all_traditional",
    "e_net_rating_percentile_away_l2w_advanced",
]
# Check if the columns in features_to_use are present in df
missing_cols = set(features_to_use) - set(df.columns)
if missing_cols:
    print(f"Columns {missing_cols} are missing in the dataframe.")
else:
    df.dropna(subset=features_to_use, inplace=True)



from src.utils.modeling_utils import ModelSetup
training_df, testing_df, model_report = ModelSetup.create_datasets(
    df, "cls", features_to_use, training_dates, testing_dates, create_report=True
)
print(training_df)
# You can print the shapes of the dataframes to verify that they are not empty
print("Training Shape: ", training_df.shape)
print("Testing Shape: ", testing_df.shape)

print("Training Shape: ", training_df.shape)
print("Testing Shape: ", testing_df.shape)
### Baselines
training_baseline_via_vegas = model_report["ind_baseline_train"]
testing_baseline_via_vegas = model_report["ind_baseline_test"]

training_baseline_via_mean = model_report["dep_baseline_train"]
testing_baseline_via_mean = model_report["dep_baseline_test"]

print(f"Training Baseline via Vegas: {training_baseline_via_vegas:.2f}")
print(f"Testing Baseline via Vegas: {testing_baseline_via_vegas:.2f}")
print(f"Training Baseline via Mean: {training_baseline_via_mean:.2f}")
print(f"Testing Baseline via Mean: {testing_baseline_via_mean:.2f}")
## Classification
X_train = training_df[features_to_use]
X_test = testing_df[features_to_use]
y_train = training_df["CLS_TARGET"]
y_test = testing_df["CLS_TARGET"]
## AutoKeras
cls = ak.StructuredDataClassifier(
    max_trials=10,
    overwrite=True,
    loss="accuracy",
)
cls.fit(X_train, y_train)
# Evaluate the best model with testing data.
print(cls.evaluate(X_test, y_test))
model = cls.export_model()
model.summary()
model_report["details"] = model.get_config
