# AutoDL - Regression
## Imports and Global Settings
import os
import sys
import datetime
import json
import pandas as pd
from sqlalchemy import create_engine
import tensorflow as tf
import autokeras as ak
from sklearn.metrics import mean_absolute_error, r2_score

here = os.getcwd()
sys.path.append(os.path.join(here, ".."))

sys.path.append('D:\\Projects\\NBA_Betting\\src')
from utils.modeling_utils import (
    ModelSetup,
    evaluate_reg_model,
    calculate_roi,
    save_model_report,
)

RDS_ENDPOINT = os.getenv("RDS_ENDPOINT")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")

# Pandas Settings
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.options.display.max_info_columns = 200
pd.options.display.precision = 5
## Load Data
username = "postgres"
password = 123654
endpoint = 'localhost'
database = "nba_betting"
port = "5432"

# Create the connection string
connection_string = (
    f"postgresql+psycopg2://{username}:{password}@{endpoint}:{port}/{database}"
)
### Games
start_date = "2020-09-01"
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
games_query = f"SELECT {', '.join(features)} FROM games WHERE CAST(LEFT(game_id, 8) AS INTEGER) >= {start_date_int};"

with create_engine(connection_string).connect() as connection:
    games = pd.read_sql_query(games_query, connection)
### Features
start_date = "2020-09-01"
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
# Drop the columns from df2 (with suffix '_drop')
games_features = games_features[
    games_features.columns.drop(list(games_features.filter(regex="_drop")))
]

## Basic Data Overview
df = games_features.copy()
df.info(verbose=True, show_counts=True)
df.head(10)
## Data Preparation
#### Drop Non-Completed Games and Games with No Line
df = df[df["game_completed"] == True]
df = df.dropna(subset=["open_line"])
### Create Targets
df = ModelSetup.add_targets(df)
### Select Features
training_seasons = [x for x in range(2020, 2022)]
training_dates, testing_dates = ModelSetup.choose_dates(training_seasons, [2022], "Reg")
print("Training Dates:")
print(training_dates)
print("Testing Dates:")
print(testing_dates)
for col in df.columns:
    print(col)
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
df.dropna(subset=features_to_use, inplace=True)
training_df, testing_df, model_report = ModelSetup.create_datasets(
    df, "reg", features_to_use, training_dates, testing_dates, create_report=True
)
print("Number of rows in training_df:", len(training_df))
print("Number of rows in testing_df:", len(testing_df))
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

## Regression
training_df.info()
X_train = training_df[features_to_use]
X_test = testing_df[features_to_use]
y_train = training_df["REG_TARGET"]
y_test = testing_df["REG_TARGET"]
## AutoKeras
reg = ak.StructuredDataRegressor(
    max_trials=10,
    overwrite=True,
    loss="mean_absolute_error",
)
reg.fit(X_train, y_train)
# Evaluate the best model with testing data.
print(reg.evaluate(X_test, y_test))
model = reg.export_model()
model.summary()
model_report["details"] = model.get_config()

### Evaluate Model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
print(f"Training MAE: {train_mae:.2f}")
print(f"Training R2: {train_r2:.2f}")
model_report["train_mae"] = train_mae
model_report["train_r2"] = train_r2
test_mae = mean_absolute_error(y_test, train_pred)
print(f"Training MAE: {train_mae:.2f}")
print(f"Training R2: {train_r2:.2f}")
model_report["train_mae"] = train_mae
model_report["train_r2"] = train_r2
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
print(f"Testing MAE: {test_mae:.2f}")
print(f"Testing R2: {test_r2:.2f}")
model_report["test_mae"] = test_mae
model_report["test_r2"] = test_r2
vegas_pred_train = -X_train["open_line"]
train_pred = train_pred.flatten()
train_pred_results = pd.DataFrame(
    {"vegas_open_hv": vegas_pred_train, "model_pred": train_pred, "actual": y_train}
)
vegas_pred_test = -X_test["open_line"]
test_pred = test_pred.flatten()
test_pred_results = pd.DataFrame(
    {"vegas_open_hv": vegas_pred_test, "model_pred": test_pred, "actual": y_test}
)
train_acc_reg, train_closer_to_target_reg, train_prediction_df_reg = evaluate_reg_model(
    train_pred_results, "vegas_open_hv", "actual", "model_pred"
)
test_acc_reg, test_closer_to_target_reg, test_prediction_df_reg = evaluate_reg_model(
    test_pred_results, "vegas_open_hv", "actual", "model_pred"
)
model_report["train_acc_reg"] = train_acc_reg
model_report["test_acc_reg"] = test_acc_reg
model_report["train_ctt"] = train_closer_to_target_reg
model_report["test_ctt"] = test_closer_to_target_reg
roi_results_reg = calculate_roi(test_prediction_df_reg, "actual_side", "pred_side")
roi_results_reg
model_report["roi_all_bets_even_amount_avg"] = roi_results_reg[
    roi_results_reg["Label"] == "All Bets, Even Amount"
]["Average ROI per Bet"].iloc[0]
model_report["roi_all_bets_typical_odds_avg"] = roi_results_reg[
    roi_results_reg["Label"] == "All Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]
### Model Storage
platform = "autokeras"
problem_type = "reg"
model_type = "dl"
datetime_str = model_report["datetime"].strftime("%Y_%m_%d_%H_%M_%S")
model_report["datetime"] = model_report["datetime"].strftime("%Y-%m-%d %H:%M:%S")

model_id = f"{platform}_{problem_type}_{model_type}_{datetime_str}"
model_id
model.save(f"../models/AutoDL/{model_id}", save_format="tf")
model_report["platform"] = platform
model_report["model_type"] = model_type
model_report["model_id"] = model_id
model_report
save_model_report(model_report)
