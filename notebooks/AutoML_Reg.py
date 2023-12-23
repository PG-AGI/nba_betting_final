# AutoML - Regression
# PyCaret
# Main Site - https://pycaret.org/
# Docs - https://pycaret.readthedocs.io/en/latest/
## Table of Contents
# [Setup and Preprocessing](#setup)  
# [Compare Models](#compare)  
# [Create Model](#create)  
# [Tune Model](#tune)  
# [Evaluate Model](#evaluate)  
# [Finalize and Store Model](#finalize_and_store)

## Imports and Global Settings
import os
import sys
import datetime
import json
import pandas as pd
from sqlalchemy import create_engine
from pycaret.regression import RegressionExperiment

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
endpoint = "localhost"
database = "nba_betting"
port = "5432"

# Create the connection string
connection_string = (
    f"postgresql+psycopg2://{username}:{password}@{endpoint}:{port}/{database}"
)
### Games
start_date = "2020-09-01"
start_date_int = int(start_date.replace("-", ""))

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
start_date_int = int(start_date.replace("-", ""))

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
py_reg = RegressionExperiment()


### Setup and Preprocessing
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

setup_params_reg = {
    "log_experiment": True,
    "log_profile": False,
    "log_plots": False,
    "experiment_name": f"REG_4_{timestamp}",
    "data": training_df,
    "test_data": testing_df,
    "target": "REG_TARGET",
    "preprocess": False,
    "normalize": False,
    "transformation": False,
    "remove_outliers": False,
    "remove_multicollinearity": False,
    "feature_selection": False,
    "pca": False,
    "pca_components": 10,
    "numeric_features": [],
    "ignore_features": ["game_id", "vegas_open_hv"],
}
py_reg.setup(**setup_params_reg)

### Compare Models
best_model_reg = py_reg.compare_models(turbo=True, sort="MAE", exclude=["catboost"])
print(best_model_reg)

### Create Selected Model
model_reg = py_reg.create_model("lasso")

### Tune Selected Model
tuned_model_reg = py_reg.tune_model(model_reg)
model_report["details"] = tuned_model_reg.get_params()


### Evaluate Model
py_reg.evaluate_model(tuned_model_reg)
# py_reg.interpret_model(tuned_model_reg)
train_predictions_reg = py_reg.predict_model(tuned_model_reg, data=training_df)
train_prediction_metrics = py_reg.pull()
model_report["train_mae"] = train_prediction_metrics["MAE"][0]
model_report["train_r2"] = train_prediction_metrics["R2"][0]
test_predictions_reg = py_reg.predict_model(tuned_model_reg, data=testing_df)
test_prediction_metrics = py_reg.pull()
model_report["test_mae"] = test_prediction_metrics["MAE"][0]
model_report["test_r2"] = test_prediction_metrics["R2"][0]
train_acc_reg, train_closer_to_target_reg, train_prediction_df_reg = evaluate_reg_model(
    train_predictions_reg, "vegas_open_hv", "REG_TARGET", "prediction_label"
)
test_acc_reg, test_closer_to_target_reg, test_prediction_df_reg = evaluate_reg_model(
    test_predictions_reg, "vegas_open_hv", "REG_TARGET", "prediction_label"
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

### Model Finalization and Storage
final_model_reg = py_reg.finalize_model(tuned_model_reg)
platform = "pycaret"
problem_type = "reg"
model_type = "lasso"
datetime_str = model_report["datetime"].strftime("%Y_%m_%d_%H_%M_%S")
model_report["datetime"] = model_report["datetime"].strftime("%Y-%m-%d %H:%M:%S")

model_id = f"{platform}_{problem_type}_{model_type}_{datetime_str}"
model_id
py_reg.save_model(final_model_reg, f"../models/AutoML/{model_id}")
model_report["platform"] = platform
model_report["model_type"] = model_type
model_report["model_id"] = model_id
model_report
save_model_report(model_report)
