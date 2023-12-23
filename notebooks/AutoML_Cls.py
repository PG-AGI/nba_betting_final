import os
import sys
import datetime
import json
import pycaret
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score
from pycaret.classification import ClassificationExperiment

here = os.getcwd()
sys.path.append(os.path.join(here, ".."))

sys.path.append('D:\\Projects\\NBA_Betting\\src')
from utils.modeling_utils import ModelSetup, calculate_roi, save_model_report

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
df.info(verbose=True, null_counts=True)
### Create Targets
df = ModelSetup.add_targets(df)
### Select Features
# training_seasons = [x for x in range(2010, 2022)]
training_dates, testing_dates = ModelSetup.choose_dates([2020, 2021], [2022], "Reg")
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
    df, "cls", features_to_use, training_dates, testing_dates, create_report=True
)
print("Training Shape: ", training_df.shape)
print("Testing Shape: ", testing_df.shape)
training_df.info(verbose=True, null_counts=True)
### Baselines
training_baseline_home_team = model_report["ind_baseline_train"]
testing_baseline_home_team = model_report["ind_baseline_test"]

training_baseline_majority_class = model_report["dep_baseline_train"]
testing_baseline_majority_class = model_report["dep_baseline_test"]

print(f"Training Baseline - Home Team: {training_baseline_home_team:.2f}")
print(f"Testing Baseline - Home Team: {testing_baseline_home_team:.2f}")
print(f"Training Baseline - Majority Class: {training_baseline_majority_class:.2f}")
print(f"Testing Baseline - Majority Class: {testing_baseline_majority_class:.2f}")

## Classification

### Setup and Preprocessing
py_cls = ClassificationExperiment()


timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

setup_params_cls = {
    "log_experiment": True,
    "log_profile": False,
    "log_plots": False,
    "experiment_name": f"CLS_4_{timestamp}",
    "data": training_df,
    "test_data": testing_df,
    "target": "CLS_TARGET",
    "preprocess": False,
    "normalize": False,  # zscore
    "transformation": False,  # yeo-johnson power transform to make data more Gaussian
    "remove_outliers": False,  # using SVD
    "remove_multicollinearity": False,
    "polynomial_features": False,
    "feature_selection": False,
    "pca": False,
    "pca_components": 10,
    "numeric_features": [],
    "ignore_features": ["game_id"],
}
py_cls.setup(**setup_params_cls)


### Compare Models
best_model_cls = py_cls.compare_models(turbo=True, sort="Accuracy", exclude=["catboost"])
print(best_model_cls)


### Create Selected Model
model_cls = py_cls.create_model("nb")


### Tune Selected Model
tuned_model_cls = py_cls.tune_model(model_cls)
print(tuned_model_cls)
model_report["details"] = tuned_model_cls.get_params()


### Evaluate Model
py_cls.evaluate_model(tuned_model_cls)
# py_cls.interpret_model(tuned_model_cls)
train_predictions_cls = py_cls.predict_model(tuned_model_cls, data=training_df)
train_prediction_metrics = py_cls.pull()
model_report["train_accuracy"] = train_prediction_metrics["Accuracy"][0]
model_report["train_auc"] = train_prediction_metrics["AUC"][0]
test_predictions_cls = py_cls.predict_model(tuned_model_cls, data=testing_df)
test_prediction_metrics = py_cls.pull()
model_report["test_accuracy"] = test_prediction_metrics["Accuracy"][0]
model_report["test_auc"] = test_prediction_metrics["AUC"][0]
roi_results_cls = calculate_roi(
    test_predictions_cls, "CLS_TARGET", "prediction_label", pred_prob="prediction_score"
)
roi_results_cls
model_report["roi_all_bets_even_amount_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "All Bets, Even Amount"
]["Average ROI per Bet"].iloc[0]

model_report["roi_all_bets_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "All Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_50_even_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 50% Bets, Even Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_50_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 50% Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_55_even_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 55% Bets, Even Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_55_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 55% Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_60_even_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 60% Bets, Even Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_60_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 60% Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_65_even_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 65% Bets, Even Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_65_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 65% Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_70_even_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 70% Bets, Even Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_cutoff_70_typical_odds_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "Cutoff 70% Bets, Typical Odds"
]["Average ROI per Bet"].iloc[0]

model_report["roi_all_bets_even_amount_kelly_criterion_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "All Bets, Even Amount, Kelly Criterion"
]["Average ROI per Bet"].iloc[0]

model_report["roi_all_bets_typical_odds_kelly_criterion_avg"] = roi_results_cls[
    roi_results_cls["Label"] == "All Bets, Typical Odds, Kelly Criterion"
]["Average ROI per Bet"].iloc[0]


### Model Finalization and Storage
final_model_cls = py_cls.finalize_model(tuned_model_cls)
platform = "pycaret"
problem_type = "cls"
model_type = "nb"
datetime_str = model_report["datetime"].strftime("%Y_%m_%d_%H_%M_%S")
model_report["datetime"] = model_report["datetime"].strftime("%Y-%m-%d %H:%M:%S")

model_id = f"{platform}_{problem_type}_{model_type}_{datetime_str}"
model_id
py_cls.save_model(final_model_cls, f"../models/AutoML/{model_id}")
model_report["platform"] = platform
model_report["model_type"] = model_type
model_report["model_id"] = model_id
model_report
save_model_report(model_report)
