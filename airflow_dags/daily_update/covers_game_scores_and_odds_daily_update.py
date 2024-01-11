import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

NBA_BETTING_BASE_DIR = os.getenv("NBA_BETTING_BASE_DIR")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")

# Define the DAG
dag = DAG(
    "Covers_Game_Scores_And_Odds_Daily_Update",
    default_args={
        "owner": "Jeff",
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
        "start_date": pendulum.datetime(2023, 5, 1),
        "email": [EMAIL_ADDRESS],
        "email_on_failure": True,
        "email_on_retry": True,
    },
    description="A DAG to run the Covers game scores and odds spider daily",
    schedule="0 16 * * *",  # 10am MT
    catchup=False,
)

command = f"cd {NBA_BETTING_BASE_DIR}/nba_betting && scrapy crawl nba -a dates=daily_update -a save_data=True -a view_data=True"

BashOperator(
    task_id=f"run_nba",
    bash_command=command,
    dag=dag,
)
