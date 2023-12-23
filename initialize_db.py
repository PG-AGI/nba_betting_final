import os
import time
import psycopg2
import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database_orm import Base  # Adjust the import path as necessary

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection settings
DB_HOST = 'db'
DB_NAME = os.getenv('DB_NAME', 'nba_betting')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '123654')

# Wait for the database to be ready
def wait_for_db():
    logging.info("Waiting for the database to be ready...")
    while True:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST
            )
            conn.close()
            logging.info("Database is ready.")
            break
        except psycopg2.OperationalError as e:
            logging.error(f"Database connection failed: {e}")
            time.sleep(2)

# Initialize the database
def initialize_db():
    logging.info("Initializing the database...")
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    Base.metadata.create_all(engine)
    logging.info("Database initialized successfully.")

def insert_data_from_csv(csv_file, table_name, engine):
    try:
        # Read data from CSV file
        CSV_FOLDER_PATH = "Data/CSV"
        data = pd.read_csv(os.path.join(CSV_FOLDER_PATH, csv_file))

        # Insert data into the database
        data.to_sql(table_name, engine, if_exists='append', index=False)
        logging.info(f"Data from {csv_file} inserted into {table_name}.")
    except Exception as e:
        logging.error(f"Error inserting data from {csv_file}: {e}")

def main():
    # Wait for DB and initialize
    wait_for_db()
    initialize_db()

    # Create database engine
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')

    # List of CSV files and corresponding table names
    # Adjust the table names based on your actual database schema
    csv_files = [
        ('games.csv', 'games'),
        ('team_fivethirtyeight_games.csv', 'team_fivethirtyeight_games'),
        ('team_nbastats_general_advanced.csv', 'team_nbastats_general_advanced'),
        ('team_nbastats_general_fourfactors.csv', 'team_nbastats_general_fourfactors'),
        ('team_nbastats_general_opponent.csv', 'team_nbastats_general_opponent'),
        ('team_nbastats_general_traditional.csv', 'team_nbastats_general_traditional')
        # Add other CSV files and table names here
    ]

    # Insert data for each CSV file
    for csv_file, table_name in csv_files:
        insert_data_from_csv(csv_file, table_name, engine)

if __name__ == "__main__":
    main()
