import os
import time
import psycopg2
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database_orm import Base  # Adjust the import path as necessary

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection settings
DB_HOST = os.getenv('DB_HOST', 'db')
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

if __name__ == "__main__":
    wait_for_db()
    initialize_db()
