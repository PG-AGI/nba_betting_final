import os
import csv
import psycopg2
from psycopg2 import sql

# Replace these with your PostgreSQL database connection details
db_params = {
    'host': 'db',
    'database': 'nba_betting',
    'user': 'nba_betting_user',
    'password': 123654,
}

# List of CSV files in the "CSV" folder
csv_folder = 'Data/CSV'
csv_files = [
    'team_nbastats_general_advanced.csv',
    'team_nbastats_general_traditional.csv',
    'team_nbastats_general_fourfactors.csv',
    'team_nbastats_general_opponent.csv',
    'games.csv',
]

# Connect to PostgreSQL database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Iterate through each CSV file
for csv_file_name in csv_files:
    # Get the path to the current CSV file
    csv_file_path = os.path.join(csv_folder, csv_file_name)

    # Open the CSV file and read data
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Fetch the header row to get column names
        header = next(csv_reader, None)

        # Skip the header row if it exists
        next(csv_reader, None)

        # Create a list of column names for the query
        columns = [sql.Identifier(col) for col in header]

        # Construct the SQL query using the dynamically obtained column names
        query = sql.SQL('''
            INSERT INTO {} ({}) 
            VALUES ({});
        ''').format(
            sql.Identifier(csv_file_name.replace('.csv', '')),  # Use file name without extension as the table name
            sql.SQL(', ').join(columns),
            sql.SQL(', ').join([sql.Placeholder() for _ in columns]),
        )

        # Iterate through rows and insert into the PostgreSQL table
        for row in csv_reader:
            cursor.execute(query, row)

# Commit the changes and close the connection
conn.commit()
conn.close()
