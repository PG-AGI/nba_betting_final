import os
import os
from nbconvert import PythonExporter
from nbformat import read
import requests
import warnings

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Define the path to the daily_update directory
    daily_update_dir = 'D:\\Projects\\NBA_Betting\\airflow_dags\\daily_update\\'

    # List of all the Python scripts in the daily_update directory
    scripts = [
        'covers_game_scores_and_odds_daily_update.py',
        'etl_daily_update.py',
        'odds_api_daily_update.py',
        'predictions_daily_update.py',
        'team_nbastats_daily_update.py'
    ]

    for script in scripts:
        try:
            # Run the script and redirect stderr to null
            os.system(f'python {daily_update_dir}{script} 2> nul')
            print(f"{script} ran successfully.")
        except Exception as e:
            print(f"Failed to run {script}.")
            print("Error:", e)

if __name__ == "__main__":
    main()

API_KEY = "dfe5c02a2c391060ebf9a7d5f69715f7"
API_ENDPOINT = "https://api.the-odds-api.com/v4/sports/basketball_nba/scores/?apiKey=dfe5c02a2c391060ebf9a7d5f69715f7"  # Replace with the actual API endpoint

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",  # Adjust content type if necessary
}

response = requests.get(API_ENDPOINT, headers=headers)

if response.status_code == 200:
    print("API Key is working!")
    # Optionally, print or process the API response data
    print(response.json())
else:
    print("API Key is not valid or there was an issue with the request.")
    print(f"Status code: {response.status_code}")
    # Optionally, print the response content for further investigation
    print(response.text)


# Get the absolute path to the 'notebook' folder
notebook_folder = os.path.abspath('notebooks')

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook_content = read(notebook_file, as_version=4)

    exporter = PythonExporter()
    (python_code, _) = exporter.from_notebook_node(notebook_content)

    # Execute the notebook code
    exec(python_code, globals())

def run_all_notebooks(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.ipynb'):
            notebook_path = os.path.join(folder_path, file_name)
            print(f"Running notebook: {file_name}")
            run_notebook(notebook_path)
            print(f"Finished running notebook: {file_name}\n")

if __name__ == "__main__":
    run_all_notebooks(notebook_folder)