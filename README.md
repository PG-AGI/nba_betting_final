# NBA Betting Project Setup Guide

This guide outlines the steps to set up the NBA Betting project on your local machine.

## Step 1: Clone the Repository

First, clone the GitHub repository using the Git Clone command. Replace `username/repository` with the actual username and repository name.

```bash
git clone https://github.com/username/repository.git
```

> **Note:** It's recommended to use GitBash for cloning.

## Step 2: Create and Activate a Virtual Environment

Create a virtual environment named `myenv` (you can choose your own name, like `nba_env`).

```bash
python3 -m venv myenv
```

### Activation:

- On macOS or Linux:
  ```bash
  source myenv/bin/activate
  ```
- On Windows:
  ```cmd
  .\myenv\Scripts\activate
  ```

## Step 3: Install Required Dependencies

Install the necessary dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

> **Note:** There are three different `requirements.txt` files:
>
> 1. The one that came with this project.
> 2. The one installed in your environment.
> 3. The one in your project folder.

## Step 4: Set Up PostgreSQL Database

Set up a PostgreSQL database with the following details:

- Database Name: `nba_betting`
- Password: `123654`

## Step 5: Run Database ORM

Run the `database_orm.py` file located at `NBA_Betting/src/database_orm.py`. This will structure the schema for the database.

## Step 6: Run Necessary Files

Now run the `TO_Database.py` file located at `NBA_Betting/Data/TO_database.py` to get necessary data in your database tables

## Step 7: Create and store all_json files. 

-> Run ETL Files:

- Run `main_etl.py` file form location `NBA_Betting/src/etl/main_etl.py`
- First check if `feature_Creation.py` file in location `NBA_Betting/src/etl/` is working properly or not

## Step 8: Run main.py

-To get started and receive updates in one place, run `main.py`.

## Optional steps:-

## Step 9: Run Necessary Files to check if working properly without error

Execute the following files in the given order:

1. Odds API App:
   - `airflow_dags/daily_update/odds_api_daily_update.py`
   - `NBA_Betting/src/data_sources/game/odds_api.py`
2. Utility Files:
   - All files in `src/utils`
3. Remaining Files:
   - As needed

> **Note:** Additional `pip` installations may be required. Ensure that modules like `pycaret`, `keras` are installed.

## Step 10: Run Models or Notebook

Execute:
-Notebooks in The notebooks folder located at `NBA_Betting/notebooks` for checking if it is working properly

> **Important:** Remove comments in the `main.py` file to enable full functionality. If any additions are needed, please inform me.

## To-Do

- Find and resolve errors in the notebook files.
- Some dependencies or modules might have not been installed might need to install them as you go further.

---

## Additional Setup Instructions

- **Upgrade Pip, Setuptools, and Wheel:**

  ```bash
  python3.8 -m venv nba_env
  ./myenv/Scripts/activate
  pip install --upgrade pip setuptools wheel
  ```

- **Install Dependencies from Requirements File:**

  ```bash
  pip install -r ./requirements.txt
  ```

- **If error 'Module not found' occured:**
  -Then use:
  ```bash
  sys.path.append('your_path_to_the_file')
  ```
  for only if accessing the files in the project.(example while running `main_etl.py` file you are getting `Module not found: feature_creation` as feature creation is a file in this project called `feature_creation.py`)
