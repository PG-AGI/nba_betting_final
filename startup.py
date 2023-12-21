import subprocess
import os

def run_etl():
    # Assuming the main ETL script is located in the 'etl' subdirectory
    # Replace 'main_etl.py' with the actual ETL script name
    subprocess.run(['python', 'src/etl/main_etl.py'])

def run_model_training():
    notebooks_path = 'notebooks'
    notebooks_to_run = ['AutoDL_Cls.ipynb', 'AutoDL_Reg.ipynb', 'AutoML_Cls.ipynb', 'AutoML_Reg.ipynb', 'Exploratory_Analysis.ipynb']

    for notebook in notebooks_to_run:
        full_path = os.path.join(notebooks_path, notebook)
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', full_path])

if __name__ == "__main__":
    run_etl()
    run_model_training()