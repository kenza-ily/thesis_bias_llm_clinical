# --- 1/ Imports
import sys
import os
from pathlib import Path
import pandas as pd
from config.repo_dir import get_repo_dir
from llm.llm_config import llms
from frameworks.fw2 import process_llms_and_df_fw2


# --- 2/ Directories

# Current script path
current_script_path = os.path.abspath(__file__)

# Repo directory
def find_repo_root(path):
    while True:
        if 'results' in os.listdir(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:  # We've reached the root directory
            raise Exception("Could not find 'results' directory in any parent directory")
        path = parent
# Find the repo root (parent of "results")
repo_dir = find_repo_root(os.path.dirname(current_script_path))

# Data directory
def get_data_path(experiment_type):
    return os.path.join(repo_dir,'data', f'{experiment_type}.csv')



# --- 3/ Experiment

def main():
    # Set up the experiment parameters
    experiment_type = input("Enter experiment type (G or GxE): ").strip()
    if experiment_type not in ["G", "GxE"]:
        print("Invalid experiment type. Please enter 'G' or 'GxE'.")
        return
    
    
    print("Load Dataset")

    # Load the dataset
    try:
        data_path=get_data_path(experiment_type)
        df = pd.read_csv(data_path)
        df=df.head(2)
        print(f"Loaded dataset with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {data_path} is empty.")
        return
    except Exception as e:
        print(f"Error loading the dataset: {str(e)}")
        return

    # Run the experiment
    try:
        
        results = process_llms_and_df_fw2(llms, df, experiment_type,repo_dir)
        print("Experiment completed successfully.")
        
        # You can add additional code here to process or analyze the results if needed
        
    except Exception as e:
        print(f"An error occurred during the experiment: {str(e)}")

if __name__ == "__main__":
    main()