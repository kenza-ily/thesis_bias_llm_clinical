import sys
import os
from pathlib import Path

# Add the directory containing experiments to the Python path
experiment_dir = Path(__file__).resolve().parent.parent.parent / 'experiments'
sys.path.append(str(experiment_dir))

# Import the necessary functions from experiment2.py
from llm.llm_config import llms
from experiment2 import process_llms_and_df_exp2

# Import other necessary libraries
import pandas as pd
# =============================================================================
import os

def find_repo_root(path):
    while True:
        if 'results' in os.listdir(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:  # We've reached the root directory
            raise Exception("Could not find 'results' directory in any parent directory")
        path = parent

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Find the repo root (parent of "results")
repo_dir = find_repo_root(os.path.dirname(current_script_path))

# Get the experiment type from the script's parent directory name
experiment_type = os.path.basename(os.path.dirname(current_script_path))

# Construct the data path
data_path = os.path.join(repo_dir, 'results', experiment_type, f'{experiment_type}.csv')


# =============================================================================


def main():
    # Set up the experiment parameters
    experiment_type = input("Enter experiment type (G or GxE): ").strip().upper()
    if experiment_type not in ["G", "GXE"]:
        print("Invalid experiment type. Please enter 'G' or 'GxE'.")
        return

    # Load the dataset
    try:
        data_path=get_data_path(experiment_type)
        df = pd.read_csv(data_path)
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
        results = process_llms_and_df_exp2(llms, df, experiment_type)
        print("Experiment completed successfully.")
        
        # You can add additional code here to process or analyze the results if needed
        
    except Exception as e:
        print(f"An error occurred during the experiment: {str(e)}")

if __name__ == "__main__":
    main()