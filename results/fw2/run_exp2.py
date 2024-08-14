import sys
import os
from pathlib import Path



# Add the directory containing experiments to the Python path
experiment_dir = Path(__file__).resolve().parent.parent.parent / 'experiments'
sys.path.append(str(experiment_dir))

main_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(main_dir))

llmdir = Path(__file__).resolve().parent.parent.parent / "llm"
sys.path.append(str(llmdir))


print("Looking for modules in ", sys.path)

# exit(0)

from llm.models import get_haiku
# from llm.models import (
#     get_haiku
#     # get_biomistral_7b
# )

# Import the necessary functions from experiment2.py
from llm.llm_config import llms
from experiment2 import process_llms_and_df_exp2

# Import other necessary libraries
import pandas as pd
# =============================================================================
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

def get_data_path(experiment_type):
    return os.path.join(repo_dir,'data', f'{experiment_type}.csv')


# # TEST
# llms = {
#     "llm_haiku": {
#         "model_name": "claude-3-haiku@20240307",
#         "model": get_haiku(),
#         "price_per_input_token":0.25,
#         "price_per_output_token": 1.25
#     }

# }

# =============================================================================


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
        results = process_llms_and_df_exp2(llms, df, experiment_type,current_script_path)
        print("Experiment completed successfully.")
        
        # You can add additional code here to process or analyze the results if needed
        
    except Exception as e:
        print(f"An error occurred during the experiment: {str(e)}")

if __name__ == "__main__":
    main()