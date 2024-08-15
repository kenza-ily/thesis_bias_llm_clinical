import sys
import os
from pathlib import Path
import argparse

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now you can import from config
from config.settings import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

os.environ['AZURE_OPENAI_API_KEY'] = AZURE_OPENAI_API_KEY
os.environ['AZURE_OPENAI_ENDPOINT'] = AZURE_OPENAI_ENDPOINT

# Rest of your imports
import pandas as pd
from config.repo_dir import get_repo_dir
from llm.llm_config import llms
from frameworks.fw2 import process_llms_and_df_fw2


# --- 2/ Directories

# Current script path
current_script_path = os.path.abspath(__file__)

# Repo directory

repo_dir = get_repo_dir()

# Data directory
def get_data_path(experiment_type):
    return os.path.join(repo_dir,'data', f'{experiment_type}.csv')



# --- 3/ Experiment

def main():

    
    
    # Set up the experiment parameters
    parser = argparse.ArgumentParser(description="Run experiments with LLMs")
    parser.add_argument("experiment_type", choices=["G", "GxE"], help="Type of experiment (G or GxE)")
    parser.add_argument("experiment_number", type=int, choices=[2, 3, 4], help="Experiment number (2, 3, or 4)")
    parser.add_argument("experiment_name", help="Name of the experiment")
    parser.add_argument("llm_type", help="Type of LLM to use")

    args = parser.parse_args()

    experiment_type = args.experiment_type
    experiment_number = args.experiment_number
    experiment_name = args.experiment_name
    llm_type = args.llm_type
    
    print("Imported llms:", llms)  # Debugging line
    
    filtered_llms = {key: value for key, value in llms.items() if value.get('type') == llm_type}
    print("Filtered llms:", filtered_llms) 
    
    
    print("Load Dataset")
    # Load the dataset
    try:
        data_path = get_data_path(experiment_type)
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
        results = process_llms_and_df_fw2(filtered_llms, df, experiment_type, repo_dir, experiment_number, experiment_name)
        print("Experiment completed successfully.")
    except Exception as e:
        print(f"An error occurred during the experiment: {str(e)}")

if __name__ == "__main__":
    main()