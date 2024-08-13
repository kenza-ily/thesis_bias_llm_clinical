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
# Load df
current_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir, 'data.csv')



# =============================================================================


def main():
    # Set up the experiment parameters
    experiment_type = input("Enter experiment type (G or GxE): ").strip().upper()
    if experiment_type not in ["G", "GXE"]:
        print("Invalid experiment type. Please enter 'G' or 'GxE'.")
        return

    # Load the dataset
    try:
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