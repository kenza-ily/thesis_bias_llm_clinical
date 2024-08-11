import sys
import os
from pathlib import Path

# Add the directory containing experiments to the Python path
experiment_dir = Path(__file__).resolve().parent.parent.parent / 'experiments'
sys.path.append(str(experiment_dir))

# Import the necessary functions from experiment2.py
from experiment2 import process_llms_and_df_exp2

# Import other necessary libraries
import pandas as pd
from llm.models import load_model  # Assuming you have this function to load LLM models

def main():
    # Set up the experiment parameters
    experiment_type = input("Enter experiment type (G or GxE): ").strip().upper()
    if experiment_type not in ["G", "GXE"]:
        print("Invalid experiment type. Please enter 'G' or 'GxE'.")
        return

    # Load the dataset
    data_path = input("Enter the path to your dataset CSV file: ").strip()
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

    # Set up the LLMs
    llms = {}
    while True:
        llm_name = input("Enter LLM name (or press Enter to finish adding LLMs): ").strip()
        if not llm_name:
            break
        try:
            llm_model = load_model(llm_name)
            llms[llm_name] = {"model": llm_model}
            
            # Add pricing information
            llms[llm_name]["price_per_input_token"] = float(input(f"Enter price per input token for {llm_name}: "))
            llms[llm_name]["price_per_output_token"] = float(input(f"Enter price per output token for {llm_name}: "))
            
            print(f"Added {llm_name} to the experiment.")
        except Exception as e:
            print(f"Error loading model {llm_name}: {str(e)}")

    if not llms:
        print("No LLMs added. Exiting.")
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