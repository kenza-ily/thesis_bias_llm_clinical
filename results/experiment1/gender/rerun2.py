import os
import sys
import pandas as pd
import time
from datetime import datetime
import glob

# Set the project root to '/app' for the Docker environment
project_root = '/app'
sys.path.insert(0, project_root)

# Import required modules
from llm.llm_config import llms
from experiments.experiment1a import process_llms_and_df

# ==============================
# Hard-coded
specific_question_type = "gender"
N_LATEST_LLMS = 2  # Number of latest LLMs to process

# Define the name of the latest CSV file (set to None if you want to use get_latest_csv)
LATEST_CSV_NAME = "results_rerun_gpt4o_20240731_122416.csv"

# -- Helper functions --
def get_latest_csv(directory):
    list_of_files = glob.glob(os.path.join(directory, 'results_*.csv'))
    list_of_files = [f for f in list_of_files if os.path.basename(f) != 'cleaned_data.csv']
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def get_latest_n_unprocessed_llms(all_llms, processed_llms, n):
    unprocessed_llms = {k: v for k, v in all_llms.items() if v['model_name'] not in processed_llms}
    sorted_unprocessed_llms = dict(sorted(unprocessed_llms.items(), key=lambda item: item[0], reverse=True))
    return dict(list(sorted_unprocessed_llms.items())[:n])

# -- Main function --
def rerun2_experiment1a():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")

    results_dir = os.path.join(project_root, 'results', 'experiment1', 'gender')
    data_path = os.path.join(results_dir, 'data.csv')

    print(f"Looking for results directory: {results_dir}")
    print(f"Looking for data file: {data_path}")

    if not os.path.exists(results_dir):
        print(f"Results directory not found. Creating it: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please ensure the data file is in the correct location.")
        return

    # Use the defined latest CSV name if provided, otherwise use get_latest_csv
    if LATEST_CSV_NAME:
        latest_csv = os.path.join(results_dir, LATEST_CSV_NAME)
        if not os.path.exists(latest_csv):
            print(f"Defined latest CSV file not found: {latest_csv}")
            print("Falling back to get_latest_csv function.")
            latest_csv = get_latest_csv(results_dir)
    else:
        latest_csv = get_latest_csv(results_dir)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Original data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {data_path}")
        print("Please check the file location and try again.")
        return

    if latest_csv:
        print(f"Latest results file: {os.path.basename(latest_csv)}")
        latest_results = pd.read_csv(latest_csv)
        print(f"Latest results shape: {latest_results.shape}")
        existing_llms = [col.split('_')[1] for col in latest_results.columns if col.startswith('llm_') and col.endswith('_performance')]
        new_llms = get_latest_n_unprocessed_llms(llms, existing_llms, N_LATEST_LLMS)
    else:
        print("No previous results found. Running the experiment for the latest 2 LLMs.")
        new_llms = dict(list(llms.items())[-N_LATEST_LLMS:])

    print(f"LLMs to process: {list(new_llms.keys())}")

    if not new_llms:
        print("No new LLMs to process. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_rerun2_{timestamp}.csv"
    result_path = os.path.join(results_dir, results_filename)

    start_time = time.time()
    new_results = process_llms_and_df(new_llms, df, specific_question_type, saving_path=result_path)
    end_time = time.time()

    if latest_csv:
        all_columns = set(latest_results.columns) | set(new_results.columns)
        for col in all_columns:
            if col not in latest_results.columns:
                latest_results[col] = None
            if col not in new_results.columns:
                new_results[col] = None
        combined_results = pd.concat([latest_results, new_results], axis=1)
    else:
        combined_results = new_results

    total_time = end_time - start_time

    combined_results.to_csv(result_path, index=False)

    print(f"\nExperiment completed in {total_time:.2f} seconds.")
    print(f"Results saved to: {result_path}")

    for llm_name, llm_info in new_llms.items():
        performance_col = f'{llm_name}_performance'
        if performance_col in combined_results.columns:
            accuracy = combined_results[performance_col].mean() * 100
            print(f"Accuracy for {llm_info['model_name']}: {accuracy:.2f}%")
        else:
            print(f"No performance data available for {llm_info['model_name']}")

        cost_col = f'{llm_name}_total_price'
        if cost_col in combined_results.columns:
            total_cost = combined_results[cost_col].sum()
            print(f"Total cost for {llm_info['model_name']}: ${total_cost:.4f}")
        else:
            print(f"No cost data available for {llm_info['model_name']}")

    print(f"Final combined results shape: {combined_results.shape}")

if __name__ == "__main__":
    rerun2_experiment1a()