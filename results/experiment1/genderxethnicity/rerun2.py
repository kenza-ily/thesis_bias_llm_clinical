import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import required modules
from llm.llm_config import llms
from experiments.experiment1b import process_llms_and_df_b

# Hard-coded
specific_question_type = "genderxethnicity"
N_LATEST_LLMS = 2  # Number of latest LLMs to process
LATEST_CSV_NAME = "results_20240731_094553.csv"  # Set to None to use the most recent CSV

def get_latest_csv(directory):
    list_of_files = [f for f in os.listdir(directory) if f.startswith('results_') and f.endswith('.csv')]
    if not list_of_files:
        return None
    return max(list_of_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))

def get_latest_n_unprocessed_llms(all_llms, processed_llms, n):
    unprocessed_llms = {k: v for k, v in all_llms.items() if k not in processed_llms}
    return dict(list(unprocessed_llms.items())[-n:])

def rerun_experiment2():
    # Define paths
    results_dir = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity')
    data_path = os.path.join(results_dir, 'data.csv')
    
    if LATEST_CSV_NAME:
        previous_results_path = os.path.join(results_dir, LATEST_CSV_NAME)
        if not os.path.exists(previous_results_path):
            print(f"Specified CSV file not found: {LATEST_CSV_NAME}")
            print("Falling back to most recent CSV.")
            previous_results_path = os.path.join(results_dir, get_latest_csv(results_dir))
    else:
        previous_results_path = os.path.join(results_dir, get_latest_csv(results_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_rerun2_{timestamp}.csv"
    result_path = os.path.join(results_dir, results_filename)

    # Load the original data and previous results
    df = pd.read_csv(data_path)
    previous_results = pd.read_csv(previous_results_path)

    print(f"Original data shape: {df.shape}")
    print(f"Previous results shape: {previous_results.shape}")

    # Determine which LLMs have been processed
    processed_llms = [col.split('_')[1] for col in previous_results.columns if col.endswith('_performance')]

    # Get the latest N unprocessed LLMs
    new_llms = get_latest_n_unprocessed_llms(llms, processed_llms, N_LATEST_LLMS)

    if not new_llms:
        print("No new LLMs to process. Exiting.")
        return

    print(f"LLMs to process: {list(new_llms.keys())}")

    # Process the data with new LLMs
    start_time = time.time()
    new_results = process_llms_and_df_b(new_llms, df, specific_question_type, saving_path=result_path)

    # Ensure all necessary columns exist in both dataframes
    all_columns = set(previous_results.columns) | set(new_results.columns)
    for col in all_columns:
        if col not in previous_results.columns:
            previous_results[col] = None
        if col not in new_results.columns:
            new_results[col] = None

    # Combine previous results with new results
    final_results = pd.concat([previous_results, new_results], axis=1)

    end_time = time.time()

    # Calculate total execution time
    total_time = end_time - start_time

    # Save the final results
    final_results.to_csv(result_path, index=False)

    # Print summary
    print(f"\nExperiment completed in {total_time:.2f} seconds.")
    print(f"Results saved to: {result_path}")

    # Calculate and print accuracy for new LLMs
    for llm_name in new_llms.keys():
        if f'{llm_name}_performance' in final_results.columns:
            accuracy = final_results[f'{llm_name}_performance'].mean() * 100
            print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        else:
            print(f"Accuracy for {llm_name}: Not available (column missing)")

        # Calculate and print total cost for new LLMs
        if f'{llm_name}_total_price' in final_results.columns:
            total_cost = final_results[f'{llm_name}_total_price'].sum()
            print(f"Total cost for {llm_name}: ${total_cost:.4f}")
        else:
            print(f"Total cost for {llm_name}: Not available (column missing)")

    print(f"Final results shape: {final_results.shape}")

    # Verification step
    if len(final_results) == len(df):
        print("All rows from the original data are present in the final results.")
    else:
        print("WARNING: The number of rows in the final results does not match the original data.")
        print("You may need to investigate this discrepancy.")

if __name__ == "__main__":
    rerun_experiment2()