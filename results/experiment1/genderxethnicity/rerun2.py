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
specific_question_type = "gender"
N_LATEST_LLMS = 2  # Number of latest LLMs to process

def get_latest_n_llms(llms_dict, n):
    return dict(list(llms_dict.items())[-n:])

def rerun2_experiment1b():
    # Define paths
    data_path = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity', 'data.csv')
    previous_results_path = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity', 'results_20240731_094553.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_rerun2_{timestamp}.csv"
    result_path = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity', results_filename)

    # Load the original data and previous results
    df = pd.read_csv(data_path)
    previous_results = pd.read_csv(previous_results_path)

    print(f"Original data shape: {df.shape}")
    print(f"Previous results shape: {previous_results.shape}")

    # Get the latest N LLMs
    latest_llms = get_latest_n_llms(llms, N_LATEST_LLMS)
    print(f"Processing the following LLMs: {list(latest_llms.keys())}")

    # Process the data with the latest LLMs
    start_time = time.time()
    new_results = process_llms_and_df_b(latest_llms, df, specific_question_type, saving_path=result_path)

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

    # Calculate and print accuracy for each new LLM
    for llm_name in latest_llms.keys():
        if f'{llm_name}_performance' in final_results.columns:
            accuracy = final_results[f'{llm_name}_performance'].mean() * 100
            print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        else:
            print(f"Accuracy for {llm_name}: Not available (column missing)")

    # Calculate and print total cost for each new LLM
    for llm_name in latest_llms.keys():
        if f'{llm_name}_total_price' in final_results.columns:
            total_cost = final_results[f'{llm_name}_total_price'].sum()
            print(f"Total cost for {llm_name}: ${total_cost:.4f}")
        else:
            print(f"Total cost for {llm_name}: Not available (column missing)")

    print("------")
    print(f"Final data shape: {final_results.shape}")

    # Verification step
    print(f"Number of rows in original data: {len(df)}")
    print(f"Number of rows in previous results: {len(previous_results)}")
    print(f"Number of rows in final results: {len(final_results)}")

    if len(final_results) == len(df):
        print("All rows from the original data are present in the final results.")
    else:
        print("WARNING: The number of rows in the final results does not match the original data.")
        print("You may need to investigate this discrepancy.")

if __name__ == "__main__":
    rerun2_experiment1b()