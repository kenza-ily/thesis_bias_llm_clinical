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
from llm.experiment1 import process_llms_and_df

# Hard-coded
specific_question_type = "gender"

def rerun_experiment1():
    # Define paths
    data_path = os.path.join(project_root, 'results', 'experiment1', 'gender', 'data.csv')
    previous_results_path = os.path.join(project_root, 'results', 'experiment1', 'gender', 'results_20240729_141017.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_rerun_gpt4o_{timestamp}.csv"
    result_path = os.path.join(project_root, 'results', 'experiment1', 'gender', results_filename)

    # Load the original data and previous results
    df = pd.read_csv(data_path)
    previous_results = pd.read_csv(previous_results_path)

    print(f"Original data shape: {df.shape}")
    print(f"Previous results shape: {previous_results.shape}")

    # Select only GPT-4
    gpt4o_llm = {'llm_gpt4o': llms['llm_gpt4o']}

    # Process the data with GPT-4
    start_time = time.time()
    new_results = process_llms_and_df(gpt4o_llm, df, specific_question_type, saving_path=result_path)
    end_time = time.time()

    # Combine previous results with new results
    all_columns = set(previous_results.columns) | set(new_results.columns)
    for col in all_columns:
        if col not in previous_results.columns:
            previous_results[col] = None
        if col not in new_results.columns:
            new_results[col] = None

    combined_results = pd.concat([previous_results, new_results], axis=1)

    # Calculate total execution time
    total_time = end_time - start_time

    # Save the combined results
    combined_results.to_csv(result_path, index=False)

    # Print summary
    print(f"\nExperiment completed in {total_time:.2f} seconds.")
    print(f"Results saved to: {result_path}")

    # Calculate and print accuracy for GPT-4
    if 'llm_gpt4o_performance' in combined_results.columns:
        accuracy = combined_results['llm_gpt4o_performance'].mean() * 100
        print(f"Accuracy for GPT-4: {accuracy:.2f}%")
    else:
        print("No performance data available for GPT-4")

    # Calculate and print total cost for GPT-4
    if 'llm_gpt4o_total_price' in combined_results.columns:
        total_cost = combined_results['llm_gpt4o_total_price'].sum()
        print(f"Total cost for GPT-4: ${total_cost:.4f}")
    else:
        print("No cost data available for GPT-4")

    print(f"Final combined results shape: {combined_results.shape}")

if __name__ == "__main__":
    rerun_experiment1()