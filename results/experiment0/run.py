import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add the project root to the Python path
project_root = '/app'  # This is the mount point in the Docker container
sys.path.insert(0, project_root)

# Import required modules
from llm.llm_config import llms
from llm.experiment0 import process_llms_and_df_0

def run_experiment0():
    # Define paths
    data_path = os.path.join(project_root, 'results', 'experiment0', 'data.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{timestamp}.csv"
    result_path = os.path.join(project_root, 'results', 'experiment0', results_filename)

    # Load the data
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")

    # Process the data with all LLMs
    start_time = time.time()
    results_df = process_llms_and_df_0(llms, df, saving_path=result_path)
    end_time = time.time()

    # Calculate total execution time
    total_time = end_time - start_time

    # Save the results
    results_df.to_csv(result_path, index=False)

    # Print summary
    print(f"\nExperiment completed in {total_time:.2f} seconds.")
    print(f"Results saved to: {result_path}")

    # Calculate and print accuracy for each LLM
    for llm_name in llms.keys():
        accuracy = results_df[f'{llm_name}_performance'].mean() * 100
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")

    # Calculate and print total cost for each LLM
    for llm_name in llms.keys():
        total_cost = results_df[f'{llm_name}_total_price'].sum()
        print(f"Total cost for {llm_name}: ${total_cost:.4f}")

if __name__ == "__main__":
    run_experiment0()