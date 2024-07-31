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
from llm.experiment1b import process_llms_and_df_b

# Hard-coded
specific_question_type = "gender"

def run_experiment1b():
    # Define paths
    data_path = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity', 'data.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{timestamp}.csv"
    result_path = os.path.join(project_root, 'results', 'experiment1', 'genderxethnicity', results_filename)

    # Load the data
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}")
    print(f"Original data shape: {df.shape}")
    print("------")

    # Process the data with all LLMs
    start_time = time.time()
    results_df = process_llms_and_df_b(llms, df, specific_question_type, saving_path=result_path)
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
    
    print("------")
    print(f"Final data shape: {df.shape}")
    
if __name__ == "__main__":
    run_experiment1b()