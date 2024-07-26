import sys
import os

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the root directory of the project (two levels up from the script)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# Add the project root to the Python path
sys.path.insert(0, project_root)

# Now you can use absolute imports
from llm.llm_config import llms
from llm.experiment1 import process_llms_and_df

import pandas as pd
import time
from datetime import datetime

def run_test():
    # Load the test data
    data_path = os.path.join(project_root, 'results', 'tests', 'data.csv')
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    df = pd.read_csv(data_path)
    
    # Set the specific question type
    specific_question_type = "gender"  # or "ethnicity"
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process the data with all LLMs
    start_time = time.time()
    results_df = process_llms_and_df(llms, df, specific_question_type)
    end_time = time.time()
    
    # Calculate total execution time
    total_time = end_time - start_time
    
    # Save the results in the current directory
    results_filename = f"results_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    
    # Print summary
    print(f"\nTest completed in {total_time:.2f} seconds.")
    print(f"Results saved to: {os.path.abspath(results_filename)}")
    
    # Calculate and print accuracy for each LLM
    for llm_name in llms.keys():
        accuracy = results_df[f'{llm_name}_performance'].mean() * 100
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
    
    # Calculate and print total cost for each LLM
    for llm_name in llms.keys():
        total_cost = results_df[f'{llm_name}_total_price'].sum()
        print(f"Total cost for {llm_name}: ${total_cost:.4f}")

if __name__ == "__main__":
    run_test()