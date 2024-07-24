import sys
import os

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the root directory of the project (assuming it's two levels up from the script)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# Add the project root to the Python path
sys.path.insert(0, project_root)

print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())
print("Contents of /app directory:", os.listdir('/app'))
print("Project root:", project_root)
print("Python path:", sys.path)

# Now try to import the config
try:
    from config.llm_config import llms
    print("Successfully imported llms from config.llm_config")
except ImportError as e:
    print(f"Failed to import llms: {e}")
    print("Contents of config directory:", os.listdir(os.path.join(project_root, 'config')))

from llm.experiment1 import process_llms_and_df
from llm.prompts import exp1_system_prompt, exp1_user_prompt, exp1_specific_question
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