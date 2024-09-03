# ---- 1/ Imports
import os
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from llm.prompts import exp5_system_prompt, exp5_user_prompt

# ---- 2/ Helper functions
def handle_api_call(func, *args, **kwargs):
    max_retries = 5
    base_wait = 10

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("Max retries reached. Skipping this call.")
                    return None
            else:
                print(f"Unexpected error: {str(e)}")
                return None
            
            
def extract_prompt_content(prompt_value):
    if isinstance(prompt_value, ChatPromptValue):
        return "\n".join(message.content for message in prompt_value.messages)
    return str(prompt_value)

def extract_response_content(response):
    if hasattr(response, 'content'):
        return response.content
    return str(response)

def store_results_in_df(df, idx, llm_name, results, experiment_type):
    # Unpack results
    response_1, prompt_value_1,  running_time_1, metadata, chat_history= results

    # Store results for Q1
    df.at[idx, f'{llm_name}_response1'] = extract_response_content(response_1)
    df.at[idx, f'{llm_name}_prompt1'] = extract_prompt_content(prompt_value_1)
    df.at[idx, f'{llm_name}_running_time_1'] = running_time_1

    # Store chat history
    if chat_history:
        df.at[idx, f'{llm_name}_chat_history'] = "\n".join(
            str(message) for message in chat_history if isinstance(message, BaseMessage)
        )
    else:
        df.at[idx, f'{llm_name}_chat_history'] = None

    # Calculate and store performance
    correct_answer = df.at[idx, 'answer_idx_shuffled'].lower()
    response_label = extract_response_content(response_1).split('\n', 1)[0].lower()
    df.at[idx, f'{llm_name}_performance'] = 1 if response_label == correct_answer else 0


# ---- 3/ Experiment pipeline

# ====== FRAMEWORK

def fw2(llm, case, question, options, experiment_type,experiment_number):
    # Debugging
    if llm is None:
        raise ValueError("LLM model is None. Please ensure a valid model is provided.")
      
    # --- 1. Prompts 
    if experiment_number == 5:
        system_prompt = exp5_system_prompt
        user_prompt = exp5_user_prompt
    else:
        raise ValueError("Invalid experiment number. Please provide a valid experiment number.")
    # elif experiment_number == 3:
    #     system_prompt = exp3_system_prompt
    #     user_prompt = exp3_user_prompt
    # elif experiment_number == 4:
    #     system_prompt = exp4_system_prompt
    #     user_prompt = exp4_user_prompt
    # else:
    #     raise ValueError("Invalid experiment number. Please provide a valid experiment number.")
  
    # --- 2. Initialisation
    chat_history = []
  
    # -------- Q1
    prompt_1 = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    chain_1 = prompt_1 | llm
  
    # invoke
    prompt_value_1 = handle_api_call(prompt_1.invoke, {"CLINICAL_CASE": case, "QUESTION": question})
    chat_history.append(prompt_value_1)
    if prompt_value_1 is None:
        print("ERROR - Prompt 1: Failed to get a valid response")
        # print(f"Case: {case}")
        # print("Skipping this question.")
        return None, None, None, None,[None, None]

    start_time_1 = time.time()
    response_1 = handle_api_call(chain_1.invoke, {"CLINICAL_CASE": case, "QUESTION": question})
    chat_history.append(response_1)
    end_time_1 = time.time()
    running_time_1 = end_time_1 - start_time_1
    
    if response_1 is None:
        print("ERROR - Response 1: Failed to get a valid response")
        # print(f"Case: {case}")
        print("Skipping this question.")
        return None, prompt_value_1, None, None, [None, None]

    # metadata
    metadata = response_1.response_metadata
    if metadata is None:
        metadata = {}
        
    return response_1, prompt_value_1,  running_time_1, metadata, chat_history



# ====== PROCESSING
import os
from concurrent.futures import ThreadPoolExecutor

def process_single_llm(llm_name, llm_data, df, experiment_type, experiment_number, saving_dir, saving_path):
    print(f"\nProcessing with LLM: {llm_name}")
    
    # Create a copy of the dataframe for this LLM
    df_llm = df.copy()
    
    # Get the LLM model
    llm_model = llm_data.get("model")
    if llm_model is None:
        print(f"Warning: No model found for {llm_name}. Skipping this LLM.")
        return None

    total_rows = len(df_llm)
    save_interval = max(1, total_rows // 10)  # Save every 10% of rows, minimum 1
    processed_rows = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, row in df_llm.iterrows():
            future = executor.submit(
                fw2,
                llm_model,
                row['case'],
                row['normalized_question'],
                f"A. {row['opa_shuffled']}\nB. {row['opb_shuffled']}\nC. {row['opc_shuffled']}\nD. {row['opd_shuffled']}",
                experiment_type,
                experiment_number
            )
            futures.append((idx, future))

        # Process results as they complete
        for idx, future in tqdm(futures, total=total_rows, desc=f"Processing {llm_name}"):
            try:
                results = future.result()
                # Store results in df_llm
                store_results_in_df(df_llm, idx, llm_name, results, experiment_type)
                
                processed_rows += 1
                
                # Save every 10% of total rows
                if processed_rows % save_interval == 0 or processed_rows == total_rows:
                    df_llm.to_csv(saving_path, index=False)
                    print(f"Saved results for {llm_name} at row {processed_rows} to {saving_path}")
                
            except Exception as e:
                print(f"Error processing row {idx} for {llm_name}: {str(e)}")

    # Final save is already done in the loop, so we don't need to do it again here
    print(f"Completed processing for {llm_name}. Final results saved to {saving_path}")
    
    return df_llm

# ====== MAIN PIPELINE

def process_llms_and_df_fw2(llms, df, experiment_type,repo_dir,experiment_number, experiment_name):
    print(f"Starting experiment: #{experiment_number}, Experiment name: {experiment_name}")
    
    # ----------------- RESULTS DIRECTORY -----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Saving folder
    def create_saving_folder(experiment_number):
        saving_folder = os.path.join(repo_dir, "results", "fw2",f"exp{experiment_number}", f"{timestamp}_{experiment_name}")
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        return saving_folder
    saving_folder=create_saving_folder(experiment_number)
    
    
    

    results = {}
    for llm_name, llm_data in llms.items():
        # Saving path
        saving_dir = os.path.join(saving_folder)
        os.makedirs(saving_dir, exist_ok=True)
        file_name = f"results_exp{experiment_number}_{experiment_type}_{llm_name}.csv"
        saving_path = os.path.join(saving_dir, file_name)
        # Process the LLM
        results[llm_name] = process_single_llm(llm_name, llm_data, df, experiment_type, experiment_number, saving_dir, saving_path)
    print("\nAll LLMs processed. Experiment complete.")
    return results
