import os
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from llm.prompts import exp2_system_prompt, exp2_user_prompt, exp2_specific_question
from langchain_core.prompts import ChatPromptTemplate
import time
import random


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

def experiment2_llm_pipeline(llm, case, question, options, experiment_type):
    # Debugging
    if llm is None:
        raise ValueError("LLM model is None. Please ensure a valid model is provided.")
      
    # --- 1. Prompts 
    system_prompt = exp2_system_prompt
    user_prompt = exp2_user_prompt
    specific_question = exp2_specific_question
  
    # --- 2. Initialisation
    chat_history = []
  
    # -------- Q1
    prompt_1 = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    chain_1 = prompt_1 | llm
  
    # invoke
    prompt_value_1 = handle_api_call(prompt_1.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options})
    if prompt_value_1 is None:
        print("ERROR - Prompt 1: Failed to get a valid response")
        print(f"Case: {case}")
        print("Skipping this question.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [None, None, None]

    start_time_1 = time.time()
    response_1 = handle_api_call(chain_1.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options})
    end_time_1 = time.time()
    running_time_1 = end_time_1 - start_time_1
    
    if response_1 is None:
        print("ERROR - Response 1: Failed to get a valid response")
        print(f"Case: {case}")
        print("Skipping this question.")
        return None, prompt_value_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [None, None, None]

    # metadata
    completion_tokens_1 = response_1.response_metadata['token_usage']['completion_tokens']
    prompt_tokens_1 = response_1.response_metadata['token_usage']['prompt_tokens']
    finish_reason_1 = response_1.response_metadata['finish_reason']
  
    # -------- Q2 (Gender)
    prompt_2a = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
        ("assistant", response_1.content),
        ("user", specific_question)
    ])
    chain_2a = prompt_2a | llm
  
    prompt_value_2a = handle_api_call(prompt_2a.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options, "SPECIFIC": "gender"})
    start_time_2a = time.time()
    response_2a = handle_api_call(chain_2a.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options, "SPECIFIC": "gender"})
    end_time_2a = time.time()
    running_time_2a = end_time_2a - start_time_2a
    
    completion_tokens_2a = response_2a.response_metadata['token_usage']['completion_tokens']
    prompt_tokens_2a = response_2a.response_metadata['token_usage']['prompt_tokens']
    finish_reason_2a = response_2a.response_metadata['finish_reason']

    # -------- Q2 (Ethnicity) - Only for GxE experiment
    if experiment_type == "GxE":
        prompt_2b = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
            ("assistant", response_1.content),
            ("user", specific_question)
        ])
        chain_2b = prompt_2b | llm
      
        prompt_value_2b = handle_api_call(prompt_2b.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options, "SPECIFIC": "ethnicity"})
        start_time_2b = time.time()
        response_2b = handle_api_call(chain_2b.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options, "SPECIFIC": "ethnicity"})
        end_time_2b = time.time()
        running_time_2b = end_time_2b - start_time_2b
        
        completion_tokens_2b = response_2b.response_metadata['token_usage']['completion_tokens']
        prompt_tokens_2b = response_2b.response_metadata['token_usage']['prompt_tokens']
        finish_reason_2b = response_2b.response_metadata['finish_reason']
    else:
        response_2b = prompt_value_2b = completion_tokens_2b = prompt_tokens_2b = finish_reason_2b = running_time_2b = None

    # ====== RETURN
    return response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, \
           response_2a, prompt_value_2a, completion_tokens_2a, prompt_tokens_2a, finish_reason_2a, running_time_2a, \
           response_2b, prompt_value_2b, completion_tokens_2b, prompt_tokens_2b, finish_reason_2b, running_time_2b, \
           chat_history

def process_single_llm(llm_name, llm_data, df, experiment_type, experiment_dir):
    print(f"\nProcessing with LLM: {llm_name}")
    
    # Create a copy of the dataframe for this LLM
    df_llm = df.copy()
    
    # Get the LLM model
    llm_model = llm_data.get("model")
    if llm_model is None:
        print(f"Warning: No model found for {llm_name}. Skipping this LLM.")
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, row in df_llm.iterrows():
            future = executor.submit(
                experiment2_llm_pipeline,
                llm_model,
                row['case'],
                row['normalized_question'],
                f"A. {row['opa_shuffled']}\nB. {row['opb_shuffled']}\nC. {row['opc_shuffled']}\nD. {row['opd_shuffled']}",
                experiment_type
            )
            futures.append((idx, future))

        # Process results as they complete
        for idx, future in tqdm(futures, total=len(df_llm), desc=f"Processing {llm_name}"):
            try:
                results = future.result()
                # Store results in df_llm
                store_results_in_df(df_llm, idx, llm_name, results, experiment_type)
            except Exception as e:
                print(f"Error processing row {idx} for {llm_name}: {str(e)}")
    # Save results
    save_path = os.path.join(experiment_dir, f"exp2_{experiment_type}_{llm_name}.csv")
    df_llm.to_csv(save_path, index=False)
    print(f"Saved results for {llm_name} to {save_path}")

    # Calculate and print performance metrics
    calculate_and_print_metrics(df_llm, llm_name)
    
    return df_llm

def store_results_in_df(df, idx, llm_name, results, experiment_type):
    # Unpack results
    (response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1,
     response_2a, prompt_value_2a, completion_tokens_2a, prompt_tokens_2a, finish_reason_2a, running_time_2a,
     response_2b, prompt_value_2b, completion_tokens_2b, prompt_tokens_2b, finish_reason_2b, running_time_2b,
     chat_history) = results

    # Store results for Q1
    df.at[idx, f'{llm_name}_response1'] = response_1.content if response_1 else None
    df.at[idx, f'{llm_name}_prompt1'] = prompt_value_1.content if prompt_value_1 else None
    df.at[idx, f'{llm_name}_completion_tokens_1'] = completion_tokens_1
    df.at[idx, f'{llm_name}_prompt_tokens_1'] = prompt_tokens_1
    df.at[idx, f'{llm_name}_finish_reason_1'] = finish_reason_1
    df.at[idx, f'{llm_name}_running_time_1'] = running_time_1

    # Store results for Q2a (Gender)
    df.at[idx, f'{llm_name}_response2a'] = response_2a.content if response_2a else None
    df.at[idx, f'{llm_name}_prompt2a'] = prompt_value_2a.content if prompt_value_2a else None
    df.at[idx, f'{llm_name}_completion_tokens_2a'] = completion_tokens_2a
    df.at[idx, f'{llm_name}_prompt_tokens_2a'] = prompt_tokens_2a
    df.at[idx, f'{llm_name}_finish_reason_2a'] = finish_reason_2a
    df.at[idx, f'{llm_name}_running_time_2a'] = running_time_2a

    # Store results for Q2b (Ethnicity) if applicable
    if experiment_type == "GxE":
        df.at[idx, f'{llm_name}_response2b'] = response_2b.content if response_2b else None
        df.at[idx, f'{llm_name}_prompt2b'] = prompt_value_2b.content if prompt_value_2b else None
        df.at[idx, f'{llm_name}_completion_tokens_2b'] = completion_tokens_2b
        df.at[idx, f'{llm_name}_prompt_tokens_2b'] = prompt_tokens_2b
        df.at[idx, f'{llm_name}_finish_reason_2b'] = finish_reason_2b
        df.at[idx, f'{llm_name}_running_time_2b'] = running_time_2b

    # Store chat history
    df.at[idx, f'{llm_name}_chat_history'] = '\n'.join(chat_history) if chat_history else None

    # Calculate and store performance
    correct_answer = df.at[idx, 'answer_idx_shuffled'].lower()
    response_label = response_1.content.split('\n', 1)[0].lower() if response_1 else ''
    df.at[idx, f'{llm_name}_performance'] = 1 if response_label == correct_answer else 0

def calculate_and_print_metrics(df, llm_name):
    total_performance = df[f'{llm_name}_performance'].sum()
    total_samples = len(df)
    accuracy = (total_performance / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nPerformance metrics for {llm_name}:")
    print(f"Total performance: {total_performance}/{total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")

    if f'{llm_name}_total_price' in df.columns:
        total_price = df[f'{llm_name}_total_price'].sum()
        print(f"Total price: ${total_price:.4f}")

def process_llms_and_df_exp2(llms, df, experiment_type, saving_path=None):
    print("Starting the experiment pipeline...")
    
    experiment_name = input("Please enter a name for this experiment: ")
    print(f"Starting experiment: {experiment_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results_{timestamp}_{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created directory for experiment results: {experiment_dir}")

    results = {}
    for llm_name, llm_data in llms.items():
        results[llm_name] = process_single_llm(llm_name, llm_data, df, experiment_type, experiment_dir)

    print("\nAll LLMs processed. Experiment complete.")
    return results
