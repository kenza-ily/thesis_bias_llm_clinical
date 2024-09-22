import json
import tiktoken
import openai
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import os
import pandas as pd
import time
import random
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


# Metadata
PRICE_PER_1M_TOKENS_INPUT = 0.3
PRICE_PER_1M_TOKENS_OUTPUT = 1.2
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


# Prompts
from llm.prompts import exp6_system_prompt_xpl, exp6_system_prompt_mcq, exp6_user_prompt_xpl, exp6_user_prompt_mcq

def create_user_prompt_function(template_str, task="MCQ"):
    if task=="MCQ":
        system_prompt= exp6_system_prompt_mcq
        def user_prompt_mcq(clinical_case, question, options):
            return template_str.format(
                CLINICAL_CASE=clinical_case,
                QUESTION=question,
                OPTIONS=options
            )
        user_prompt_fct= user_prompt_mcq
    elif task=="XPL":
        system_prompt= exp6_system_prompt_xpl
        def user_prompt_mcq(clinical_case, question, options, solution):
            return template_str.format(
                CLINICAL_CASE=clinical_case,
                QUESTION=question,
                OPTIONS=options,
                SOLUTION=solution
            )
        user_prompt_fct= user_prompt_mcq
    else:
        raise ValueError("task must be 'MCQ' or 'XPL'")
    return system_prompt, user_prompt_fct


# Initialize the tokenizer (using gpt-4 as a proxy for gpt-4o-mini)
enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text):
    """Count the number of tokens in the text."""
    return len(enc.encode(text))

# API handling
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

# Processing
def process_batch(batch, model,system_prompt, user_prompt_fct,llm_used,ft_or_baseline):
    results = []
    batch_input_tokens = 0
    batch_output_tokens = 0
    
    for _, row in batch.iterrows():
        case = row['case']
        question = row['normalized_question']
        options = f"A. {row['opa_shuffled']}\nB. {row['opb_shuffled']}\nC. {row['opc_shuffled']}\nD. {row['opd_shuffled']}"
        solution=f"{row['answer_idx_shuffled'].upper()} {row['answer']}"
        user_prompt = user_prompt_fct(case, question, options,solution)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
        batch_input_tokens += input_tokens
        
        start_time = time.time()
        completion = handle_api_call(
            client.chat.completions.create,
            model=model,
            messages=messages
        )
        end_time = time.time()
        running_time = end_time - start_time
        
        if completion is not None:
            response = completion.choices[0].message.content
            output_tokens = completion.usage.completion_tokens
            batch_output_tokens += output_tokens
        else:
            response = "Error: API call failed"
            output_tokens = 0
        
        results.append({
            f'llm_{llm_used}_{ft_or_baseline}_running_time': running_time,
            f'llm_{llm_used}_{ft_or_baseline}_prompt': str(messages),
            f'llm_{llm_used}_{ft_or_baseline}_response': response,
            f'llm_{llm_used}_{ft_or_baseline}_input_tokens': input_tokens,
            f'llm_{llm_used}_{ft_or_baseline}_output_tokens': output_tokens
        })
    
    return results, batch_input_tokens, batch_output_tokens

def process_csv(file_path, save_dir, model, model_name, batch_size=10):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    print(f"Processing file: {file_path}")
    print(f"Using model: {model}")
    print(f"Model name: {model_name}")
    
    total_input_tokens = 0
    total_output_tokens = 0
    num_calls = 0

    
    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    # Initialize new columns
    llm_used=model_name
    ft_or_baseline= "ft" if model.startswith("ft") else "baseline"
    new_columns = [f'llm_{llm_used}_{ft_or_baseline}_running_time', f'llm_{llm_used}_{ft_or_baseline}_prompt', f'llm_{llm_used}_{ft_or_baseline}_response', 
                  f'llm_{llm_used}_{ft_or_baseline}_input_tokens',f'llm_{llm_used}_{ft_or_baseline}_output_tokens']
    for col in new_columns:
        df[col] = None
    
    for i in tqdm(range(0, total_rows, batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_batch, batch, model)
            results, batch_input_tokens, batch_output_tokens = future.result()
        
        total_input_tokens += batch_input_tokens
        total_output_tokens += batch_output_tokens
        
        for j, result in enumerate(results):
            idx = i + j
            for col, value in result.items():
                df.loc[idx, col] = value
        
        num_calls += len(batch)
        
        # Save progress at row 1 and every 10 rows after that
        if i == 0 or (i + batch_size) % 10 == 0 or (i + batch_size) >= total_rows:
            save_path = os.path.join(save_dir, "results_fw4_GxE_gpt4omini.csv")
            df.to_csv(save_path, index=False)
            
            progress_percentage = ((i + batch_size) / total_rows) * 100
            input_cost = (total_input_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_INPUT
            output_cost = (total_output_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_OUTPUT
            total_cost = input_cost + output_cost
            
            print(f"\nProgress: {progress_percentage:.1f}%")
            print(f"Input tokens: {total_input_tokens}, Cost: ${input_cost:.4f}")
            print(f"Output tokens: {total_output_tokens}, Cost: ${output_cost:.4f}")
            print(f"Total cost so far: ${total_cost:.4f}")
    
    # Final statistics
    print("\nProcessing complete!")
    print(f"Total calls: {num_calls}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Final input cost: ${(total_input_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_INPUT:.4f}")
    print(f"Final output cost: ${(total_output_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_OUTPUT:.4f}")
    print(f"Total final cost: ${((total_input_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_INPUT + (total_output_tokens / 1_000_000) * PRICE_PER_1M_TOKENS_OUTPUT):.4f}")
