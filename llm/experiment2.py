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


# =========== Heart of the experiment
def experiment1_llm_pipeline_b(llm,case,question,options,specific_question_type):
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
  # prompt
  prompt_1 = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])
  # chain
  chain_1 = prompt_1 | llm
  
  # invoke
  # PROMPT1
  prompt_value_1 = handle_api_call(prompt_1.invoke, {"CLINICAL_CASE": case, "QUESTION": question, "OPTIONS": options})
  if prompt_value_1 is None:
        print("ERROR - Prompt 1: Failed to get a valid response")
        print(f"Case: {case}")
        print("Skipping this question.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [None, None, None]
  # CHAIN1
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
  if prompt_value_1 is not None:
    prompt_tokens_1 = response_1.response_metadata['token_usage']['prompt_tokens']
    completion_tokens_1 = response_1.response_metadata['token_usage']['completion_tokens']
    finish_reason_1=response_1.response_metadata['finish_reason']
  else:
    prompt_tokens_1 = None
    completion_tokens_1 = None
    finish_reason_1 = None
  
  # -------- Q2
  # question selection
  if specific_question_type=='gender':
    specific='gender'
  elif specific_question_type=='ethnicity':
    specific='ethnicity'
  else:
    raise ValueError("Unrecognised question type")
  
  # prompt
  prompt_2 = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
    ("assistant", response_1.content),
    ("user", specific_question)
])
  # chain
  chain_2 = prompt_2 | llm
  
  # invoke
  # -------- Q2
  try:
      prompt_value_2 = handle_api_call(prompt_2.invoke, {"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options,"SPECIFIC":specific})
      start_time_2 = time.time()
      response_2 = handle_api_call(chain_2.invoke, {"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options, "SPECIFIC":specific})
      end_time_2 = time.time()
      running_time_2 = end_time_2 - start_time_2
      chat_history.extend([prompt_value_2.messages[3].content, response_2.content])
  except Exception as e:
      print(f"ERROR - Prompt 2: {str(e)}")
      running_time_2 = None
      response_2 = None
      prompt_value_2 = None
      chat_history.extend([None, None])
  
  
  # metadata
  if prompt_value_2 is not None:
    completion_tokens_2 = response_2.response_metadata['token_usage']['completion_tokens']
    prompt_tokens_2 = response_2.response_metadata['token_usage']['prompt_tokens']
    finish_reason_2=response_2.response_metadata['finish_reason']
  else:
    completion_tokens_2 = None
    prompt_tokens_2 = None
    finish_reason_2 = None

  # ====== RETURN
  return response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2, prompt_value_2, completion_tokens_2, prompt_tokens_2, finish_reason_2, running_time_2, chat_history


# =========== Experiment pipeline
def process_llms_and_df_b(llms, df, specific_question_type, saving_path=None):
    # Create df_results as a copy of df
    df_results = df.copy()

    # Initialization
    total_rows = len(df)
    progress_interval = max(1, total_rows // 10)  # Calculate 10% of total rows

    # LLM loop
    for llm_name, llm_data in llms.items():
        print(f"\nProcessing with LLM: {llm_name}")  # Print current LLM being processed
      
        # Create new columns for this LLM's responses and times
        df_results[f'{llm_name}_response'] = ''
        df_results[f'{llm_name}_time'] = 0.0
        
        # Get the LLM model
        llm_model = llm_data.get("model")
        if llm_model is None:
            print(f"Warning: No model found for {llm_name}. Skipping this LLM.")
            continue

        # df loop
        for idx_val, row_val in df_results.iterrows():
            # Extracting the data
            clinical_case = row_val['case']
            question=row_val['normalized_question']
            options = f"A. {row_val['opa_shuffled']}\nB. {row_val['opb_shuffled']}\nC. {row_val['opc_shuffled']}\nD. {row_val['opd_shuffled']}"
            correct_answer=row_val['answer_idx_shuffled']; correct_answer_lower = correct_answer.lower()

            # Run the LLM
            try:
              response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2, prompt_value_2, completion_tokens_2, prompt_tokens_2, finish_reason_2, running_time_2, chat_history = experiment1_llm_pipeline_b(
              llm=llm_model,
              case=clinical_case,
              question=question,
              options=options,
              specific_question_type=specific_question_type
          )
            except KeyError as e:
              print(f"KeyError in llm_data for {llm_name}: {e}")
              print(f"Available keys in llm_data: {llm_data.keys()}")
              continue
            except Exception as e:
              print(f"Unexpected error occurred: {str(e)}")
              print(f"Question: {question}")
              response_1 = None
              prompt_value_1 = None
              completion_tokens_1 = None
              prompt_tokens_1 = None
              finish_reason_1 = None
              running_time_1 = None
              response_2 = None
              prompt_value_2 = None
              completion_tokens_2 = None
              prompt_tokens_2 = None
              finish_reason_2 = None
              running_time_2 = None
              chat_history = [None, None, None, None, None, None, None, None]
              print("Skipping this question.")
            
            
            # WORK
            if prompt_value_1 is not None and hasattr(prompt_value_1, 'messages'):
                prompt_value_1_str = f"System_prompt: {prompt_value_1.messages[0].content}\nUser Prompt: {prompt_value_1.messages[1].content}"
                
                response_1_str = response_1.content if response_1 else ''
                response_1_parts = response_1_str.split('\n', 1)
                response_1_label = response_1_parts[0] if len(response_1_parts) > 0 else ''
                response_1_explanation = response_1_parts[1] if len(response_1_parts) > 1 else ''
                
                response_1_label_lower = response_1_label.lower()
                row_performance = 1 if response_1_label_lower == correct_answer_lower else 0

                # Store prompt_value_1 related data
                df_results.loc[idx_val, f'{llm_name}_prompt1'] = prompt_value_1_str
                df_results.loc[idx_val, f'{llm_name}_response1'] = response_1_str
                df_results.loc[idx_val, f'{llm_name}_label1'] = response_1_label
                df_results.loc[idx_val, f'{llm_name}_explanation1'] = response_1_explanation
                df_results.loc[idx_val, f'{llm_name}_performance'] = row_performance
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_1'] = completion_tokens_1
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_1'] = prompt_tokens_1
                df_results.loc[idx_val, f'{llm_name}_finish_reason_1'] = finish_reason_1
                df_results.loc[idx_val, f'{llm_name}_running_time_1'] = running_time_1
            else:
                df_results.loc[idx_val, f'{llm_name}_prompt1'] = None
                df_results.loc[idx_val, f'{llm_name}_response1'] = None
                df_results.loc[idx_val, f'{llm_name}_label1'] = None
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_1'] = None
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_1'] = None
                df_results.loc[idx_val, f'{llm_name}_finish_reason_1'] = None
                df_results.loc[idx_val, f'{llm_name}_running_time_1'] = None
                df_results.loc[idx_val, f'{llm_name}_explanation1'] = None
                df_results.loc[idx_val, f'{llm_name}_performance'] = None

            # Processing for prompt_value_2
            if prompt_value_2 is not None and hasattr(prompt_value_2, 'messages'):
                prompt_value_2_str = "\n".join([msg.content for msg in prompt_value_2.messages if hasattr(msg, 'content')])
                
                response_2_str = response_2.content if response_2 else ''
                response_2_parts = response_2_str.split('\n', 1)
                response_2_label = response_2_parts[0] if len(response_2_parts) > 0 else ''
                response_2_explanation = response_2_parts[1] if len(response_2_parts) > 1 else ''

                # Store prompt_value_2 related data
                df_results.loc[idx_val, f'{llm_name}_prompt2'] = prompt_value_2_str
                df_results.loc[idx_val, f'{llm_name}_response2'] = response_2_str
                df_results.loc[idx_val, f'{llm_name}_label2'] = response_2_label
                df_results.loc[idx_val, f'{llm_name}_explanation2'] = response_2_explanation
                # Metadata
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2'] = completion_tokens_2
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2'] = prompt_tokens_2
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2'] = finish_reason_2
                df_results.loc[idx_val, f'{llm_name}_running_time_2'] = running_time_2
            else:
                df_results.loc[idx_val, f'{llm_name}_prompt2'] = None
                df_results.loc[idx_val, f'{llm_name}_response2'] = None
                df_results.loc[idx_val, f'{llm_name}_label2'] = None
                df_results.loc[idx_val, f'{llm_name}_explanation2'] = None
                # metadata
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2'] = None
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2'] = None
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2'] = None
                df_results.loc[idx_val, f'{llm_name}_running_time_2'] = None

            
            
            # ----- Print progress every 10%
            if (idx_val + 1) % progress_interval == 0:
                progress_percentage = ((idx_val + 1) / total_rows) * 100
                print(f"----- Progress: {progress_percentage:.1f}% complete")
                if saving_path is not None:
                    df_results.to_csv(saving_path, index=False)
                    print(f"Saved progress to {saving_path}")

        # You can also keep a running total if needed
        total_performance = df_results[f'{llm_name}_performance'].sum()
        accuracy = total_performance / len(df_results) * 100
        print(f"Total performance for {llm_name}: {total_performance}/{len(df_results)}")
        print(f"Total price for {llm_name}: {df_results[f'{llm_name}_total_price'].sum()}")
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        print(f"Finished processing with LLM: {llm_name}")  # Print when finished with current LLM
            
    print("\nAll LLMs processed. Returning results.")
    return df_results
