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
  
  # -------- Q2a
  # question selection
  if specific_question_type=='gender':
    specific='gender'
  elif specific_question_type=='ethnicity':
    specific='ethnicity'
  else:
    raise ValueError("Unrecognised question type")
  
  # prompt
  prompt_2a = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
    ("assistant", response_1.content),
    ("user", specific_question)
])
  # chain
  chain_2a = prompt_2a | llm
  
  # invoke
  # -------- Q2a
  try:
      prompt_value_2a = handle_api_call(prompt_2a.invoke, {"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options,"SPECIFIC":specific})
      start_time_2a = time.time()
      response_2a = handle_api_call(chain_2a.invoke, {"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options, "SPECIFIC":specific})
      end_time_2a = time.time()
      running_time_2a = end_time_2a - start_time_2a
      chat_history.extend([prompt_value_2a.messages[3].content, response_2a.content])
  except Exception as e:
      print(f"ERROR - Prompt 2a: {str(e)}")
      running_time_2a = None
      response_2a = None
      prompt_value_2a = None
      chat_history.extend([None, None])
  
  
  # metadata
  if prompt_value_2a is not None:
    completion_tokens_2a = response_2a.response_metadata['token_usage']['completion_tokens']
    prompt_tokens_2a = response_2a.response_metadata['token_usage']['prompt_tokens']
    finish_reason_2a=response_2a.response_metadata['finish_reason']
  else:
    completion_tokens_2a = None
    prompt_tokens_2a = None
    finish_reason_2a = None
  
  # -------- Q2b - first one
  # question selection
  if specific_question_type=='gender':
    specific='ethnicity'
  elif specific_question_type=='ethnicity':
    specific='gender'
  else:
    raise ValueError("Unrecognised question type")
  
  # prompt
  prompt_2b = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
    ("assistant", response_1.content),
    ("user", specific_question)
])
  # chain
  chain_2b = prompt_2b | llm
  
  # invoke
  try:
    prompt_value_2b = handle_api_call(prompt_2b.invoke,{"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options,"SPECIFIC":specific})
    start_time_2b = time.time()
    response_2b = handle_api_call(chain_2b.invoke,{"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options, "SPECIFIC":specific})  # Pass chat history to question 2
    end_time_2b=time.time()
    running_time_2b=end_time_2b-start_time_2b
    chat_history.extend([prompt_value_2b.messages[3].content, response_2b.content])
  except Exception as e:
    print(f"ERROR - Prompt 2b: {str(e)}")
    print(f"Case: {case}")
    running_time_2b = None
    response_2b = None
    prompt_value_2b = None
    chat_history.extend([None, None])
    print("Skipping this question due to error.")
    
  # metadata
  if response_2b is not None:
    completion_tokens_2b = response_2b.response_metadata['token_usage']['completion_tokens']
    prompt_tokens_2b = response_2b.response_metadata['token_usage']['prompt_tokens']
    finish_reason_2b=response_2b.response_metadata['finish_reason']
  else:
    completion_tokens_2b = None
    prompt_tokens_2b = None
    finish_reason_2b = None

  # ====== RETURN
  return response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2a, prompt_value_2a, completion_tokens_2a, prompt_tokens_2a, finish_reason_2a, running_time_2a, response_2b, prompt_value_2b, completion_tokens_2b, prompt_tokens_2b, finish_reason_2b, running_time_2b, chat_history


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
              response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2a, prompt_value_2a, completion_tokens_2a, prompt_tokens_2a, finish_reason_2a, running_time_2a, response_2b, prompt_value_2b, completion_tokens_2b, prompt_tokens_2b, finish_reason_2b, running_time_2b, chat_history = experiment1_llm_pipeline_b(
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
              response_2a = None
              prompt_value_2a = None
              completion_tokens_2a = None
              prompt_tokens_2a = None
              finish_reason_2a = None
              running_time_2a = None
              response_2b = None
              prompt_value_2b = None
              completion_tokens_2b = None
              prompt_tokens_2b = None
              finish_reason_2b = None
              running_time_2b = None
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

            # Processing for prompt_value_2a
            if prompt_value_2a is not None and hasattr(prompt_value_2a, 'messages'):
                prompt_value_2a_str = "\n".join([msg.content for msg in prompt_value_2a.messages if hasattr(msg, 'content')])
                
                response_2a_str = response_2a.content if response_2a else ''
                response_2a_parts = response_2a_str.split('\n', 1)
                response_2a_label = response_2a_parts[0] if len(response_2a_parts) > 0 else ''
                response_2a_explanation = response_2a_parts[1] if len(response_2a_parts) > 1 else ''

                # Store prompt_value_2a related data
                df_results.loc[idx_val, f'{llm_name}_prompt2a'] = prompt_value_2a_str
                df_results.loc[idx_val, f'{llm_name}_response2a'] = response_2a_str
                df_results.loc[idx_val, f'{llm_name}_label2a'] = response_2a_label
                df_results.loc[idx_val, f'{llm_name}_explanation2a'] = response_2a_explanation
                # Metadata
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2a'] = completion_tokens_2a
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2a'] = prompt_tokens_2a
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2a'] = finish_reason_2a
                df_results.loc[idx_val, f'{llm_name}_running_time_2a'] = running_time_2a
            else:
                df_results.loc[idx_val, f'{llm_name}_prompt2a'] = None
                df_results.loc[idx_val, f'{llm_name}_response2a'] = None
                df_results.loc[idx_val, f'{llm_name}_label2a'] = None
                df_results.loc[idx_val, f'{llm_name}_explanation2a'] = None
                # metadata
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2a'] = None
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2a'] = None
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2a'] = None
                df_results.loc[idx_val, f'{llm_name}_running_time_2a'] = None

            # Processing for prompt_value_2b
            if prompt_value_2b is not None and hasattr(prompt_value_2b, 'messages'):
                prompt_value_2b_str = "\n".join([msg.content for msg in prompt_value_2b.messages if hasattr(msg, 'content')])
                
                response_2b_str = response_2b.content if response_2b else ''
                response_2b_parts = response_2b_str.split('\n', 1)
                response_2b_label = response_2b_parts[0] if len(response_2b_parts) > 0 else ''
                response_2b_explanation = response_2b_parts[1] if len(response_2b_parts) > 1 else ''

                # Store prompt_value_2b related data
                df_results.loc[idx_val, f'{llm_name}_prompt2b'] = prompt_value_2b_str
                df_results.loc[idx_val, f'{llm_name}_response2b'] = response_2b_str
                df_results.loc[idx_val, f'{llm_name}_label2b'] = response_2b_label
                df_results.loc[idx_val, f'{llm_name}_explanation2b'] = response_2b_explanation
                # metadata
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2b'] = completion_tokens_2b
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2b'] = prompt_tokens_2b
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2b'] = finish_reason_2b
                df_results.loc[idx_val, f'{llm_name}_running_time_2b'] = running_time_2b
            else:
                df_results.loc[idx_val, f'{llm_name}_prompt2b'] = None
                df_results.loc[idx_val, f'{llm_name}_response2b'] = None
                df_results.loc[idx_val, f'{llm_name}_label2b'] = None
                df_results.loc[idx_val, f'{llm_name}_explanation2b'] = None
                df_results.loc[idx_val, f'{llm_name}_completion_tokens_2b'] = None
                df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2b'] = None
                df_results.loc[idx_val, f'{llm_name}_finish_reason_2b'] = None
                df_results.loc[idx_val, f'{llm_name}_running_time_2b'] = None
                  
                
                
            
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
