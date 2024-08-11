from llm.prompts import exp0_system_prompt,exp0_user_prompt
from langchain_core.prompts import ChatPromptTemplate
import time


# =========== Heart of the experiment
def experiment0_llm_pipeline(llm,question_original,answer_choices):
  # Debugging
  if llm is None:
        raise ValueError("LLM model is None. Please ensure a valid model is provided.")
      
  # --- 1. Prompts 
  system_prompt=exp0_system_prompt
  user_prompt=exp0_user_prompt
  
  # --- 2. Initialisation
  chat_history = []
  
  # -------- Q1
  # prompt
  prompt_1 = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
])
  # chain
  chain_1 = prompt_1 | llm
  
  # invoke
  examples="" #zero shot because no difference 
  try: 
    prompt_value_1 = prompt_1.invoke({"question": question_original,"answer_choices":answer_choices,"few-shot examples":examples})
    start_time_1 = time.time()
    response_1 = chain_1.invoke({"question": question_original,"answer_choices":answer_choices,"few-shot examples":examples})
    end_time_1 = time.time()
    running_time_1=end_time_1-start_time_1
    chat_history.extend([prompt_value_1.messages[0].content,prompt_value_1.messages[1].content, response_1.content])
  except ValueError as e: 
    if "Azure rate limit" in str(e):
      print("Azure rate limit reached. Waiting for 10 seconds before retrying.")
      time.sleep(10)
      prompt_value_1 = prompt_1.invoke({"question": question_original,"answer_choices":answer_choices,"few-shot examples":examples})
      start_time_1 = time.time()
      response_1 = chain_1.invoke({"question": question_original,"answer_choices":answer_choices,"few-shot examples":examples})
      end_time_1 = time.time()
      running_time_1=end_time_1-start_time_1
      chat_history.extend([prompt_value_1.messages[0].content,prompt_value_1.messages[1].content, response_1.content])
    else:
      print(f"Unexpected error occurred: {str(e)}")
      print(f"Question: {question_original}")
      running_time_1 = None
      response_1 = None
      prompt_value_1 = None
      chat_history.extend([None, None, None])
      print("Skipping this question.")

  
  # metadata
  if prompt_value_1 is not None:
    prompt_tokens_1 = response_1.response_metadata['token_usage']['prompt_tokens']
    completion_tokens_1 = response_1.response_metadata['token_usage']['completion_tokens']
    finish_reason_1=response_1.response_metadata['finish_reason']
  else:
    prompt_tokens_1 = None
    completion_tokens_1 = None
    finish_reason_1=None
  
  # ======= RETURN
  return response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, chat_history


# =========== Experiment pipeline
def process_llms_and_df_0(llms, df,saving_path=None):
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
            question_original = row_val['question']
            answer_choices = f"A. {row_val['opa_shuffled']}\nB. {row_val['opb_shuffled']}\nC. {row_val['opc_shuffled']}\nD. {row_val['opd_shuffled']}"
            correct_answer=row_val['answer_idx_shuffled']; correct_answer_lower = correct_answer.lower()

            # Run the LLM
            try:
              response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1,chat_history = experiment0_llm_pipeline(
              llm=llm_model,
              question_original=question_original,
              answer_choices=answer_choices,
          )
            except KeyError as e:
              print(f"KeyError in llm_data for {llm_name}: {e}")
              print(f"Available keys in llm_data: {llm_data.keys()}")
              continue
            
            # POSTPROCESSING
            
            # chat history
            if all(item is not None for item in chat_history):
              chat_history = '\n'.join(chat_history)
            else:
              chat_history = None

            
            
            
            # ----- Store experiment parameters in df_results
            
            # Responses

            # Chat History
            df_results.loc[idx_val, f'{llm_name}_chat_history'] = chat_history
            # Metadata
            if prompt_value_1 is not None:
              ## Q1
              # postprocessing
              prompt_value_1_str = f"System_prompt: {prompt_value_1.messages[0].content}\nUser Prompt: {prompt_value_1.messages[1].content}"
              response_1_str = response_1.content
              response_1_parts = response_1_str.split('\n', 1)
              response_1_label = response_1_parts[0] if len(response_1_parts) > 0 else ''
              response_1_explanation = response_1_parts[1] if len(response_1_parts) > 1 else ''
              response_1_label_lower = response_1_label.lower()
              row_performance = 1 if response_1_label_lower == correct_answer_lower else 0
              # Store
              df_results.loc[idx_val, f'{llm_name}_prompt1'] = prompt_value_1_str
              df_results.loc[idx_val, f'{llm_name}_response1'] = response_1_str
              df_results.loc[idx_val, f'{llm_name}_finish_reason_1'] = finish_reason_1
              df_results.loc[idx_val, f'{llm_name}_prompt_tokens_1'] = prompt_tokens_1
              df_results.loc[idx_val, f'{llm_name}_completion_tokens_1'] = completion_tokens_1
              df_results.loc[idx_val, f'{llm_name}_running_time_1'] = running_time_1
              # Pricing
              ## Q1
              df_results.loc[idx_val, f'{llm_name}_input_price_1'] = llm_data["price_per_input_token"] * prompt_tokens_1
              df_results.loc[idx_val, f'{llm_name}_output_price_1'] = llm_data["price_per_output_token"] * completion_tokens_1
              ## Total
              df_results.loc[idx_val, f'{llm_name}_total_price'] = df_results.loc[idx_val, f'{llm_name}_input_price_1'] + df_results.loc[idx_val, f'{llm_name}_output_price_1']
              # ---- Store experiment results in df_results
              df_results.loc[idx_val, f'{llm_name}_label1'] = response_1_label
              df_results.loc[idx_val, f'{llm_name}_explanation1'] = response_1_explanation
              # Performance
              df_results.loc[idx_val, f'{llm_name}_performance'] = row_performance
            else:
              df_results.loc[idx_val, f'{llm_name}_finish_reason_1'] = None
              df_results.loc[idx_val, f'{llm_name}_prompt_tokens_1'] = None
              df_results.loc[idx_val, f'{llm_name}_completion_tokens_1'] = None
              df_results.loc[idx_val, f'{llm_name}_running_time_1'] = None
              df_results.loc[idx_val, f'{llm_name}_input_price_1'] = None
              df_results.loc[idx_val, f'{llm_name}_output_price_1'] = None
              df_results.loc[idx_val, f'{llm_name}_total_price'] = None
              df_results.loc[idx_val, f'{llm_name}_label1'] = None
              df_results.loc[idx_val, f'{llm_name}_explanation1'] = None
              df_results.loc[idx_val, f'{llm_name}_performance'] = None
            
            # ----- Print progress every 10%
            if (idx_val + 1) % progress_interval == 0:
                progress_percentage = ((idx_val + 1) / total_rows) * 100
                print(f"Progress: {progress_percentage:.1f}% complete")
                if saving_path is not None:
                    df_results.to_csv(saving_path, index=False)

        # You can also keep a running total if needed
        total_performance = df_results[f'{llm_name}_performance'].sum()
        accuracy = total_performance / len(df_results) * 100
        print(f"Total performance for {llm_name}: {total_performance}/{len(df_results)}")
        print(f"Total price for {llm_name}: {df_results[f'{llm_name}_total_price'].sum()}")
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        print(f"Finished processing with LLM: {llm_name}")  # Print when finished with current LLM
            
    print("\nAll LLMs processed. Returning results.")
    return df_results
