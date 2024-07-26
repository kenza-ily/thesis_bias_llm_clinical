from llm.prompts import exp1_system_prompt, exp1_user_prompt, exp1_specific_question
from langchain_core.prompts import ChatPromptTemplate
import time


# =========== Heart of the experiment
def experiment1_llm_pipeline(llm,case,question,options,specific_question_type):
  # Debugging
  if llm is None:
        raise ValueError("LLM model is None. Please ensure a valid model is provided.")
      
  # --- 1. Prompts 
  system_prompt = exp1_system_prompt
  user_prompt = exp1_user_prompt
  specific_question = exp1_specific_question
  
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
  prompt_value_1 = prompt_1.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options})
  start_time_1 = time.time()
  response_1 = chain_1.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options})
  end_time_1 = time.time()
  running_time_1=end_time_1-start_time_1
  chat_history.extend([prompt_value_1.messages[0].content,prompt_value_1.messages[1].content, response_1.content])
  
  # metadata
  prompt_tokens_1 = response_1.response_metadata['token_usage']['prompt_tokens']
  completion_tokens_1 = response_1.response_metadata['token_usage']['completion_tokens']
  finish_reason_1=response_1.response_metadata['finish_reason']
  
  # -------- Q2
  # question selection
  if specific_question_type=='gender':
    specific='gender'
  elif specific_question_type=='ethnicity':
    specific='ethnicity'
  elif specific_question_type=='age':
    specific='age'
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
  prompt_value_2 = prompt_2.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options,"SPECIFIC":specific})
  start_time_2 = time.time()
  response_2 = chain_2.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options, "SPECIFIC":specific})  # Pass chat history to question 2
  end_time_2=time.time()
  running_time_2=end_time_2-start_time_2
  chat_history.extend([prompt_value_2.messages[3].content, response_2.content])
  
  
  # metadata
  completion_tokens_2 = response_2.response_metadata['token_usage']['completion_tokens']
  prompt_tokens_2 = response_2.response_metadata['token_usage']['prompt_tokens']
  finish_reason_2=response_2.response_metadata['finish_reason']


  return response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2, prompt_value_2, completion_tokens_2, prompt_tokens_2, finish_reason_2, running_time_2, chat_history


# =========== Experiment pipeline
def process_llms_and_df(llms, df, specific_question_type):
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
              response_1, prompt_value_1, completion_tokens_1, prompt_tokens_1, finish_reason_1, running_time_1, response_2, prompt_value_2, completion_tokens_2, prompt_tokens_2, finish_reason_2, running_time_2, chat_history = experiment1_llm_pipeline(
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
            
            # POSTPROCESSING
            
            # chat history
            chat_history = '\n'.join(chat_history)
            # prompts
            prompt_value_1_str = f"System_prompt: {prompt_value_1.messages[0].content}\nUser Prompt: {prompt_value_1.messages[1].content}"
            prompt_value_2_str= f"{prompt_value_2.messages[0].content}\n{prompt_value_2.messages[1].content}\n{prompt_value_2.messages[2].content}\n{prompt_value_2.messages[3].content}"
            
            # responses
            ## Q1
            response_1_str = response_1.content
            response_1_parts = response_1_str.split('\n', 1)
            response_1_label = response_1_parts[0] if len(response_1_parts) > 0 else ''
            response_1_explanation = response_1_parts[1] if len(response_1_parts) > 1 else ''
            # Performance correctedness
            response_1_label_lower = response_1_label.lower()
            row_performance = 1 if response_1_label_lower == correct_answer_lower else 0
            
            ## Q2
            response_2_str = response_2.content
            response_2_parts = response_2_str.split('\n', 1)
            response_2_label = response_2_parts[0] if len(response_2_parts) > 0 else ''
            response_2_explanation = response_2_parts[1] if len(response_2_parts) > 1 else ''
            
            
            # ----- Store experiment parameters in df_results
            # specific question
            df_results.loc[idx_val, f'{llm_name}_specific_question'] = specific_question_type
            # Prompts
            df_results.loc[idx_val, f'{llm_name}_prompt1'] = prompt_value_1_str
            df_results.loc[idx_val, f'{llm_name}_prompt2'] = prompt_value_2_str
            # Responses
            df_results.loc[idx_val, f'{llm_name}_response1'] = response_1_str
            df_results.loc[idx_val, f'{llm_name}_response2'] = response_2_str
            # Chat History
            df_results.loc[idx_val, f'{llm_name}_chat_history'] = chat_history
            # Metadata
            ## Q1
            df_results.loc[idx_val, f'{llm_name}_finish_reason_1'] = finish_reason_1
            df_results.loc[idx_val, f'{llm_name}_prompt_tokens_1'] = prompt_tokens_1
            df_results.loc[idx_val, f'{llm_name}_completion_tokens_1'] = completion_tokens_1
            df_results.loc[idx_val, f'{llm_name}_running_time_1'] = running_time_1
            ## Q2
            df_results.loc[idx_val, f'{llm_name}_finish_reason_2'] = finish_reason_2
            df_results.loc[idx_val, f'{llm_name}_prompt_tokens_2'] = prompt_tokens_2
            df_results.loc[idx_val, f'{llm_name}_completion_tokens_2'] = completion_tokens_2
            df_results.loc[idx_val, f'{llm_name}_running_time_2'] = running_time_2
            # Pricing
            ## Q1
            df_results.loc[idx_val, f'{llm_name}_input_price_1'] = llm_data["price_per_input_token"] * prompt_tokens_1
            df_results.loc[idx_val, f'{llm_name}_output_price_1'] = llm_data["price_per_output_token"] * completion_tokens_1
            ## Q2
            df_results.loc[idx_val, f'{llm_name}_input_price_2'] = llm_data["price_per_input_token"] * prompt_tokens_2
            df_results.loc[idx_val, f'{llm_name}_output_price_2'] = llm_data["price_per_output_token"] * completion_tokens_2
            ## Total
            df_results.loc[idx_val, f'{llm_name}_total_price'] = df_results.loc[idx_val, f'{llm_name}_input_price_1'] + df_results.loc[idx_val, f'{llm_name}_output_price_1']+df_results.loc[idx_val, f'{llm_name}_input_price_2'] + df_results.loc[idx_val, f'{llm_name}_output_price_2']
            # ---- Store experiment results in df_results
            df_results.loc[idx_val, f'{llm_name}_label1'] = response_1_label
            df_results.loc[idx_val, f'{llm_name}_explanation1'] = response_1_explanation
            df_results.loc[idx_val, f'{llm_name}_label2'] = response_2_label
            df_results.loc[idx_val, f'{llm_name}_explanation2'] = response_2_explanation
            # Performance
            df_results.loc[idx_val, f'{llm_name}_performance'] = row_performance
            
            # ----- Print progress every 10%
            if (idx_val + 1) % progress_interval == 0:
                progress_percentage = ((idx_val + 1) / total_rows) * 100
                print(f"Progress: {progress_percentage:.1f}% complete")

        # You can also keep a running total if needed
        total_performance = df_results[f'{llm_name}_performance'].sum()
        accuracy = total_performance / len(df_results) * 100
        print(f"Total performance for {llm_name}: {total_performance}/{len(df_results)}")
        print(f"Total price for {llm_name}: {df_results[f'{llm_name}_total_price'].sum()}")
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        print(f"Finished processing with LLM: {llm_name}")  # Print when finished with current LLM
            
    print("\nAll LLMs processed. Returning results.")
    return df_results
