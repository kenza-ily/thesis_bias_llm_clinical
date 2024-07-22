from llm.prompts import exp1_system_prompt, exp1_user_prompt, exp1_specific_question

def experiment1(llm, system_prompt,user_prompt,case,question,options,specific_question_type):
    # ===== Initialisation
    chat_history = []
    
    # ======1 / QUESTION 1
    # Define the prompt
    prompt_1 = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("user", user_prompt)
  ])
    chain_1 = prompt_1 | llm
    
    # Question 1
    prompt_value_1 = prompt_1.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options})
    response_1 = chain_1.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options})
    chat_history.extend([prompt_value_1.messages[0].content,prompt_value_1.messages[1].content, response_1.content])
    
  #   # ======2 / QUESTION 2
    # ===== Select question 2
    if specific_question_type=='gender':
      specific='gender'
    elif specific_question_type=='ethnicity':
      specific='ethnicity'
    else:
      raise ValueError("Unrecognised question type")
    # Define the prompt
    prompt_2 = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("user", user_prompt),
      ("assistant", response_1.content),
      ("user", exp1_specific_question)
  ])
    chain_2 = prompt_2 | llm
    
    # Question 2
    prompt_value_2 = prompt_2.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options,"SPECIFIC":specific})
    response_2 = chain_2.invoke({"CLINICAL_CASE": case,"QUESTION":question,"OPTIONS":options, "SPECIFIC":specific})  # Pass chat history to question 2
    chat_history.extend([prompt_value_2.messages[3].content, response_2.content])
    
    
    # METADATA
    completion_tokens = response_1.response_metadata['token_usage']['completion_tokens']
    prompt_tokens = response_1.response_metadata['token_usage']['prompt_tokens']
    finish_reason=response_1.response_metadata['finish_reason']


    return response_1, prompt_value_1, response_2, prompt_value_2, chat_history, completion_tokens, prompt_tokens, finish_reason


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

        # df loop
        for idx_val, row_val in df_results.iterrows():
            # Extracting the data
            clinical_case = row_val['case']
            question=row_val['normalized_question']
            options = f"A. {row_val['opa_shuffled']}\nB. {row_val['opb_shuffled']}\nC. {row_val['opc_shuffled']}\nD. {row_val['opd_shuffled']}"
            correct_answer=row_val['answer_idx_shuffled']; correct_answer_lower = correct_answer.lower()

            # Run the LLM
            response_1, prompt_value_1, response_2, prompt_value_2, chat_history, completion_tokens, prompt_tokens, finish_reason = experiment1(
            llm=llm_data["model"],
            system_prompt=exp1_system_prompt,
            user_prompt=exp1_user_prompt,
            case=clinical_case,
            question=question,
            options=options,
            specific_question_type=specific_question_type
        )
            # POSTPROCESSING
            # ---- Prompts
            prompt_value_1_str = f"System_prompt: {prompt_value_1.messages[0].content}\nUser Prompt: {prompt_value_1.messages[1].content}"
            prompt_value_2_str= f"{prompt_value_2.messages[0].content}\n{prompt_value_2.messages[1].content}\n{prompt_value_2.messages[2].content}\n{prompt_value_2.messages[3].content}"
            # ----- Responses
            response_1_str = response_1.content
            response_2_str = response_2.content
            # Split responses into label and explanation
            response_1_parts = response_1_str.split('\n', 1)
            response_2_parts = response_2_str.split('\n', 1)
            
            response_1_label = response_1_parts[0] if len(response_1_parts) > 0 else ''
            response_1_explanation = response_1_parts[1] if len(response_1_parts) > 1 else ''
            
            response_2_label = response_2_parts[0] if len(response_2_parts) > 0 else ''
            response_2_explanation = response_2_parts[1] if len(response_2_parts) > 1 else ''
            # Inside the loop where you process each row
            
            response_1_label_lower = response_1_label.lower()
            row_performance = 1 if response_1_label_lower == correct_answer_lower else 0
            
            # ----- Store experiment parameters in df_results
            # Prompts
            df_results.at[idx_val, f'{llm_name}_prompt1'] = prompt_value_1_str
            df_results.at[idx_val, f'{llm_name}_prompt2'] = prompt_value_2_str
            # Responses
            df_results.at[idx_val, f'{llm_name}_response1'] = response_1_str
            df_results.at[idx_val, f'{llm_name}_response2'] = response_2_str
            # Chat History
            df_results.at[idx_val, f'{llm_name}_chat_history'] = chat_history
            # Metadata
            df_results.at[idx_val, f'{llm_name}_finish_reason'] = finish_reason
            df_results.at[idx_val, f'{llm_name}_prompt_tokens'] = prompt_tokens
            df_results.at[idx_val, f'{llm_name}_completion_tokens'] = completion_tokens
            # Pricing
            df_results.at[idx_val, f'{llm_name}_input_price'] = llm_data["price_per_input_token"] * prompt_tokens
            df_results.at[idx_val, f'{llm_name}_output_price'] = llm_data["price_per_output_token"] * completion_tokens
            df_results.at[idx_val, f'{llm_name}_total_price'] = df_results.at[idx_val, f'{llm_name}_input_price'] + df_results.at[idx_val, f'{llm_name}_output_price']
            # ---- Store experiment results in df_results
            df_results.at[idx_val, f'{llm_name}_label1'] = response_1_label
            df_results.at[idx_val, f'{llm_name}_explanation1'] = response_1_explanation
            df_results.at[idx_val, f'{llm_name}_label2'] = response_2_label
            df_results.at[idx_val, f'{llm_name}_explanation2'] = response_2_explanation
            # Performance
            df_results.at[idx_val, f'{llm_name}_performance'] = row_performance
            
            # ----- Print progress every 10%
            if (idx_val + 1) % progress_interval == 0:
                progress_percentage = ((idx_val + 1) / total_rows) * 100
                print(f"Progress: {progress_percentage:.1f}% complete")

        # You can also keep a running total if needed
        total_performance = df_results[f'{llm_name}_performance'].sum()
        accuracy = total_performance / len(df_results) * 100
        print(f"Accuracy for {llm_name}: {accuracy:.2f}%")
        print(f"Finished processing with LLM: {llm_name}")  # Print when finished with current LLM
            
    print("\nAll LLMs processed. Returning results.")
    return df_results
