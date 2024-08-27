from llm.models import (
    get_gpt3_model,
    get_gpt4o_model,
    # get_gpt4omini_model,
    get_gpt4turbo_model,
    get_mistral_nemo,
    get_mixtral_8x22b,
    get_mistral_7b,
    get_llama3_8b,
    get_llama3_70b,
    get_llama3_1_8b,
    get_gemma2_2b,
    get_gemma2_9b,
    get_haiku,
    get_sonnet3_5,
    get_gemini_3_5_flash
)

# ---- 2/ LLM definition ----

llms = {
    # ----- OpenAI models -----
    "llm_gpt3": {
        "model_name": "gpt3.5",
        "model": get_gpt3_model(),
        "type":'closed'
    },
    "llm_gpt4o": {
        "model_name": "gpt4o",
        "model": get_gpt4o_model(),
        "type":'closed'
    },
    # "llm_gpt4omini": {
    #     "model_name": "gpt4o-mini",
    #     "model": get_gpt4omini_model(),
    #     "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4omini", costs_content),
    #     "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4omini", costs_content)
    # },
    "llm_gpt4turbo": {
        "model_name": "gpt4-turbo",
        "model": get_gpt4turbo_model(),
        "type":'closed'
    },
    # ---- claude ----
    "llm_haiku": {
        "model_name": "claude-3-haiku",
        "model": get_haiku(),
        "type":'closed'
    },
    "llm_sonnet3_5": {
        "model_name": "claude-3-sonnet3.5",
        "model": get_sonnet3_5(),
        "type":'closed'
    },
    # ---- gemini flash
    "llm_gemini_3_5_flash": {
        "model_name": "gemini-3-5-flash",
        "model": get_gemini_3_5_flash(),
        "type":'closed'
    },
    # ==== OLLAMA
   # ----- Mixtral -----
    "llm_mixtral_nemo": {
        "model_name": "mistral-nemo",
        "model": get_mistral_nemo(),
        "type":'open'
    },
    # "llm_mixtral_8x22b": {
    #     "model_name": "Mixtral-8x22B",
    #     "model": get_mixtral_8x22b(),
    #     "type":'open'
    # },
    "llm_mistral_7b": {
        "model_name": "mistral-7b",
        "model": get_mistral_7b(),
        "type":'open'
    },
    # ----- LLaMas -----
    "llm_llama3_8b": {
        "model_name": "llama3_8b",
        "model": get_llama3_8b(),
        "type":'open'
    },
    # "llm_llama3_70b": {
    #     "model_name": "llama3_70b",
    #     "model": get_llama3_70b(),
    #     "type":'open'
    # },
    "llm_llama3_1_8b": {
        "model_name": "llama3_1_8b",
        "model": get_llama3_1_8b(),
        "type":'open'
    },
    # ----- GEMMA -----
    "llm_llm_gemma2_2b": {
        "model_name": "gemma-2-2b",
        "model": get_gemma2_2b(),
        "type":'open'
    },
    "llm_gemma2_9b": {
        "model_name": "gemma-2-9b",
        "model": get_gemma2_9b(),
        "type":'open'
    }
}