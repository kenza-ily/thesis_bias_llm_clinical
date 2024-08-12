from llm.models import (
    get_gpt3_model,
    get_gpt4o_model,
    get_gpt4omini_model,
    get_gpt4turbo_model,
    get_mixtral_8x22b,
    get_llama3_8b,
    get_llama3_70b,
    get_llama3_1_8b,
    get_gemma2_2b,
    get_gemma2_9b,
    get_biomistral_7b
)
import re

# ---- 1/ Helper functions ----

# Price for OpenAI models

def extract_price(variable_name, costs_content):
    pattern = rf'{variable_name}\s*=\s*(\d+(?:\.\d+)?)/\(1e6\)'
    match = re.search(pattern, costs_content)
    return float(match.group(1)) / 1e6 if match else None

with open('data/config/costs.txt', 'r') as file:
    costs_content = file.read()

# ---- 2/ LLM definition ----

llms = {
    # ----- OpenAI models -----
    "llm_gpt3": {
        "model_name": "gpt3.5",
        "model": get_gpt3_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT3", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT3", costs_content)
    },
    "llm_gpt4o": {
        "model_name": "gpt4o",
        "model": get_gpt4o_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4o", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4o", costs_content)
    },
    "llm_gpt4omini": {
        "model_name": "gpt4o-mini",
        "model": get_gpt4omini_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4omini", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4omini", costs_content)
    },
    "llm_gpt4turbo": {
        "model_name": "gpt4-turbo",
        "model": get_gpt4turbo_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4turbo", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4turbo", costs_content)
    },
    # ----- Mixtral -----
    "llm_mixtral_8x22b": {
        "model_name": "Mixtral-8x22B",
        "model": get_mixtral_8x22b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    # ----- LLaMas -----
    "llm_llama3_8b": {
        "model_name": "llama3_8b",
        "model": get_llama3_8b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    "llm_llama3_70b": {
        "model_name": "llama3_70b",
        "model": get_llama3_70b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    "llm_llama3_1_8b": {
        "model_name": "llama3_1_8b",
        "model": get_llama3_1_8b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    # ----- GEMMA -----
    "llm_llm_gemma2_2b": {
        "model_name": "gemma-2-2b",
        "model": get_gemma2_2b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    "llm_gemma2_9b": {
        "model_name": "gemma-2-9b",
        "model": get_gemma2_9b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    },
    # ===== Biomedical models =====
    "llm_biomistral_7b": {
        "model_name": "BioMistral-7B",
        "model": get_biomistral_7b(),
        "price_per_input_token": 0,
        "price_per_output_token": 0
    }
}