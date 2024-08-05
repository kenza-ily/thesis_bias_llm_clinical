from llm.models import get_gpt3_model , get_gpt4o_model, get_gpt4omini_model, get_gpt4turbo_model
import re

def extract_price(variable_name, costs_content):
    pattern = rf'{variable_name}\s*=\s*(\d+(?:\.\d+)?)/\(1e6\)'
    match = re.search(pattern, costs_content)
    return float(match.group(1)) / 1e6 if match else None

with open('data/costs.txt', 'r') as file:
    costs_content = file.read()

llms = {
    "llm_gpt3": {
        "model_name": "gpt-3.5-turbo",
        "model": get_gpt3_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT3", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT3", costs_content)
    },
    "llm_gpt4o": {
        "model_name": "gpt-4.0-turbo",
        "model": get_gpt4o_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4o", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4o", costs_content)
    },
    "llm_gpt4omini": {
        "model_name": "gpt-4omini-turbo",
        "model": get_gpt4omini_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4omini", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4omini", costs_content)
    },
    "llm_gpt4turbo": {
        "model_name": "gpt-4-turbo",
        "model": get_gpt4turbo_model(),
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT4turbo", costs_content),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT4turbo", costs_content)
    }  
    # Add other LLM configurations here
}