import re

def extract_price(variable_name):
    with open('costs.txt', 'r') as file:
        costs_content = file.read()
    pattern = rf'{variable_name}\s*=\s*(\d+(?:\.\d+)?)/\(1e6\)'
    match = re.search(pattern, costs_content)
    return float(match.group(1)) / 1e6 if match else None

# LLM-specific configurations
LLM_CONFIGS = {
    "llm_gpt3": {
        "model_name": "gpt-3.5-turbo",
        "price_per_input_token": extract_price("PRICE_PER_INPUT_TOKEN_GPT3"),
        "price_per_output_token": extract_price("PRICE_PER_OUTPUT_TOKEN_GPT3")
    },
    # Add other LLM configurations here
}