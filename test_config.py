from config.settings import *
from config.llm_config import llms

print("API Keys loaded:")
print(f"Azure OpenAI API Key: {'*' * len(AZURE_OPENAI_API_KEY)}")
print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")

print("\nLLM Configurations:")
for llm_name, llm_config in llms.items():
    print(f"{llm_name}: {llm_config['model_name']}")