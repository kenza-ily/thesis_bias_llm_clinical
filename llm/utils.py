from llm.models import get_gpt3_model , get_gpt4o_model
from llm.llm_config import llms

def get_llms():
    return {
        # GPT-3
        "llm_gpt3": {
            "model": get_gpt3_model(),
            **llms["llm_gpt3"]
        },
        # GPT4o
        "llm_gpt4o": {
            "model": get_gpt4o_model(),
            **llms["llm_gpt4o"]
        }
        # Add more LLMs here if needed
    }

# Add other utility functions here if needed