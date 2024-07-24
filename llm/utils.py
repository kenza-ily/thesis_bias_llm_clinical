from llm.models import get_gpt3_model #!TODO COMPLETE
from config.llm_config import llms

def get_llms():
    return {
        "llm_gpt3": {
            "model": get_gpt3_model(),
            **llms["llm_gpt3"]
        },
        # Add other LLMs here
    }

# Add other utility functions here