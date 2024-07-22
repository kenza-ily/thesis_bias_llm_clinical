import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Global settings
TEMPERATURE = 0

# API keys and versions
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3 = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3')

# Add other global settings as needed