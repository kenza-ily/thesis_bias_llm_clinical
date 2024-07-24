from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3 = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3')

# LLM hyperparameters
TEMPERATURE = 0