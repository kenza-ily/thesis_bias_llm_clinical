from langchain_openai import AzureChatOpenAI
from config.settings import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3, TEMPERATURE

def get_gpt3_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3,
        temperature=TEMPERATURE
    )