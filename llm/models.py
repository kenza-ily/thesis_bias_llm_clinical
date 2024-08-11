from langchain_openai import AzureChatOpenAI
from config.settings import TEMPERATURE
from config.settings import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4omini
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4turbo

def get_gpt3_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3,
        temperature=TEMPERATURE
    )
    
def get_gpt4o_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o,
        temperature=TEMPERATURE
    )
    
def get_gpt4omini_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4omini,
        temperature=TEMPERATURE
    )
    
def get_gpt4turbo_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4turbo,
        temperature=TEMPERATURE
    )
    