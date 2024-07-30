from langchain_openai import AzureChatOpenAI
from config.settings import TEMPERATURE
from config.settings import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o

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