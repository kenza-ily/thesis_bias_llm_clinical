from config.settings import TEMPERATURE
#---- 1/ OpenAI models ----

from langchain_openai import AzureChatOpenAI
from config.settings import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4omini
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4turbo
# -------

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

#---- 2/ Open-source models ----

from langchain_huggingface import HuggingFaceEndpoint
from config.settings import HF_TOKEN
# -------

# Function to load a HuggingFace model
def load_huggingface_model(model_name):
    return HuggingFaceEndpoint(
    repo_id=model_name,
    temperature=TEMPERATURE,
    huggingfacehub_api_token=HF_TOKEN,
)
# -------

def get_mixtral_8x22b():
    return load_huggingface_model("Mixtral-8x22B")

def get_llama3_8b():
    return load_huggingface_model("meta-llama/Meta-Llama-3-8B")

def get_llama3_70b():
    return load_huggingface_model("meta-llama/Meta-Llama-3-70B")

def get_llama3_1_8b():
    return load_huggingface_model("meta-llama/Meta-Llama-3.1-8B")

def get_gemma2_2b():
    return load_huggingface_model("google/gemma-2-2b")

def get_gemma2_9b():
    return load_huggingface_model("google/gemma-2-9b")

# - optional
def get_biomistral_7b():
    return load_huggingface_model("BioMistral/BioMistral-7B")