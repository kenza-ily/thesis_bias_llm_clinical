from config.settings import TEMPERATURE
#---- 1/ OpenAI models ----

from langchain_openai import AzureChatOpenAI
from config.settings import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o
# from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4omini
from config.settings import AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4turbo
# -------

def get_gpt3_model():
    print(f"API Version: {AZURE_OPENAI_API_VERSION}")
    print(f"Deployment: {AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3}")
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3,
        temperature=TEMPERATURE
    )
    
# def get_gpt3_model():
#     return AzureChatOpenAI(
#         openai_api_version=AZURE_OPENAI_API_VERSION,
#         azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT3,
#         temperature=TEMPERATURE
#     )
    
def get_gpt4o_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4o,
        temperature=TEMPERATURE
    )
    
# def get_gpt4omini_model():
#     return AzureChatOpenAI(
#         openai_api_version=AZURE_OPENAI_API_VERSION,
#         azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4omini,
#         temperature=TEMPERATURE
#     )
    
def get_gpt4turbo_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4turbo,
        temperature=TEMPERATURE
    )

#---- 2/ Open-source models ----

from langchain_community.chat_models import ChatOllama

# Function to load a HuggingFace model
def load_ollama_model(model_name):
    return ChatOllama(
    model=model_name,
    temperature=TEMPERATURE
)
# -------

# ----- Mistral -----
def get_mistral_nemo():
    return load_ollama_model("mistral:nemo")

def get_mixtral_8x22b():
    return load_ollama_model("mixtral:8x22b")

def get_mistral_7b():
    return load_ollama_model("mistral:7b")
# ----- Llama -----
def get_llama3_8b():
    return load_ollama_model("llama3:8b")

def get_llama3_70b():
    return load_ollama_model("llama3:70b")

def get_llama3_1_8b():
    return load_ollama_model("llama3.1")

# ---- Gemma ----
def get_gemma2_2b():
    return load_ollama_model("gemma2:2b")

def get_gemma2_9b():
    return load_ollama_model("gemma2")

# --- 3/Vertex models ---

from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI


project ="gifted-course-432415-m9"
location="europe-west1"

def load_anthropic_model(name):
    return ChatAnthropicVertex(model_name=name, temperature=TEMPERATURE, project=project, location=location)

# ----- Anthropic ----
def get_haiku():
    return load_anthropic_model("claude-3-haiku@20240307")


def get_sonnet3_5():
    return load_anthropic_model("claude-3-5-sonnet@20240620")

# ----- Gemini ----

def load_gemini_model(name):
    return ChatVertexAI(model_name=name, temperature=TEMPERATURE)

def get_gemini_3_5_flash():
    return load_gemini_model("gemini-1.5-flash")

# from langchain_google_vertexai.model_garden import CohereChat
# def load_cohere_model(name):
#     return CohereChat(model_name=name, temperature=TEMPERATURE, project=project, location=location)

# def get_commandr_plus():
#     return load_cohere_model

# def get_commandr_plus():
#     return load_cohere_model


# ==========================
