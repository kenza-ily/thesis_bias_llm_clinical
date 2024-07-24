from llm.models import get_gpt3_model
from langchain_core.messages import HumanMessage

def test_llm():
    print("Initializing LLM...")
    llm = get_gpt3_model()
    
    print("Sending a test message to the LLM...")
    response = llm.invoke([HumanMessage(content="Hello, can you hear me?")])
    
    print("LLM Response:")
    print(response.content)

if __name__ == "__main__":
    test_llm()