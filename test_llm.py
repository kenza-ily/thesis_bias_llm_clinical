from llm.utils import get_llms

def main():
    llms = get_llms()
    gpt3_model = llms["llm_gpt3"]["model"]
    
    # Test the model
    response = gpt3_model.invoke("Hello, how are you?")
    print(response.content)

if __name__ == "__main__":
    main()