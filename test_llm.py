from llm.models import get_gpt3_model
from langchain_core.messages import HumanMessage

llm = get_gpt3_model()
response = llm.invoke([HumanMessage(content="Hello, how are you?")])
print(response.content)