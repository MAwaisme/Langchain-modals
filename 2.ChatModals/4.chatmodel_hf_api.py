# import os

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# huggingfacehub_api_token = os.getenv("HF_TOKEN")

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     huggingfacehub_api_token=huggingfacehub_api_token
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("Who created the first LLM model?")

# print(result.content)

import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

huggingfacehub_api_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
    huggingfacehub_api_token=huggingfacehub_api_token,
)

model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]
# result = model.invoke("Who created the first LLM model?")
while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    # if user_input.lower() in ["exit", "quit"]:
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)