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
from dotenv import load_dotenv

load_dotenv()

huggingfacehub_api_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
    huggingfacehub_api_token=huggingfacehub_api_token,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who created the first LLM model?")

print(result.content)