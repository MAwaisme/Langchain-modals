import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who created the first LLM model?"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
