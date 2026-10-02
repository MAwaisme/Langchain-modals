import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

chat_messages = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} assistant."),
    ("user", "Explain {topic} in the market.")
])

# prompt = chat_messages.invoke({"domain": "crypto", "topic": "trends"})
# print(prompt)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)
model = ChatHuggingFace(llm=llm)
response = (chat_messages | model).invoke({
    "domain": "crypto",
    "topic": "trends",
})
print(response.content)



# Here is using the deepseek-ai/DeepSeek-V3-0324 model from Hugging Face for chat-based interactions. The code sets up a chat history and allows the user to input messages, which are then processed by the model to generate responses. The conversation continues until the user types 'exit'.
# import os

# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load_dotenv()

# chat_messages = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful {domain} assistant."),
#     ("user", "Explain {topic} in the market."),
# ])

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V3-0324",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )
# model = ChatHuggingFace(llm=llm)

# chain = chat_messages | model
# response = chain.invoke({"domain": "crypto", "topic": "trends"})
# print(response.content)