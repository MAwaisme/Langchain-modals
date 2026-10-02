import ast

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load the existing HumanMessage(...) and AIMessage(...) records.
chat_history = []
with open("chat_histroy.txt") as f:
    chat_history.extend(f.readlines())
    print(chat_history)
# with open("chat_histroy.txt", encoding="utf-8") as history_file:
#     for line in history_file:
#         message_type, separator, content_repr = line.strip().partition("(Content=")
#         if not separator or not content_repr.endswith(")"):
#             continue

#         role = {"HumanMessage": "human", "AIMessage": "ai"}.get(message_type)
#         if role is not None:
#             chat_history.append((role, ast.literal_eval(content_repr[:-1])))

# create prompt
# prompt = chat_template.invoke({
#     "chat_history": chat_history,
#     "query": "Can you tell me the status of my refund?",
# })
# print(prompt)


prompt = chat_template.invoke({"chat_history": chat_history, "query": "Can you tell me the status of my refund?"})
print(prompt)