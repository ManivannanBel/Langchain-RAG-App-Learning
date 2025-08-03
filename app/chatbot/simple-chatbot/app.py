### Simple Chatbot

# This is a simple chatbot that uses the LangChain and LangGraph libraries to create a chatbot.
# It uses the GPT-4o-mini model from OpenAI and the MemorySaver to store the chat history.
# It uses the StateGraph to create a chatbot that can handle multiple users.

# Importing libraries
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

# Initialize the model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': response}

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("call_model", call_model)
workflow.add_edge(START, "call_model")

# Graph Visualization:
# ┌─────────┐    ┌─────────────┐    ┌─────────┐
# │  START  │───▶│  call_model │───▶│  END    │
# └─────────┘    └─────────────┘    └─────────┘
#                │
#                ▼
#           call_model()
#           - Processes messages
#           - Returns AI response
#           - Maintains conversation state


# Add memory
memory = MemorySaver()
# Compile the graph with the memory
app = workflow.compile(checkpointer=memory)

# Configs
# User 1
config1 = {"configurable": {"thread_id": "user1"}}
# User 2
config2 = {"configurable": {"thread_id": "user2"}}

# User 1
output = app.invoke({"messages": [HumanMessage("Hi! I'm Bob")]}, config=config1)
print("User 1: "+output["messages"][-1].pretty_print())

# User 2
output = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=config2)
print("User 2: "+output["messages"][-1].pretty_print())

# User 1
output = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=config1)
print("User 1: "+output["messages"][-1].pretty_print())

