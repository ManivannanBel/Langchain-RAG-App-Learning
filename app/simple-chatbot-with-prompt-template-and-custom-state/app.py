# =============================================================================
# Simple Chatbot with Prompt Template and Custom State
# =============================================================================
# This application demonstrates a more advanced chatbot implementation using:
# - Custom state management with TypedDict
# - Prompt templates for dynamic system messages
# - Language-specific responses
# - Enhanced conversation flow with predefined messages
# =============================================================================

# Importing required libraries
from dotenv import load_dotenv  # For loading environment variables
from langchain.chat_models import init_chat_model  # For initializing chat models
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # Message types
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt templates
from langgraph.checkpoint.memory import MemorySaver  # For conversation memory
from langgraph.graph import START, MessagesState, StateGraph  # Graph components
from langgraph.graph.message import add_messages  # Message handling utilities
from typing_extensions import TypedDict, Annotated  # Type hints for custom state
from typing import Sequence  # Type hints for sequences

# Load environment variables from .env file
# This loads API keys and other configuration
load_dotenv()

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
# Initialize the OpenAI GPT-4o-mini model for chat interactions
# This model will be used to generate responses based on the conversation context
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# =============================================================================
# CUSTOM STATE DEFINITION
# =============================================================================
# Define a custom state class that extends the basic MessagesState
# This allows us to add additional fields beyond just messages
class CustomState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    language: str  # Language preference for responses

# =============================================================================
# PROMPT TEMPLATE SETUP
# =============================================================================
# Create a dynamic prompt template that includes:
# 1. System message with language parameter
# 2. Placeholder for conversation history
# This allows the assistant to respond in the specified language
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),  # Placeholder for conversation history
    ]
)

# =============================================================================
# MODEL CALLING FUNCTION
# =============================================================================
# This function processes the current state and generates a response
# It combines the prompt template with the model to create contextual responses
def call_model(state: CustomState):
    # Create the prompt using the template and current state
    prompt = prompt_template.invoke(state)
    # Generate response using the model
    response = model.invoke(prompt)
    # Return the response wrapped in the expected format
    return {'messages': [response]}

# =============================================================================
# WORKFLOW GRAPH SETUP
# =============================================================================
# Create the conversation workflow using StateGraph
# This defines how messages flow through the system
workflow = StateGraph(state_schema=CustomState)  # Use our custom state schema
workflow.add_node("call_model", call_model)  # Add the model processing node
workflow.add_edge(START, "call_model")  # Connect start to the model node

# =============================================================================
# MEMORY AND APP COMPILATION
# =============================================================================
# Set up memory saver for persistent conversation history
# This allows the chatbot to remember previous interactions
memory = MemorySaver()
# Compile the workflow with memory checkpointing
# This creates the final executable application
app = workflow.compile(checkpointer=memory)

# =============================================================================
# CONVERSATION CONFIGURATION
# =============================================================================
# Configure the conversation thread for user identification
# This ensures conversation continuity for the specific user
config = {"configurable": {"thread_id": "user1"}}

# =============================================================================
# CONVERSATION EXECUTION
# =============================================================================
# Invoke the chatbot with initial conversation state
# This includes:
# - Predefined conversation history (Bob's introduction and initial exchange)
# - Language preference (Tamil)
# - User configuration for thread management
output = app.invoke(
    {
        "messages": [
            HumanMessage("Hi! I'm Bob"),  # User introduces themselves
            AIMessage("Hello Bob! How can I assist you today?"),  # AI's initial response
            HumanMessage("What's my name?")  # User asks for their name
        ],
        "language": "Tamil"  # Specify language for responses
    },
    config=config  # Use the configured thread
)

# =============================================================================
# OUTPUT DISPLAY
# =============================================================================
# Print all messages in the conversation for review
# This shows the complete conversation flow including the new response
for message in output["messages"]:
    print(message.pretty_print())






