# Simple Chatbot

A simple chatbot application built with LangChain and LangGraph that demonstrates multi-user conversation management with memory persistence.

## Overview

This chatbot uses OpenAI's GPT-4o-mini model to handle conversations while maintaining separate conversation threads for different users. It showcases the power of LangGraph for building conversational AI applications with state management.

## Features

- **Multi-user Support**: Handles conversations for multiple users simultaneously
- **Memory Persistence**: Uses MemorySaver to maintain conversation history
- **State Management**: Leverages LangGraph's StateGraph for conversation flow
- **OpenAI Integration**: Powered by GPT-4o-mini model
- **Thread-based Isolation**: Each user has their own conversation thread

## app.py Description

The `app.py` file contains a complete chatbot implementation with the following key components:

### Core Components

1. **Model Initialization**: Sets up GPT-4o-mini model from OpenAI
2. **State Management**: Uses MessagesState to track conversation history
3. **Graph Workflow**: Implements a simple StateGraph with a single model node
4. **Memory System**: Integrates MemorySaver for persistent conversation storage

### Workflow Structure

```
┌─────────┐    ┌─────────────┐    ┌─────────┐
│  START  │───▶│  call_model │───▶│  END    │
└─────────┘    └─────────────┘    └─────────┘
               │
               ▼
          call_model()
          - Processes messages
          - Returns AI response
          - Maintains conversation state
```

### Key Functions

- `call_model()`: Processes incoming messages and generates AI responses
- `workflow.compile()`: Compiles the graph with memory checkpointing
- `app.invoke()`: Handles user interactions with thread-specific configurations

### Usage Example

The application demonstrates multi-user conversation handling:

```python
# User 1 conversation
output = app.invoke({"messages": [HumanMessage("Hi! I'm Bob")]}, config=config1)

# User 2 conversation (separate thread)
output = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=config2)
```

## Requirements

- Python 3.8+
- OpenAI API key (set in .env file)
- Required packages: langchain, langgraph, python-dotenv

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with your OpenAI API key
3. Run the application: `python app.py`

## Architecture

This implementation showcases a minimal but functional chatbot that can be extended with additional nodes for more complex conversation flows, such as:
- Tool calling
- Conditional routing
- Multi-step reasoning
- External API integrations
