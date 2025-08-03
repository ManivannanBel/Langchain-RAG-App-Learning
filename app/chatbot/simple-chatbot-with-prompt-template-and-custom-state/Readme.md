# Advanced Chatbot with Prompt Templates and Custom State

A sophisticated chatbot implementation that demonstrates advanced LangChain and LangGraph concepts including custom state management, dynamic prompt templates, message trimming, and language-specific responses.

## Concepts Covered

### 1. **Prompt Templates**
- **Dynamic System Messages**: Uses `ChatPromptTemplate` to create dynamic system prompts that can include variables
- **Language Parameterization**: Allows the assistant to respond in different languages based on the `{language}` parameter
- **Message Placeholders**: Uses `MessagesPlaceholder` to inject conversation history into the prompt
- **Template Structure**: Combines system instructions with conversation context for contextual responses

### 2. **Custom State Management**
- **TypedDict Extension**: Extends the basic `MessagesState` with additional fields using `TypedDict`
- **Language Preference**: Adds a `language` field to track user's preferred response language
- **Type Safety**: Uses `Annotated` types with `add_messages` for proper message handling
- **State Persistence**: Maintains both conversation history and user preferences across interactions

### 3. **Message Trimming**
- **Token Management**: Implements `trim_messages` to prevent conversations from exceeding token limits
- **Smart Trimming Strategy**: Uses "last" strategy to keep the most recent messages
- **Token Counting**: Uses the model itself to count tokens accurately
- **Partial Message Handling**: Configures trimming to avoid cutting messages mid-sentence

## app.py Code Structure

### **Core Components**

#### 1. **Model Initialization**
```python
model = init_chat_model("gpt-4o-mini", model_provider="openai")
```
- Initializes OpenAI's GPT-4o-mini model for chat interactions
- Provides the foundation for generating contextual responses

#### 2. **Custom State Definition**
```python
class CustomState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
```
- Defines a custom state schema that extends basic message handling
- Includes conversation history and language preference
- Uses type annotations for better code safety and IDE support

#### 3. **Message Trimming Setup**
```python
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    allow_partial=False,
    start_on="human"
)
```
- Creates a message trimmer to manage conversation length
- Prevents token limit exceeded errors
- Keeps the most recent and relevant conversation context

#### 4. **Prompt Template Configuration**
```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages")
])
```
- Defines a dynamic prompt template with language parameterization
- Includes system instructions and conversation history placeholder
- Enables language-specific responses

#### 5. **Model Processing Function**
```python
def call_model(state: CustomState):
    trimmed_messages = trimmer.invoke(state['messages'])
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {'messages': [response]}
```
- Processes incoming state and generates responses
- Applies message trimming before processing
- Combines prompt template with model invocation
- Returns properly formatted response

#### 6. **Workflow Graph Setup**
```python
workflow = StateGraph(state_schema=CustomState)
workflow.add_node("call_model", call_model)
workflow.add_edge(START, "call_model")
```
- Creates a StateGraph with custom state schema
- Defines the conversation flow from START to model processing
- Establishes the basic workflow structure

#### 7. **Memory and Compilation**
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
- Sets up persistent conversation memory
- Compiles the workflow with memory checkpointing
- Creates the final executable application

### **Workflow Visualization**

```
┌─────────┐    ┌─────────────┐    ┌─────────┐
│  START  │───▶│  call_model │───▶│  END    │
└─────────┘    └─────────────┘    └─────────┘
               │
               ▼
          call_model()
          ├─── Trim messages (1000 tokens)
          ├─── Apply prompt template
          ├─── Generate response
          └─── Return formatted output
```

### **Key Features**

1. **Language Flexibility**: Responds in user-specified languages
2. **Memory Management**: Maintains conversation history with smart trimming
3. **Type Safety**: Uses proper type annotations throughout
4. **Scalable Architecture**: Easy to extend with additional nodes
5. **Token Efficiency**: Prevents token limit issues with message trimming

### **Usage Example**

```python
output = app.invoke({
    "messages": [
        HumanMessage("Hi! I'm Bob"),
        AIMessage("Hello Bob! How can I assist you today?"),
        HumanMessage("What's my name?")
    ],
    "language": "Tamil"
}, config={"configurable": {"thread_id": "user1"}})
```

This demonstrates:
- Predefined conversation history
- Language-specific responses (Tamil)
- Thread-based conversation management
- Complete conversation flow with memory

### **Advanced Concepts Demonstrated**

- **Dynamic Prompting**: System messages that adapt based on state
- **State Extension**: Custom state beyond basic message handling
- **Token Management**: Intelligent conversation length control
- **Memory Persistence**: Long-term conversation memory
- **Type Safety**: Proper type annotations for better development experience