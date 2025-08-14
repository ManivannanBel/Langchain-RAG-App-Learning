# Website RAG Chat Application

A powerful RAG (Retrieval-Augmented Generation) based chat application built with LangChain and LangGraph that can fetch multiple websites, index them in a vector database, and answer questions based on the indexed content.

## Features

- 🌐 **Multi-website crawling**: Fetch and process content from multiple websites
- 📚 **Vector indexing**: Store website content in Chroma vector database for efficient retrieval
- 🤖 **RAG-powered chat**: Answer questions based on indexed website content
- 🔄 **LangGraph workflow**: Structured conversation flow with state management
- 📝 **Text processing**: Smart text splitting and cleaning for optimal retrieval
- ➕ **Dynamic website addition**: Add new websites to the knowledge base on-the-fly

## Architecture

The application uses a sophisticated architecture combining:

1. **Website Crawler**: Fetches and parses website content using BeautifulSoup
2. **Text Processing**: Splits content into chunks using RecursiveCharacterTextSplitter
3. **Vector Database**: Stores embeddings in Chroma for similarity search
4. **RAG Chain**: Combines retrieval and generation for accurate answers
5. **LangGraph Workflow**: Manages conversation state and flow

## Setup

### Prerequisites

- Python 3.13+
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pipenv install
```

2. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the application:
```bash
python app/agents/website-read-rag/app.py
```

## Usage

### Starting the Application

The application starts with example websites (python.org, langchain.com, openai.com) and provides an interactive chat interface.

### Commands

- **Chat**: Simply type your question to get answers based on indexed websites
- **Add website**: Type `add <url>` to add a new website to the knowledge base
- **Quit**: Type `quit` to exit the application

### Example Usage

```
🌐 Website RAG Chat Application
==================================================
Initializing with example websites...
Fetching content from: https://python.org
Fetching content from: https://langchain.com
Fetching content from: https://openai.com
Indexed 45 document chunks in vector store

💬 Chat with your RAG system!
Commands:
- Type 'add <url>' to add a new website
- Type 'quit' to exit
- Type your question to chat
--------------------------------------------------

You: What is Python?

🤖 Assistant: Based on the context from python.org, Python is a programming language that emphasizes code readability with its notable use of significant whitespace. It features a dynamic type system and automatic memory management, supporting multiple programming paradigms including structured, object-oriented, and functional programming.

You: add https://pytorch.org

✅ Successfully added https://pytorch.org

You: What is PyTorch?

🤖 Assistant: Based on the context from pytorch.org, PyTorch is an open source machine learning framework that accelerates the path from research prototyping to production deployment. It provides a comprehensive ecosystem for machine learning development including tools for computer vision, natural language processing, and more.
```

## Key Components

### WebsiteRAGChatApp Class

The main class that orchestrates the entire RAG system:

- `fetch_website_content()`: Crawls and parses website content
- `process_websites()`: Handles multiple websites
- `create_documents_from_websites()`: Converts website data to documents
- `setup_vectorstore()`: Initializes Chroma vector database
- `search_similar_content()`: Performs similarity search
- `create_rag_chain()`: Builds the RAG pipeline
- `chat()`: Main chat interface

### LangGraph Workflow

The application includes a LangGraph workflow with the following nodes:

1. **process_websites_node**: Handles website processing
2. **retrieve_context_node**: Retrieves relevant context
3. **generate_answer_node**: Generates answers using RAG
4. **update_messages_node**: Updates conversation history

### RAG Chain

The RAG chain combines:
- Context retrieval from vector database
- Prompt template for structured responses
- LLM (GPT-3.5-turbo) for answer generation

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for LLM and embeddings

### Model Configuration

- **LLM**: GPT-3.5-turbo with temperature 0.1
- **Embeddings**: OpenAI text-embedding-ada-002
- **Text Splitter**: RecursiveCharacterTextSplitter with 1000 chunk size, 200 overlap

### Vector Database

- **Database**: Chroma
- **Collection**: "website_rag"
- **Similarity Search**: Top 5 most relevant chunks

## Advanced Features

### Custom Prompt Template

The application uses a custom prompt template that:
- Provides context from websites
- Asks for source citations
- Handles cases where context is insufficient

### Error Handling

- Graceful handling of website fetch failures
- Timeout protection for web requests
- User-friendly error messages

### Text Processing

- Removes script and style elements
- Cleans whitespace and formatting
- Extracts meaningful text content
- Splits into optimal chunks for retrieval

## Extending the Application

### Adding New Vector Stores

You can easily switch to other vector stores by modifying the `setup_vectorstore()` method:

```python
# For Pinecone
from langchain_community.vectorstores import Pinecone
self.vectorstore = Pinecone.from_documents(...)

# For Weaviate
from langchain_community.vectorstores import Weaviate
self.vectorstore = Weaviate.from_documents(...)
```

### Adding New LLMs

Replace the ChatOpenAI instance with other LLM providers:

```python
# For Anthropic Claude
from langchain_anthropic import ChatAnthropic
self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# For local models
from langchain_community.llms import Ollama
self.llm = Ollama(model="llama2")
```

### Custom Text Processing

Modify the `fetch_website_content()` method to add custom text processing:

```python
# Add custom cleaning
text = custom_text_cleaner(text)

# Add metadata extraction
metadata = extract_metadata(soup)
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Ensure your API key is set in the `.env` file
2. **Website Fetch Failures**: Some websites may block automated requests
3. **Memory Issues**: Large websites may consume significant memory during processing

### Performance Tips

- Use smaller chunk sizes for faster processing
- Implement caching for frequently accessed websites
- Consider using async processing for multiple websites

## License

This project is part of the LangChain RAG App Learning repository. 