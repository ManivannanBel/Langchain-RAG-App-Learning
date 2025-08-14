#!/usr/bin/env python3
"""
Example usage of the WebsiteRAGChatApp class.

This script demonstrates how to use the RAG application programmatically
for integration into other applications or for automated processing.
"""

import os
from dotenv import load_dotenv
from app import WebsiteRAGChatApp

# Load environment variables
load_dotenv()

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize the RAG app
    app = WebsiteRAGChatApp()
    
    # Define websites to index
    websites = [
        "https://python.org",
        "https://langchain.com"
    ]
    
    # Add websites to the knowledge base
    print("Adding websites to knowledge base...")
    results = app.add_websites(websites)
    
    # Print results
    for url, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        print(f"{status} {url}: {result.get('title', 'N/A')}")
    
    # Ask questions
    questions = [
        "What is Python?",
        "What is LangChain?",
        "How does Python compare to other programming languages?"
    ]
    
    print("\n=== Asking Questions ===")
    for question in questions:
        print(f"\nQ: {question}")
        answer = app.chat(question)
        print(f"A: {answer}")

def example_advanced_usage():
    """Advanced usage with custom processing"""
    print("\n=== Advanced Usage Example ===")
    
    app = WebsiteRAGChatApp()
    
    # Add a specific website
    custom_websites = ["https://pytorch.org"]
    app.add_websites(custom_websites)
    
    # Search for specific content
    query = "What is PyTorch and what are its main features?"
    print(f"\nSearching for: {query}")
    
    # Get similar documents
    similar_docs = app.search_similar_content(query, k=3)
    print(f"\nFound {len(similar_docs)} similar documents:")
    
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. Source: {doc.metadata.get('url', 'Unknown')}")
        print(f"   Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"   Content preview: {doc.page_content[:200]}...")
    
    # Get answer using RAG
    answer = app.chat(query)
    print(f"\nRAG Answer: {answer}")

def example_langgraph_workflow():
    """Example using the LangGraph workflow"""
    print("\n=== LangGraph Workflow Example ===")
    
    app = WebsiteRAGChatApp()
    
    # Create the workflow
    workflow = app.create_langgraph_workflow()
    
    # Initialize state
    initial_state = {
        "messages": [],
        "current_question": "What is machine learning?",
        "context": [],
        "answer": "",
        "websites_processed": False,
        "websites_to_process": ["https://scikit-learn.org"]
    }
    
    # Run the workflow
    print("Running LangGraph workflow...")
    result = workflow.invoke(initial_state)
    
    print(f"Final answer: {result['answer']}")
    print(f"Number of messages: {len(result['messages'])}")

def example_batch_processing():
    """Example of batch processing multiple websites"""
    print("\n=== Batch Processing Example ===")
    
    app = WebsiteRAGChatApp()
    
    # Large list of websites
    websites = [
        "https://python.org",
        "https://numpy.org",
        "https://pandas.pydata.org",
        "https://matplotlib.org",
        "https://scikit-learn.org"
    ]
    
    print(f"Processing {len(websites)} websites...")
    results = app.process_websites(websites)
    
    # Analyze results
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = len(results) - successful
    
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed to process: {failed}")
    
    # Create documents and index
    documents = app.create_documents_from_websites(results)
    app.setup_vectorstore(documents)
    
    print(f"📚 Indexed {len(documents)} document chunks")
    
    # Test retrieval
    test_queries = [
        "What is NumPy?",
        "How to use pandas?",
        "What is machine learning?"
    ]
    
    for query in test_queries:
        answer = app.chat(query)
        print(f"\nQ: {query}")
        print(f"A: {answer[:200]}...")

def main():
    """Run all examples"""
    print("🚀 Website RAG Chat Application Examples")
    print("=" * 50)
    
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Please set your OpenAI API key in the .env file")
            return
        
        # Run examples
        example_basic_usage()
        example_advanced_usage()
        example_langgraph_workflow()
        example_batch_processing()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Installed all dependencies with 'pipenv install'")
        print("3. Have an active internet connection")

if __name__ == "__main__":
    main() 