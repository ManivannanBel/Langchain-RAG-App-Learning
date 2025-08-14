#!/usr/bin/env python3
"""
Test script for the Website RAG Chat Application.

This script tests various functionality of the RAG application
to ensure it works correctly with different configurations.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, validate_config, TestingConfig
from app import WebsiteRAGChatApp

# Load environment variables
load_dotenv()

def test_configuration():
    """Test configuration validation"""
    print("🧪 Testing Configuration...")
    
    # Test default configuration
    config = get_config()
    errors = validate_config(config)
    assert len(errors) == 0, f"Default config has errors: {errors}"
    print("✅ Default configuration is valid")
    
    # Test testing configuration
    test_config = get_config("testing")
    errors = validate_config(test_config)
    assert len(errors) == 0, f"Testing config has errors: {errors}"
    print("✅ Testing configuration is valid")
    
    # Test with missing API key
    original_key = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    errors = validate_config(config)
    assert len(errors) > 0, "Should have errors when API key is missing"
    print("✅ Configuration validation works correctly")
    
    # Restore API key
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key

def test_app_initialization():
    """Test app initialization with different configs"""
    print("\n🧪 Testing App Initialization...")
    
    try:
        # Test with default config
        app = WebsiteRAGChatApp()
        print("✅ App initialized with default config")
        
        # Test with testing config
        app = WebsiteRAGChatApp(get_config("testing"))
        print("✅ App initialized with testing config")
        
        # Test with custom config
        custom_config = TestingConfig()
        custom_config.CHUNK_SIZE = 500
        app = WebsiteRAGChatApp(custom_config)
        print("✅ App initialized with custom config")
        
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        return False
    
    return True

def test_website_fetching():
    """Test website content fetching"""
    print("\n🧪 Testing Website Fetching...")
    
    app = WebsiteRAGChatApp(get_config("testing"))
    
    # Test with a simple website
    test_url = "https://httpbin.org/html"
    result = app.fetch_website_content(test_url)
    
    assert result["status"] == "success", f"Failed to fetch {test_url}"
    assert result["url"] == test_url
    assert len(result["content"]) > 0
    print("✅ Website fetching works correctly")
    
    # Test with invalid URL
    invalid_url = "https://invalid-url-that-does-not-exist.com"
    result = app.fetch_website_content(invalid_url)
    
    assert result["status"] == "error", "Should fail for invalid URL"
    print("✅ Error handling works correctly")

def test_document_creation():
    """Test document creation from website data"""
    print("\n🧪 Testing Document Creation...")
    
    app = WebsiteRAGChatApp(get_config("testing"))
    
    # Create mock website data
    websites_data = {
        "https://example.com": {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.",
            "status": "success"
        }
    }
    
    documents = app.create_documents_from_websites(websites_data)
    
    assert len(documents) > 0, "Should create at least one document"
    assert documents[0].metadata["url"] == "https://example.com"
    assert documents[0].metadata["title"] == "Example Domain"
    print("✅ Document creation works correctly")

def test_vectorstore_operations():
    """Test vector store operations"""
    print("\n🧪 Testing Vector Store Operations...")
    
    app = WebsiteRAGChatApp(get_config("testing"))
    
    # Create test documents
    from langchain_core.documents import Document
    test_documents = [
        Document(
            page_content="Python is a programming language",
            metadata={"url": "https://python.org", "title": "Python"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence",
            metadata={"url": "https://ml.org", "title": "Machine Learning"}
        )
    ]
    
    # Setup vector store
    app.setup_vectorstore(test_documents)
    
    # Test similarity search
    results = app.search_similar_content("Python programming")
    assert len(results) > 0, "Should find similar documents"
    print("✅ Vector store operations work correctly")

def test_rag_chain():
    """Test RAG chain creation and execution"""
    print("\n🧪 Testing RAG Chain...")
    
    app = WebsiteRAGChatApp(get_config("testing"))
    
    # Create test documents and setup vector store
    from langchain_core.documents import Document
    test_documents = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"url": "https://python.org", "title": "Python"}
        )
    ]
    app.setup_vectorstore(test_documents)
    
    # Test RAG chain creation
    rag_chain = app.create_rag_chain()
    assert rag_chain is not None, "RAG chain should be created"
    print("✅ RAG chain creation works correctly")
    
    # Note: We can't test the actual execution without an API key
    # but we can test that the chain is properly structured

def test_langgraph_workflow():
    """Test LangGraph workflow creation"""
    print("\n🧪 Testing LangGraph Workflow...")
    
    app = WebsiteRAGChatApp(get_config("testing"))
    
    try:
        workflow = app.create_langgraph_workflow()
        assert workflow is not None, "Workflow should be created"
        print("✅ LangGraph workflow creation works correctly")
    except ValueError as e:
        if "disabled" in str(e):
            print("✅ LangGraph workflow correctly disabled in testing config")
        else:
            print(f"❌ LangGraph workflow error: {e}")

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Website RAG Chat Application Tests")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found. Some tests may be limited.")
        print("Set your OpenAI API key in the .env file for full testing.")
    
    tests = [
        test_configuration,
        test_app_initialization,
        test_website_fetching,
        test_document_creation,
        test_vectorstore_operations,
        test_rag_chain,
        test_langgraph_workflow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 