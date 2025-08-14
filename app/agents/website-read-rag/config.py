"""
Configuration settings for the Website RAG Chat Application.

This module contains all the configurable parameters for the RAG application,
making it easy to customize the behavior without modifying the main code.
"""

import os
from typing import Dict, Any, List

class RAGConfig:
    """Configuration class for the RAG application"""
    
    # OpenAI Configuration
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 1000
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL = "text-embedding-ada-002"
    
    # Text Processing Configuration
    CHUNK_SIZE = 1000 # Number of characters in each chunk
    CHUNK_OVERLAP = 200 # Number of characters to overlap between chunks
    MAX_TEXT_LENGTH = 100000  # Maximum text length to process per website
    
    # Vector Store Configuration
    VECTOR_STORE_COLLECTION_NAME = "website_rag"
    SIMILARITY_SEARCH_K = 5  # Number of similar documents to retrieve
    
    # Website Crawling Configuration
    REQUEST_TIMEOUT = 10  # seconds
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    MAX_RETRIES = 3
    
    # RAG Chain Configuration
    RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context from websites.

Context from websites:
{context}

Question: {question}

Please answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Always cite the source URLs when possible.

Answer:"""
    
    # LangGraph Configuration
    ENABLE_LANGGRAPH_WORKFLOW = True
    MEMORY_SAVER_ENABLED = True
    
    # Logging Configuration
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"
    
    # Performance Configuration
    BATCH_SIZE = 5  # Number of websites to process in parallel
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # Cache TTL in seconds
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    
    @classmethod
    def get_embeddings_config(cls) -> Dict[str, Any]:
        """Get embeddings configuration"""
        return {
            "model": cls.EMBEDDINGS_MODEL,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    
    @classmethod
    def get_text_splitter_config(cls) -> Dict[str, Any]:
        """Get text splitter configuration"""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "length_function": len,
        }
    
    @classmethod
    def get_vectorstore_config(cls) -> Dict[str, Any]:
        """Get vector store configuration"""
        return {
            "collection_name": cls.VECTOR_STORE_COLLECTION_NAME,
            "similarity_search_k": cls.SIMILARITY_SEARCH_K,
        }
    
    @classmethod
    def get_crawler_config(cls) -> Dict[str, Any]:
        """Get website crawler configuration"""
        return {
            "timeout": cls.REQUEST_TIMEOUT,
            "user_agent": cls.USER_AGENT,
            "max_retries": cls.MAX_RETRIES,
            "max_text_length": cls.MAX_TEXT_LENGTH,
        }
    
    @classmethod
    def get_rag_chain_config(cls) -> Dict[str, Any]:
        """Get RAG chain configuration"""
        return {
            "prompt_template": cls.RAG_PROMPT_TEMPLATE,
            "enable_langgraph": cls.ENABLE_LANGGRAPH_WORKFLOW,
        }

class DevelopmentConfig(RAGConfig):
    """Development-specific configuration"""
    ENABLE_LOGGING = True
    LOG_LEVEL = "DEBUG"
    CACHE_ENABLED = False

class ProductionConfig(RAGConfig):
    """Production-specific configuration"""
    ENABLE_LOGGING = True
    LOG_LEVEL = "WARNING"
    CACHE_ENABLED = True
    BATCH_SIZE = 10
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

class TestingConfig(RAGConfig):
    """Testing-specific configuration"""
    ENABLE_LOGGING = False
    CACHE_ENABLED = False
    BATCH_SIZE = 1
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

# Configuration presets
CONFIG_PRESETS = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": RAGConfig,
}

def get_config(environment: str = "default") -> RAGConfig:
    """Get configuration for the specified environment"""
    return CONFIG_PRESETS.get(environment, RAGConfig)

def validate_config(config: RAGConfig) -> List[str]:
    """Validate configuration and return list of errors"""
    errors = []
    
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable is required")
    
    # Validate numeric parameters
    if config.CHUNK_SIZE <= 0:
        errors.append("CHUNK_SIZE must be positive")
    
    if config.CHUNK_OVERLAP < 0:
        errors.append("CHUNK_OVERLAP must be non-negative")
    
    if config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if config.SIMILARITY_SEARCH_K <= 0:
        errors.append("SIMILARITY_SEARCH_K must be positive")
    
    if config.REQUEST_TIMEOUT <= 0:
        errors.append("REQUEST_TIMEOUT must be positive")
    
    if config.BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    return errors

def print_config_summary(config: RAGConfig):
    """Print a summary of the current configuration"""
    print("🔧 RAG Application Configuration Summary")
    print("=" * 50)
    print(f"OpenAI Model: {config.OPENAI_MODEL}")
    print(f"Temperature: {config.OPENAI_TEMPERATURE}")
    print(f"Chunk Size: {config.CHUNK_SIZE}")
    print(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
    print(f"Similarity Search K: {config.SIMILARITY_SEARCH_K}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Cache Enabled: {config.CACHE_ENABLED}")
    print(f"Logging Enabled: {config.ENABLE_LOGGING}")
    print(f"LangGraph Workflow: {config.ENABLE_LANGGRAPH_WORKFLOW}")
    print("=" * 50) 