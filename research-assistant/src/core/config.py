"""
Configuration management for the research assistant.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from .exceptions import ConfigurationError

# Load .env file from the project root (parent directory of src)
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / '.env'
print('Env file path:', env_path)
print('Env file exists:', env_path.exists())

# Load the .env file directly
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print('Loaded SERPER_API_KEY:', os.getenv('SERPER_API_KEY'))


class Config:
    """Configuration class for the research assistant"""
    
    # API Keys
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    
    # Models
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # System Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    MAX_DOCUMENTS: int = int(os.getenv("MAX_DOCUMENTS", "1000"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Search Settings
    DEFAULT_DOC_RESULTS: int = 5
    DEFAULT_WEB_RESULTS: int = 5
    HYBRID_ALPHA: float = 0.5  # Weight for dense vs sparse retrieval
    
    # Quality Thresholds
    MIN_CREDIBILITY_SCORE: float = 0.3
    MIN_RELEVANCE_SCORE: float = 0.2
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "research_documents")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration and raise ConfigurationError if invalid"""
        errors = []
        
        if not cls.SERPER_API_KEY:
            errors.append("SERPER_API_KEY is required")
        
        if cls.CACHE_TTL < 0:
            errors.append("CACHE_TTL must be non-negative")
        
        if not 0 <= cls.HYBRID_ALPHA <= 1:
            errors.append("HYBRID_ALPHA must be between 0 and 1")
        
        if cls.MAX_CHUNK_SIZE <= 0:
            errors.append("MAX_CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")
        
        if cls.CHUNK_OVERLAP >= cls.MAX_CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than MAX_CHUNK_SIZE")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    @classmethod
    def get_serper_api_key(cls) -> str:
        """Get Serper API key with validation"""
        if not cls.SERPER_API_KEY:
            raise ConfigurationError("SERPER_API_KEY is not configured")
        return cls.SERPER_API_KEY
    
    @classmethod
    def get_chroma_settings(cls) -> dict:
        """Get ChromaDB settings"""
        return {
            'persist_directory': cls.CHROMA_PERSIST_DIRECTORY,
            'collection_name': cls.CHROMA_COLLECTION_NAME
        } 