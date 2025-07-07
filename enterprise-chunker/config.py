# config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemConfig:
    """Configuration for the Enterprise Knowledge Management System"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-3.5-turbo-instruct"
    
    # ChromaDB Configuration
    chroma_db_path: str = "./enterprise_knowledge_db"
    collection_name: str = "enterprise_docs"
    
    # Processing Configuration
    max_document_size: int = 1000000  # 1MB limit
    batch_size: int = 10
    
    # Chunking Configuration
    default_chunk_size: int = 800
    default_chunk_overlap: int = 150
    
    # Evaluation Configuration
    eval_sample_size: int = 50
    similarity_threshold: float = 0.7
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return True