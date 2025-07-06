"""
ChromaDB document store for vector search.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface
from langchain_chroma import Chroma  # pip install -U langchain-chroma

from ..core.models import SearchResult
from ..core.config import Config
from ..core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class ChromaDocumentStore:
    """ChromaDB-based document store for vector search"""
    
    def __init__(self, embedding_model: str| None = None):
        """Initialize ChromaDB document store"""
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.MAX_CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Initialize ChromaDB
        chroma_settings = Config.get_chroma_settings()
        self.persist_directory = chroma_settings['persist_directory']
        self.collection_name = chroma_settings['collection_name']
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
    
    def load_and_process_pdf(self, pdf_path: str) -> List[Document]:
        """Load and process PDF document"""
        try:
            if not os.path.exists(pdf_path):
                raise DocumentProcessingError(f"PDF file not found: {pdf_path}")
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, doc in enumerate(splits):
                doc.metadata.update({
                    'source_file': pdf_path,
                    'chunk_id': i,
                    'processed_at': datetime.now().isoformat(),
                    'file_name': os.path.basename(pdf_path)
                })
            
            logger.info(f"Processed {len(splits)} chunks from {pdf_path}")
            return splits
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise DocumentProcessingError(f"Failed to process PDF {pdf_path}: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise DocumentProcessingError(f"Failed to add documents: {str(e)}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search documents using vector similarity"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'error': str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False 