import os
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPDFLoader as PDFLoader
)
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

# ChromaDB
import chromadb
from chromadb.config import Settings

# Additional imports
import tiktoken
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Document type classifications"""
    TECHNICAL_DOC = "technical_documentation"
    API_REFERENCE = "api_reference"
    SUPPORT_TICKET = "support_ticket"
    POLICY = "policy_document"
    TUTORIAL = "tutorial"
    CODE_SNIPPET = "code_snippet"
    TROUBLESHOOTING = "troubleshooting_guide"
    FAQ = "faq"
    GENERAL = "general"

@dataclass
class ChunkingStrategy:
    """Configuration for chunking strategies"""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    keep_separator: bool = True
    
class DocumentClassifier:
    """Classifies documents using OpenAI GPT-3.5-turbo"""
    
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        self.classification_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            Analyze the following document content and classify it into one of these categories:
            
            Categories:
            1. technical_documentation - Technical specifications, architecture docs, design documents
            2. api_reference - API documentation, endpoint descriptions, parameter lists
            3. support_ticket - Bug reports, customer issues, support conversations
            4. policy_document - Company policies, procedures, compliance documents
            5. tutorial - Step-by-step guides, how-to instructions, learning materials
            6. code_snippet - Code examples, programming snippets, scripts
            7. troubleshooting_guide - Problem-solving guides, diagnostic procedures
            8. faq - Frequently asked questions and answers
            9. general - General documents that don't fit other categories
            
            Document content (first 1000 characters):
            {content}
            
            Classification (respond with only the category name):
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.classification_prompt)
    
    def classify_document(self, content: str) -> DocumentType:
        """Classify a document based on its content"""
        try:
            # Truncate content for classification
            truncated_content = content[:1000]
            
            with get_openai_callback() as cb:
                result = self.chain.run(content=truncated_content)
                logger.info(f"Classification cost: ${cb.total_cost:.4f}")
            
            # Clean and parse result
            classification = result.strip().lower()
            
            # Map to DocumentType enum
            for doc_type in DocumentType:
                if doc_type.value in classification:
                    return doc_type
            
            # Default fallback
            return DocumentType.GENERAL
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return DocumentType.GENERAL

class AdaptiveChunker:
    """Adaptive chunking system based on document type"""
    
    def __init__(self, openai_api_key: str):
        self.classifier = DocumentClassifier(openai_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Define chunking strategies for each document type
        self.strategies = {
            DocumentType.TECHNICAL_DOC: ChunkingStrategy(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            ),
            DocumentType.API_REFERENCE: ChunkingStrategy(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "```", "```python", "```javascript", "```json"]
            ),
            DocumentType.SUPPORT_TICKET: ChunkingStrategy(
                chunk_size=600,
                chunk_overlap=50,
                separators=["\n\n", "\n", "---", "Reply:", "Comment:", "Status:"]
            ),
            DocumentType.POLICY: ChunkingStrategy(
                chunk_size=1200,
                chunk_overlap=300,
                separators=["\n\n", "\n", "Section", "Article", "Clause", ".", " "]
            ),
            DocumentType.TUTORIAL: ChunkingStrategy(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\nStep", "\n1.", "\n2.", "\n3.", "\n", ".", " "]
            ),
            DocumentType.CODE_SNIPPET: ChunkingStrategy(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\ndef ", "\nclass ", "\nif ", "\nfor ", "\n", " "]
            ),
            DocumentType.TROUBLESHOOTING: ChunkingStrategy(
                chunk_size=900,
                chunk_overlap=200,
                separators=["\n\n", "\nProblem:", "\nSolution:", "\nStep", "\n", ".", " "]
            ),
            DocumentType.FAQ: ChunkingStrategy(
                chunk_size=400,
                chunk_overlap=50,
                separators=["\n\n", "\nQ:", "\nA:", "\n", "?", ".", " "]
            ),
            DocumentType.GENERAL: ChunkingStrategy(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        }
    
    def get_text_splitter(self, doc_type: DocumentType) -> RecursiveCharacterTextSplitter:
        """Get appropriate text splitter based on document type"""
        strategy = self.strategies.get(doc_type, self.strategies[DocumentType.GENERAL])
        
        if doc_type == DocumentType.CODE_SNIPPET:
            return PythonCodeTextSplitter(
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap
            )
        elif doc_type == DocumentType.API_REFERENCE:
            return MarkdownTextSplitter(
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap,
                separators=strategy.separators,
                keep_separator=strategy.keep_separator
            )
    
    def chunk_document(self, document: Document) -> Tuple[List[Document], DocumentType]:
        """Chunk a document using adaptive strategy"""
        # Classify document
        doc_type = self.classifier.classify_document(document.page_content)
        logger.info(f"Document classified as: {doc_type.value}")
        
        # Get appropriate text splitter
        text_splitter = self.get_text_splitter(doc_type)
        
        # Split document
        chunks = text_splitter.split_documents([document])
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'document_type': doc_type.value,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunking_strategy': self.strategies[doc_type].__dict__,
                'original_doc_id': document.metadata.get('source', 'unknown')
            })
        
        return chunks, doc_type

class EnterpriseKnowledgeManager:
    """Main knowledge management system"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "enterprise_docs"):
        self.openai_api_key = openai_api_key
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.chunker = AdaptiveChunker(openai_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.metrics = {
            'total_documents': 0,
            'total_chunks': 0,
            'document_types': {},
            'processing_time': 0
        }
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client=self.chroma_client
            )
            logger.info("Vector store initialized successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def load_sample_documents(self) -> List[Document]:
        """Load sample documents for testing"""
        sample_docs = [
            Document(
                page_content="""
                # API Documentation - User Authentication
                
                ## Overview
                The User Authentication API provides secure access to user accounts.
                
                ## Endpoints
                
                ### POST /auth/login
                Authenticate user credentials and return access token.
                
                **Parameters:**
                - username (string, required): User's email address
                - password (string, required): User's password
                
                **Response:**
                ```json
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "Bearer",
                    "expires_in": 3600
                }
                ```
                
                ### GET /auth/user
                Get current user information.
                
                **Headers:**
                - Authorization: Bearer {access_token}
                
                **Response:**
                ```json
                {
                    "id": 123,
                    "username": "user@example.com",
                    "role": "user"
                }
                ```
                """,
                metadata={"source": "auth_api_docs.md", "document_type": "api_documentation"}
            ),
            
            Document(
                page_content="""
                # Company Data Privacy Policy
                
                ## Section 1: Introduction
                This policy outlines our commitment to protecting personal data and privacy rights of our users and employees.
                
                ## Section 2: Data Collection
                We collect information in the following ways:
                
                Article 2.1: Information you provide directly
                - Account registration details
                - Communication preferences
                - Support requests and feedback
                
                Article 2.2: Information collected automatically
                - Usage analytics and performance metrics
                - Device information and IP addresses
                - Cookies and similar technologies
                
                ## Section 3: Data Processing
                We process personal data for legitimate business purposes including:
                - Service delivery and customer support
                - Security and fraud prevention
                - Legal compliance and regulatory requirements
                
                ## Section 4: Data Sharing
                We do not sell personal data to third parties. Limited sharing occurs only for:
                - Service providers under strict agreements
                - Legal requirements and law enforcement
                - Business transfers with appropriate safeguards
                """,
                metadata={"source": "privacy_policy.md", "document_type": "policy"}
            ),
            
            Document(
                page_content="""
                # Troubleshooting Guide: Application Won't Start
                
                ## Problem: Application crashes on startup
                
                This guide helps resolve common startup issues with our enterprise application.
                
                ## Step 1: Check System Requirements
                Verify your system meets minimum requirements:
                - Operating System: Windows 10 or higher, macOS 10.15+, Ubuntu 18.04+
                - Memory: 8GB RAM minimum, 16GB recommended
                - Disk Space: 2GB available space
                
                ## Step 2: Verify Installation
                Check if the application is properly installed:
                1. Navigate to installation directory
                2. Verify all core files are present
                3. Check file permissions
                
                ## Step 3: Review Log Files
                Application logs are located in:
                - Windows: %APPDATA%/CompanyName/AppName/logs/
                - macOS: ~/Library/Application Support/CompanyName/AppName/logs/
                - Linux: ~/.local/share/CompanyName/AppName/logs/
                
                Look for error messages containing:
                - "Failed to initialize"
                - "Permission denied"
                - "Missing dependency"
                
                ## Step 4: Clear Cache and Temporary Files
                Sometimes corrupted cache files cause startup issues:
                1. Close the application completely
                2. Delete cache directory contents
                3. Restart the application
                
                ## Step 5: Reinstall Application
                If previous steps don't resolve the issue:
                1. Uninstall the current version
                2. Download latest version from official website
                3. Install with administrator privileges
                """,
                metadata={"source": "troubleshooting_startup.md", "document_type": "troubleshooting"}
            ),
            
            Document(
                page_content="""
                # Tutorial: Getting Started with Our API
                
                ## Overview
                This tutorial will guide you through making your first API calls and integrating our service into your application.
                
                ## Prerequisites
                - Basic knowledge of HTTP and REST APIs
                - A valid API key (sign up at our developer portal)
                - Python 3.7+ or Node.js 14+
                
                ## Step 1: Setup Your Environment
                First, install the required dependencies:
                
                For Python:
                ```bash
                pip install requests python-dotenv
                ```
                
                For Node.js:
                ```bash
                npm install axios dotenv
                ```
                
                ## Step 2: Configure Authentication
                Create a .env file in your project root:
                ```
                API_KEY=your_api_key_here
                API_BASE_URL=https://api.example.com/v1
                ```
                
                ## Step 3: Make Your First API Call
                Here's how to authenticate and make a basic request:
                
                Python example:
                ```python
                import os
                import requests
                from dotenv import load_dotenv
                
                load_dotenv()
                
                headers = {
                    'Authorization': f'Bearer {os.getenv("API_KEY")}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(
                    f'{os.getenv("API_BASE_URL")}/users/me',
                    headers=headers
                )
                
                print(response.json())
                ```
                
                ## Step 4: Handle API Responses
                Always check the response status and handle errors appropriately:
                
                ```python
                if response.status_code == 200:
                    data = response.json()
                    print(f"Hello, {data['name']}!")
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                ```
                
                ## Step 5: Implement Rate Limiting
                Our API has rate limits. Implement proper retry logic:
                
                ```python
                import time
                
                def make_request_with_retry(url, headers, max_retries=3):
                    for attempt in range(max_retries):
                        response = requests.get(url, headers=headers)
                        if response.status_code == 429:  # Rate limited
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return response
                    return None
                ```
                """,
                metadata={"source": "api_tutorial.md", "document_type": "tutorial"}
            ),
            
            Document(
                page_content="""
                Support Ticket #12345
                
                Status: Resolved
                Priority: High
                Created: 2024-01-15 10:30 AM
                Customer: TechCorp Solutions
                
                Issue Description:
                Customer reports that their API integration is returning 500 errors intermittently. 
                The errors started occurring after the recent system update on January 12th.
                
                Initial Report:
                "We're seeing random 500 Internal Server Error responses from the /api/v1/data endpoint. 
                This is affecting our production systems and causing data sync failures. 
                Error rate is approximately 15% of all requests."
                
                ---
                
                Comment by Support Agent (2024-01-15 11:15 AM):
                Thank you for reporting this issue. I've escalated this to our engineering team 
                as a high priority incident. Initial investigation shows increased load on our 
                database servers following the recent update.
                
                Temporary workaround: Please implement retry logic with exponential backoff 
                for your API calls. This should help mitigate the impact while we resolve 
                the underlying issue.
                
                ---
                
                Comment by Engineering Team (2024-01-15 2:45 PM):
                Root cause identified: Database connection pool exhaustion due to inefficient 
                query optimization in the recent update. 
                
                Fix deployed: Updated database connection pooling configuration and reverted 
                problematic query optimizations.
                
                Status: Monitoring for 24 hours to ensure stability.
                
                ---
                
                Comment by Customer (2024-01-16 9:00 AM):
                Confirmed that error rates have returned to normal levels. 
                No 500 errors observed since yesterday afternoon. Thank you for the quick resolution.
                
                Final Resolution:
                Issue resolved through database configuration updates and query optimization rollback. 
                Customer confirmed normal operations restored.
                
                Resolution Time: 22 hours
                """,
                metadata={"source": "support_ticket_12345.txt", "document_type": "support_ticket"}
            )
        ]
        
        return sample_docs
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Process and store documents in the knowledge base"""
        start_time = datetime.now()
        all_chunks = []
        processing_stats = {
            'total_documents': len(documents),
            'document_types': {},
            'chunk_counts': {},
            'processing_errors': []
        }
        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('source', 'unknown')}")
                chunks, doc_type = self.chunker.chunk_document(doc)
                processing_stats['document_types'][doc_type.value] = \
                    processing_stats['document_types'].get(doc_type.value, 0) + 1
                processing_stats['chunk_counts'][doc.metadata.get('source', f'doc_{i}')] = len(chunks)
                all_chunks.extend(chunks)
            except Exception as e:
                error_msg = f"Error processing document {i}: {str(e)}"
                logger.error(error_msg)
                processing_stats['processing_errors'].append(error_msg)
        if all_chunks:
            for chunk in all_chunks:
                chunk.metadata = {k: v for k, v in chunk.metadata.items() if not isinstance(v, (dict, list))}
            try:
                if self.vector_store is not None:
                    self.vector_store.add_documents(all_chunks)
                    logger.info(f"Successfully stored {len(all_chunks)} chunks in vector database")
            except Exception as e:
                logger.error(f"Failed to store chunks in vector database: {e}")
                raise
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics.update({
            'total_documents': self.metrics['total_documents'] + processing_stats['total_documents'],
            'total_chunks': self.metrics['total_chunks'] + len(all_chunks),
            'processing_time': self.metrics['processing_time'] + processing_time,
            'document_types': {**self.metrics['document_types'], **processing_stats['document_types']}
        })
        processing_stats['processing_time_seconds'] = processing_time
        processing_stats['total_chunks'] = len(all_chunks)
        return processing_stats
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Document]:
        """Search the knowledge base for relevant documents"""
        try:
            if self.vector_store is not None:
                results = self.vector_store.similarity_search(query, k=k)
                return results
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def evaluate_retrieval_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval quality using test queries"""
        if not test_queries:
            return {}
        
        total_queries = len(test_queries)
        successful_retrievals = 0
        relevance_scores = []
        
        for query_data in test_queries:
            query = query_data['query']
            expected_doc_types = query_data.get('expected_types', [])
            
            results = self.search_knowledge_base(query, k=3)
            
            if results:
                successful_retrievals += 1
                
                # Calculate relevance score based on document types
                if expected_doc_types:
                    retrieved_types = [doc.metadata.get('document_type') for doc in results]
                    relevance = len(set(retrieved_types) & set(expected_doc_types)) / len(expected_doc_types)
                    relevance_scores.append(relevance)
                else:
                    relevance_scores.append(1.0)  # Default score if no expected types
        
        metrics = {
            'retrieval_success_rate': successful_retrievals / total_queries,
            'average_relevance_score': np.mean(relevance_scores) if relevance_scores else 0.0,
            'total_queries_evaluated': total_queries
        }
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'processing_metrics': self.metrics,
            'vector_store_stats': {
                'collection_name': self.collection_name,
                'total_documents_in_store': self.vector_store._collection.count() if self.vector_store else 0
            },
            'chunking_strategies': {
                doc_type.value: strategy.__dict__ 
                for doc_type, strategy in self.chunker.strategies.items()
            }
        }
    
    def export_processed_data(self, export_path: str):
        """Export processed data and metrics"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.get_system_metrics(),
            'document_processing_stats': self.metrics
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Data exported to {export_path}")

def main():
    """Main function to demonstrate the system"""
    # Configuration
    OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual API key
    CHROMA_DB_PATH = "./enterprise_knowledge_db"
    
    # Initialize the system
    print("🚀 Initializing Enterprise Knowledge Management System...")
    knowledge_manager = EnterpriseKnowledgeManager(
        openai_api_key=OPENAI_API_KEY,
        chroma_db_path=CHROMA_DB_PATH
    )
    
    # Load sample documents
    print("\n📄 Loading sample documents...")
    sample_docs = knowledge_manager.load_sample_documents()
    print(f"Loaded {len(sample_docs)} sample documents")
    
    # Process documents
    print("\n🔄 Processing documents with adaptive chunking...")
    processing_stats = knowledge_manager.process_documents(sample_docs)
    
    print(f"\n📊 Processing Complete!")
    print(f"- Total documents processed: {processing_stats['total_documents']}")
    print(f"- Total chunks created: {processing_stats['total_chunks']}")
    print(f"- Processing time: {processing_stats['processing_time_seconds']:.2f} seconds")
    print(f"- Document types detected: {processing_stats['document_types']}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    test_queries = [
        {
            'query': 'How to authenticate with the API?',
            'expected_types': ['api_reference', 'tutorial']
        },
        {
            'query': 'What is the company data privacy policy?',
            'expected_types': ['policy_document']
        },
        {
            'query': 'Application won\'t start troubleshooting',
            'expected_types': ['troubleshooting_guide']
        },
        {
            'query': 'Support ticket database errors',
            'expected_types': ['support_ticket']
        }
    ]
    
    for query_data in test_queries:
        query = query_data['query']
        print(f"\n🔎 Query: '{query}'")
        results = knowledge_manager.search_knowledge_base(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.metadata.get('document_type', 'unknown')}] "
                  f"{result.metadata.get('source', 'unknown')}")
            print(f"     Preview: {result.page_content[:100]}...")
    
    # Evaluate retrieval quality
    print("\n📈 Evaluating retrieval quality...")
    quality_metrics = knowledge_manager.evaluate_retrieval_quality(test_queries)
    print(f"- Retrieval success rate: {quality_metrics['retrieval_success_rate']:.2%}")
    print(f"- Average relevance score: {quality_metrics['average_relevance_score']:.2f}")
    
    # Display system metrics
    print("\n📊 System Metrics:")
    metrics = knowledge_manager.get_system_metrics()
    print(f"- Total documents processed: {metrics['processing_metrics']['total_documents']}")
    print(f"- Total chunks created: {metrics['processing_metrics']['total_chunks']}")
    print(f"- Document types distribution: {metrics['processing_metrics']['document_types']}")
    
    # Export results
    export_path = "enterprise_knowledge_system_results.json"
    knowledge_manager.export_processed_data(export_path)
    print(f"\n💾 Results exported to: {export_path}")
    
    # Interactive search demo
    print("\n6. Interactive Search Demo:")
    demo_queries = [
        "How to configure database connections?",
        "API rate limiting information",
        "Support ticket about login issues",
        "Application performance problems",
        # Add an exact match query from a document chunk
        "The User Authentication API provides secure access to user accounts.",
    ]
    for query in demo_queries:
        print(f"\n   🔍 Query: '{query}'")
        results = knowledge_manager.search_knowledge_base(query, k=2)
        for i, result in enumerate(results, 1):
            doc_type = result.metadata.get('document_type', 'unknown')
            source = result.metadata.get('source', 'unknown')
            preview = result.page_content[:150].replace('\n', ' ')
            print(f"      {i}. [{doc_type}] {source}")
            print(f"         {preview}...")
    
    print("\n✅ Enterprise Knowledge Management System Demo Complete!")

if __name__ == "__main__":
    main()