"""
Main research assistant class that orchestrates all components.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from .core.config import Config
from .core.exceptions import ResearchAssistantError, ConfigurationError
from .storage.chroma_store import ChromaDocumentStore
from .storage.session_manager import SessionManager
from .search.web_search import SerperWebSearch
from .search.document_search import DocumentSearch
from .search.hybrid_retriever import HybridRetriever
from .synthesis.response_synthesizer import ResponseSynthesizer
from .monitoring.quality_monitor import QualityMonitor

logger = logging.getLogger(__name__)


class HybridSearchAssistant:
    """Main research assistant that combines document and web search"""
    
    def __init__(self, serper_api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the research assistant"""
        try:
            # Validate configuration
            Config.validate()
            
            # Initialize components
            self._initialize_components(serper_api_key, model_name)
            
            logger.info("HybridSearchAssistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
            raise ResearchAssistantError(f"Initialization failed: {str(e)}")
    
    def _initialize_components(self, serper_api_key: Optional[str], model_name: Optional[str]) -> None:
        """Initialize all assistant components"""
        # Get API key
        api_key = serper_api_key or Config.get_serper_api_key()
        
        # Initialize storage
        self.chroma_store = ChromaDocumentStore()
        self.session_manager = SessionManager()
        
        # Initialize search components
        self.web_search = SerperWebSearch(api_key)
        self.document_search = DocumentSearch(self.chroma_store)
        self.retriever = HybridRetriever(self.document_search, self.web_search)
        
        # Initialize synthesis and monitoring
        self.synthesizer = ResponseSynthesizer(model_name)
        self.quality_monitor = QualityMonitor()
        
        logger.info("All components initialized")
    
    def add_document(self, pdf_path: str) -> bool:
        """Add PDF document to knowledge base"""
        try:
            # Process PDF
            documents = self.chroma_store.load_and_process_pdf(pdf_path)
            if not documents:
                logger.warning(f"No documents extracted from {pdf_path}")
                return False
            
            # Add to search indexes
            success = self.document_search.add_documents(documents)
            if not success:
                logger.error(f"Failed to add documents from {pdf_path}")
                return False
            
            logger.info(f"Successfully added {len(documents)} chunks from {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {pdf_path}: {e}")
            return False
    
    def query(self, query: str, session_id: str = "default", use_cache: bool = True) -> Dict[str, Any]:
        """Process query and return comprehensive response"""
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_results = self.session_manager.get_cached_query(query)
                if cached_results:
                    logger.info("Using cached results")
                    response = self.synthesizer.synthesize_response(query, cached_results)
                    response.response_time = time.time() - start_time
                    response.session_id = session_id
                    
                    # Log metrics
                    self.quality_monitor.log_query(query, response.response_time, cached_results)
                    
                    # Add to session
                    self.session_manager.add_query(session_id, query, response.__dict__)
                    
                    return response.__dict__
            
            # Perform hybrid search
            search_results = self.retriever.combined_search(query)
            
            # Cache results
            if use_cache:
                self.session_manager.cache_query(query, search_results)
            
            # Generate response
            response = self.synthesizer.synthesize_response(query, search_results)
            response.response_time = time.time() - start_time
            response.session_id = session_id
            
            # Log metrics
            self.quality_monitor.log_query(query, response.response_time, search_results)
            
            # Add to session
            self.session_manager.add_query(session_id, query, response.__dict__)
            
            return response.__dict__
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Query processing error: {e}")
            
            # Log failed query
            self.quality_monitor.log_query(query, response_time, [], success=False, error=str(e))
            
            return {
                'response': f"Error processing query: {str(e)}",
                'sources_used': 0,
                'source_breakdown': {'documents': 0, 'web': 0},
                'average_credibility': 0.0,
                'response_time': response_time,
                'session_id': session_id,
                'search_results': [],
                'error': str(e)
            }
    
    def benchmark(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run benchmark tests"""
        results = {
            'total_queries': len(test_queries),
            'avg_response_time': 0.0,
            'success_rate': 0.0,
            'source_coverage': {'documents': 0, 'web': 0},
            'individual_results': []
        }
        
        successful_queries = 0
        total_time = 0.0
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = self.query(query, session_id="benchmark", use_cache=False)
                response_time = time.time() - start_time
                
                total_time += response_time
                successful_queries += 1
                
                results['individual_results'].append({
                    'query': query,
                    'response_time': response_time,
                    'success': True,
                    'sources_used': response.get('sources_used', 0)
                })
                
                # Update source coverage
                breakdown = response.get('source_breakdown', {})
                results['source_coverage']['documents'] += breakdown.get('documents', 0)
                results['source_coverage']['web'] += breakdown.get('web', 0)
                
            except Exception as e:
                results['individual_results'].append({
                    'query': query,
                    'response_time': 0.0,
                    'success': False,
                    'error': str(e)
                })
        
        results['avg_response_time'] = total_time / len(test_queries) if test_queries else 0.0
        results['success_rate'] = successful_queries / len(test_queries) if test_queries else 0.0
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'documents_indexed': self.document_search.get_stats()['total_documents'],
            'sessions_active': len(self.session_manager.get_all_sessions()),
            'cache_stats': self.session_manager.get_cache_stats(),
            'quality_metrics': self.quality_monitor.get_metrics(),
            'retriever_stats': self.retriever.get_stats(),
            'model_info': self.synthesizer.get_model_info(),
            'web_search_available': self.web_search.test_connection()
        }
    
    def update_settings(self, doc_k: Optional[int] = None, web_k: Optional[int] = None, 
                       alpha: Optional[float] = None) -> None:
        """Update search settings"""
        self.retriever.update_settings(doc_k, web_k, alpha)
    
    def clear_cache(self) -> int:
        """Clear all cached queries"""
        return self.session_manager.clear_cache()
    
    def clear_documents(self) -> bool:
        """Clear all documents from the knowledge base"""
        return self.chroma_store.clear_collection()
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get query history for a session"""
        session = self.session_manager.get_session(session_id)
        return session.get('queries', []) if session else []
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all system metrics"""
        return {
            'system_stats': self.get_system_stats(),
            'quality_metrics': self.quality_monitor.export_metrics(),
            'export_timestamp': time.time()
        } 