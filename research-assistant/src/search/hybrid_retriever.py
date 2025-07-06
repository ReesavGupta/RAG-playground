"""
Hybrid retriever that combines document and web search.
"""

import logging
from typing import List, Dict, Any, Optional

from ..core.models import SearchResult
from ..core.config import Config
from .web_search import SerperWebSearch
from .document_search import DocumentSearch
from ..core.exceptions import SearchError

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines document and web search with intelligent ranking"""
    
    def __init__(self, document_search: DocumentSearch, web_search: SerperWebSearch):
        """Initialize hybrid retriever"""
        self.document_search = document_search
        self.web_search = web_search
        self.doc_k = Config.DEFAULT_DOC_RESULTS
        self.web_k = Config.DEFAULT_WEB_RESULTS
        self.hybrid_alpha = Config.HYBRID_ALPHA
        
        logger.info("HybridRetriever initialized")
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        """Search documents using hybrid retrieval"""
        k = k or self.doc_k
        return self.document_search.hybrid_search(query, k, self.hybrid_alpha)
    
    def search_web(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        """Search web using Serper API"""
        k = k or self.web_k
        
        try:
            web_results = self.web_search.search(query, k)
            
            # Score web results based on query relevance
            for result in web_results:
                result.relevance_score = self._calculate_relevance(query, result.content)
            
            # Sort by relevance score
            web_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return web_results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def combined_search(self, query: str, doc_k: Optional[int] = None, web_k: Optional[int] = None) -> List[SearchResult]:
        """Combine document and web search results with intelligent ranking"""
        doc_k = doc_k or self.doc_k
        web_k = web_k or self.web_k
        
        try:
            # Get results from both sources
            doc_results = self.search_documents(query, doc_k)
            web_results = self.search_web(query, web_k)
            
            # Combine results
            all_results = doc_results + web_results
            
            # Re-rank based on relevance and credibility
            for result in all_results:
                # Combine relevance and credibility scores
                combined_score = (result.relevance_score * 0.8 + 
                                result.credibility_score * 0.2)
                result.relevance_score = combined_score
            
            # Sort by combined score
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Combined search returned {len(all_results)} results "
                       f"({len(doc_results)} documents, {len(web_results)} web)")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Combined search error: {e}")
            raise SearchError(f"Combined search failed: {str(e)}")
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score based on term overlap"""
        if not query or not content:
            return 0.0
        
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(content_terms))
        base_score = overlap / len(query_terms)
        
        # Boost score for exact phrase matches
        if query.lower() in content.lower():
            base_score += 0.3
        
        return min(base_score, 1.0)
    
    def update_settings(self, doc_k: Optional[int] = None, web_k: Optional[int] = None, alpha: Optional[float] = None) -> None:
        """Update search settings"""
        if doc_k is not None:
            self.doc_k = doc_k
        if web_k is not None:
            self.web_k = web_k
        if alpha is not None:
            self.hybrid_alpha = alpha
        
        logger.info(f"Updated settings: doc_k={self.doc_k}, web_k={self.web_k}, alpha={self.hybrid_alpha}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'document_search_stats': self.document_search.get_stats(),
            'web_search_available': self.web_search.test_connection(),
            'settings': {
                'doc_k': self.doc_k,
                'web_k': self.web_k,
                'hybrid_alpha': self.hybrid_alpha
            }
        } 