"""
Document search functionality with BM25 sparse retrieval.
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from ..core.models import SearchResult
from ..storage.chroma_store import ChromaDocumentStore
from ..core.exceptions import SearchError

logger = logging.getLogger(__name__)


class DocumentSearch:
    """Document search with hybrid dense and sparse retrieval"""
    
    def __init__(self, chroma_store: ChromaDocumentStore):
        """Initialize document search"""
        self.chroma_store = chroma_store
        self.bm25 = None
        self.documents = []
        self.document_metadata = {}
        
        logger.info("DocumentSearch initialized")
    
    def add_documents(self, documents: List[Any]) -> bool:
        """Add documents to both vector store and BM25 index"""
        try:
            # Add to ChromaDB
            success = self.chroma_store.add_documents(documents)
            if not success:
                return False
            
            # Add to BM25 index
            self._update_bm25_index(documents)
            
            # Store document metadata
            for doc in documents:
                doc_id = id(doc)
                self.document_metadata[doc_id] = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source_file', 'Unknown')
                }
            
            logger.info(f"Added {len(documents)} documents to search indexes")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise SearchError(f"Failed to add documents: {str(e)}")
    
    def _update_bm25_index(self, documents: List[Any]) -> None:
        """Update BM25 index with new documents"""
        # Store document references first
        self.documents.extend(documents)
        
        # Prepare tokenized documents for BM25 (all documents)
        tokenized_docs = []
        for doc in self.documents:
            # Simple tokenization - split on whitespace and lowercase
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        # Recreate BM25 index with all documents
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Updated BM25 index with {len(documents)} new documents (total: {len(self.documents)})")
    
    def dense_search(self, query: str, k: int = 10) -> List[Tuple[Any, float]]:
        """Semantic search using vector embeddings"""
        try:
            results = self.chroma_store.search(query, k=k)
            logger.info(f"Dense search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []
    
    def sparse_search(self, query: str, k: int = 10) -> List[Tuple[Any, float]]:
        """Keyword-based search using BM25"""
        if self.bm25 is None or not self.documents:
            logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            
            for idx in top_indices:
                if idx < len(self.documents):
                    results.append((self.documents[idx], scores[idx]))
            
            logger.info(f"Sparse search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[SearchResult]:
        """Combine dense and sparse retrieval with configurable weighting"""
        try:
            # Get results from both methods
            dense_results = self.dense_search(query, k)
            sparse_results = self.sparse_search(query, k)
            
            # Normalize scores
            dense_scores = self._normalize_scores([score for _, score in dense_results])
            sparse_scores = self._normalize_scores([score for _, score in sparse_results])
            
            # Combine results
            doc_results = {}
            
            # Add dense results
            for i, (doc, _) in enumerate(dense_results):
                doc_id = id(doc)
                doc_results[doc_id] = {
                    'document': doc,
                    'dense_score': dense_scores[i],
                    'sparse_score': 0.0
                }
            
            # Add sparse results
            for i, (doc, _) in enumerate(sparse_results):
                doc_id = id(doc)
                if doc_id in doc_results:
                    doc_results[doc_id]['sparse_score'] = sparse_scores[i]
                else:
                    doc_results[doc_id] = {
                        'document': doc,
                        'dense_score': 0.0,
                        'sparse_score': sparse_scores[i]
                    }
            
            # Calculate hybrid scores
            hybrid_results = []
            for doc_id, data in doc_results.items():
                hybrid_score = (alpha * data['dense_score'] + 
                              (1 - alpha) * data['sparse_score'])
                
                result = SearchResult(
                    content=data['document'].page_content,
                    source=data['document'].metadata.get('source_file', 'Unknown'),
                    relevance_score=hybrid_score,
                    source_type='document',
                    credibility_score=1.0,  # Documents are assumed credible
                    snippet=data['document'].page_content[:200] + "..."
                )
                hybrid_results.append(result)
            
            # Sort by relevance score
            hybrid_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            raise SearchError(f"Hybrid search failed: {str(e)}")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'total_documents': len(self.documents),
            'bm25_available': self.bm25 is not None,
            'chroma_stats': self.chroma_store.get_collection_stats()
        } 