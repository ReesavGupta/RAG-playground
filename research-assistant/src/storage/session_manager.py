"""
Session management and caching for the research assistant.
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.models import SearchResult
from ..core.config import Config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage user sessions and query caching"""
    
    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = Config.CACHE_TTL
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create new user session"""
        session_data = {
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'queries': [],
            'documents_processed': [],
            'total_queries': 0,
            'cache_hits': 0
        }
        
        self.sessions[session_id] = session_data
        logger.info(f"Created new session: {session_id}")
        return session_data
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data and update last accessed time"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session['last_accessed'] = datetime.now().isoformat()
        return session
    
    def add_query(self, session_id: str, query: str, response: Dict[str, Any]) -> None:
        """Add query to session history"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        session['queries'].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        session['total_queries'] += 1
        session['last_accessed'] = datetime.now().isoformat()
        
        logger.info(f"Added query to session {session_id}")
    
    def add_document_to_session(self, session_id: str, document_path: str) -> None:
        """Add processed document to session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        if document_path not in session['documents_processed']:
            session['documents_processed'].append(document_path)
            logger.info(f"Added document {document_path} to session {session_id}")
    
    def cache_query(self, query: str, results: List[SearchResult], ttl: Optional[int] = None) -> None:
        """Cache query results"""
        cache_key = self._generate_cache_key(query)
        cache_ttl = ttl or self.cache_ttl
        
        self.query_cache[cache_key] = {
            'results': results,
            'cached_at': time.time(),
            'ttl': cache_ttl
        }
        
        logger.info(f"Cached query results for key: {cache_key[:8]}...")
    
    def get_cached_query(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached query results if still valid"""
        cache_key = self._generate_cache_key(query)
        cached = self.query_cache.get(cache_key)
        
        if cached and (time.time() - cached['cached_at']) < cached['ttl']:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached['results']
        
        if cached:
            # Remove expired cache entry
            del self.query_cache[cache_key]
            logger.info(f"Removed expired cache entry: {cache_key[:8]}...")
        
        return None
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'],
            'last_accessed': session['last_accessed'],
            'total_queries': session['total_queries'],
            'documents_processed': len(session['documents_processed']),
            'cache_hits': session.get('cache_hits', 0)
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return self.sessions.copy()
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def clear_expired_sessions(self, max_age_hours: int = 24) -> int:
        """Clear sessions that haven't been accessed for specified hours"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            last_accessed = datetime.fromisoformat(session['last_accessed'])
            age_hours = (current_time - last_accessed).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        logger.info(f"Cleared {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def clear_cache(self) -> int:
        """Clear all cached queries"""
        cache_size = len(self.query_cache)
        self.query_cache.clear()
        logger.info(f"Cleared {cache_size} cached queries")
        return cache_size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for cache_data in self.query_cache.values():
            if (current_time - cache_data['cached_at']) < cache_data['ttl']:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.query_cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_size_mb': len(str(self.query_cache)) / (1024 * 1024)  # Rough estimate
        } 