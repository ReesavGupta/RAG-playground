"""
Data models for the research assistant.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class SearchResult:
    """Structured search result with metadata"""
    content: str
    source: str
    relevance_score: float
    source_type: str  # 'document' or 'web'
    timestamp: Optional[str] = None
    credibility_score: float = 0.0
    snippet: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QueryResponse:
    """Response from a query with metadata"""
    response: str
    sources_used: int
    source_breakdown: dict
    average_credibility: float
    response_time: float
    session_id: str
    search_results: list 