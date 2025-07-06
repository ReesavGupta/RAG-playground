"""
Web search functionality using Serper API.
"""

import logging
from typing import List
import requests

from ..core.models import SearchResult
from ..core.exceptions import WebSearchError

logger = logging.getLogger(__name__)


class SerperWebSearch:
    """Serper API integration for web search"""
    
    def __init__(self, api_key: str):
        """Initialize web search with API key"""
        if not api_key:
            raise WebSearchError("Serper API key is required")
        
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        
        # Credible domains for credibility assessment
        self.credible_domains = [
            'wikipedia.org', 'britannica.com', 'nature.com', 'sciencedirect.com',
            'ieee.org', 'acm.org', 'springer.com', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov',
            'edu', 'gov', 'org', 'researchgate.net', 'scholar.google.com'
        ]
        
        logger.info("SerperWebSearch initialized")
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform web search using Serper API"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': min(num_results, 20),  # Serper API limit
            'gl': 'us',
            'hl': 'en'
        }
        
        try:
            logger.info(f"Performing web search for: {query}")
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('organic', []):
                result = SearchResult(
                    content=item.get('snippet', ''),
                    source=item.get('link', ''),
                    relevance_score=0.0,  # Will be computed later
                    source_type='web',
                    snippet=item.get('snippet', ''),
                    credibility_score=self._assess_credibility(item.get('link', ''))
                )
                results.append(result)
            
            logger.info(f"Web search returned {len(results)} results")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Web search request error: {e}")
            raise WebSearchError(f"Web search request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            raise WebSearchError(f"Web search failed: {str(e)}")
    
    def _assess_credibility(self, url: str) -> float:
        """Assess credibility of a URL based on domain"""
        if not url:
            return 0.3  # Low credibility for empty URLs
        
        score = 0.5  # Base score
        
        # Check for credible domains
        for domain in self.credible_domains:
            if domain in url.lower():
                score += 0.3
                break
        
        # Additional checks
        if 'https://' in url:
            score += 0.1
        
        if '.edu' in url or '.gov' in url:
            score += 0.2
        
        if 'wikipedia.org' in url:
            score += 0.1
        
        return min(score, 1.0)
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            test_query = "test"
            results = self.search(test_query, num_results=1)
            return len(results) > 0
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False 