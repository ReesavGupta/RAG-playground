"""
Quality monitoring for response quality and system performance.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from ..core.models import SearchResult

logger = logging.getLogger(__name__)


class QualityMonitor:
    """Monitor response quality and system performance"""
    
    def __init__(self):
        """Initialize quality monitor"""
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'source_distribution': {'documents': 0, 'web': 0},
            'credibility_scores': [],
            'response_times': [],
            'query_success_rate': 1.0,
            'failed_queries': 0
        }
        
        logger.info("QualityMonitor initialized")
    
    def log_query(self, query: str, response_time: float, search_results: List[SearchResult], 
                  success: bool = True, error: Optional[str] = None) -> None:
        """Log query metrics"""
        self.metrics['total_queries'] += 1
        
        if success:
            # Update average response time
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['total_queries'] - 1) + response_time) /
                self.metrics['total_queries']
            )
            
            # Update source distribution
            for result in search_results:
                self.metrics['source_distribution'][result.source_type] += 1
                self.metrics['credibility_scores'].append(result.credibility_score)
            
            # Store response time
            self.metrics['response_times'].append(response_time)
        else:
            self.metrics['failed_queries'] += 1
        
        # Update success rate
        self.metrics['query_success_rate'] = (
            (self.metrics['total_queries'] - self.metrics['failed_queries']) / 
            self.metrics['total_queries']
        )
        
        logger.info(f"Logged query: success={success}, time={response_time:.2f}s, "
                   f"results={len(search_results)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        avg_credibility = np.mean(self.metrics['credibility_scores']) if self.metrics['credibility_scores'] else 0.0
        
        # Calculate response time statistics
        response_times = self.metrics['response_times']
        response_time_stats = {
            'avg': np.mean(response_times) if response_times else 0.0,
            'min': np.min(response_times) if response_times else 0.0,
            'max': np.max(response_times) if response_times else 0.0,
            'std': np.std(response_times) if response_times else 0.0
        }
        
        return {
            **self.metrics,
            'avg_credibility': avg_credibility,
            'response_time_stats': response_time_stats,
            'total_sources': sum(self.metrics['source_distribution'].values()),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        metrics = self.get_metrics()
        
        return {
            'performance_score': self._calculate_performance_score(metrics),
            'quality_score': self._calculate_quality_score(metrics),
            'reliability_score': metrics['query_success_rate'],
            'avg_response_time': metrics['avg_response_time'],
            'avg_credibility': metrics['avg_credibility']
        }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on response times"""
        avg_time = metrics['avg_response_time']
        
        # Score based on response time (lower is better)
        if avg_time < 2.0:
            return 1.0
        elif avg_time < 5.0:
            return 0.8
        elif avg_time < 10.0:
            return 0.6
        else:
            return 0.4
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate quality score based on credibility and source diversity"""
        avg_credibility = metrics['avg_credibility']
        source_dist = metrics['source_distribution']
        total_sources = sum(source_dist.values())
        
        if total_sources == 0:
            return 0.0
        
        # Credibility component (50%)
        credibility_score = avg_credibility
        
        # Source diversity component (50%)
        diversity_score = min(len([s for s in source_dist.values() if s > 0]) / 2.0, 1.0)
        
        return (credibility_score * 0.5) + (diversity_score * 0.5)
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'source_distribution': {'documents': 0, 'web': 0},
            'credibility_scores': [],
            'response_times': [],
            'query_success_rate': 1.0,
            'failed_queries': 0
        }
        logger.info("Quality metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for external analysis"""
        return {
            'metrics': self.get_metrics(),
            'performance_summary': self.get_performance_summary(),
            'export_timestamp': datetime.now().isoformat()
        } 