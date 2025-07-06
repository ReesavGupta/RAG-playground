#!/usr/bin/env python3
"""
Main entry point for the Hybrid Search Research Assistant.

This is the refactored version with modular components and ChromaDB integration.
"""

import json
import logging
import sys
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our refactored components
from src.assistant import HybridSearchAssistant
from src.core.config import Config


def main():
    """Main function to run the research assistant"""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize assistant
        assistant = HybridSearchAssistant()
        logger.info("Research assistant initialized")
        
        # Example usage
        run_example_queries(assistant)
        
    except Exception as e:
        logger.error(f"Failed to start research assistant: {e}")
        sys.exit(1)


def run_example_queries(assistant: HybridSearchAssistant):
    """Run example queries to demonstrate functionality"""
    
    # Example test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does natural language processing work?",
        "What are the latest developments in AI?",
        "Compare supervised and unsupervised learning"
    ]
    
    print("\n" + "="*60)
    print("🔍 HYBRID SEARCH RESEARCH ASSISTANT")
    print("="*60)
    
    # Run benchmark
    print("\n📊 Running benchmark tests...")
    benchmark_results = assistant.benchmark(test_queries)
    
    print(f"\nBenchmark Results:")
    print(f"  Total Queries: {benchmark_results['total_queries']}")
    print(f"  Success Rate: {benchmark_results['success_rate']:.2%}")
    print(f"  Avg Response Time: {benchmark_results['avg_response_time']:.2f}s")
    print(f"  Source Coverage: {benchmark_results['source_coverage']}")
    
    # Interactive query example
    print("\n" + "-"*60)
    print("💬 Interactive Query Example")
    print("-"*60)
    
    query = "What is the difference between machine learning and deep learning?"
    print(f"\nQuery: {query}")
    
    response = assistant.query(query)
    
    print(f"\nResponse: {response['response'][:200]}...")
    print(f"Sources used: {response['sources_used']}")
    print(f"Average credibility: {response['average_credibility']:.2f}")
    print(f"Response time: {response['response_time']:.2f}s")
    
    # System statistics
    print("\n" + "-"*60)
    print("📈 System Statistics")
    print("-"*60)
    
    stats = assistant.get_system_stats()
    print(f"Documents indexed: {stats['documents_indexed']}")
    print(f"Active sessions: {stats['sessions_active']}")
    print(f"Cache entries: {stats['cache_stats']['total_entries']}")
    print(f"Web search available: {stats['web_search_available']}")
    print(f"Model connection: {stats['model_info']['connection_ok']}")
    
    print("\n✅ Research assistant is ready for use!")


def run_webapp():
    """Run the Streamlit web application"""
    import subprocess
    import sys
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "webapp/app.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start webapp: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "webapp":
        run_webapp()
    else:
        main() 