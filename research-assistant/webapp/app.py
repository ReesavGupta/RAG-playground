# app_new.py - Streamlit Web Application (Refactored)
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to Python path to find src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our refactored components
from src.assistant  import HybridSearchAssistant
from src.core.config import Config
from src.core.models import SearchResult

# Page configuration
st.set_page_config(
    page_title="Hybrid Search Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .source-document {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .source-web {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .credibility-high {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
    .credibility-medium {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .credibility-low {
        background-color: #ffebee;
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.init_session_state()
        self.assistant = self.get_assistant()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'assistant' not in st.session_state:
            st.session_state.assistant = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'documents_uploaded' not in st.session_state:
            st.session_state.documents_uploaded = []
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = None
    
    @st.cache_resource
    def get_assistant(_self):
        """Initialize and cache the assistant"""
        try:
            Config.validate()
            return HybridSearchAssistant()
        except Exception as e:
            st.error(f"Failed to initialize assistant: {e}")
            return None
    
    def render_sidebar(self):
        """Render the sidebar with configuration and controls"""
        st.sidebar.markdown("## 🔧 Configuration")
        
        # API Key status
        if Config.SERPER_API_KEY:
            st.sidebar.success("✅ Serper API Key configured")
        else:
            st.sidebar.error("❌ Serper API Key missing")
            st.sidebar.info("Please set SERPER_API_KEY in your .env file")
        
        # Model settings
        st.sidebar.markdown("### Model Settings")
        st.sidebar.info(f"**Ollama Model:** {Config.OLLAMA_MODEL}")
        st.sidebar.info(f"**Embedding Model:** {Config.EMBEDDING_MODEL}")
        
        # Search settings
        st.sidebar.markdown("### Search Settings")
        doc_results = st.sidebar.slider("Document Results", 1, 20, Config.DEFAULT_DOC_RESULTS)
        web_results = st.sidebar.slider("Web Results", 1, 20, Config.DEFAULT_WEB_RESULTS)
        hybrid_alpha = st.sidebar.slider("Dense/Sparse Weight", 0.0, 1.0, Config.HYBRID_ALPHA, 0.1)
        
        # Document upload
        st.sidebar.markdown("### 📄 Document Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            self.handle_file_upload(uploaded_files)
        
        # System stats
        if self.assistant:
            st.sidebar.markdown("### 📊 System Stats")
            stats = self.assistant.get_system_stats()
            st.sidebar.metric("Documents Indexed", stats['documents_indexed'])
            st.sidebar.metric("Active Sessions", stats['sessions_active'])
            st.sidebar.metric("Cache Size", stats['cache_stats']['total_entries'])
        
        return doc_results, web_results, hybrid_alpha
    
    def handle_file_upload(self, uploaded_files):
        """Handle PDF file uploads"""
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents_uploaded:
                # Save uploaded file
                file_path = f"documents/{uploaded_file.name}"
                os.makedirs("documents", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to assistant
                if self.assistant and self.assistant.add_document(file_path):
                    st.session_state.documents_uploaded.append(uploaded_file.name)
                    st.sidebar.success(f"✅ Added {uploaded_file.name}")
                else:
                    st.sidebar.error(f"❌ Failed to add {uploaded_file.name}")
    
    def render_main_interface(self, doc_results, web_results, hybrid_alpha):
        """Render the main query interface"""
        st.markdown('<div class="main-header">🔍 Hybrid Search Research Assistant</div>', 
                   unsafe_allow_html=True)
        
        # Query input
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., What are the latest developments in machine learning?",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        with col2:
            use_cache = st.checkbox("Use Cache", value=True)
        
        with col3:
            session_id = st.text_input("Session ID", value="default")
        
        if search_button and query and self.assistant:
            self.process_query(query, session_id, use_cache, doc_results, web_results, hybrid_alpha)
        
        # Display recent queries
        if st.session_state.query_history:
            st.markdown("### 📝 Recent Queries")
            for i, query_data in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {query_data['query'][:50]}..."):
                    self.display_query_result(query_data)
    
    def process_query(self, query: str, session_id: str, use_cache: bool, doc_results: int, web_results: int, hybrid_alpha: float):
        """Process a search query"""
        with st.spinner("Searching and analyzing..."):
            try:
                # Update retriever settings
                if not self.assistant:
                    return

                self.assistant.update_settings(doc_results, web_results, hybrid_alpha)
                
                # Process query
                response = self.assistant.query(query, session_id, use_cache)
                
                # Store in history
                query_data = {
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.query_history.append(query_data)
                
                # Display results
                self.display_query_result(query_data)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    def display_query_result(self, query_data: Dict[str, Any]):
        """Display query results"""
        response = query_data['response']
        
        # Display response
        st.markdown("### Response")
        st.write(response['response'])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sources Used", response['sources_used'])
        
        with col2:
            st.metric("Response Time", f"{response['response_time']:.2f}s")
        
        with col3:
            st.metric("Avg Credibility", f"{response['average_credibility']:.2f}")
        
        with col4:
            st.metric("Session ID", response['session_id'])
        
        # Display source breakdown
        if 'source_breakdown' in response:
            st.markdown("### Source Breakdown")
            breakdown = response['source_breakdown']
            self.render_source_breakdown(breakdown)
    
    def render_source_breakdown(self, breakdown: Dict[str, int]):
        """Render source breakdown chart"""
        if breakdown:
            fig = px.pie(
                values=list(breakdown.values()),
                names=list(breakdown.keys()),
                title="Source Distribution"
            )
            st.plotly_chart(fig, use_container_width=True, key="source_breakdown")
    
    def render_benchmark_tab(self):
        """Render benchmark testing tab"""
        st.markdown("## 🧪 Benchmark Testing")
        
        # Test queries
        test_queries = st.text_area(
            "Enter test queries (one per line):",
            value="What is machine learning?\nExplain neural networks\nHow does NLP work?",
            height=150
        )
        
        if st.button("Run Benchmark"):
            if self.assistant and test_queries.strip():
                queries = [q.strip() for q in test_queries.split('\n') if q.strip()]
                
                with st.spinner("Running benchmark..."):
                    results = self.assistant.benchmark(queries)
                    st.session_state.benchmark_results = results
                
                self.display_benchmark_results(results)
    
    def display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results"""
        st.markdown("### Benchmark Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", results['total_queries'])
        
        with col2:
            st.metric("Success Rate", f"{results['success_rate']:.2%}")
        
        with col3:
            st.metric("Avg Response Time", f"{results['avg_response_time']:.2f}s")
        
        with col4:
            st.metric("Source Coverage", sum(results['source_coverage'].values()))
        
        # Individual results
        st.markdown("### Individual Query Results")
        for i, result in enumerate(results['individual_results']):
            with st.expander(f"Query {i+1}: {result['query']}"):
                if result['success']:
                    st.success(f"✅ Success - {result['response_time']:.2f}s")
                    st.info(f"Sources used: {result['sources_used']}")
                else:
                    st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.markdown("## 📊 Analytics")
        
        if self.assistant:
            # System statistics
            stats = self.assistant.get_system_stats()
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents Indexed", stats['documents_indexed'])
                st.metric("Active Sessions", stats['sessions_active'])
            
            with col2:
                st.metric("Cache Entries", stats['cache_stats']['total_entries'])
                st.metric("Web Search Available", "✅" if stats['web_search_available'] else "❌")
            
            with col3:
                st.metric("Model Connection", "✅" if stats['model_info']['connection_ok'] else "❌")
                st.metric("Model Name", stats['model_info']['model_name'])
            
            # Quality metrics
            if 'quality_metrics' in stats:
                st.markdown("### Quality Metrics")
                quality = stats['quality_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", quality['total_queries'])
                
                with col2:
                    st.metric("Avg Response Time", f"{quality['avg_response_time']:.2f}s")
                
                with col3:
                    st.metric("Success Rate", f"{quality['query_success_rate']:.2%}")
                
                with col4:
                    st.metric("Avg Credibility", f"{quality['avg_credibility']:.2f}")
                
                # Source distribution chart
                if 'source_distribution' in quality:
                    st.markdown("### Source Distribution")
                    source_data = quality['source_distribution']
                    fig = px.bar(
                        x=list(source_data.keys()),
                        y=list(source_data.values()),
                        title="Source Distribution Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="analytics_source_distribution")
    
    def render_help_tab(self):
        """Render help tab"""
        st.markdown("## ❓ Help & Documentation")
        
        st.markdown("""
        ### How to Use
        
        1. **Upload Documents**: Use the sidebar to upload PDF documents for local search
        2. **Ask Questions**: Enter your research question in the main interface
        3. **Configure Search**: Adjust document/web results and hybrid weights in the sidebar
        4. **View Results**: See comprehensive responses with source citations
        
        ### Features
        
        - **Hybrid Search**: Combines dense (semantic) and sparse (keyword) retrieval
        - **Web Search**: Integrates with Serper API for real-time web results
        - **Document Search**: Uses ChromaDB for vector storage and BM25 for keyword search
        - **Response Synthesis**: Generates comprehensive answers using LLM
        - **Quality Monitoring**: Tracks performance and credibility metrics
        - **Session Management**: Maintains query history and caching
        
        ### Configuration
        
        Set the following environment variables:
        - `SERPER_API_KEY`: Your Serper API key for web search
        - `OLLAMA_MODEL`: Ollama model name (default: llama3.2)
        - `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
        
        ### Architecture
        
        The system is built with modular components:
        - **Core**: Configuration and data models
        - **Storage**: ChromaDB integration and session management
        - **Search**: Web search, document search, and hybrid retrieval
        - **Synthesis**: LLM-based response generation
        - **Monitoring**: Quality and performance tracking
        """)
    
    def run(self):
        """Run the Streamlit application"""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "🧪 Benchmark", "📊 Analytics", "❓ Help"])
        
        # Sidebar
        doc_results, web_results, hybrid_alpha = self.render_sidebar()
        
        # Main interface tab
        with tab1:
            self.render_main_interface(doc_results, web_results, hybrid_alpha)
        
        # Benchmark tab
        with tab2:
            self.render_benchmark_tab()
        
        # Analytics tab
        with tab3:
            self.render_analytics_tab()
        
        # Help tab
        with tab4:
            self.render_help_tab()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run() 