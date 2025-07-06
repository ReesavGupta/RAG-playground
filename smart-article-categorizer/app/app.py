import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.classification import ArticleClassifier 

class StreamlitApp:
    def __init__(self):
        self.classifier = ArticleClassifier()
        self.categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        
    def load_models(self):
        """Load pre-trained models"""
        # This would load your saved models
        pass
    
    def main(self):
        st.set_page_config(
            page_title="Smart Article Categorizer",
            page_icon="📰",
            layout="wide"
        )
        
        st.title("🔍 Smart Article Categorizer")
        st.markdown("Classify news articles using different embedding approaches")
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🔮 Classify Article", "📊 Model Comparison", "🧠 Embedding Analysis"])
        
        with tab1:
            self.classification_tab()
            
        with tab2:
            self.comparison_tab()
            
        with tab3:
            self.analysis_tab()
    
    def classification_tab(self):
        """Article classification interface"""
        st.header("Article Classification")
        
        # Text input
        article_text = st.text_area(
            "Enter article text:",
            height=200,
            placeholder="Paste your news article here..."
        )
        
        if st.button("Classify Article", type="primary"):
            if article_text:
                # Show loading
                with st.spinner("Analyzing article..."):
                    # Get predictions from all models
                    results = self.predict_all_models(article_text)
                    
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Predictions")
                    for model_type, (prediction, confidence) in results.items():
                        st.metric(
                            label=f"{model_type.upper()} Model",
                            value=prediction,
                            delta=f"{confidence:.2%} confidence"
                        )
                
                with col2:
                    st.subheader("Confidence Scores")
                    # Create confidence chart
                    confidence_data = {
                        'Model': list(results.keys()),
                        'Confidence': [conf for _, conf in results.values()]
                    }
                    fig = px.bar(
                        confidence_data,
                        x='Model',
                        y='Confidence',
                        title="Model Confidence Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    def predict_all_models(self, text):
        """Get predictions from all models"""
        results = {}
        
        # Simulate predictions (replace with actual model calls)
        models = ['word2vec', 'bert', 'sentence_bert', 'openai']
        for model in models:
            # Mock prediction
            prediction = np.random.choice(self.categories)
            confidence = np.random.uniform(0.6, 0.95)
            results[model] = (prediction, confidence)
            
        return results
    
    def comparison_tab(self):
        """Model comparison interface"""
        st.header("Model Performance Comparison")
        
        # Mock performance data
        performance_data = {
            'Model': ['Word2Vec', 'BERT', 'Sentence-BERT', 'OpenAI'],
            'Accuracy': [0.78, 0.85, 0.88, 0.91],
            'Precision': [0.77, 0.84, 0.87, 0.90],
            'Recall': [0.76, 0.83, 0.86, 0.89],
            'F1-Score': [0.77, 0.84, 0.87, 0.90]
        }
        
        df = pd.DataFrame(performance_data)
        
        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(df, use_container_width=True)
        
        # Create comparison chart
        st.subheader("Performance Visualization")
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=df[metric],
                text=df[metric],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def analysis_tab(self):
        """Embedding analysis interface"""
        st.header("Embedding Analysis")
        
        st.subheader("Embedding Clusters Visualization")
        
        # Generate sample data for visualization
        np.random.seed(42)
        n_samples = 100
        
        # Create sample embeddings using PCA
        sample_embeddings = np.random.randn(n_samples, 50)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(sample_embeddings)
        
        # Create sample labels
        labels = np.random.choice(self.categories, n_samples)
        
        # Create scatter plot
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=labels,
            title="Article Embeddings Visualization (PCA)",
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison metrics
        st.subheader("Embedding Model Characteristics")
        
        model_info = {
            'Model': ['Word2Vec', 'BERT', 'Sentence-BERT', 'OpenAI'],
            'Embedding Size': [100, 768, 384, 1536],
            'Training Time': ['Fast', 'Slow', 'Medium', 'API Call'],
            'Context Awareness': ['Low', 'High', 'High', 'Very High'],
            'Best Use Case': ['Basic similarity', 'Complex NLP', 'Sentence similarity', 'General purpose']
        }
        
        st.dataframe(pd.DataFrame(model_info), use_container_width=True)

if __name__ == "__main__":
    app = StreamlitApp()
    app.main()