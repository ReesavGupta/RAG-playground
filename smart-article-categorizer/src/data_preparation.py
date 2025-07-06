import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

class DataPreprocessor:
    def __init__(self) -> None:
        self.categories = ["Tech", 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    def load_news_dataset(self):
        dataset = load_dataset("ag_news")
        return self.create_sample_dataset()

    def create_sample_dataset(self):
        """Create sample dataset for demonstration"""
        # This would be replaced with real news data
        sample_data = {
            'text': [
                "Apple releases new iPhone with advanced AI features...",
                "Stock market shows volatility amid inflation concerns...",
                "New cancer treatment shows promising results in trials...",
                "Basketball championship finals draw record viewers...",
                "Senate passes new infrastructure bill...",
                "Hollywood blockbuster breaks box office records..."
            ],
            'category': ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        }
        return pd.DataFrame(sample_data)

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self):
        """Main data preparation pipeline"""
        df = self.load_news_dataset()
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['category'], 
            test_size=0.2, random_state=42, stratify=df['category']
        )
        
        return X_train, X_test, y_train, y_test   