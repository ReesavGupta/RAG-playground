import nltk
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

class Word2VecEmbedder:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def train_word2vec(self, texts: str):
        tokenized_texts = []
        for text in texts:
            token =word_tokenize(text=text.lower())
            tokenized_texts.append(token)

        self.model = Word2Vec(
            tokenized_texts,
            vector_size=self.vector_size,
            min_count=self.min_count,
            window=self.window,
            workers=4
        )

    def get_document_embedding(self, text):
        """Get document embedding by averaging word vectors"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_word2vec first.")
            
        tokens = word_tokenize(text.lower())
        vectors = []
        
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
                
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def get_embeddings_batch(self, texts):
        """Get embeddings for multiple texts"""
        return np.array([self.get_document_embedding(text) for text in texts])

