from openai_embedding_model import OpenAIEmbedder
from word2vec_embedding_model import Word2VecEmbedder
from bert_embedding_model import BERTEmbedder, SentenceBERTEmbedder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class ArticleClassifier:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.embedders = {}
        
    def initialize_embedders(self, openai_key=None):
        """Initialize all embedding models"""
        self.embedders['word2vec'] = Word2VecEmbedder()
        self.embedders['bert'] = BERTEmbedder()
        self.embedders['sentence_bert'] = SentenceBERTEmbedder()
        if openai_key:
            self.embedders['openai'] = OpenAIEmbedder(openai_key)
    
    def train_classifier(self, X_train, y_train, embedding_type):
        """Train logistic regression classifier"""
        # Get embeddings
        if embedding_type == 'word2vec':
            self.embedders['word2vec'].train_word2vec(X_train)
            X_train_emb = self.embedders['word2vec'].get_embeddings_batch(X_train)
        elif embedding_type == 'bert':
            X_train_emb = self.embedders['bert'].get_embeddings_batch(X_train)
        elif embedding_type == 'sentence_bert':
            X_train_emb = self.embedders['sentence_bert'].get_embeddings_batch(X_train)
        elif embedding_type == 'openai':
            X_train_emb = self.embedders['openai'].get_embeddings_batch(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train_emb, y_train_encoded)
        
        self.models[embedding_type] = classifier
        
        return X_train_emb
    
    def predict(self, texts, embedding_type):
        """Make predictions"""
        if embedding_type not in self.models:
            raise ValueError(f"Model {embedding_type} not trained")
            
        # Get embeddings
        if embedding_type == 'word2vec':
            X_emb = self.embedders['word2vec'].get_embeddings_batch(texts)
        elif embedding_type == 'bert':
            X_emb = self.embedders['bert'].get_embeddings_batch(texts)
        elif embedding_type == 'sentence_bert':
            X_emb = self.embedders['sentence_bert'].get_embeddings_batch(texts)
        elif embedding_type == 'openai':
            X_emb = self.embedders['openai'].get_embeddings_batch(texts)
        
        # Predict
        predictions = self.models[embedding_type].predict(X_emb)
        probabilities = self.models[embedding_type].predict_proba(X_emb)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def save_models(self, directory):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for model_type, model in self.models.items():
            with open(f"{directory}/{model_type}_classifier.pkl", 'wb') as f:
                pickle.dump(model, f)
                
        with open(f"{directory}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)