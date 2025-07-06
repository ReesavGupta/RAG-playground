import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_bert_embedding(self, text):
        """Get BERT [CLS] token embedding"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return cls_embedding.squeeze()
    
    def get_embeddings_batch(self, texts, batch_size=16):
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = [self.get_bert_embedding(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

from sentence_transformers import SentenceTransformer

class SentenceBERTEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings_batch(self, texts):
        """Get sentence embeddings"""
        return self.model.encode(texts)