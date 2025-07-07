import time
import numpy as np
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def get_embedding(self, text):
        """Get OpenAI embedding"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts, batch_size=100):
        """Get embeddings in batches with rate limiting"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
                
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)