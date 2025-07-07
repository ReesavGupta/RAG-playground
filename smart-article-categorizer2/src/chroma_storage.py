import chromadb

class ChromaDataManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.db = chromadb.PersistentClient(path=persist_directory)

    def create_collection(self):
        """create collections for different embeddings"""
        collections = {}
        for model_type in ["word2vec", "bert", "sentence_bart", "openai"]:
            collections[model_type] = self.db.get_or_create_collection(name=f"articles_{model_type}")
            return collections
    
    def store_embeddings(self, collection_name, texts, embeddings, categories, ids):
        """Store embeddings in ChromaDB"""
        collection = self.db.get_collection(collection_name)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"category": cat} for cat in categories],
            ids=ids
        )