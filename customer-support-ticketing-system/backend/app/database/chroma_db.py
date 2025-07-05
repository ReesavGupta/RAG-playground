import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class ChromaDBService:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.tickets_collection = self.client.get_or_create_collection("tickets")
        self.knowledge_collection = self.client.get_or_create_collection("knowledge_base")
    
    def add_ticket(self, ticket_id: str, content: str, metadata: Dict):
        """Add ticket to vector database"""
        self.tickets_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[ticket_id]
        )
    
    def add_knowledge_item(self, kb_id: str, content: str, metadata: Dict):
        """Add knowledge base item to vector database"""
        self.knowledge_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[kb_id]
        )
    
    def search_similar_tickets(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar tickets"""
        results = self.tickets_collection.query(
            query_texts=[query],
            n_results=n_results
        ) 
        return results #type: ignore
    
    def search_knowledge_base(self, query: str, n_results: int = 3) -> Dict:
        """Search knowledge base"""
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results #type: ignore