from typing import List, Dict, Optional
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os

class RAGService:
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.3,
            max_tokens=500
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Custom prompt template for support responses
        self.prompt_template = PromptTemplate(
            template="""
            You are a helpful customer support agent. Based on the following context from previous tickets and knowledge base, 
            provide a helpful response to the customer's inquiry.
            
            Context:
            {context}
            
            Customer Inquiry:
            {question}
            
            Instructions:
            1. Be empathetic and professional
            2. Provide specific solutions when possible
            3. If you need more information, ask relevant questions
            4. If the issue requires human intervention, mention escalation
            5. Keep the response concise but comprehensive
            
            Response:
            """,
            input_variables=["context", "question"]
        )
    
    def generate_response(self, 
                         ticket_content: str, 
                         similar_tickets: List[Dict], 
                         knowledge_items: List[Dict]) -> Dict:
        """Generate response using RAG pipeline"""
        
        # Combine context from similar tickets and knowledge base
        context_parts = []
        
        # Add similar ticket resolutions
        for ticket in similar_tickets:
            if ticket.get('resolution'):
                context_parts.append(f"Similar issue resolution: {ticket['resolution']}")
        
        # Add knowledge base information
        for kb_item in knowledge_items:
            context_parts.append(f"Knowledge base: {kb_item['content']}")
        
        context = "\n".join(context_parts)
        
        # Generate response using the LLM
        response = self.llm(
            self.prompt_template.format(
                context=context,
                question=ticket_content
            )
        )
        
        # Calculate confidence score based on context quality
        confidence = self._calculate_confidence(similar_tickets, knowledge_items)
        
        return {
            "response": response.strip(),
            "confidence": confidence,
            "sources": {
                "similar_tickets": len(similar_tickets),
                "knowledge_items": len(knowledge_items)
            }
        }
    
    def _calculate_confidence(self, similar_tickets: List[Dict], knowledge_items: List[Dict]) -> float:
        """Calculate confidence score for the response"""
        base_confidence = 0.4
        
        # Boost confidence based on similar tickets
        if similar_tickets:
            ticket_boost = min(0.3, len(similar_tickets) * 0.1)
            base_confidence += ticket_boost
        
        # Boost confidence based on knowledge base matches
        if knowledge_items:
            kb_boost = min(0.3, len(knowledge_items) * 0.1)
            base_confidence += kb_boost
        
        return min(base_confidence, 1.0)