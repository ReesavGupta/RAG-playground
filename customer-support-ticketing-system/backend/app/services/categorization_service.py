from typing import Dict, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

class TicketCategorizationService:
    def __init__(self):
        self.categories = {
            "shipping_issue": ["shipping", "delivery", "tracking", "package", "arrived", "delayed"],
            "payment_issue": ["payment", "charged", "refund", "billing", "credit card", "transaction"],
            "product_issue": ["defective", "broken", "damaged", "quality", "not working", "malfunction"],
            "return_refund": ["return", "refund", "exchange", "money back", "dissatisfied"],
            "account_issue": ["login", "password", "account", "access", "locked", "forgot"],
            "general_inquiry": ["question", "information", "help", "support", "how to"],
            "complaint": ["angry", "frustrated", "terrible", "worst", "disappointed", "complaint"]
        }
        
        self.priority_keywords = {
            "urgent": ["urgent", "emergency", "asap", "immediately", "critical"],
            "high": ["important", "priority", "soon", "quickly", "fast"],
            "medium": ["normal", "standard", "regular"],
            "low": ["whenever", "no rush", "low priority"]
        }
    
    def categorize_ticket(self, subject: str, description: str) -> Tuple[str, float]:
        """Categorize ticket based on content"""
        text = f"{subject} {description}".lower()
        
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score / len(keywords)
        
        if not category_scores:
            return "general_inquiry", 0.3
        
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        confidence = category_scores[best_category]
        
        return best_category, confidence
    
    def determine_priority(self, subject: str, description: str) -> str:
        """Determine ticket priority"""
        text = f"{subject} {description}".lower()
        
        for priority, keywords in self.priority_keywords.items():
            if any(keyword in text for keyword in keywords):
                return priority
        
        return "medium"
    
    def extract_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (replace with proper sentiment model)"""
        positive_words = ["happy", "satisfied", "good", "great", "excellent", "love", "perfect"]
        negative_words = ["angry", "frustrated", "terrible", "awful", "hate", "disappointed", "broken"]
        
        text = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)