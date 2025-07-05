import re
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TextProcessor:
    def __init__(self) -> None:
        self.text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len, separators=[
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
            ","
            "",
            " ",
            ""
        ])
    
    def clean_text(self, text:str) -> str:
        """clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
         # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
        return text.strip()
    
    def chunk_document(self , text : str, metadata: Dict | None= None) -> List[Document]:
        """Chunk document using recursive character splitting"""
        cleaned_text = self.clean_text(text)
        chunks = self.text_splitter.split_text(cleaned_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                "chunk_id": i,
                "chunk_size": len(chunk),
                **(metadata or {})
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        # This is a simplified version - in production, use more sophisticated NLP
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:top_k]
