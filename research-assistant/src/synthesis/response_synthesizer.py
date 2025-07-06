"""
Response synthesis using LLM for generating comprehensive answers.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.models import SearchResult, QueryResponse
from ..core.config import Config
from ..core.exceptions import SynthesisError

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """Generate comprehensive responses from multiple sources"""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize response synthesizer"""
        self.model_name = model_name or Config.OLLAMA_MODEL
        
        try:
            self.llm = OllamaLLM(model=self.model_name)
            
            # Create synthesis prompt
            self.synthesis_prompt = PromptTemplate(
                input_variables=["query", "sources"],
                template="""
Based on the following sources, provide a comprehensive answer to the question: {query}

Sources:
{sources}

Please synthesize the information from these sources to provide a well-structured, accurate response. 
Include proper citations using [Source X] format where X is the source number.
If sources conflict, note the discrepancy.
If information is incomplete, state what additional information would be helpful.

Answer:
"""
            )
            
            self.chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
            logger.info(f"ResponseSynthesizer initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResponseSynthesizer: {e}")
            raise SynthesisError(f"Failed to initialize synthesizer: {str(e)}")
    
    def synthesize_response(self, query: str, search_results: List[SearchResult]) -> QueryResponse:
        """Generate synthesized response from search results"""
        try:
            if not search_results:
                return QueryResponse(
                    response="No relevant sources found for your query.",
                    sources_used=0,
                    source_breakdown={'documents': 0, 'web': 0},
                    average_credibility=0.0,
                    response_time=0.0,
                    session_id="",
                    search_results=[]
                )
            
            # Prepare sources text
            sources_text = ""
            for i, result in enumerate(search_results[:10], 1):
                source_type = result.source_type.capitalize()
                sources_text += f"Source {i} ({source_type}): {result.content}\n"
                sources_text += f"Source: {result.source}\n"
                sources_text += f"Credibility: {result.credibility_score:.2f}\n\n"
            
            # Generate response using invoke instead of deprecated run method
            response = self.chain.invoke({"query": query, "sources": sources_text})
            
            # Calculate metrics
            source_breakdown = {
                'documents': len([r for r in search_results if r.source_type == 'document']),
                'web': len([r for r in search_results if r.source_type == 'web'])
            }
            
            avg_credibility = float(np.mean([r.credibility_score for r in search_results]))
            
            return QueryResponse(
                response=response,
                sources_used=len(search_results),
                source_breakdown=source_breakdown,
                average_credibility=avg_credibility,
                response_time=0.0,  # Will be set by caller
                session_id="",  # Will be set by caller
                search_results=search_results
            )
            
        except Exception as e:
            logger.error(f"Response synthesis error: {e}")
            error_response = f"Error generating response: {str(e)}"
            
            return QueryResponse(
                response=error_response,
                sources_used=0,
                source_breakdown={'documents': 0, 'web': 0},
                average_credibility=0.0,
                response_time=0.0,
                session_id="",
                search_results=[]
            )
    
    def test_model_connection(self) -> bool:
        """Test if the LLM model is accessible"""
        try:
            test_response = self.llm.invoke("Hello")
            return bool(test_response and len(str(test_response)) > 0)
        except Exception as e:
            logger.error(f"Model connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'connection_ok': self.test_model_connection()
        } 