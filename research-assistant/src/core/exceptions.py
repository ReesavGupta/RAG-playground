"""
Custom exceptions for the research assistant.
"""


class ResearchAssistantError(Exception):
    """Base exception for research assistant errors"""
    pass


class SearchError(ResearchAssistantError):
    """Exception raised for search-related errors"""
    pass


class SynthesisError(ResearchAssistantError):
    """Exception raised for response synthesis errors"""
    pass


class ConfigurationError(ResearchAssistantError):
    """Exception raised for configuration errors"""
    pass


class DocumentProcessingError(ResearchAssistantError):
    """Exception raised for document processing errors"""
    pass


class WebSearchError(ResearchAssistantError):
    """Exception raised for web search errors"""
    pass 