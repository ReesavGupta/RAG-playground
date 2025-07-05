"""
Configuration settings for the Customer Support Ticketing System
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = "your_password"
    
    # ChromaDB settings
    chroma_persist_directory: str = "./chroma_db"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    
    # JWT settings
    secret_key: str = "your_secret_key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # App settings
    debug: bool = True
    api_prefix: str = "/api"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Database URL
DATABASE_URL = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
