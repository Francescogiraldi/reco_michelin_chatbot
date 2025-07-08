"""Configuration module for Michelin Chatbot."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation."""

    # OpenAI Configuration
    openai_api_key: str = Field(default="demo_key", description="OpenAI API key")
    embedding_model: str = Field(
        default="text-embedding-3-large", env="EMBEDDING_MODEL"
    )
    chat_model: str = Field(default="gpt-4o-mini", env="CHAT_MODEL")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, env="TEMPERATURE")

    # RAG Configuration
    top_k: int = Field(default=4, ge=1, le=20, env="TOP_K")
    chunk_size: int = Field(default=1000, ge=100, le=4000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, env="CHUNK_OVERLAP")

    # File Paths
    catalog_path: Path = Field(default=Path("catalog.csv"), env="CATALOG_PATH")
    index_dir: Path = Field(default=Path("faiss_index"), env="INDEX_DIR")
    logs_dir: Path = Field(default=Path("logs"), env="LOGS_DIR")

    # Streamlit Configuration
    streamlit_host: str = Field(default="localhost", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, ge=1024, le=65535, env="STREAMLIT_PORT")

    # Application Configuration
    app_title: str = Field(default="Michelin Tire Recommendation Chatbot", env="APP_TITLE")
    app_description: str = Field(
        default="AI-powered assistant for Michelin tire recommendations",
        env="APP_DESCRIPTION"
    )
    max_chat_history: int = Field(default=50, ge=1, le=200, env="MAX_CHAT_HISTORY")
    
    # Language and Localization
    default_language: str = Field(default="fr", env="DEFAULT_LANGUAGE")
    language: str = Field(default="en", env="LANGUAGE")
    supported_languages: list[str] = Field(
        default=["fr", "en", "it", "es", "de"], env="SUPPORTED_LANGUAGES"
    )

    # Security
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    max_requests_per_minute: int = Field(default=30, ge=1, le=1000, env="MAX_REQUESTS_PER_MINUTE")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        env="LOG_FORMAT"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __post_init__(self):
        """Post-initialization setup."""
        # Create necessary directories
        self.logs_dir.mkdir(exist_ok=True)
        
        # Validate OpenAI API key format (allow demo keys)
        if self.openai_api_key != "demo_key" and not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def update_setting(key: str, value: any) -> None:
    """Update a specific setting."""
    if hasattr(settings, key):
        setattr(settings, key, value)
    else:
        raise ValueError(f"Setting '{key}' does not exist")