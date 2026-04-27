"""
Central configuration for Smart Teaching Assistant.
Loads settings from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # App security
    APP_SECRET_KEY: str = os.getenv("APP_SECRET_KEY", "change-me-in-production")

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "vector_store")

    # Guardrails
    MIN_QUERY_LENGTH: int = 3
    MAX_QUERY_LENGTH: int = 1000

    # Demo credentials (replace with a real auth system in production)
    DEMO_USERNAME: str = os.getenv("DEMO_USERNAME", "admin")
    DEMO_PASSWORD: str = os.getenv("DEMO_PASSWORD", "password123")


settings = Settings()
