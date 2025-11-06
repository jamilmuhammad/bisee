import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Allow extra environment variables (backwards-compat with older .env keys)
    model_config = {"extra": "ignore"}
    # LLM & Embeddings
    GROQ_API_KEY: str
    MODEL_NAME_FOR_EMBEDDING: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    LLM_MODEL_NAME_GROQ: str = "llama-3.3-70b-versatile"

    # Langsmith
    # LANGSMITH_TRACING: bool = False
    # LANGSMITH_ENDPOINT: str
    # LANGSMITH_API_KEY: str
    # LANGSMITH_PROJECT: str

    # MongoDB
    # Make Mongo optional at import time; real deployments should set this.
    MONGODB_URI: Optional[str] = None
    MONGODB_DATABASE_NAME: str = "halobol"
    MONGODB_USERS_COLLECTION: str = "db_bisee.users"
    MONGODB_CHUNKS_COLLECTION: str = "db_bisee.chunks"
    MONGODB_ATLAS_VECTOR_INDEX_NAME: str = "db_bisee.vector_index"
    MONGODB_CHAT_SESSIONS_COLLECTION: str = "db_bisee.sessions"

    # JWT Authentication
    JWT_SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Google OAuth
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    # PostgreSQL Database (for RAG chatbot)
    POSTGRES_URL: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

    # Pydantic v2 configuration
    # Keep .env loading behavior and ignore extra keys from legacy env files
    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()

# Basic Logging Setup (can be more sophisticated)
import logging

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validate critical settings
if not settings.JWT_SECRET_KEY:
    logger.critical("JWT_SECRET_KEY not set. Authentication will fail.")
if not settings.GOOGLE_CLIENT_ID:
    logger.critical("GOOGLE_CLIENT_ID not set. Google Sign-In verification will fail.")
