"""
config.py - Application Configuration

Manages all environment variables using Pydantic Settings.
Pydantic validates every variable at startup — if a required
variable is missing, the app fails immediately with a clear
error instead of breaking mysteriously later.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    All environment variables for the application.
    Values are read from the .env file automatically.
    """

    # ── App ────────────────────────────────────────────────────────────────
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # ── Gemini ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str

    # ── Supabase ───────────────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SECRET_KEY: str
    SUPABASE_PUBLISHABLE_KEY: str

    # ── Vector Store ───────────────────────────────────────────────────────
    VECTOR_STORE: str = "chroma"
    CHROMA_DB_PATH: str = "./chroma_db"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "quizify"

    # ── CORS ───────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ── Quiz Generation ────────────────────────────────────────────────────
    DEFAULT_QUESTION_COUNT: int = 10
    MAX_QUESTION_COUNT: int = 30
    DEFAULT_DIFFICULTY: str = "medium"

    # ── File Upload ────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Single instance used across the entire app
settings = Settings()