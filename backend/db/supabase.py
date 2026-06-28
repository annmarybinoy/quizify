"""
db/supabase.py - Supabase Client

Handles Supabase database and storage connections.
"""

from supabase import create_client, Client
from loguru import logger
from config import settings

# Global client instance
_supabase_client = None


def get_supabase_client() -> Client:
    """
    Returns the Supabase client instance.
    Creates it if it doesn't exist yet.
    """
    global _supabase_client

    if _supabase_client is None:
        try:
            _supabase_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SECRET_KEY
            )
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    return _supabase_client