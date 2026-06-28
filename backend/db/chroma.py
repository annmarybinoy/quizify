"""
db/chroma.py - ChromaDB Client and Initialization

Handles all ChromaDB operations:
- Initializing the ChromaDB client
- Creating/getting collections
- Storing and retrieving embeddings
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from config import settings

# Global client instance — initialized once at startup
_chroma_client = None


def init_chroma() -> None:
    """
    Initialize ChromaDB client at app startup.
    Creates the local storage directory if it doesn't exist.
    Called once from main.py lifespan.
    """
    global _chroma_client

    try:
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(
                anonymized_telemetry=False  # disable usage tracking
            )
        )
        logger.info(f"ChromaDB initialized at path: {settings.CHROMA_DB_PATH}")

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns the ChromaDB client instance.
    Raises an error if called before init_chroma().
    """
    if _chroma_client is None:
        raise RuntimeError("ChromaDB not initialized. Call init_chroma() first.")
    return _chroma_client


def get_or_create_collection(doc_id: str) -> chromadb.Collection:
    """
    Gets an existing collection or creates a new one for a document.
    Each uploaded document gets its own isolated collection.

    Args:
        doc_id: Unique identifier for the document

    Returns:
        ChromaDB collection for that document
    """
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text embeddings
    )

    logger.debug(f"Got/created collection for doc_id: {doc_id}")
    return collection


def delete_collection(doc_id: str) -> None:
    """
    Deletes a document's collection from ChromaDB.
    Called when a quiz is deleted or expires.

    Args:
        doc_id: Unique identifier for the document to delete
    """
    client = get_chroma_client()

    try:
        client.delete_collection(name=f"doc_{doc_id}")
        logger.info(f"Deleted ChromaDB collection for doc_id: {doc_id}")
    except Exception as e:
        logger.warning(f"Could not delete collection for doc_id {doc_id}: {e}")