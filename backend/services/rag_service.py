"""
services/rag_service.py - RAG Pipeline Service

Handles the complete RAG (Retrieval Augmented Generation) pipeline:
1. Chunking  — splits large text into smaller overlapping pieces
2. Embedding — converts text chunks into numbers using Gemini
3. Storing   — saves embeddings in ChromaDB with doc_id isolation
4. Retrieving — searches ChromaDB for most relevant chunks
"""

import uuid
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings
from db.chroma import get_or_create_collection, delete_collection

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)


# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500        # each chunk is ~500 words
CHUNK_OVERLAP = 50      # 50 word overlap between chunks so context isn't lost
TOP_K_CHUNKS = 10       # retrieve top 10 most relevant chunks for quiz generation
EMBEDDING_MODEL = "models/text-embedding-004"   # Gemini's free embedding model


# ── Step 1: Chunking ───────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Splits large text into smaller overlapping chunks.

    Why overlapping? If a sentence is split across two chunks,
    the overlap ensures the context isn't lost at boundaries.

    Args:
        text: Plain text extracted from PDF, image, or user input

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
        # tries to split at paragraph breaks first,
        # then newlines, then sentences, then words
    )

    chunks = splitter.split_text(text)

    logger.debug(f"Text split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Step 2: Embedding ──────────────────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts text chunks into embeddings (lists of numbers)
    using Gemini's text-embedding-004 model.

    Args:
        chunks: List of text chunks

    Returns:
        List of embeddings — each embedding is a list of floats
    """
    logger.info(f"Embedding {len(chunks)} chunks using Gemini...")

    embeddings = []

    for i, chunk in enumerate(chunks):
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunk,
            task_type="retrieval_document",  # optimized for storage/retrieval
        )
        embeddings.append(result["embedding"])

        if (i + 1) % 10 == 0:
            logger.debug(f"Embedded {i + 1}/{len(chunks)} chunks")

    logger.info(f"Successfully embedded {len(chunks)} chunks")
    return embeddings


# ── Step 3: Storing ────────────────────────────────────────────────────────────
def store_chunks(
    text: str,
    doc_id: str | None = None
) -> str:
    """
    Full pipeline: chunks text, embeds it, and stores in ChromaDB.
    This is the main function called after parsing a document.

    Args:
        text: Plain text to process
        doc_id: Optional existing doc_id. If None, generates a new one.

    Returns:
        doc_id — the unique identifier for this document's vectors
    """
    # Generate a unique doc_id if not provided
    if doc_id is None:
        doc_id = str(uuid.uuid4()).replace("-", "")[:16]

    logger.info(f"Starting RAG indexing for doc_id: {doc_id}")

    # Step 1 — chunk
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No chunks generated from text. Text may be too short.")

    # Step 2 — embed
    embeddings = embed_chunks(chunks)

    # Step 3 — store in ChromaDB
    collection = get_or_create_collection(doc_id)

    # Prepare data for ChromaDB
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    logger.info(f"Stored {len(chunks)} chunks in ChromaDB for doc_id: {doc_id}")
    return doc_id


# ── Step 4: Retrieving ─────────────────────────────────────────────────────────
def retrieve_chunks(
    query: str,
    doc_id: str,
    top_k: int = TOP_K_CHUNKS
) -> list[str]:
    """
    Searches ChromaDB for the most relevant chunks for a given query.
    Used during quiz generation to get the best context from the document.

    Args:
        query: The search query (usually the quiz topic or document title)
        doc_id: Which document to search in
        top_k: How many chunks to retrieve

    Returns:
        List of the most relevant text chunks
    """
    logger.info(f"Retrieving top {top_k} chunks for doc_id: {doc_id}")

    # Embed the query using the same model
    # task_type is "retrieval_query" for search queries
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )["embedding"]

    # Search ChromaDB for similar chunks
    collection = get_or_create_collection(doc_id)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # don't request more than available
        include=["documents", "distances"]
    )

    chunks = results["documents"][0]  # list of matching text chunks
    distances = results["distances"][0]  # similarity scores

    logger.info(f"Retrieved {len(chunks)} chunks (best similarity: {1 - distances[0]:.3f})")
    return chunks


# ── Cleanup ────────────────────────────────────────────────────────────────────
def delete_document_vectors(doc_id: str) -> None:
    """
    Deletes all vectors for a document from ChromaDB.
    Called when a quiz is deleted or expires.

    Args:
        doc_id: Document whose vectors should be deleted
    """
    logger.info(f"Deleting vectors for doc_id: {doc_id}")
    delete_collection(doc_id)
    logger.info(f"Vectors deleted for doc_id: {doc_id}")