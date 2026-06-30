"""
services/rag_service.py - RAG Pipeline Service

Handles the complete RAG (Retrieval Augmented Generation) pipeline:
1. Chunking  — splits large text into smaller overlapping pieces
2. Embedding — converts text chunks into numbers using Gemini
3. Storing   — saves embeddings in ChromaDB with doc_id isolation
4. Retrieving — searches ChromaDB for most relevant chunks
"""

import uuid
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings
from db.chroma import get_or_create_collection, delete_collection

# Initialize Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

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
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk,
            config={"task_type": "retrieval_document"},
        )
        embeddings.append(result.embeddings[0].values)

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
# ── Step 5: Retrieval ──────────────────────────────────────────────────────────
def retrieve_chunks_for_quiz(
    doc_id: str,
    text: str,
    specific_topic: str | None = None,
) -> list[str]:
    """
    Retrieves relevant chunks based on retrieval strategy:

    Strategy 1 — specific_topic provided:
        User wants quiz focused on a specific topic.
        Retrieve chunks most relevant to that topic only.

    Strategy 2 — no specific_topic:
        User wants quiz from whole document.
        Extract key topics first, retrieve chunks per topic,
        combine for full document coverage.

    Args:
        doc_id: Which document to search in
        text: Full document text (used for topic extraction)
        specific_topic: Optional topic to focus on

    Returns:
        List of relevant chunks
    """
    collection = get_or_create_collection(doc_id)
    total_chunks = collection.count()

    # ── Strategy 1: Specific topic retrieval ───────────────────────────────
    if specific_topic:
        logger.info(f"Strategy: specific topic retrieval → '{specific_topic}'")

        query_embedding = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=topic,
            config={"task_type": "retrieval_query"},
        ).embeddings[0].values

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10, total_chunks),
            include=["documents"]
        )

        chunks = results["documents"][0]
        logger.info(f"Retrieved {len(chunks)} chunks for topic: '{specific_topic}'")
        return chunks

    # ── Strategy 2: Full document coverage ─────────────────────────────────
    logger.info("Strategy: full document coverage via topic extraction")

    topics = extract_key_topics(text)

    all_chunks = []
    seen_chunks = set()

    for topic in topics:
        logger.debug(f"Retrieving chunks for topic: '{topic}'")

        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=topic,
            task_type="retrieval_query",
        )["embedding"]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K_PER_TOPIC, total_chunks),
            include=["documents"]
        )

        for chunk in results["documents"][0]:
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                all_chunks.append(chunk)

    logger.info(
        f"Retrieved {len(all_chunks)} unique chunks "
        f"across {len(topics)} topics for doc_id: {doc_id}"
    )
    return all_chunks

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