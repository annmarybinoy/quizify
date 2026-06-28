"""
services/parser_service.py - Content Parser Service

Handles extraction of plain text from different input types:
- PDF files (using PyMuPDF)
- Image files (using Gemini Vision)
- Raw text (validation only)

All three return plain text that feeds into the RAG pipeline.
"""

import magic
import pymupdf
import google.generativeai as genai
from PIL import Image
from loguru import logger
from fastapi import UploadFile, HTTPException
import io

from config import settings

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024
MIN_TEXT_LENGTH = 50  # minimum characters to generate a quiz from

# Magic bytes for file type verification
MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
}


# ── Validation ─────────────────────────────────────────────────────────────────
async def validate_file(file: UploadFile) -> bytes:
    """
    Validates an uploaded file before processing.
    Checks file size, MIME type, and actual file signature (magic bytes).

    Args:
        file: The uploaded file from FastAPI

    Returns:
        file contents as bytes if valid

    Raises:
        HTTPException: If file fails any validation check
    """
    # Read file contents into memory
    contents = await file.read()

    # Check 1 — file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    # Check 2 — claimed MIME type
    if file.content_type not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' is not allowed. Allowed types: PDF, JPEG, PNG, WEBP."
        )

    # Check 3 — actual file signature (magic bytes)
    # Prevents renamed malicious files from slipping through
    is_valid_signature = False
    for magic_bytes, mime_type in MAGIC_BYTES.items():
        if contents.startswith(magic_bytes):
            is_valid_signature = True
            # Also verify signature matches claimed type
            if mime_type != file.content_type:
                raise HTTPException(
                    status_code=400,
                    detail="File content does not match its claimed type."
                )
            break

    if not is_valid_signature:
        raise HTTPException(
            status_code=400,
            detail="File signature is invalid or unrecognized."
        )

    logger.debug(f"File validated: {file.filename} ({file.content_type}, {len(contents)} bytes)")
    return contents


def validate_text(text: str) -> str:
    """
    Validates raw text input before processing.

    Args:
        text: Raw text from user

    Returns:
        Stripped text if valid

    Raises:
        HTTPException: If text is empty or too short
    """
    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text content cannot be empty."
        )

    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too short to generate a quiz from. Please provide at least {MIN_TEXT_LENGTH} characters."
        )

    return text


# ── Parsers ────────────────────────────────────────────────────────────────────
async def parse_pdf(file: UploadFile) -> str:
    """
    Extracts plain text from a PDF file using PyMuPDF.

    Args:
        file: Uploaded PDF file

    Returns:
        Extracted plain text from all pages
    """
    logger.info(f"Parsing PDF: {file.filename}")

    # Validate first
    contents = await validate_file(file)

    try:
        # Open PDF from bytes in memory (no temp file needed)
        pdf_document = pymupdf.open(stream=contents, filetype="pdf")

        extracted_text = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()

            if text.strip():  # only add non-empty pages
                extracted_text.append(f"[Page {page_num + 1}]\n{text}")

        pdf_document.close()

        full_text = "\n\n".join(extracted_text)

        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The file may be scanned or image-based."
            )

        logger.info(f"PDF parsed successfully: {len(full_text)} characters extracted from {len(pdf_document)} pages")
        return full_text

    except HTTPException:
        raise  # re-raise our own exceptions as is
    except Exception as e:
        logger.error(f"Failed to parse PDF {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse PDF. Please ensure the file is not corrupted."
        )


async def parse_image(file: UploadFile) -> str:
    """
    Extracts text and content description from an image using Gemini Vision.

    Args:
        file: Uploaded image file

    Returns:
        Text description and content extracted from the image
    """
    logger.info(f"Parsing image: {file.filename}")

    # Validate first
    contents = await validate_file(file)

    try:
        # Open image using Pillow
        image = Image.open(io.BytesIO(contents))

        # Initialize Gemini Vision model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Prompt Gemini to extract all content from the image
        prompt = """
        You are a content extractor. Analyze this image and extract ALL text, 
        information, and educational content from it.
        
        If it contains:
        - Handwritten or printed text → transcribe it exactly
        - Diagrams or charts → describe them in detail
        - Tables → reproduce them as text
        - Equations or formulas → write them out
        - Any other educational content → describe it thoroughly
        
        Return ONLY the extracted content with no additional commentary.
        The output will be used to generate quiz questions, so be thorough and accurate.
        """

        response = model.generate_content([prompt, image])
        extracted_text = response.text

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any content from the image."
            )

        logger.info(f"Image parsed successfully: {len(extracted_text)} characters extracted")
        return extracted_text

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse image {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze image. Please try again."
        )


async def parse_text(text: str) -> str:
    """
    Validates and returns raw text input.

    Args:
        text: Raw text from user

    Returns:
        Validated and cleaned text
    """
    logger.info("Parsing raw text input")
    validated_text = validate_text(text)
    logger.info(f"Text validated: {len(validated_text)} characters")
    return validated_text