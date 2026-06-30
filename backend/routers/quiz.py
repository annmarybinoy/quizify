"""
routers/quiz.py - Quiz Generation API Endpoints

Three quiz creation paths:
- POST /api/quiz/create/document → file upload (PDF or image)
- POST /api/quiz/create/text     → raw text input
- POST /api/quiz/create/topic    → topic only, no document

Plus:
- GET    /api/quiz/{quiz_id}  → retrieve a quiz
- DELETE /api/quiz/{quiz_id}  → delete a quiz and its vectors
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from typing import Optional
from datetime import datetime
import json

from services import parser_service, rag_service, quiz_service
from services.quiz_service import Difficulty, QuestionType
from models.quiz import (
    QuizFromTopicRequest,
    RawTextRequest,
    QuizResponse,
    QuestionResponse,
    ErrorResponse,
)
from db.supabase import get_supabase_client
from config import settings

router = APIRouter()


# ── Helper ─────────────────────────────────────────────────────────────────────
def _format_quiz_response(quiz_data: dict, doc_id: Optional[str] = None) -> QuizResponse:
    """
    Converts raw quiz dictionary from quiz_service
    into a structured QuizResponse Pydantic model.

    Args:
        quiz_data: Raw dictionary from quiz_service
        doc_id: Optional document ID if quiz was from a document

    Returns:
        QuizResponse Pydantic model
    """
    questions = [
        QuestionResponse(
            id=q["id"],
            type=q["type"],
            question=q["question"],
            options=q.get("options", {}),
            correct_answer=q["correct_answer"],
            explanation=q["explanation"],
        )
        for q in quiz_data["questions"]
    ]

    return QuizResponse(
        quiz_id=quiz_data["quiz_id"],
        title=quiz_data["title"],
        topic=quiz_data["topic"],
        difficulty=quiz_data["difficulty"],
        input_type=quiz_data["input_type"],
        question_count=quiz_data["question_count"],
        questions=questions,
        doc_id=doc_id,
        created_at=datetime.utcnow(),
    )


async def _save_quiz_to_supabase(quiz_response: QuizResponse, doc_id: Optional[str] = None) -> None:
    """
    Saves generated quiz to Supabase for later retrieval.
    Fails silently with a warning — quiz is still returned
    to user even if saving fails.

    Args:
        quiz_response: The formatted quiz response
        doc_id: Optional document ID
    """
    try:
        supabase = get_supabase_client()

        quiz_dict = quiz_response.model_dump()
        quiz_dict["created_at"] = quiz_dict["created_at"].isoformat()
        quiz_dict["questions"] = json.dumps(quiz_dict["questions"])

        supabase.table("quizzes").insert(quiz_dict).execute()
        logger.info(f"Quiz saved to Supabase: {quiz_response.quiz_id}")

    except Exception as e:
        # Don't fail the request if saving fails
        # User gets their quiz regardless
        logger.warning(f"Failed to save quiz to Supabase: {e}")


# ── Endpoint 1: Document Upload ────────────────────────────────────────────────
@router.post(
    "/create/document",
    response_model=QuizResponse,
    summary="Generate quiz from PDF or image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    }
)
async def create_quiz_from_document(
    file: UploadFile = File(..., description="PDF or image file to generate quiz from"),
    num_questions: int = Form(default=10, ge=1, le=30),
    difficulty: Difficulty = Form(default=Difficulty.MEDIUM),
    question_type: QuestionType = Form(default=QuestionType.MIXED),
    specific_topic: Optional[str] = Form(default=None),
):
    """
    Generate a quiz from an uploaded PDF or image file.

    The file is parsed, chunked, embedded into ChromaDB via RAG,
    and then Gemini generates questions strictly from the document content.
    """
    logger.info(f"Quiz generation requested from document: {file.filename}")

    # Step 1 — Parse file based on type
    if file.content_type == "application/pdf":
        extracted_text = await parser_service.parse_pdf(file)
    elif file.content_type in ["image/jpeg", "image/png", "image/webp"]:
        extracted_text = await parser_service.parse_image(file)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or image files."
        )

    # Step 2 — RAG: chunk, embed, store
    doc_id = rag_service.store_chunks(extracted_text)

    # Step 3 — RAG: retrieve relevant chunks
    chunks = rag_service.retrieve_chunks_for_quiz(
        doc_id=doc_id,
        text=extracted_text,
        specific_topic=specific_topic,
    )

    # Step 4 — Generate quiz from chunks
    quiz_data = quiz_service.generate_quiz_from_document(
        chunks=chunks,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
        specific_topic=specific_topic,
    )

    # Step 5 — Format and save
    quiz_response = _format_quiz_response(quiz_data, doc_id=doc_id)
    await _save_quiz_to_supabase(quiz_response, doc_id=doc_id)

    logger.info(f"Quiz created successfully: {quiz_response.quiz_id}")
    return quiz_response


# ── Endpoint 2: Raw Text ───────────────────────────────────────────────────────
@router.post(
    "/create/text",
    response_model=QuizResponse,
    summary="Generate quiz from pasted text",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid text"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    }
)
async def create_quiz_from_text(request: RawTextRequest):
    """
    Generate a quiz from raw text pasted by the user.

    Text is validated, chunked, embedded into ChromaDB via RAG,
    and Gemini generates questions strictly from the provided text.
    """
    logger.info(f"Quiz generation requested from raw text ({len(request.text)} chars)")

    # Step 1 — Validate text
    validated_text = await parser_service.parse_text(request.text)

    # Step 2 — RAG: chunk, embed, store
    doc_id = rag_service.store_chunks(validated_text)

    # Step 3 — RAG: retrieve relevant chunks
    chunks = rag_service.retrieve_chunks_for_quiz(
        doc_id=doc_id,
        text=validated_text,
        specific_topic=request.specific_topic,
    )

    # Step 4 — Generate quiz from chunks
    quiz_data = quiz_service.generate_quiz_from_document(
        chunks=chunks,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        question_type=request.question_type,
        specific_topic=request.specific_topic,
    )

    # Step 5 — Format and save
    quiz_response = _format_quiz_response(quiz_data, doc_id=doc_id)
    await _save_quiz_to_supabase(quiz_response, doc_id=doc_id)

    logger.info(f"Quiz created successfully: {quiz_response.quiz_id}")
    return quiz_response


# ── Endpoint 3: Topic Only ─────────────────────────────────────────────────────
@router.post(
    "/create/topic",
    response_model=QuizResponse,
    summary="Generate quiz from a topic",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid topic"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    }
)
async def create_quiz_from_topic(request: QuizFromTopicRequest):
    """
    Generate a quiz from a topic using Gemini's own knowledge.
    No file upload or RAG involved — pure LLM generation.
    """
    logger.info(f"Quiz generation requested for topic: '{request.topic}'")

    # No RAG — direct Gemini call
    quiz_data = quiz_service.generate_quiz_from_topic(
        topic=request.topic,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        question_type=request.question_type,
    )

    # Format and save
    quiz_response = _format_quiz_response(quiz_data)
    await _save_quiz_to_supabase(quiz_response)

    logger.info(f"Quiz created successfully: {quiz_response.quiz_id}")
    return quiz_response


# ── Endpoint 4: Get Quiz ───────────────────────────────────────────────────────
@router.get(
    "/{quiz_id}",
    response_model=QuizResponse,
    summary="Get a quiz by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Quiz not found"},
    }
)
async def get_quiz(quiz_id: str):
    """
    Retrieve a previously generated quiz by its ID.
    """
    logger.info(f"Fetching quiz: {quiz_id}")

    try:
        supabase = get_supabase_client()
        result = supabase.table("quizzes").select("*").eq("quiz_id", quiz_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Quiz with ID '{quiz_id}' not found."
            )

        quiz_data = result.data[0]
        quiz_data["questions"] = json.loads(quiz_data["questions"])

        return QuizResponse(**quiz_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch quiz {quiz_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quiz."
        )


# ── Endpoint 5: Delete Quiz ────────────────────────────────────────────────────
@router.delete(
    "/{quiz_id}",
    summary="Delete a quiz and its vectors",
    responses={
        404: {"model": ErrorResponse, "description": "Quiz not found"},
        500: {"model": ErrorResponse, "description": "Deletion failed"},
    }
)
async def delete_quiz(quiz_id: str):
    """
    Deletes a quiz from Supabase and removes its vectors from ChromaDB.
    This frees up vector storage for document-based quizzes.
    """
    logger.info(f"Deleting quiz: {quiz_id}")

    try:
        supabase = get_supabase_client()

        # First check quiz exists and get doc_id
        result = supabase.table("quizzes").select("doc_id").eq("quiz_id", quiz_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Quiz with ID '{quiz_id}' not found."
            )

        doc_id = result.data[0].get("doc_id")

        # Delete from Supabase
        supabase.table("quizzes").delete().eq("quiz_id", quiz_id).execute()
        logger.info(f"Quiz deleted from Supabase: {quiz_id}")

        # Delete vectors from ChromaDB if document-based quiz
        if doc_id:
            rag_service.delete_document_vectors(doc_id)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Quiz {quiz_id} deleted successfully.",
                "vectors_deleted": doc_id is not None,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete quiz {quiz_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete quiz."
        )