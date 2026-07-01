"""
services/quiz_service.py - Quiz Generation Service

Handles all quiz generation logic:
- Document-based quiz generation (uses RAG chunks as context)
- Topic-based quiz generation (uses Gemini's own knowledge)
- Structured output parsing and validation
- Retry logic for unreliable LLM responses

All generation paths return the same QuizOutput structure
so the rest of the app doesn't need to know which path was taken.
"""

import json
import re
import uuid
from enum import Enum
from typing import Optional
from loguru import logger
from google import genai
from config import settings

# Initialize Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.5-flash"
MAX_RETRIES = 3


# ── Enums ──────────────────────────────────────────────────────────────────────
class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    MIXED = "mixed"


# ── Prompt Builders ────────────────────────────────────────────────────────────
def _build_difficulty_instruction(difficulty: Difficulty) -> str:
    """
    Returns difficulty-specific instructions for the prompt.
    Different difficulties change the cognitive level of questions.
    """
    instructions = {
        Difficulty.EASY: """
            - Ask straightforward factual questions
            - Use simple, clear language
            - Wrong options should be obviously incorrect
            - Focus on basic definitions and simple facts
        """,
        Difficulty.MEDIUM: """
            - Mix factual and conceptual questions
            - Wrong options should be plausible but clearly wrong with understanding
            - Include some questions that require connecting two concepts
            - Avoid trick questions but make students think
        """,
        Difficulty.HARD: """
            - Focus on deep conceptual understanding and application
            - Wrong options should be common misconceptions
            - Include scenario-based questions
            - Questions should require synthesis of multiple concepts
            - Avoid pure memorization questions
        """
    }
    return instructions[difficulty]


def _build_question_type_instruction(question_type: QuestionType, num_questions: int) -> str:
    """
    Returns question type distribution instructions.
    For MIXED, distributes questions across all types.
    """
    if question_type == QuestionType.MCQ:
        return f"Generate exactly {num_questions} multiple choice questions (MCQ) with 4 options each (A, B, C, D)."

    elif question_type == QuestionType.TRUE_FALSE:
        return f"Generate exactly {num_questions} true/false questions."

    elif question_type == QuestionType.SHORT_ANSWER:
        return f"Generate exactly {num_questions} short answer questions."

    elif question_type == QuestionType.MIXED:
        # Calculate distribution
        mcq_count = max(1, int(num_questions * 0.6))           # 60% MCQ
        tf_count = max(1, int(num_questions * 0.25))            # 25% True/False
        sa_count = max(1, num_questions - mcq_count - tf_count) # remaining Short Answer

        return f"""Generate exactly {num_questions} questions with this distribution:
        - {mcq_count} multiple choice questions (MCQ) with 4 options each (A, B, C, D)
        - {tf_count} true/false questions
        - {sa_count} short answer questions
        """


def _build_output_format_instruction() -> str:
    """
    Returns the exact JSON format Gemini must follow.
    Being extremely specific here reduces parsing failures.
    """
    return """
    Return ONLY a valid JSON object. No markdown, no backticks, no explanation.
    Follow this EXACT structure:

    {
        "title": "Quiz title based on the content",
        "topic": "Main topic of the quiz",
        "questions": [
            {
                "id": 1,
                "type": "mcq",
                "question": "Question text here?",
                "options": {
                    "A": "First option",
                    "B": "Second option", 
                    "C": "Third option",
                    "D": "Fourth option"
                },
                "correct_answer": "B",
                "explanation": "Why this answer is correct"
            },
            {
                "id": 2,
                "type": "true_false",
                "question": "Statement to evaluate?",
                "options": {
                    "A": "True",
                    "B": "False"
                },
                "correct_answer": "A",
                "explanation": "Why this is true/false"
            },
            {
                "id": 3,
                "type": "short_answer",
                "question": "Question requiring a short answer?",
                "options": {},
                "correct_answer": "The expected answer",
                "explanation": "Explanation of the answer"
            }
        ]
    }

    Rules:
    - Every question MUST have: id, type, question, options, correct_answer, explanation
    - MCQ correct_answer must be the letter (A, B, C, or D)
    - True/false correct_answer must be A (True) or B (False)
    - Short answer correct_answer is the expected answer text
    - Explanations must be clear and educational
    - Do NOT include any text outside the JSON object
    """


def _build_document_prompt(
    chunks: list[str],
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
    specific_topic: Optional[str] = None,
) -> str:
    """
    Builds prompt for document-based quiz generation.
    Instructs Gemini to generate questions ONLY from provided chunks.
    """
    context = "\n\n---\n\n".join(chunks)

    topic_instruction = (
        f"Focus ONLY on the topic: '{specific_topic}'"
        if specific_topic
        else "Cover all key topics present in the content"
    )

    difficulty_instruction = _build_difficulty_instruction(difficulty)
    type_instruction = _build_question_type_instruction(question_type, num_questions)
    format_instruction = _build_output_format_instruction()

    prompt = f"""
    You are an expert quiz generator. Generate a quiz based STRICTLY on the 
    provided content below. Do not use any outside knowledge.
    If the content doesn't have enough information for a question, skip it.

    CONTENT:
    {context}

    INSTRUCTIONS:
    - {topic_instruction}
    - Difficulty level: {difficulty.value}
    {difficulty_instruction}
    - {type_instruction}

    OUTPUT FORMAT:
    {format_instruction}
    """

    return prompt


def _build_topic_prompt(
    topic: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> str:
    """
    Builds prompt for topic-based quiz generation.
    Gemini uses its own knowledge — no document context needed.
    """
    difficulty_instruction = _build_difficulty_instruction(difficulty)
    type_instruction = _build_question_type_instruction(question_type, num_questions)
    format_instruction = _build_output_format_instruction()

    prompt = f"""
    You are an expert quiz generator. Generate a high quality quiz about: {topic}

    Use your knowledge to create accurate, educational questions.

    INSTRUCTIONS:
    - Topic: {topic}
    - Difficulty level: {difficulty.value}
    {difficulty_instruction}
    - {type_instruction}
    - Ensure questions are factually accurate
    - Cover different aspects of the topic

    OUTPUT FORMAT:
    {format_instruction}
    """

    return prompt


# ── Response Parser ────────────────────────────────────────────────────────────
def _parse_and_validate_response(response_text: str, num_questions: int) -> dict:
    """
    Parses and validates Gemini's JSON response.
    Handles common LLM output issues like markdown backticks.

    Args:
        response_text: Raw text response from Gemini
        num_questions: Expected number of questions for validation

    Returns:
        Validated quiz dictionary

    Raises:
        ValueError: If response cannot be parsed or fails validation
    """
    # Clean up common LLM formatting issues
    cleaned = response_text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    # Parse JSON
    try:
        quiz_data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Gemini: {e}")

    # Validate required top-level fields
    required_fields = ["title", "topic", "questions"]
    for field in required_fields:
        if field not in quiz_data:
            raise ValueError(f"Missing required field: '{field}'")

    # Validate questions
    questions = quiz_data["questions"]

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Questions must be a non-empty list")

    if len(questions) < num_questions * 0.7:
        # Allow up to 30% fewer questions than requested
        # sometimes content isn't rich enough for exact count
        raise ValueError(
            f"Too few questions generated: got {len(questions)}, expected {num_questions}"
        )

    # Validate each question
    for i, q in enumerate(questions):
        required_q_fields = ["id", "type", "question", "options", "correct_answer", "explanation"]
        for field in required_q_fields:
            if field not in q:
                raise ValueError(f"Question {i+1} missing field: '{field}'")

        # Validate correct_answer for MCQ
        if q["type"] == "mcq" and q["correct_answer"] not in ["A", "B", "C", "D"]:
            raise ValueError(f"Question {i+1} MCQ correct_answer must be A, B, C, or D")

        # Validate correct_answer for true/false
        if q["type"] == "true_false" and q["correct_answer"] not in ["A", "B"]:
            raise ValueError(f"Question {i+1} true_false correct_answer must be A or B")

    logger.info(f"Quiz validated: {len(questions)} questions, title: '{quiz_data['title']}'")
    return quiz_data


# ── Main Generation Functions ──────────────────────────────────────────────────
def _call_gemini_with_retry(prompt: str, num_questions: int) -> dict:
    """
    Calls Gemini with error-guided retry logic.
    On each retry, tells Gemini exactly what was wrong with
    its previous response so it can correct itself.

    Args:
        prompt: The complete prompt to send
        num_questions: Expected number of questions for validation

    Returns:
        Validated quiz dictionary

    Raises:
        RuntimeError: If all retries are exhausted
    """
    

    last_error = None
    last_response = None
    current_prompt = prompt

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Calling Gemini for quiz generation (attempt {attempt}/{MAX_RETRIES})")

            response = client.interactions.create(
                model=GEMINI_MODEL,
                input=current_prompt,
            )
            last_response = response.output_text
            quiz_data = _parse_and_validate_response(response.output_text, num_questions)

            logger.info(f"Quiz generated successfully on attempt {attempt}")
            return quiz_data

        except ValueError as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed validation: {e}")

            if attempt < MAX_RETRIES:
                # Build error-guided retry prompt
                # Tell Gemini exactly what went wrong and show its previous response
                current_prompt = f"""
                {prompt}

                ---
                CORRECTION NEEDED:
                Your previous response had the following error:
                ERROR: {str(e)}

                Your previous response was:
                {last_response}

                Please fix this specific error and return a valid JSON response.
                Remember: Return ONLY valid JSON, nothing else.
                """
                logger.info(f"Retrying with error context: {e}")
            continue

        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt} failed with unexpected error: {e}")

            if attempt < MAX_RETRIES:
                current_prompt = f"""
                {prompt}

                ---
                CORRECTION NEEDED:
                Your previous response caused an unexpected error: {str(e)}
                Please try again and return ONLY valid JSON.
                """
                logger.info("Retrying with error context...")
            continue

    raise RuntimeError(
        f"Quiz generation failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )

def generate_quiz_from_document(
    chunks: list[str],
    num_questions: int = 10,
    difficulty: Difficulty = Difficulty.MEDIUM,
    question_type: QuestionType = QuestionType.MIXED,
    specific_topic: Optional[str] = None,
) -> dict:
    """
    Generates a quiz from document chunks retrieved via RAG.
    Questions are grounded strictly in the provided content.

    Args:
        chunks: List of relevant text chunks from RAG retrieval
        num_questions: How many questions to generate
        difficulty: easy / medium / hard
        question_type: mcq / true_false / short_answer / mixed
        specific_topic: Optional topic to focus questions on

    Returns:
        Structured quiz dictionary
    """
    logger.info(
        f"Generating document-based quiz: "
        f"{num_questions} questions, {difficulty.value}, {question_type.value}"
    )

    # Validate question count
    num_questions = min(num_questions, settings.MAX_QUESTION_COUNT)
    num_questions = max(num_questions, 1)

    prompt = _build_document_prompt(
        chunks=chunks,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
        specific_topic=specific_topic,
    )

    quiz_data = _call_gemini_with_retry(prompt, num_questions)

    # Add metadata
    quiz_data["quiz_id"] = str(uuid.uuid4())
    quiz_data["input_type"] = "document"
    quiz_data["difficulty"] = difficulty.value
    quiz_data["question_count"] = len(quiz_data["questions"])

    return quiz_data


def generate_quiz_from_topic(
    topic: str,
    num_questions: int = 10,
    difficulty: Difficulty = Difficulty.MEDIUM,
    question_type: QuestionType = QuestionType.MIXED,
) -> dict:
    """
    Generates a quiz from a topic using Gemini's own knowledge.
    No document or RAG involved — pure LLM generation.

    Args:
        topic: The topic to generate quiz about
        num_questions: How many questions to generate
        difficulty: easy / medium / hard
        question_type: mcq / true_false / short_answer / mixed

    Returns:
        Structured quiz dictionary
    """
    logger.info(
        f"Generating topic-based quiz: '{topic}', "
        f"{num_questions} questions, {difficulty.value}, {question_type.value}"
    )

    # Validate question count
    num_questions = min(num_questions, settings.MAX_QUESTION_COUNT)
    num_questions = max(num_questions, 1)

    prompt = _build_topic_prompt(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
    )

    quiz_data = _call_gemini_with_retry(prompt, num_questions)

    # Add metadata
    quiz_data["quiz_id"] = str(uuid.uuid4())
    quiz_data["input_type"] = "topic"
    quiz_data["difficulty"] = difficulty.value
    quiz_data["question_count"] = len(quiz_data["questions"])

    return quiz_data