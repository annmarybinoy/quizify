"""
models/quiz.py - Quiz Pydantic Models

Defines the data contracts for all quiz-related requests and responses.
Pydantic automatically validates all incoming data against these models
and FastAPI uses them to generate accurate API documentation.

Two categories:
- Request models  → what the frontend sends to us
- Response models → what we send back to the frontend
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


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


class InputType(str, Enum):
    DOCUMENT = "document"
    TOPIC = "topic"


# ── Request Models ─────────────────────────────────────────────────────────────
class QuizFromTopicRequest(BaseModel):
    """
    Request model for topic-based quiz generation.
    Frontend sends this when user just types a topic.
    File upload is NOT involved here.
    """
    topic: str = Field(
        ...,                                    # ... means required
        min_length=3,
        max_length=200,
        description="The topic to generate quiz about",
        examples=["Photosynthesis", "World War 2", "Python programming"]
    )
    num_questions: int = Field(
        default=10,
        ge=1,                                   # ge = greater than or equal to
        le=30,                                  # le = less than or equal to
        description="Number of questions to generate"
    )
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM,
        description="Difficulty level of the quiz"
    )
    question_type: QuestionType = Field(
        default=QuestionType.MIXED,
        description="Type of questions to generate"
    )

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_empty(cls, v: str) -> str:
        """Ensures topic is not just whitespace."""
        if not v.strip():
            raise ValueError("Topic cannot be empty or just whitespace")
        return v.strip()


class QuizFromDocumentRequest(BaseModel):
    """
    Request model for document-based quiz generation.
    Note: The actual file is sent as multipart form data separately.
    This model handles the configuration options only.
    """
    num_questions: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of questions to generate"
    )
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM,
        description="Difficulty level of the quiz"
    )
    question_type: QuestionType = Field(
        default=QuestionType.MIXED,
        description="Type of questions to generate"
    )
    specific_topic: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional: focus quiz on a specific topic within the document"
    )

    @field_validator("specific_topic")
    @classmethod
    def clean_specific_topic(cls, v: Optional[str]) -> Optional[str]:
        """Strips whitespace and converts empty string to None."""
        if v is not None:
            v = v.strip()
            if not v:
                return None     # treat empty string as no topic
        return v


class RawTextRequest(BaseModel):
    """
    Request model for raw text quiz generation.
    User pastes text directly instead of uploading a file.
    """
    text: str = Field(
        ...,
        min_length=50,
        description="Raw text content to generate quiz from"
    )
    num_questions: int = Field(
        default=10,
        ge=1,
        le=30,
    )
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM)
    question_type: QuestionType = Field(default=QuestionType.MIXED)
    specific_topic: Optional[str] = Field(default=None, max_length=200)

    @field_validator("text")
    @classmethod
    def text_must_have_content(cls, v: str) -> str:
        """Ensures text has actual content not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()


# ── Response Models ────────────────────────────────────────────────────────────
class QuestionResponse(BaseModel):
    """
    Response model for a single quiz question.
    Every question type uses this same model —
    options is empty dict for short answer questions.
    """
    id: int = Field(description="Question number")
    type: str = Field(description="Question type: mcq, true_false, short_answer")
    question: str = Field(description="The question text")
    options: dict = Field(
        description="Answer options. Empty for short answer questions.",
        examples=[{"A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4"}]
    )
    correct_answer: str = Field(description="The correct answer (letter for MCQ/TF, text for short answer)")
    explanation: str = Field(description="Explanation of why the answer is correct")


class QuizResponse(BaseModel):
    """
    Response model for a complete generated quiz.
    Returned after successful quiz generation.
    """
    quiz_id: str = Field(description="Unique identifier for this quiz")
    title: str = Field(description="Quiz title")
    topic: str = Field(description="Main topic of the quiz")
    difficulty: str = Field(description="Difficulty level used")
    input_type: str = Field(description="How quiz was generated: document or topic")
    question_count: int = Field(description="Number of questions in the quiz")
    questions: list[QuestionResponse] = Field(description="List of quiz questions")
    doc_id: Optional[str] = Field(
        default=None,
        description="Document ID if quiz was generated from a document"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the quiz was generated"
    )


class QuizSummary(BaseModel):
    """
    Lightweight quiz summary for listing quizzes.
    Used when showing a user's quiz history —
    we don't need all questions, just the metadata.
    """
    quiz_id: str
    title: str
    topic: str
    difficulty: str
    input_type: str
    question_count: int
    created_at: datetime


# ── Error Response ─────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """
    Standard error response structure.
    Every error from our API follows this format
    so the frontend always knows what to expect.
    """
    error: str = Field(description="Error type")
    detail: str = Field(description="Human readable error message")
    status_code: int = Field(description="HTTP status code")