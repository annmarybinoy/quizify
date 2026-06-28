"""
routers/quiz.py - Quiz Generation Endpoints

Handles all quiz-related API routes:
- Creating quizzes from PDF, image, text, or topic
- Retrieving quiz details
- Deleting quizzes
"""

from fastapi import APIRouter

router = APIRouter()


# Placeholder — full implementation coming in Phase 2
@router.get("/ping")
async def ping():
    return {"message": "Quiz router is working"}