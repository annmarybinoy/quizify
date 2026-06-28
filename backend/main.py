"""
main.py - Quizify Backend Entry Point

This is the root of the FastAPI application. It handles:
- App initialization and configuration
- Middleware setup (CORS, logging)
- Router registration
- Startup and shutdown events
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import os

from routers import quiz, room, ws
from db.chroma import init_chroma
from config import settings


# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)  # create logs folder if it doesn't exist

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/quizify.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


# ── Lifespan (startup + shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Startup: initialize database connections.
    Shutdown: clean up connections.
    """
    # Startup
    logger.info("Starting Quizify backend...")
    init_chroma()
    logger.info("ChromaDB initialized successfully")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("Quizify backend is ready")

    yield  # app runs here

    # Shutdown
    logger.info("Shutting down Quizify backend...")


# ── App initialization ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Quizify API",
    description="AI-powered quiz generation and real-time multiplayer platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ─────────────────────────────────────────────────────────────────────
app.include_router(quiz.router, prefix="/api/quiz", tags=["Quiz"])
app.include_router(room.router, prefix="/api/room", tags=["Room"])
app.include_router(ws.router, prefix="/ws", tags=["WebSocket"])


# ── Health check ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    Used by deployment platforms to verify the server is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }