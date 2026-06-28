"""
routers/ws.py - WebSocket Endpoints

Handles real-time WebSocket connections for live quiz sessions.
"""

from fastapi import APIRouter

router = APIRouter()


# Placeholder — full implementation coming in Phase 3
@router.get("/ping")
async def ping():
    return {"message": "WebSocket router is working"}