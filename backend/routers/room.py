"""
routers/room.py - Room Management Endpoints

Handles all room-related API routes:
- Creating quiz rooms
- Joining rooms
- Starting and controlling live sessions
"""

from fastapi import APIRouter

router = APIRouter()


# Placeholder — full implementation coming in Phase 3
@router.get("/ping")
async def ping():
    return {"message": "Room router is working"}