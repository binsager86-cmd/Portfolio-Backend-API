"""API v2 router aggregate."""

from fastapi import APIRouter

from app.api.v2.simulator import router as simulator_router

v2_router = APIRouter(prefix="/api/v2")
v2_router.include_router(simulator_router)