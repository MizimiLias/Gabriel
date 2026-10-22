"""Health and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.configs.dependencies import get_db
from app.exceptions.chat import DatabaseUnavailableError
from app.schemas.chat import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """Report process liveness without touching dependencies."""

    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def ready(db: Annotated[AsyncSession, Depends(get_db)]) -> HealthResponse:
    """Report readiness by checking database connectivity."""

    try:
        await db.execute(text("SELECT 1"))
    except (SQLAlchemyError, TimeoutError, OSError) as exc:
        raise DatabaseUnavailableError("The database is not ready.") from exc
    return HealthResponse(status="ok")
