"""FastAPI exception handlers returning ProblemDetail responses."""

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.exceptions.chat import (
    ChatSessionNotFoundError,
    ChatUpstreamError,
    DatabaseUnavailableError,
    PersonaConflictError,
    PersonaNotFoundError,
    PersonaValidationError,
)
from app.schemas.error import ProblemDetail

logger = structlog.get_logger(__name__)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _problem(
    request: Request,
    *,
    code: str,
    title: str,
    status: int,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    body = ProblemDetail(
        type=f"https://gabriel.local/problems/{code}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        request_id=_request_id(request),
        errors=errors,
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


async def chat_session_not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="chat-session-not-found",
        title="Chat session not found",
        status=404,
        detail=str(exc) or "The requested chat session does not exist.",
    )


async def persona_not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="persona-not-found",
        title="Persona not found",
        status=404,
        detail=str(exc) or "The requested persona does not exist.",
    )


async def persona_conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="persona-conflict",
        title="Persona conflict",
        status=409,
        detail=str(exc) or "The persona conflicts with an existing one.",
    )


async def persona_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="persona-validation-error",
        title="Persona validation failed",
        status=422,
        detail=str(exc) or "The persona input is invalid.",
    )


async def chat_upstream_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="llm-upstream-error",
        title="Language model unavailable",
        status=502,
        detail=str(exc) or "The language model could not complete the request.",
    )


async def database_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    return _problem(
        request,
        code="database-unavailable",
        title="Database unavailable",
        status=503,
        detail=str(exc) or "The database is not ready.",
    )


async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("database_operation_failed", error_type=type(exc).__name__)
    return await database_unavailable_handler(request, exc)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return await unhandled_error_handler(request, exc)
    errors = [
        {
            "loc": list(error.get("loc", ())),
            "msg": str(error.get("msg", "Invalid value")),
            "type": str(error.get("type", "validation_error")),
        }
        for error in exc.errors()
    ]
    return _problem(
        request,
        code="validation-error",
        title="Request validation failed",
        status=422,
        detail="One or more request fields are invalid.",
        errors=errors,
    )


async def http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    status_code = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
    detail = str(exc.detail) if isinstance(exc, StarletteHTTPException) else str(exc)
    return _problem(
        request,
        code="http-error",
        title="HTTP error",
        status=status_code,
        detail=detail,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", path=request.url.path, error_type=type(exc).__name__)
    return _problem(
        request,
        code="internal-server-error",
        title="Internal server error",
        status=500,
        detail="An unexpected error occurred.",
    )


def register_exception_handlers(application: FastAPI) -> None:
    """Register all application exception handlers."""

    application.add_exception_handler(ChatSessionNotFoundError, chat_session_not_found_handler)
    application.add_exception_handler(PersonaNotFoundError, persona_not_found_handler)
    application.add_exception_handler(PersonaConflictError, persona_conflict_handler)
    application.add_exception_handler(PersonaValidationError, persona_validation_handler)
    application.add_exception_handler(ChatUpstreamError, chat_upstream_error_handler)
    application.add_exception_handler(DatabaseUnavailableError, database_unavailable_handler)
    application.add_exception_handler(SQLAlchemyError, database_error_handler)
    application.add_exception_handler(TimeoutError, database_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(StarletteHTTPException, http_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)
