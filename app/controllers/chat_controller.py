"""Chat JSON and SSE endpoints."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.configs.dependencies import get_chat_service, get_request_id
from app.entities.chat_message import ChatMessage
from app.exceptions.chat import ChatUpstreamError
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    StreamDoneEvent,
    StreamStartEvent,
    StreamTokenEvent,
)
from app.schemas.error import SSEErrorEvent
from app.services.chat_service import ChatService, StreamDone, StreamStart, StreamToken

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _message_response(message: ChatMessage) -> ChatMessageResponse:
    if message.role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported persisted message role: {message.role}")
    return ChatMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


@router.post("", response_model=ChatResponse)
async def send_message(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Generate one complete assistant response."""

    message = await service.send_message(payload.session_id, payload.message)
    return ChatResponse(session_id=payload.session_id, message=_message_response(message))


@router.post("/stream")
async def stream_message(
    request: Request,
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> EventSourceResponse:
    """Generate an assistant response over Server-Sent Events."""

    request_id = get_request_id(request)

    async def events() -> AsyncIterator[ServerSentEvent]:
        try:
            async for event in service.stream_message(payload.session_id, payload.message):
                if isinstance(event, StreamStart):
                    start = StreamStartEvent(
                        session_id=event.session_id,
                        message_id=event.message_id,
                    )
                    yield ServerSentEvent(event="start", data=start.model_dump_json())
                elif isinstance(event, StreamToken):
                    yield ServerSentEvent(
                        event="token",
                        data=StreamTokenEvent(delta=event.delta).model_dump_json(),
                    )
                elif isinstance(event, StreamDone):
                    done = StreamDoneEvent(
                        session_id=event.session_id,
                        message_id=event.message_id,
                        created_at=event.created_at,
                    )
                    yield ServerSentEvent(event="done", data=done.model_dump_json())
        except asyncio.CancelledError:
            raise
        except ChatUpstreamError as exc:
            error = SSEErrorEvent(
                code="LLM_UPSTREAM_ERROR",
                message=str(exc),
                request_id=request_id,
            )
            yield ServerSentEvent(event="error", data=error.model_dump_json())
        except Exception:
            logger.exception(
                "chat_stream_failed",
                request_id=request_id,
                session_id=str(payload.session_id),
            )
            error = SSEErrorEvent(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred while streaming the response.",
                request_id=request_id,
            )
            yield ServerSentEvent(event="error", data=error.model_dump_json())

    return EventSourceResponse(
        events(),
        ping=15,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
async def get_history(
    session_id: Annotated[uuid.UUID, Path()],
    service: Annotated[ChatService, Depends(get_chat_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ChatHistoryResponse:
    """Return recent messages for one session."""

    messages = await service.get_history(session_id, limit)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[_message_response(message) for message in messages],
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: Annotated[uuid.UUID, Path()],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> Response:
    """Delete a session and its messages."""

    await service.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
