"""Chat API request and response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Incoming message for both JSON and SSE chat endpoints."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: UUID
    message: str = Field(min_length=1, max_length=32_000)


class ChatMessageResponse(BaseModel):
    """Persisted chat message returned to a client."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    """Complete non-streaming chat response."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message: ChatMessageResponse


class ChatHistoryResponse(BaseModel):
    """Ordered history for one chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    messages: list[ChatMessageResponse]


class StreamStartEvent(BaseModel):
    """SSE event emitted before the model starts producing tokens."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message_id: UUID


class StreamTokenEvent(BaseModel):
    """One generated text fragment."""

    model_config = ConfigDict(extra="forbid")

    delta: str


class StreamDoneEvent(BaseModel):
    """SSE event emitted after the completed exchange is persisted."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message_id: UUID
    created_at: datetime


class HealthResponse(BaseModel):
    """Service health response."""

    status: Literal["ok"]
