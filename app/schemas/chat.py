"""Chat API request and response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.entities.chat_session import SessionSource


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
    title: str
    message: ChatMessageResponse


class ChatHistoryResponse(BaseModel):
    """Ordered history for one chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    title: str
    messages: list[ChatMessageResponse]


class ChatSessionSummary(BaseModel):
    """A session row used by the conversation sidebar."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    title: str
    source: SessionSource
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str


class ChatSessionListResponse(BaseModel):
    """All sessions, ordered by most recent activity."""

    model_config = ConfigDict(extra="forbid")

    items: list[ChatSessionSummary]


class ChatSessionUpdateRequest(BaseModel):
    """Editable session metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=80)


class StreamStartEvent(BaseModel):
    """SSE event emitted before the model starts producing tokens."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message_id: UUID


class StreamTokenEvent(BaseModel):
    """One generated text fragment."""

    model_config = ConfigDict(extra="forbid")

    delta: str


class StreamReasoningEvent(BaseModel):
    """One reasoning/thinking fragment (never persisted)."""

    model_config = ConfigDict(extra="forbid")

    delta: str


class StreamDoneEvent(BaseModel):
    """SSE event emitted after the completed exchange is persisted."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    message_id: UUID
    title: str
    created_at: datetime


class HealthResponse(BaseModel):
    """Service health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
