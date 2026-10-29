"""Chat API request and response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.entities.chat_session import SessionSource
from app.schemas.content import ContentPart, TextPart


class ChatRequest(BaseModel):
    """Incoming message for both JSON and SSE chat endpoints."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: UUID
    parts: list[ContentPart] = Field(min_length=1)
    persona_id: UUID | None = None

    @model_validator(mode="after")
    def validate_total_text_length(self) -> ChatRequest:
        """Keep one request within the legacy total text budget."""

        total = sum(len(part.text) for part in self.parts if isinstance(part, TextPart))
        if total > 32_000:
            raise ValueError("Text parts exceed the 32000 character limit.")
        return self


class ChatMessageResponse(BaseModel):
    """Persisted chat message returned to a client."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    role: Literal["user", "assistant"]
    parts: list[ContentPart]
    created_at: datetime


class ChatResponse(BaseModel):
    """Complete non-streaming chat response."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    title: str | None
    message: ChatMessageResponse


class ChatHistoryResponse(BaseModel):
    """Ordered history for one chat session."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    title: str | None
    messages: list[ChatMessageResponse]


class ChatSessionSummary(BaseModel):
    """A session row used by the conversation sidebar."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    title: str | None
    source: SessionSource
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str
    persona_id: UUID | None = None
    persona_name: str | None = None


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
    title: str | None
    created_at: datetime


class HealthResponse(BaseModel):
    """Service health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
