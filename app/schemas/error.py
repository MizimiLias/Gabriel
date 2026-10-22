"""Problem-detail response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ProblemDetail(BaseModel):
    """RFC 7807-style error response."""

    model_config = ConfigDict(extra="forbid")

    type: str
    title: str
    status: int
    detail: str
    instance: str
    request_id: str
    errors: list[dict[str, Any]] | None = None


class SSEErrorEvent(BaseModel):
    """Error sent inside an already-started SSE response."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str
