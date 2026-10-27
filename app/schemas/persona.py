"""Persona API request and response schemas."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.personas import (
    MAX_PERSONA_MESSAGE_LENGTH,
    MAX_PERSONA_MESSAGES,
    MAX_PERSONA_NAME_LENGTH,
    MAX_PERSONA_PRESET_TOTAL_LENGTH,
    MAX_PERSONA_SYSTEM_PROMPT_LENGTH,
)


class PersonaMessageInput(BaseModel):
    """One preset message supplied by a client."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_PERSONA_MESSAGE_LENGTH)


class _PersonaMessagesPayload(BaseModel):
    messages: list[PersonaMessageInput] = Field(
        default_factory=list, max_length=MAX_PERSONA_MESSAGES
    )

    @model_validator(mode="after")
    def _check_total_length(self) -> Self:
        total = sum(len(message.content) for message in self.messages)
        if total > MAX_PERSONA_PRESET_TOTAL_LENGTH:
            raise ValueError(
                "Preset dialogue cannot exceed "
                f"{MAX_PERSONA_PRESET_TOTAL_LENGTH} characters in total."
            )
        return self


class PersonaCreate(_PersonaMessagesPayload):
    """Create a persona with optional preset dialogue."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=MAX_PERSONA_NAME_LENGTH)
    description: str | None = Field(default=None, max_length=255)
    system_prompt: str = Field(min_length=1, max_length=MAX_PERSONA_SYSTEM_PROMPT_LENGTH)


class PersonaUpdate(BaseModel):
    """Partially update persona metadata without changing preset dialogue."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=MAX_PERSONA_NAME_LENGTH)
    description: str | None = Field(default=None, max_length=255)
    system_prompt: str | None = Field(
        default=None, min_length=1, max_length=MAX_PERSONA_SYSTEM_PROMPT_LENGTH
    )

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> Self:
        for field_name in ("name", "system_prompt"):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")
        return self


class PersonaMessagesReplace(_PersonaMessagesPayload):
    """Replace a persona's preset dialogue wholesale."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PersonaMessageResponse(BaseModel):
    """A stored preset message."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    position: int
    role: Literal["user", "assistant"]
    content: str


class PersonaResponse(BaseModel):
    """Full persona detail."""

    model_config = ConfigDict(extra="forbid")

    persona_id: UUID
    name: str
    description: str | None
    system_prompt: str
    messages: list[PersonaMessageResponse]
    created_at: datetime
    updated_at: datetime


class PersonaSummaryResponse(BaseModel):
    """Persona summary used by the management list."""

    model_config = ConfigDict(extra="forbid")

    persona_id: UUID
    name: str
    description: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime


class PersonaListResponse(BaseModel):
    """All personas."""

    model_config = ConfigDict(extra="forbid")

    items: list[PersonaSummaryResponse]
