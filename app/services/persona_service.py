"""Persona orchestration and validation."""

from collections.abc import Sequence
from uuid import UUID

from app.exceptions.chat import PersonaNotFoundError, PersonaValidationError
from app.personas import (
    MAX_PERSONA_MESSAGE_LENGTH,
    MAX_PERSONA_MESSAGES,
    MAX_PERSONA_NAME_LENGTH,
    MAX_PERSONA_PRESET_TOTAL_LENGTH,
    MAX_PERSONA_SYSTEM_PROMPT_LENGTH,
)
from app.repositories.persona_repository import (
    PersonaMessageDraft,
    PersonaMessageMutation,
    PersonaRecord,
    PersonaRepository,
    PersonaSummaryRow,
)


class PersonaService:
    """Coordinates persona validation and persistence."""

    def __init__(self, repository: PersonaRepository) -> None:
        self._repository = repository

    async def create(
        self,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
        messages: Sequence[PersonaMessageDraft],
    ) -> PersonaRecord:
        """Create a persona after validating its limits."""

        _validate(name=name, system_prompt=system_prompt, messages=messages)
        return await self._repository.create(
            name=name,
            description=description,
            system_prompt=system_prompt,
            messages=messages,
        )

    async def list_all(self) -> list[PersonaSummaryRow]:
        """Return summaries for every persona."""

        return await self._repository.list_all()

    async def get(self, persona_id: UUID) -> PersonaRecord:
        """Return one persona or raise when it does not exist."""

        record = await self._repository.get(persona_id)
        if record is None:
            raise PersonaNotFoundError("The requested persona does not exist.")
        return record

    async def update(
        self,
        persona_id: UUID,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
        messages: Sequence[PersonaMessageMutation],
    ) -> PersonaRecord:
        """Atomically update metadata and synchronize preset dialogue."""

        _validate(name=name, system_prompt=system_prompt, messages=messages)
        return await self._repository.update_with_messages(
            persona_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            messages=messages,
        )

    async def delete(self, persona_id: UUID) -> None:
        """Delete a persona and let the database clear references."""

        await self._repository.delete(persona_id)


def _validate(
    *,
    name: str,
    system_prompt: str,
    messages: Sequence[PersonaMessageDraft | PersonaMessageMutation],
) -> None:
    if not name.strip():
        raise PersonaValidationError("A persona name is required.")
    if len(name) > MAX_PERSONA_NAME_LENGTH:
        raise PersonaValidationError(
            f"A persona name cannot exceed {MAX_PERSONA_NAME_LENGTH} characters."
        )
    if not system_prompt.strip():
        raise PersonaValidationError("A persona system prompt is required.")
    if len(system_prompt) > MAX_PERSONA_SYSTEM_PROMPT_LENGTH:
        raise PersonaValidationError(
            f"A persona system prompt cannot exceed {MAX_PERSONA_SYSTEM_PROMPT_LENGTH} characters."
        )
    _validate_messages(messages)


def _validate_messages(
    messages: Sequence[PersonaMessageDraft | PersonaMessageMutation],
) -> None:
    if len(messages) > MAX_PERSONA_MESSAGES:
        raise PersonaValidationError(
            f"A persona cannot have more than {MAX_PERSONA_MESSAGES} preset messages."
        )

    total = 0
    for message in messages:
        if message.role not in ("user", "assistant"):
            raise PersonaValidationError("Preset messages must be user or assistant messages.")
        if not message.content.strip():
            raise PersonaValidationError("Preset message content cannot be empty.")
        if len(message.content) > MAX_PERSONA_MESSAGE_LENGTH:
            raise PersonaValidationError(
                f"A preset message cannot exceed {MAX_PERSONA_MESSAGE_LENGTH} characters."
            )
        total += len(message.content)

    if total > MAX_PERSONA_PRESET_TOTAL_LENGTH:
        raise PersonaValidationError(
            f"Preset dialogue cannot exceed {MAX_PERSONA_PRESET_TOTAL_LENGTH} characters in total."
        )
