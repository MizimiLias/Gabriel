"""Persistence operations for personas and their preset dialogue."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities.persona import Persona
from app.entities.persona_message import PersonaMessage
from app.exceptions.chat import PersonaConflictError, PersonaNotFoundError

PersonaRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class PersonaMessageDraft:
    """A preset message supplied by the API before it is stored."""

    role: PersonaRole
    content: str


@dataclass(frozen=True, slots=True)
class PersonaMessageRecord:
    """A stored preset message."""

    id: UUID
    position: int
    role: PersonaRole
    content: str


@dataclass(frozen=True, slots=True)
class PersonaRecord:
    """A persona and its ordered preset dialogue."""

    persona_id: UUID
    name: str
    description: str | None
    system_prompt: str
    messages: tuple[PersonaMessageRecord, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PersonaSummaryRow:
    """Database projection used to list personas."""

    persona_id: UUID
    name: str
    description: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime


class PersonaRepository:
    """Database gateway for personas."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, persona_id: UUID) -> PersonaRecord | None:
        """Return one persona with its ordered preset messages."""

        statement = (
            select(Persona, PersonaMessage)
            .outerjoin(PersonaMessage, PersonaMessage.persona_id == Persona.persona_id)
            .where(Persona.persona_id == persona_id)
            .order_by(PersonaMessage.position)
        )
        rows = (await self._session.execute(statement)).all()
        if not rows:
            return None

        persona = rows[0][0]
        messages = tuple(
            PersonaMessageRecord(
                id=message.id,
                position=message.position,
                role=cast(PersonaRole, message.role),
                content=message.content,
            )
            for _, message in rows
            if message is not None
        )
        return PersonaRecord(
            persona_id=persona.persona_id,
            name=persona.name,
            description=persona.description,
            system_prompt=persona.system_prompt,
            messages=messages,
            created_at=persona.created_at,
            updated_at=persona.updated_at,
        )

    async def list_all(self) -> list[PersonaSummaryRow]:
        """Return every persona summary ordered by name."""

        message_count = (
            select(func.count(PersonaMessage.id))
            .where(PersonaMessage.persona_id == Persona.persona_id)
            .correlate(Persona)
            .scalar_subquery()
        )
        statement = select(
            Persona.persona_id,
            Persona.name,
            Persona.description,
            message_count.label("message_count"),
            Persona.created_at,
            Persona.updated_at,
        ).order_by(Persona.name)
        rows = (await self._session.execute(statement)).all()
        return [
            PersonaSummaryRow(
                persona_id=row.persona_id,
                name=row.name,
                description=row.description,
                message_count=row.message_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    async def create(
        self,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
        messages: Sequence[PersonaMessageDraft],
    ) -> PersonaRecord:
        """Create a persona with its preset dialogue in one transaction."""

        now = datetime.now(UTC)
        persona_id = uuid4()
        self._session.add(
            Persona(
                persona_id=persona_id,
                name=name,
                description=description,
                system_prompt=system_prompt,
                created_at=now,
                updated_at=now,
            )
        )
        self._session.add_all(
            [
                PersonaMessage(
                    persona_id=persona_id,
                    position=position,
                    role=draft.role,
                    content=draft.content,
                    created_at=now,
                )
                for position, draft in enumerate(messages)
            ]
        )
        try:
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            _raise_conflict(exc)
        return await self._require(persona_id)

    async def update(
        self,
        persona_id: UUID,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
    ) -> PersonaRecord:
        """Update persona metadata without touching preset dialogue."""

        statement = (
            update(Persona)
            .where(Persona.persona_id == persona_id)
            .values(
                name=name,
                description=description,
                system_prompt=system_prompt,
                updated_at=datetime.now(UTC),
            )
            .returning(Persona.persona_id)
        )
        try:
            updated = await self._session.scalar(statement)
            if updated is None:
                await self._session.rollback()
                raise PersonaNotFoundError("The requested persona does not exist.")
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            _raise_conflict(exc)
        return await self._require(persona_id)

    async def replace_messages(
        self, persona_id: UUID, messages: Sequence[PersonaMessageDraft]
    ) -> PersonaRecord:
        """Replace all preset messages for one persona."""

        if not await self._session.scalar(select(exists().where(Persona.persona_id == persona_id))):
            raise PersonaNotFoundError("The requested persona does not exist.")

        now = datetime.now(UTC)
        try:
            await self._session.execute(
                delete(PersonaMessage).where(PersonaMessage.persona_id == persona_id)
            )
            self._session.add_all(
                [
                    PersonaMessage(
                        persona_id=persona_id,
                        position=position,
                        role=draft.role,
                        content=draft.content,
                        created_at=now,
                    )
                    for position, draft in enumerate(messages)
                ]
            )
            await self._session.execute(
                update(Persona).where(Persona.persona_id == persona_id).values(updated_at=now)
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return await self._require(persona_id)

    async def delete(self, persona_id: UUID) -> None:
        """Delete a persona; the database cascades presets and clears sessions."""

        statement = (
            delete(Persona).where(Persona.persona_id == persona_id).returning(Persona.persona_id)
        )
        deleted = await self._session.scalar(statement)
        if deleted is None:
            await self._session.rollback()
            raise PersonaNotFoundError("The requested persona does not exist.")
        await self._session.commit()

    async def _require(self, persona_id: UUID) -> PersonaRecord:
        record = await self.get(persona_id)
        if record is None:
            raise PersonaNotFoundError("The requested persona does not exist.")
        return record


def _raise_conflict(exc: IntegrityError) -> None:
    detail = str(exc.orig)
    constraint = getattr(exc.orig, "constraint_name", None)
    if constraint == "uq_personas_name" or "uq_personas_name" in detail:
        raise PersonaConflictError("A persona with this name already exists.") from exc
    raise exc
