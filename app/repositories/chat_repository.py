"""Persistence operations for chat sessions and messages."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities.chat_message import ChatMessage
from app.entities.chat_session import ChatSession, SessionSource
from app.entities.inbound_event import InboundEvent
from app.entities.persona import Persona
from app.exceptions.chat import ChatSessionNotFoundError


@dataclass(frozen=True, slots=True)
class ChatSessionSummaryRow:
    """Database projection used to build the session sidebar."""

    session_id: UUID
    title: str | None
    source: SessionSource
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str
    persona_id: UUID | None = None
    persona_name: str | None = None


@dataclass(frozen=True, slots=True)
class ChatSessionMeta:
    """Session attributes needed before assembling a model request."""

    title: str | None
    source: SessionSource
    persona_id: UUID | None


class ChatRepository:
    """Database gateway for chat state."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_recent_messages(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        """Return the most recent messages in chronological order."""

        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence.desc())
            .limit(limit)
        )
        result = await self._session.scalars(statement)
        return list(reversed(result.all()))

    async def get_history(
        self, session_id: UUID, limit: int
    ) -> tuple[str | None, list[ChatMessage]]:
        """Return a session title and recent history, raising when unknown."""

        row = (
            await self._session.execute(
                select(ChatSession.session_id, ChatSession.title).where(
                    ChatSession.session_id == session_id
                )
            )
        ).first()
        if row is None:
            raise ChatSessionNotFoundError("The requested chat session does not exist.")
        return row.title, await self.get_recent_messages(session_id, limit)

    async def get_session_title(self, session_id: UUID) -> str | None:
        """Return a session title; None means no title or no such session."""

        return cast(
            str | None,
            await self._session.scalar(
                select(ChatSession.title).where(ChatSession.session_id == session_id)
            ),
        )

    async def get_session_meta(self, session_id: UUID) -> ChatSessionMeta | None:
        """Return session attributes; None means the session does not exist."""

        row = (
            await self._session.execute(
                select(
                    ChatSession.title,
                    ChatSession.source,
                    ChatSession.persona_id,
                ).where(ChatSession.session_id == session_id)
            )
        ).first()
        if row is None:
            return None
        return ChatSessionMeta(
            title=row.title,
            source=SessionSource(row.source),
            persona_id=row.persona_id,
        )

    async def session_exists(self, session_id: UUID) -> bool:
        """Return whether a session exists."""

        value = await self._session.scalar(
            select(exists().where(ChatSession.session_id == session_id))
        )
        return bool(value)

    async def list_sessions(
        self, source: SessionSource = SessionSource.WEB
    ) -> list[ChatSessionSummaryRow]:
        """Return session summaries for one source ordered by recent activity."""

        message_count = (
            select(func.count(ChatMessage.id))
            .where(ChatMessage.session_id == ChatSession.session_id)
            .correlate(ChatSession)
            .scalar_subquery()
        )
        latest_message = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == ChatSession.session_id)
            .order_by(ChatMessage.sequence.desc())
            .limit(1)
            .correlate(ChatSession)
            .scalar_subquery()
        )
        statement = (
            select(
                ChatSession.session_id,
                ChatSession.title,
                ChatSession.source,
                ChatSession.persona_id,
                Persona.name.label("persona_name"),
                ChatSession.created_at,
                ChatSession.updated_at,
                message_count.label("message_count"),
                func.coalesce(func.left(latest_message, 160), "").label("last_message_preview"),
            )
            .outerjoin(Persona, Persona.persona_id == ChatSession.persona_id)
            .where(ChatSession.source == source.value)
            .order_by(ChatSession.updated_at.desc())
        )
        rows = (await self._session.execute(statement)).all()
        return [
            ChatSessionSummaryRow(
                session_id=row.session_id,
                title=row.title,
                source=SessionSource(row.source),
                created_at=row.created_at,
                updated_at=row.updated_at,
                message_count=row.message_count,
                last_message_preview=row.last_message_preview,
                persona_id=row.persona_id,
                persona_name=row.persona_name,
            )
            for row in rows
        ]

    async def save_exchange(
        self,
        *,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_message_id: UUID,
        title: str | None = None,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        persona_id: UUID | None = None,
    ) -> ChatMessage:
        """Atomically create the session when needed and store one completed exchange."""

        now = datetime.now(UTC)
        upsert_session = (
            postgresql_insert(ChatSession)
            .values(
                session_id=session_id,
                title=title,
                source=source.value,
                external_key=external_key,
                persona_id=persona_id,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[ChatSession.session_id],
                set_={"updated_at": now},
            )
        )

        user_message = ChatMessage(
            session_id=session_id,
            role="user",
            content=user_content,
            created_at=now,
        )
        assistant_message = ChatMessage(
            id=assistant_message_id,
            session_id=session_id,
            role="assistant",
            content=assistant_content,
            created_at=now,
        )

        try:
            await self._session.execute(upsert_session)
            self._session.add_all([user_message, assistant_message])
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return assistant_message

    async def update_session_title(self, session_id: UUID, title: str) -> ChatSessionSummaryRow:
        """Rename one session and return its updated summary."""

        now = datetime.now(UTC)
        statement = (
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(title=title, updated_at=now)
            .returning(ChatSession)
        )
        try:
            result = await self._session.scalar(statement)
            if result is None:
                raise ChatSessionNotFoundError("The requested chat session does not exist.")

            row = next(
                (
                    session
                    for session in await self.list_sessions(SessionSource(result.source))
                    if session.session_id == session_id
                ),
                None,
            )
            if row is None:
                raise ChatSessionNotFoundError("The requested chat session does not exist.")
            await self._session.commit()
            return row
        except Exception:
            await self._session.rollback()
            raise

    async def delete_session(self, session_id: UUID) -> None:
        """Delete a session and all messages through the database cascade."""

        try:
            await self._session.execute(
                delete(ChatSession).where(ChatSession.session_id == session_id)
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise


class InboundEventRepository:
    """Database gateway for external event idempotency claims."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim(self, source: SessionSource, external_event_id: str) -> bool:
        """Claim an event, returning false when it was already received."""

        now = datetime.now(UTC)
        statement = (
            postgresql_insert(InboundEvent)
            .values(
                source=source.value,
                external_event_id=external_event_id,
                status="processing",
                received_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=[InboundEvent.source, InboundEvent.external_event_id]
            )
            .returning(InboundEvent.external_event_id)
        )
        try:
            claimed = await self._session.scalar(statement)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return claimed is not None

    async def mark_completed(self, source: SessionSource, external_event_id: str) -> None:
        """Record successful processing of a claimed event."""

        await self._update_status(source, external_event_id, status="completed")

    async def mark_failed(
        self, source: SessionSource, external_event_id: str, error_type: str
    ) -> None:
        """Record failed processing without allowing an automatic duplicate retry."""

        await self._update_status(
            source,
            external_event_id,
            status="failed",
            error_type=error_type,
        )

    async def _update_status(
        self,
        source: SessionSource,
        external_event_id: str,
        *,
        status: str,
        error_type: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        statement = (
            update(InboundEvent)
            .where(
                InboundEvent.source == source.value,
                InboundEvent.external_event_id == external_event_id,
            )
            .values(status=status, processed_at=now, updated_at=now, error_type=error_type)
        )
        try:
            await self._session.execute(statement)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
