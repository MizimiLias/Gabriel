"""Persistence operations for chat sessions and messages."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, exists, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities.chat_message import ChatMessage
from app.entities.chat_session import ChatSession
from app.exceptions.chat import ChatSessionNotFoundError


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

    async def get_history(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        """Return recent history, raising when the session is unknown."""

        session_exists = await self._session.scalar(
            select(exists().where(ChatSession.session_id == session_id))
        )
        if not session_exists:
            raise ChatSessionNotFoundError("The requested chat session does not exist.")
        return await self.get_recent_messages(session_id, limit)

    async def session_exists(self, session_id: UUID) -> bool:
        """Return whether a session exists."""

        value = await self._session.scalar(
            select(exists().where(ChatSession.session_id == session_id))
        )
        return bool(value)

    async def save_exchange(
        self,
        *,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_message_id: UUID,
    ) -> ChatMessage:
        """Atomically create the session when needed and store one completed exchange."""

        now = datetime.now(UTC)
        upsert_session = (
            postgresql_insert(ChatSession)
            .values(session_id=session_id, created_at=now, updated_at=now)
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
