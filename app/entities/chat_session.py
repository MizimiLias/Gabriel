"""Chat session entity."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.base import Base


class SessionSource(StrEnum):
    """The channel that owns a chat session."""

    WEB = "web"
    QQ_C2C = "qq_c2c"
    QQ_GROUP = "qq_group"


class ChatSession(Base):
    """A chat session identified by a client-provided UUID."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        UniqueConstraint("source", "external_key", name="uq_chat_sessions_source_external_key"),
        Index("ix_chat_sessions_source_updated_at", "source", "updated_at"),
    )

    session_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    title: Mapped[str | None] = mapped_column(String(80), nullable=True)
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=SessionSource.WEB.value,
        server_default=SessionSource.WEB.value,
    )
    external_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
        index=True,
    )
