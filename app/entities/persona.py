"""Persona entity."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.base import Base


class Persona(Base):
    """A named system prompt with optional preset dialogue."""

    __tablename__ = "personas"
    __table_args__ = (UniqueConstraint("name", name="uq_personas_name"),)

    persona_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
    )
