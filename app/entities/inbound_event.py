"""Persistent idempotency records for messages received from integrations."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.entities.base import Base


class InboundEvent(Base):
    """One external message event claimed for processing."""

    __tablename__ = "inbound_events"

    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    external_event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="processing", server_default="processing"
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
    )
