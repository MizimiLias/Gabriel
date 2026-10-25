"""Add session sources and inbound event idempotency records.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-23 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column(
            "source",
            sa.String(length=32),
            server_default="web",
            nullable=False,
        ),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("external_key", sa.String(length=255), nullable=True),
    )
    op.create_unique_constraint(
        "uq_chat_sessions_source_external_key",
        "chat_sessions",
        ["source", "external_key"],
    )
    op.create_index(
        "ix_chat_sessions_source_updated_at",
        "chat_sessions",
        ["source", "updated_at"],
        unique=False,
    )
    op.create_table(
        "inbound_events",
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("external_event_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="processing",
            nullable=False,
        ),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_type", sa.String(length=80), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source", "external_event_id", name="pk_inbound_events"),
    )


def downgrade() -> None:
    op.drop_table("inbound_events")
    op.drop_index("ix_chat_sessions_source_updated_at", table_name="chat_sessions")
    op.drop_constraint(
        "uq_chat_sessions_source_external_key",
        "chat_sessions",
        type_="unique",
    )
    op.drop_column("chat_sessions", "external_key")
    op.drop_column("chat_sessions", "source")
