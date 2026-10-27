"""Add personas and persona preset dialogue.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-27 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "personas",
        sa.Column("persona_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("persona_id", name="pk_personas"),
        sa.UniqueConstraint("name", name="uq_personas_name"),
    )
    op.create_table(
        "persona_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("persona_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_persona_messages"),
        sa.ForeignKeyConstraint(
            ["persona_id"],
            ["personas.persona_id"],
            name="fk_persona_messages_persona_id_personas",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "persona_id",
            "position",
            name="uq_persona_messages_persona_id_position",
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="valid_role"),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("persona_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_chat_sessions_persona_id_personas",
        "chat_sessions",
        "personas",
        ["persona_id"],
        ["persona_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_chat_sessions_persona_id",
        "chat_sessions",
        ["persona_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_persona_id", table_name="chat_sessions")
    op.drop_constraint(
        "fk_chat_sessions_persona_id_personas",
        "chat_sessions",
        type_="foreignkey",
    )
    op.drop_column("chat_sessions", "persona_id")
    op.drop_table("persona_messages")
    op.drop_table("personas")
