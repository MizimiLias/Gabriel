"""Store chat messages as multimodal content parts.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-29 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "parts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE chat_messages
            SET parts = jsonb_build_array(
                jsonb_build_object('type', 'text', 'text', content)
            )
            """
        )
    )
    op.alter_column("chat_messages", "parts", nullable=False)
    op.create_check_constraint(
        "parts_array",
        "chat_messages",
        "jsonb_typeof(parts) = 'array' AND jsonb_array_length(parts) > 0",
    )
    op.drop_column("chat_messages", "content")


def downgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("content", sa.Text(), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE chat_messages AS message
            SET content = COALESCE((
                SELECT string_agg(
                    part->>'text',
                    E'\\n\\n'
                    ORDER BY ordinality
                )
                FROM jsonb_array_elements(message.parts)
                    WITH ORDINALITY AS item(part, ordinality)
                WHERE part->>'type' = 'text'
            ), '')
            """
        )
    )
    op.alter_column("chat_messages", "content", nullable=False)
    op.drop_constraint("parts_array", "chat_messages", type_="check")
    op.drop_column("chat_messages", "parts")
