"""Add session titles and recent-activity index.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-22 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column(
            "title",
            sa.String(length=80),
            server_default="New chat",
            nullable=False,
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE chat_sessions AS session
            SET title = COALESCE(
                (
                    SELECT LEFT(REGEXP_REPLACE(message.content, '\\s+', ' ', 'g'), 80)
                    FROM chat_messages AS message
                    WHERE message.session_id = session.session_id
                      AND message.role = 'user'
                    ORDER BY message.sequence
                    LIMIT 1
                ),
                'New chat'
            )
            """
        )
    )
    op.create_index(
        "ix_chat_sessions_updated_at",
        "chat_sessions",
        ["updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_updated_at", table_name="chat_sessions")
    op.drop_column("chat_sessions", "title")
