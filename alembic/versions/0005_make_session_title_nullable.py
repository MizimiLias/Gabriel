"""Allow sessions without a title for non-web channels.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-26 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "chat_sessions",
        "title",
        existing_type=sa.String(length=80),
        nullable=True,
        server_default=None,
    )


def downgrade() -> None:
    op.execute("UPDATE chat_sessions SET title = 'New chat' WHERE title IS NULL")
    op.alter_column(
        "chat_sessions",
        "title",
        existing_type=sa.String(length=80),
        nullable=False,
        server_default="New chat",
    )