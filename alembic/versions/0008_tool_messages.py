"""Allow tool messages in chat history.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-30 00:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("valid_role", "chat_messages", type_="check")
    op.create_check_constraint(
        "valid_role",
        "chat_messages",
        "role IN ('user', 'assistant', 'tool')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM chat_messages WHERE role = 'tool'")
    op.execute(
        "DELETE FROM chat_messages "
        "WHERE role = 'assistant' AND parts @> '[{\"type\": \"tool_call\"}]'::jsonb"
    )
    op.drop_constraint("valid_role", "chat_messages", type_="check")
    op.create_check_constraint(
        "valid_role",
        "chat_messages",
        "role IN ('user', 'assistant')",
    )
