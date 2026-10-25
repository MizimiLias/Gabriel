"""Expand inbound event IDs for QQ group messages.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-23 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "inbound_events",
        "external_event_id",
        existing_type=sa.String(length=128),
        type_=sa.Text(),
        existing_nullable=False,
    )


def downgrade() -> None:
    max_length = op.get_bind().scalar(
        sa.text("SELECT max(length(external_event_id)) FROM inbound_events")
    )
    if max_length is not None and max_length > 128:
        raise RuntimeError(
            "Cannot downgrade: inbound_events contains external_event_id values longer than 128"
        )
    op.alter_column(
        "inbound_events",
        "external_event_id",
        existing_type=sa.Text(),
        type_=sa.String(length=128),
        existing_nullable=False,
    )
