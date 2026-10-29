"""Alembic migration smoke test against a disposable PostgreSQL database."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]


def test_migrations_create_expected_tables() -> None:
    assert TEST_DATABASE_URL is not None
    url = make_url(TEST_DATABASE_URL)
    environment = {
        **os.environ,
        "DB_HOST": url.host or "localhost",
        "DB_PORT": str(url.port or 5432),
        "DB_NAME": url.database or "gabriel_db_test",
        "DB_USERNAME": url.username or "postgres",
        "DB_PASSWORD": url.password or "",
    }
    alembic = Path(sys.executable).with_name("alembic")

    subprocess.run(
        [str(alembic), "downgrade", "base"],
        check=True,
        cwd=Path(__file__).parents[2],
        env=environment,
    )
    subprocess.run(
        [str(alembic), "upgrade", "head"],
        check=True,
        cwd=Path(__file__).parents[2],
        env=environment,
    )

    tables = asyncio.run(_inspect_tables(TEST_DATABASE_URL))
    assert {
        "chat_sessions",
        "chat_messages",
        "inbound_events",
        "personas",
        "persona_messages",
    }.issubset(tables)
    external_event_id_type = asyncio.run(_inspect_external_event_id_type(TEST_DATABASE_URL))
    assert isinstance(external_event_id_type, sa.Text)
    persona_id_nullable = asyncio.run(_inspect_persona_id_nullable(TEST_DATABASE_URL))
    assert persona_id_nullable is True
    chat_message_columns = asyncio.run(_inspect_chat_message_columns(TEST_DATABASE_URL))
    assert isinstance(chat_message_columns["parts"], JSONB)
    assert "content" not in chat_message_columns
    assert asyncio.run(_count_personas(TEST_DATABASE_URL)) == 0


async def _inspect_tables(database_url: str) -> set[str]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return set(
                await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_table_names()
                )
            )
    finally:
        await engine.dispose()


async def _inspect_external_event_id_type(database_url: str) -> object:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: next(
                    column["type"]
                    for column in inspect(sync_connection).get_columns("inbound_events")
                    if column["name"] == "external_event_id"
                )
            )
    finally:
        await engine.dispose()


async def _inspect_persona_id_nullable(database_url: str) -> bool:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return bool(
                await connection.run_sync(
                    lambda sync_connection: next(
                        column["nullable"]
                        for column in inspect(sync_connection).get_columns("chat_sessions")
                        if column["name"] == "persona_id"
                    )
                )
            )
    finally:
        await engine.dispose()


async def _count_personas(database_url: str) -> int:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return int(await connection.scalar(sa.text("SELECT count(*) FROM personas")) or 0)
    finally:
        await engine.dispose()


async def _inspect_chat_message_columns(database_url: str) -> dict[str, object]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: {
                    column["name"]: column["type"]
                    for column in inspect(sync_connection).get_columns("chat_messages")
                }
            )
    finally:
        await engine.dispose()
