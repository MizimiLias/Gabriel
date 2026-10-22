"""Alembic migration smoke test against a disposable PostgreSQL database."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect
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
    assert {"chat_sessions", "chat_messages"}.issubset(tables)


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
