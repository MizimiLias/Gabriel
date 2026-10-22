"""PostgreSQL repository integration tests."""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.exceptions.chat import ChatSessionNotFoundError
from app.repositories.chat_repository import ChatRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    assert TEST_DATABASE_URL is not None
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def test_save_load_and_delete_exchange(db_session: AsyncSession) -> None:
    repository = ChatRepository(db_session)
    session_id = uuid4()

    with pytest.raises(ChatSessionNotFoundError):
        await repository.get_history(session_id, 50)

    await repository.save_exchange(
        session_id=session_id,
        user_content="hello",
        assistant_content="hi",
        assistant_message_id=uuid4(),
    )

    history = await repository.get_history(session_id, 50)
    assert [message.role for message in history] == ["user", "assistant"]

    await repository.delete_session(session_id)
    with pytest.raises(ChatSessionNotFoundError):
        await repository.get_history(session_id, 50)
