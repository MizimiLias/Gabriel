"""PostgreSQL repository integration tests."""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.entities.chat_session import SessionSource
from app.exceptions.chat import ChatSessionNotFoundError
from app.repositories.chat_repository import ChatRepository, InboundEventRepository

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


async def test_save_load_list_rename_and_delete_exchange(db_session: AsyncSession) -> None:
    repository = ChatRepository(db_session)
    session_id = uuid4()

    with pytest.raises(ChatSessionNotFoundError):
        await repository.get_history(session_id, 50)

    await repository.save_exchange(
        session_id=session_id,
        user_content="hello",
        assistant_content="hi",
        assistant_message_id=uuid4(),
        title="A greeting",
    )

    title, history = await repository.get_history(session_id, 50)
    assert title == "A greeting"
    assert [message.role for message in history] == ["user", "assistant"]

    sessions = await repository.list_sessions()
    session = next(row for row in sessions if row.session_id == session_id)
    assert session.message_count == 2
    assert session.last_message_preview == "hi"

    renamed = await repository.update_session_title(session_id, "Renamed")
    assert renamed.title == "Renamed"
    assert await repository.get_session_title(session_id) == "Renamed"

    await repository.delete_session(session_id)
    with pytest.raises(ChatSessionNotFoundError):
        await repository.get_history(session_id, 50)


async def test_qq_sessions_are_hidden_from_default_list(db_session: AsyncSession) -> None:
    repository = ChatRepository(db_session)
    web_session_id = uuid4()
    qq_session_id = uuid4()
    await repository.save_exchange(
        session_id=web_session_id,
        user_content="web",
        assistant_content="answer",
        assistant_message_id=uuid4(),
    )
    await repository.save_exchange(
        session_id=qq_session_id,
        user_content="qq",
        assistant_content="answer",
        assistant_message_id=uuid4(),
        source=SessionSource.QQ_GROUP,
        external_key="app:group",
    )

    web_sessions = await repository.list_sessions()
    qq_sessions = await repository.list_sessions(SessionSource.QQ_GROUP)

    assert [row.session_id for row in web_sessions] == [web_session_id]
    assert [row.session_id for row in qq_sessions] == [qq_session_id]
    assert qq_sessions[0].source == SessionSource.QQ_GROUP

    await repository.delete_session(web_session_id)
    await repository.delete_session(qq_session_id)


async def test_inbound_event_claim_is_idempotent(db_session: AsyncSession) -> None:
    repository = InboundEventRepository(db_session)
    event_id = f"ROBOT1.0_{'x' * 128}"
    assert len(event_id) == 137

    assert await repository.claim(SessionSource.QQ_C2C, event_id) is True
    assert await repository.claim(SessionSource.QQ_C2C, event_id) is False

    await repository.mark_completed(SessionSource.QQ_C2C, event_id)
