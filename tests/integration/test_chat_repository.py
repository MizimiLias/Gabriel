"""PostgreSQL repository integration tests."""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.entities.chat_session import SessionSource
from app.entities.inbound_event import InboundEvent
from app.exceptions.chat import ChatSessionNotFoundError
from app.repositories.chat_repository import ChatRepository, InboundEventRepository
from app.repositories.persona_repository import PersonaRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def _text_parts(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


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
        user_parts=_text_parts("hello"),
        assistant_parts=_text_parts("hi"),
        assistant_message_id=uuid4(),
        title="A greeting",
    )

    title, history = await repository.get_history(session_id, 50)
    assert title == "A greeting"
    assert [message.role for message in history] == ["user", "assistant"]
    assert history[0].parts == _text_parts("hello")
    assert history[1].parts == _text_parts("hi")

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
        user_parts=_text_parts("web"),
        assistant_parts=_text_parts("answer"),
        assistant_message_id=uuid4(),
    )
    await repository.save_exchange(
        session_id=qq_session_id,
        user_parts=_text_parts("qq"),
        assistant_parts=_text_parts("answer"),
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


async def test_session_without_title_round_trips(db_session: AsyncSession) -> None:
    repository = ChatRepository(db_session)
    session_id = uuid4()

    await repository.save_exchange(
        session_id=session_id,
        user_parts=_text_parts("qq"),
        assistant_parts=_text_parts("answer"),
        assistant_message_id=uuid4(),
        source=SessionSource.QQ_GROUP,
        external_key="app:untitled-group",
    )

    title, history = await repository.get_history(session_id, 50)
    assert title is None
    assert [message.role for message in history] == ["user", "assistant"]
    assert await repository.get_session_title(session_id) is None

    sessions = await repository.list_sessions(SessionSource.QQ_GROUP)
    row = next(item for item in sessions if item.session_id == session_id)
    assert row.title is None

    await repository.delete_session(session_id)


async def test_inbound_event_claim_is_idempotent(db_session: AsyncSession) -> None:
    repository = InboundEventRepository(db_session)
    event_id = f"ROBOT1.0_{uuid4().hex}" + "x" * 96
    assert len(event_id) == 137

    assert await repository.claim(SessionSource.QQ_C2C, event_id) is True
    assert await repository.claim(SessionSource.QQ_C2C, event_id) is False

    await repository.mark_completed(SessionSource.QQ_C2C, event_id)

    await db_session.execute(
        delete(InboundEvent).where(
            InboundEvent.source == SessionSource.QQ_C2C.value,
            InboundEvent.external_event_id == event_id,
        )
    )
    await db_session.commit()


async def test_persona_is_persisted_only_when_session_is_created(
    db_session: AsyncSession,
) -> None:
    personas = PersonaRepository(db_session)
    first = await personas.create(
        name=f"first-{uuid4().hex[:8]}",
        description=None,
        system_prompt="First persona.",
        messages=[],
    )
    second = await personas.create(
        name=f"second-{uuid4().hex[:8]}",
        description=None,
        system_prompt="Second persona.",
        messages=[],
    )
    repository = ChatRepository(db_session)
    session_id = uuid4()

    await repository.save_exchange(
        session_id=session_id,
        user_parts=_text_parts("hello"),
        assistant_parts=_text_parts("hi"),
        assistant_message_id=uuid4(),
        persona_id=first.persona_id,
    )
    meta = await repository.get_session_meta(session_id)
    assert meta is not None
    assert meta.persona_id == first.persona_id
    assert meta.source is SessionSource.WEB

    await repository.save_exchange(
        session_id=session_id,
        user_parts=_text_parts("again"),
        assistant_parts=_text_parts("hi again"),
        assistant_message_id=uuid4(),
        persona_id=second.persona_id,
    )
    meta = await repository.get_session_meta(session_id)
    assert meta is not None
    assert meta.persona_id == first.persona_id

    await personas.delete(first.persona_id)
    meta = await repository.get_session_meta(session_id)
    assert meta is not None
    assert meta.persona_id is None

    await repository.delete_session(session_id)
    await personas.delete(second.persona_id)


async def test_get_session_meta_returns_none_for_missing_session(
    db_session: AsyncSession,
) -> None:
    repository = ChatRepository(db_session)

    assert await repository.get_session_meta(uuid4()) is None


async def test_list_sessions_exposes_persona_name(db_session: AsyncSession) -> None:
    personas = PersonaRepository(db_session)
    persona = await personas.create(
        name=f"listed-{uuid4().hex[:8]}",
        description=None,
        system_prompt="Listed persona.",
        messages=[],
    )
    repository = ChatRepository(db_session)
    session_id = uuid4()

    await repository.save_exchange(
        session_id=session_id,
        user_parts=_text_parts("hello"),
        assistant_parts=_text_parts("hi"),
        assistant_message_id=uuid4(),
        persona_id=persona.persona_id,
    )

    sessions = await repository.list_sessions()
    row = next(item for item in sessions if item.session_id == session_id)
    assert row.persona_id == persona.persona_id
    assert row.persona_name == persona.name

    await repository.delete_session(session_id)
    await personas.delete(persona.persona_id)
