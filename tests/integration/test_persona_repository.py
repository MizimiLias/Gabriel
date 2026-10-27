"""PostgreSQL persona repository integration tests."""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.entities.persona_message import PersonaMessage
from app.exceptions.chat import PersonaConflictError, PersonaNotFoundError
from app.repositories.persona_repository import PersonaMessageDraft, PersonaRepository

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


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


async def test_create_get_list_replace_and_delete(db_session: AsyncSession) -> None:
    repository = PersonaRepository(db_session)
    name = _unique_name("Ada")

    created = await repository.create(
        name=name,
        description="concise engineering assistant",
        system_prompt="You are Ada.",
        messages=[
            PersonaMessageDraft(role="user", content="Who are you?"),
            PersonaMessageDraft(role="assistant", content="I am Ada."),
        ],
    )

    assert created.name == name
    assert created.description == "concise engineering assistant"
    assert [message.position for message in created.messages] == [0, 1]
    assert [message.role for message in created.messages] == ["user", "assistant"]

    loaded = await repository.get(created.persona_id)
    assert loaded == created

    rows = await repository.list_all()
    row = next(item for item in rows if item.persona_id == created.persona_id)
    assert row.message_count == 2

    replaced = await repository.replace_messages(
        created.persona_id,
        [PersonaMessageDraft(role="user", content="Hi")],
    )
    assert [(message.position, message.role, message.content) for message in replaced.messages] == [
        (0, "user", "Hi")
    ]

    updated = await repository.update(
        created.persona_id,
        name=_unique_name("Ada-updated"),
        description=None,
        system_prompt="You are Ada, briefly.",
    )
    assert updated.description is None
    assert updated.system_prompt == "You are Ada, briefly."

    await repository.delete(created.persona_id)
    assert await repository.get(created.persona_id) is None

    with pytest.raises(PersonaNotFoundError):
        await repository.delete(created.persona_id)


async def test_duplicate_name_is_reported_as_conflict(db_session: AsyncSession) -> None:
    repository = PersonaRepository(db_session)
    name = _unique_name("Gabriel")

    created = await repository.create(
        name=name,
        description=None,
        system_prompt="You are Gabriel.",
        messages=[],
    )

    with pytest.raises(PersonaConflictError):
        await repository.create(
            name=name,
            description=None,
            system_prompt="Another Gabriel.",
            messages=[],
        )

    await repository.delete(created.persona_id)


async def test_delete_cascades_preset_messages(db_session: AsyncSession) -> None:
    repository = PersonaRepository(db_session)

    created = await repository.create(
        name=_unique_name("Cascade"),
        description=None,
        system_prompt="Cascade test.",
        messages=[PersonaMessageDraft(role="user", content="hello")],
    )

    await repository.delete(created.persona_id)

    count = await db_session.scalar(
        select(func.count())
        .select_from(PersonaMessage)
        .where(PersonaMessage.persona_id == created.persona_id)
    )
    assert count == 0


async def test_missing_persona_operations_raise_not_found(db_session: AsyncSession) -> None:
    repository = PersonaRepository(db_session)
    missing = uuid4()

    assert await repository.get(missing) is None

    with pytest.raises(PersonaNotFoundError):
        await repository.update(
            missing,
            name="Missing",
            description=None,
            system_prompt="Missing.",
        )

    with pytest.raises(PersonaNotFoundError):
        await repository.replace_messages(
            missing, [PersonaMessageDraft(role="user", content="hello")]
        )
