"""PostgreSQL persona repository integration tests."""

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.entities.persona_message import PersonaMessage
from app.exceptions.chat import (
    PersonaConflictError,
    PersonaNotFoundError,
    PersonaValidationError,
)
from app.repositories.persona_repository import (
    PersonaMessageDraft,
    PersonaMessageMutation,
    PersonaRepository,
)

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


async def test_create_get_list_update_and_delete(db_session: AsyncSession) -> None:
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

    first_id = created.messages[0].id
    second_id = created.messages[1].id
    updated = await repository.update_with_messages(
        created.persona_id,
        name=_unique_name("Ada-updated"),
        description="updated",
        system_prompt="You are Ada, briefly.",
        messages=[
            PersonaMessageMutation(id=first_id, role="user", content="Hello again"),
            PersonaMessageMutation(id=second_id, role="assistant", content="Hello!"),
            PersonaMessageMutation(id=None, role="user", content="Be concise."),
        ],
    )
    assert updated.description == "updated"
    assert updated.system_prompt == "You are Ada, briefly."
    assert updated.messages[0].id == first_id
    assert updated.messages[1].id == second_id
    assert updated.messages[2].id not in {first_id, second_id}
    assert [message.content for message in updated.messages] == [
        "Hello again",
        "Hello!",
        "Be concise.",
    ]

    trimmed = await repository.update_with_messages(
        created.persona_id,
        name=updated.name,
        description=updated.description,
        system_prompt=updated.system_prompt,
        messages=[
            PersonaMessageMutation(id=first_id, role="user", content="Hello again"),
        ],
    )
    assert [(message.id, message.position) for message in trimmed.messages] == [(first_id, 0)]

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
        await repository.update_with_messages(
            missing,
            name="Missing",
            description=None,
            system_prompt="Missing.",
            messages=[PersonaMessageMutation(id=None, role="user", content="hello")],
        )


async def test_update_rejects_reordering_middle_deletion_and_foreign_ids(
    db_session: AsyncSession,
) -> None:
    repository = PersonaRepository(db_session)
    created = await repository.create(
        name=_unique_name("Ordered"),
        description=None,
        system_prompt="Ordered test.",
        messages=[
            PersonaMessageDraft(role="user", content="one"),
            PersonaMessageDraft(role="assistant", content="two"),
            PersonaMessageDraft(role="user", content="three"),
        ],
    )
    first, second, third = created.messages

    with pytest.raises(PersonaValidationError):
        await repository.update_with_messages(
            created.persona_id,
            name=created.name,
            description=None,
            system_prompt=created.system_prompt,
            messages=[
                PersonaMessageMutation(id=first.id, role=first.role, content=first.content),
                PersonaMessageMutation(id=third.id, role=third.role, content=third.content),
            ],
        )

    with pytest.raises(PersonaValidationError):
        await repository.update_with_messages(
            created.persona_id,
            name=created.name,
            description=None,
            system_prompt=created.system_prompt,
            messages=[
                PersonaMessageMutation(id=second.id, role=second.role, content=second.content),
                PersonaMessageMutation(id=first.id, role=first.role, content=first.content),
                PersonaMessageMutation(id=third.id, role=third.role, content=third.content),
            ],
        )

    with pytest.raises(PersonaValidationError):
        await repository.update_with_messages(
            created.persona_id,
            name=created.name,
            description=None,
            system_prompt=created.system_prompt,
            messages=[
                PersonaMessageMutation(id=uuid4(), role="user", content="foreign"),
            ],
        )

    await repository.delete(created.persona_id)


async def test_update_rolls_back_messages_when_metadata_conflicts(
    db_session: AsyncSession,
) -> None:
    repository = PersonaRepository(db_session)
    first = await repository.create(
        name=_unique_name("First"),
        description=None,
        system_prompt="First.",
        messages=[PersonaMessageDraft(role="user", content="original")],
    )
    second = await repository.create(
        name=_unique_name("Second"),
        description=None,
        system_prompt="Second.",
        messages=[],
    )

    with pytest.raises(PersonaConflictError):
        await repository.update_with_messages(
            first.persona_id,
            name=second.name,
            description=None,
            system_prompt="Changed.",
            messages=[
                PersonaMessageMutation(
                    id=first.messages[0].id,
                    role="user",
                    content="changed",
                ),
                PersonaMessageMutation(id=None, role="assistant", content="new"),
            ],
        )

    loaded = await repository.get(first.persona_id)
    assert loaded is not None
    assert loaded.system_prompt == "First."
    assert [(message.id, message.content) for message in loaded.messages] == [
        (first.messages[0].id, "original")
    ]

    await repository.delete(first.persona_id)
    await repository.delete(second.persona_id)
