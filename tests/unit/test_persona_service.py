"""Persona service validation and orchestration tests."""

from typing import cast

import pytest

from app.exceptions.chat import PersonaConflictError, PersonaNotFoundError, PersonaValidationError
from app.personas import (
    MAX_PERSONA_MESSAGE_LENGTH,
    MAX_PERSONA_MESSAGES,
    MAX_PERSONA_SYSTEM_PROMPT_LENGTH,
)
from app.repositories.persona_repository import PersonaMessageDraft, PersonaRepository
from app.services.persona_service import PersonaService
from tests.support import FakePersonaRepository


def build_service(repository: FakePersonaRepository) -> PersonaService:
    return PersonaService(cast(PersonaRepository, repository))


async def test_create_and_get_persona() -> None:
    service = build_service(FakePersonaRepository())

    created = await service.create(
        name="Ada",
        description="engineering assistant",
        system_prompt="You are Ada.",
        messages=[
            PersonaMessageDraft(role="user", content="Who are you?"),
            PersonaMessageDraft(role="assistant", content="I am Ada."),
        ],
    )

    loaded = await service.get(created.persona_id)
    assert loaded.name == "Ada"
    assert [message.position for message in loaded.messages] == [0, 1]


async def test_create_rejects_duplicate_name() -> None:
    service = build_service(FakePersonaRepository())
    await service.create(name="Ada", description=None, system_prompt="You are Ada.", messages=[])

    with pytest.raises(PersonaConflictError):
        await service.create(
            name="Ada", description=None, system_prompt="Another Ada.", messages=[]
        )


async def test_update_keeps_unset_fields_and_messages() -> None:
    service = build_service(FakePersonaRepository())
    created = await service.create(
        name="Ada",
        description="original",
        system_prompt="You are Ada.",
        messages=[PersonaMessageDraft(role="user", content="Hi")],
    )

    updated = await service.update(created.persona_id, system_prompt="You are Ada, briefly.")
    assert updated.name == "Ada"
    assert updated.description == "original"
    assert updated.system_prompt == "You are Ada, briefly."
    assert [message.content for message in updated.messages] == ["Hi"]

    cleared = await service.update(created.persona_id, description=None)
    assert cleared.description is None


async def test_empty_update_returns_current_persona() -> None:
    service = build_service(FakePersonaRepository())
    created = await service.create(
        name="Ada", description=None, system_prompt="You are Ada.", messages=[]
    )

    unchanged = await service.update(created.persona_id)
    assert unchanged == created


async def test_replace_messages_preserves_order() -> None:
    service = build_service(FakePersonaRepository())
    created = await service.create(
        name="Ada", description=None, system_prompt="You are Ada.", messages=[]
    )

    replaced = await service.replace_messages(
        created.persona_id,
        [
            PersonaMessageDraft(role="user", content="first"),
            PersonaMessageDraft(role="assistant", content="second"),
            PersonaMessageDraft(role="user", content="third"),
        ],
    )

    assert [(message.position, message.content) for message in replaced.messages] == [
        (0, "first"),
        (1, "second"),
        (2, "third"),
    ]


async def test_delete_and_missing_persona_errors() -> None:
    service = build_service(FakePersonaRepository())
    created = await service.create(
        name="Ada", description=None, system_prompt="You are Ada.", messages=[]
    )

    await service.delete(created.persona_id)

    with pytest.raises(PersonaNotFoundError):
        await service.get(created.persona_id)
    with pytest.raises(PersonaNotFoundError):
        await service.update(created.persona_id, system_prompt="Missing.")
    with pytest.raises(PersonaNotFoundError):
        await service.delete(created.persona_id)


async def test_validation_rejects_oversized_inputs() -> None:
    service = build_service(FakePersonaRepository())

    with pytest.raises(PersonaValidationError):
        await service.create(name="   ", description=None, system_prompt="x", messages=[])
    with pytest.raises(PersonaValidationError):
        await service.create(name="x" * 81, description=None, system_prompt="x", messages=[])
    with pytest.raises(PersonaValidationError):
        await service.create(
            name="Ada",
            description=None,
            system_prompt="x" * (MAX_PERSONA_SYSTEM_PROMPT_LENGTH + 1),
            messages=[],
        )
    with pytest.raises(PersonaValidationError):
        await service.create(
            name="Ada",
            description=None,
            system_prompt="You are Ada.",
            messages=[
                PersonaMessageDraft(role="user", content="x")
                for _ in range(MAX_PERSONA_MESSAGES + 1)
            ],
        )
    with pytest.raises(PersonaValidationError):
        await service.create(
            name="Ada",
            description=None,
            system_prompt="You are Ada.",
            messages=[
                PersonaMessageDraft(role="user", content="x" * (MAX_PERSONA_MESSAGE_LENGTH + 1))
            ],
        )
    with pytest.raises(PersonaValidationError):
        await service.create(
            name="Ada",
            description=None,
            system_prompt="You are Ada.",
            messages=[
                PersonaMessageDraft(role="user", content="x" * MAX_PERSONA_MESSAGE_LENGTH)
                for _ in range(5)
            ],
        )
