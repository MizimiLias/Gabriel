"""Pydantic request model tests."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.persona import PersonaCreate, PersonaUpdate


def test_valid_chat_request() -> None:
    session_id = uuid4()

    request = ChatRequest(session_id=session_id, message="  hello  ")

    assert request.session_id == session_id
    assert request.message == "hello"


@pytest.mark.parametrize(
    "message",
    ["", "   ", "x" * 32_001],
    ids=["empty", "whitespace", "too-long"],
)
def test_invalid_message_is_rejected(message: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id=uuid4(), message=message)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {"session_id": str(uuid4()), "message": "hello", "unexpected": True}
        )


def test_chat_request_accepts_persona_id() -> None:
    persona_id = uuid4()

    request = ChatRequest(session_id=uuid4(), message="hello", persona_id=persona_id)

    assert request.persona_id == persona_id


def test_persona_create_rejects_oversized_preset_dialogue() -> None:
    with pytest.raises(ValidationError):
        PersonaCreate(
            name="Ada",
            system_prompt="You are Ada.",
            messages=[{"role": "user", "content": "x" * 8_000} for _ in range(5)],
        )


def test_persona_create_rejects_system_role_preset() -> None:
    with pytest.raises(ValidationError):
        PersonaCreate(
            name="Ada",
            system_prompt="You are Ada.",
            messages=[{"role": "system", "content": "hello"}],
        )


def test_persona_update_rejects_explicit_null_system_prompt() -> None:
    with pytest.raises(ValidationError):
        PersonaUpdate(system_prompt=None)
