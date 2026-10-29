"""Pydantic request model tests."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.persona import PersonaCreate, PersonaUpdate


def test_valid_chat_request() -> None:
    session_id = uuid4()

    request = ChatRequest(
        session_id=session_id,
        parts=[{"type": "text", "text": "  hello  "}],
    )

    assert request.session_id == session_id
    assert request.parts[0].type == "text"
    assert request.parts[0].text == "hello"


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [{"type": "text", "text": ""}],
        [{"type": "text", "text": "   "}],
        [{"type": "text", "text": "x" * 32_001}],
    ],
    ids=["empty-parts", "empty-text", "whitespace", "too-long"],
)
def test_invalid_parts_are_rejected(parts: list[object]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id=uuid4(), parts=parts)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "session_id": str(uuid4()),
                "parts": [{"type": "text", "text": "hello"}],
                "unexpected": True,
            }
        )


def test_chat_request_accepts_persona_id() -> None:
    persona_id = uuid4()

    request = ChatRequest(
        session_id=uuid4(),
        parts=[{"type": "text", "text": "hello"}],
        persona_id=persona_id,
    )

    assert request.persona_id == persona_id


def test_chat_request_accepts_text_and_image_parts() -> None:
    request = ChatRequest(
        session_id=uuid4(),
        parts=[
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ],
    )

    assert [part.type for part in request.parts] == ["text", "image_url"]


def test_chat_request_accepts_image_only_message() -> None:
    request = ChatRequest(
        session_id=uuid4(),
        parts=[
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            }
        ],
    )

    assert request.parts[0].type == "image_url"


def test_chat_request_rejects_invalid_image_url() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id=uuid4(),
            parts=[
                {
                    "type": "image_url",
                    "image_url": {"url": "file:///tmp/image.png"},
                }
            ],
        )


def test_chat_request_rejects_unknown_part_type() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id=uuid4(),
            parts=[{"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}],
        )


def test_chat_request_rejects_excessive_total_text() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id=uuid4(),
            parts=[
                {"type": "text", "text": "x" * 20_000},
                {"type": "text", "text": "y" * 20_000},
            ],
        )


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


def test_persona_update_requires_full_metadata() -> None:
    with pytest.raises(ValidationError):
        PersonaUpdate.model_validate({"system_prompt": "You are Ada.", "messages": []})


def test_persona_update_accepts_existing_and_new_messages() -> None:
    message_id = uuid4()

    request = PersonaUpdate(
        name="Ada",
        description=None,
        system_prompt="You are Ada.",
        messages=[
            {"id": message_id, "role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Ada."},
        ],
    )

    assert request.messages[0].id == message_id
    assert request.messages[1].id is None


def test_persona_update_rejects_new_message_before_existing_id() -> None:
    with pytest.raises(ValidationError):
        PersonaUpdate(
            name="Ada",
            description=None,
            system_prompt="You are Ada.",
            messages=[
                {"role": "assistant", "content": "I am Ada."},
                {"id": uuid4(), "role": "user", "content": "Who are you?"},
            ],
        )


def test_persona_update_rejects_duplicate_message_ids() -> None:
    message_id = uuid4()

    with pytest.raises(ValidationError):
        PersonaUpdate(
            name="Ada",
            description=None,
            system_prompt="You are Ada.",
            messages=[
                {"id": message_id, "role": "user", "content": "one"},
                {"id": message_id, "role": "assistant", "content": "two"},
            ],
        )
