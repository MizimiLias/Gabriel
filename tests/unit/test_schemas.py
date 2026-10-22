"""Pydantic request model tests."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest


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
