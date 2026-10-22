"""Chat service memory and persistence tests."""

from typing import cast
from uuid import uuid4

import pytest

from app.repositories.chat_repository import ChatRepository
from app.services.chat_service import ChatService, StreamDone, StreamStart, StreamToken
from tests.support import FakeChatModel, FakeChatRepository, make_settings


def build_service(
    *,
    repository: FakeChatRepository,
    model: FakeChatModel,
    context_limit: int = 20,
) -> ChatService:
    return ChatService(
        cast(ChatRepository, repository),
        model.as_base_chat_model(),
        make_settings(chat_context_message_limit=context_limit),
    )


async def test_second_turn_receives_persisted_history() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(response="first answer")
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    await service.send_message(session_id, "My name is Ada")
    model.response = "Your name is Ada"
    await service.send_message(session_id, "What is my name?")

    serialized = [(message.type, message.text) for message in model.last_messages]
    assert serialized == [
        ("system", model.last_messages[0].text),
        ("human", "My name is Ada"),
        ("ai", "first answer"),
        ("human", "What is my name?"),
    ]


async def test_context_limit_keeps_only_recent_messages() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, context_limit=2)
    session_id = uuid4()

    await service.send_message(session_id, "one")
    await service.send_message(session_id, "two")
    await service.send_message(session_id, "three")

    assert [message.text for message in model.last_messages[1:]] == [
        "two",
        model.response,
        "three",
    ]


async def test_stream_persists_only_after_completion() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(chunks=["Hello", " world"])
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    events = [event async for event in service.stream_message(session_id, "Hi")]

    assert isinstance(events[0], StreamStart)
    assert [event.delta for event in events if isinstance(event, StreamToken)] == [
        "Hello",
        " world",
    ]
    assert isinstance(events[-1], StreamDone)
    assert len(repository.messages[session_id]) == 2


async def test_stream_failure_does_not_persist() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(fail=True)
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    with pytest.raises(Exception, match="could not complete"):
        _ = [event async for event in service.stream_message(session_id, "Hi")]

    assert session_id not in repository.messages
