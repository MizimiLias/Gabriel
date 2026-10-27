"""Chat service memory, title, and persistence tests."""

from typing import cast
from uuid import uuid4

import pytest

from app.entities.chat_session import SessionSource
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


async def test_first_exchange_generates_title_once() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Ada's name")
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, "My name is Ada")
    assert reply.title == "Ada's name"
    assert repository.titles[session_id] == "Ada's name"

    model.title_response = "A different title"
    await service.send_message(session_id, "What is my name?")
    assert repository.titles[session_id] == "Ada's name"


async def test_title_generation_failure_uses_first_message() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(fail_title=True)
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, "  Plan a quiet weekend  ")
    assert reply.title == "Plan a quiet weekend"
    assert len(repository.messages[session_id]) == 2


@pytest.mark.parametrize("source", [SessionSource.QQ_C2C, SessionSource.QQ_GROUP])
async def test_non_web_source_skips_title_generation(source: SessionSource) -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, "hello", source=source)

    assert reply.title is None
    assert repository.titles[session_id] is None
    assert model.title_calls == 0


@pytest.mark.parametrize("source", [SessionSource.QQ_C2C, SessionSource.QQ_GROUP])
async def test_non_web_stream_skips_title_generation(source: SessionSource) -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    events = [event async for event in service.stream_message(session_id, "Hi", source=source)]

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.title is None
    assert repository.titles[session_id] is None
    assert model.title_calls == 0


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
    assert events[-1].title == "Test conversation"
    assert len(repository.messages[session_id]) == 2


async def test_stream_failure_does_not_persist() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(fail=True)
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    with pytest.raises(Exception, match="could not complete"):
        _ = [event async for event in service.stream_message(session_id, "Hi")]

    assert session_id not in repository.messages


async def test_external_session_identity_is_persisted() -> None:
    repository = FakeChatRepository()
    service = build_service(repository=repository, model=FakeChatModel())
    session_id = uuid4()

    await service.send_message(
        session_id,
        "hello",
        source=SessionSource.QQ_C2C,
        external_key="app:user",
    )

    assert repository.sources[session_id] == SessionSource.QQ_C2C
    assert repository.external_keys[session_id] == "app:user"
    assert repository.titles[session_id] is None
