"""QQ message routing, idempotency, and reply formatting tests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import HumanMessage

from app.entities.chat_session import SessionSource
from app.integrations.qq.handler import (
    QQInboundMessage,
    QQMessageHandler,
    resolve_qq_session,
    split_qq_reply,
)
from app.services.chat_service import ChatService
from tests.support import FakeChatModel, FakeChatRepository, make_settings


@dataclass
class FakeEventStore:
    """In-memory idempotency store matching the handler protocol."""

    claimed: set[tuple[SessionSource, str]] = field(default_factory=set)
    statuses: dict[tuple[SessionSource, str], str] = field(default_factory=dict)
    errors: dict[tuple[SessionSource, str], str] = field(default_factory=dict)

    async def claim(self, source: SessionSource, external_event_id: str) -> bool:
        key = (source, external_event_id)
        if key in self.claimed:
            return False
        self.claimed.add(key)
        return True

    async def mark_completed(self, source: SessionSource, external_event_id: str) -> None:
        self.statuses[(source, external_event_id)] = "completed"

    async def mark_failed(
        self, source: SessionSource, external_event_id: str, error_type: str
    ) -> None:
        key = (source, external_event_id)
        self.statuses[key] = "failed"
        self.errors[key] = error_type


def _inbound_message(
    *,
    source: SessionSource,
    event_id: str,
    conversation_id: str,
    content: str,
    replies: list[tuple[str, int]],
    has_attachments: bool = False,
    sender_name: str | None = None,
    sender_id: str | None = None,
) -> QQInboundMessage:
    async def reply(text: str, sequence: int) -> object:
        replies.append((text, sequence))
        return object()

    return QQInboundMessage(
        source=source,
        event_id=event_id,
        conversation_id=conversation_id,
        content=content,
        has_attachments=has_attachments,
        reply=reply,
        sender_name=sender_name,
        sender_id=sender_id,
    )


def _handler(
    repository: FakeChatRepository,
    model: FakeChatModel,
    store: FakeEventStore,
) -> QQMessageHandler:
    service = ChatService(
        repository=repository,  # type: ignore[arg-type]
        model=model.as_base_chat_model(),
        settings=make_settings(),
    )

    @asynccontextmanager
    async def service_scope() -> AsyncIterator[ChatService]:
        yield service

    @asynccontextmanager
    async def store_scope() -> AsyncIterator[FakeEventStore]:
        yield store

    return QQMessageHandler(
        app_id="app-1",
        service_scope=service_scope,
        event_store_scope=store_scope,
    )


def test_session_identity_is_stable_per_qq_conversation() -> None:
    first = resolve_qq_session("app-1", SessionSource.QQ_C2C, "user-1")
    repeated = resolve_qq_session("app-1", SessionSource.QQ_C2C, "user-1")
    other_user = resolve_qq_session("app-1", SessionSource.QQ_C2C, "user-2")
    group = resolve_qq_session("app-1", SessionSource.QQ_GROUP, "group-1")

    assert first == repeated
    assert first.session_id != other_user.session_id
    assert first.session_id != group.session_id
    assert first.external_key == "app-1:user-1"


def test_reply_splitter_obeys_part_limit_and_marks_truncation() -> None:
    parts = split_qq_reply("x" * 9_000, max_parts=5)

    assert len(parts) == 5
    assert all(len(part) <= 1_500 for part in parts)
    assert parts[-1].endswith("已截断）")  # noqa: RUF001


async def test_group_messages_share_one_session() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), store)

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="first",
            replies=replies,
        )
    )
    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-2",
            conversation_id="group-1",
            content="second",
            replies=replies,
        )
    )

    assert len(repository.messages) == 1
    session_id = next(iter(repository.messages))
    assert len(repository.messages[session_id]) == 4
    assert repository.sources[session_id] == SessionSource.QQ_GROUP
    assert repository.external_keys[session_id] == "app-1:group-1"


async def test_c2c_messages_are_isolated_by_user() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), store)

    for event_id, user_id in (("event-1", "user-1"), ("event-2", "user-2")):
        await handler.handle(
            _inbound_message(
                source=SessionSource.QQ_C2C,
                event_id=event_id,
                conversation_id=user_id,
                content="hello",
                replies=replies,
            )
        )

    assert len(repository.messages) == 2
    assert set(repository.sources.values()) == {SessionSource.QQ_C2C}


async def test_duplicate_event_is_processed_once() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), store)
    message = _inbound_message(
        source=SessionSource.QQ_C2C,
        event_id="event-1",
        conversation_id="user-1",
        content="hello",
        replies=replies,
    )

    await handler.handle(message)
    await handler.handle(message)

    assert len(repository.saved_exchanges) == 1
    assert replies == [("answer", 1)]
    assert store.statuses[(SessionSource.QQ_C2C, "event-1")] == "completed"


async def test_empty_or_attachment_message_does_not_call_model() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="unused"), store)

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="   ",
            replies=replies,
            has_attachments=True,
        )
    )

    assert repository.messages == {}
    assert replies == [("目前仅支持文本消息。", 1)]
    assert store.statuses[(SessionSource.QQ_GROUP, "event-1")] == "completed"


async def test_upstream_failure_sends_fallback_without_persisting() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(fail=True), store)

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_C2C,
            event_id="event-1",
            conversation_id="user-1",
            content="hello",
            replies=replies,
        )
    )

    assert repository.messages == {}
    assert replies == [("Gabriel 暂时无法回复，请稍后再试。", 1)]  # noqa: RUF001
    assert store.statuses[(SessionSource.QQ_C2C, "event-1")] == "failed"
    assert store.errors[(SessionSource.QQ_C2C, "event-1")] == "ChatUpstreamError"


async def test_long_reply_increments_sequence() -> None:
    repository = FakeChatRepository()
    store = FakeEventStore()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="x" * 9_000), store)

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_C2C,
            event_id="event-1",
            conversation_id="user-1",
            content="hello",
            replies=replies,
        )
    )

    assert len(replies) == 4
    assert [sequence for _, sequence in replies] == [1, 2, 3, 4]
    assert all(len(content) <= 1_500 for content, _ in replies)
    assert replies[-1][0].endswith("已截断）")  # noqa: RUF001


@pytest.mark.parametrize("source", [SessionSource.QQ_C2C, SessionSource.QQ_GROUP])
def test_splitter_returns_single_short_message(source: SessionSource) -> None:
    assert split_qq_reply(" hello ", 4 if source == SessionSource.QQ_C2C else 5) == ["hello"]


async def test_group_message_content_includes_sender_name() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="hello",
            replies=replies,
            sender_name="Alice",
            sender_id="member-1",
        )
    )

    assert repository.saved_exchanges[0][1] == "<Alice>: hello"


async def test_group_message_falls_back_to_sender_id() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="hello",
            replies=replies,
            sender_id="member-123456789",
        )
    )

    assert repository.saved_exchanges[0][1] == "<member-1>: hello"


async def test_sender_name_is_sanitized() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="hello",
            replies=replies,
            sender_name="  <Alice>\n\tAdmin  ",
        )
    )

    assert repository.saved_exchanges[0][1] == "<Alice Admin>: hello"


async def test_sender_name_is_truncated() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="hello",
            replies=replies,
            sender_name="A" * 100,
        )
    )

    assert repository.saved_exchanges[0][1] == f"<{'A' * 64}>: hello"


async def test_c2c_message_has_no_sender_prefix() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_C2C,
            event_id="event-1",
            conversation_id="user-1",
            content="hello",
            replies=replies,
            sender_name="Alice",
            sender_id="user-123456789",
        )
    )

    assert repository.saved_exchanges[0][1] == "hello"


async def test_group_message_without_sender_identity_has_no_prefix() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="hello",
            replies=replies,
        )
    )

    assert repository.saved_exchanges[0][1] == "hello"


async def test_group_history_replays_sender_prefix() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(response="answer")
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, model, FakeEventStore())

    for event_id, sender_name, content in (
        ("event-1", "Alice", "first"),
        ("event-2", "Bob", "second"),
    ):
        await handler.handle(
            _inbound_message(
                source=SessionSource.QQ_GROUP,
                event_id=event_id,
                conversation_id="group-1",
                content=content,
                replies=replies,
                sender_name=sender_name,
            )
        )

    human_contents = [
        message.content for message in model.last_messages if isinstance(message, HumanMessage)
    ]
    assert human_contents == ["<Alice>: first", "<Bob>: second"]


async def test_oversized_group_message_does_not_persist_prefixed_content() -> None:
    repository = FakeChatRepository()
    replies: list[tuple[str, int]] = []
    handler = _handler(repository, FakeChatModel(response="answer"), FakeEventStore())

    await handler.handle(
        _inbound_message(
            source=SessionSource.QQ_GROUP,
            event_id="event-1",
            conversation_id="group-1",
            content="x" * 32_001,
            replies=replies,
            sender_name="Alice",
        )
    )

    assert repository.messages == {}
    assert replies == [("消息过长，请缩短后再试。", 1)]  # noqa: RUF001
