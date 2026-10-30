"""ChatService tool-calling integration tests."""

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.runtime import (
    AgentResult,
    AgentToken,
    AgentToolCalls,
    AgentToolResult,
)
from app.entities.chat_message import ChatMessage
from app.exceptions.chat import ChatUpstreamError
from app.repositories.chat_repository import ChatRepository, StoredMessageDraft
from app.services.chat_service import (
    ChatService,
    StreamDone,
    StreamStart,
    StreamToken,
    StreamToolCalls,
    StreamToolResult,
)
from tests.support import (
    FakeChatAgent,
    FakeChatModel,
    FakeChatRepository,
    make_settings,
    text_parts,
    visible_message_text,
)

_TOOL_CALL = {
    "name": "get_weather",
    "args": {"city": "北京"},
    "id": "call-1",
    "type": "tool_call",
}
_TOOL_PARTS = {
    "type": "tool_call",
    "id": "call-1",
    "name": "get_weather",
    "args": {"city": "北京"},
}
_TOOL_RESULT_PARTS = {
    "type": "tool_result",
    "tool_call_id": "call-1",
    "name": "get_weather",
    "content": "晴 20 度",
    "is_error": False,
}


def build_service(
    *,
    repository: FakeChatRepository,
    model: FakeChatModel,
    agent: FakeChatAgent | None = None,
) -> ChatService:
    return ChatService(
        cast(ChatRepository, repository),
        model.as_base_chat_model(),
        make_settings(),
        agent=agent,
    )


def _agent_result(text: str) -> AgentResult:
    return AgentResult(
        text=text,
        trace=[
            AIMessage(content="", tool_calls=[_TOOL_CALL]),
            ToolMessage(content="晴 20 度", tool_call_id="call-1", name="get_weather"),
        ],
    )


async def test_send_message_persists_tool_trace_and_final_answer() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Weather chat")
    agent = FakeChatAgent(result=_agent_result("北京晴, 20 度。"))
    service = build_service(repository=repository, model=model, agent=agent)
    session_id = uuid4()

    reply = await service.send_message(session_id, text_parts("北京天气?"))

    stored = repository.messages[session_id]
    assert [message.role for message in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1].parts == [_TOOL_PARTS]
    assert stored[2].parts == [_TOOL_RESULT_PARTS]
    assert stored[3].parts == [{"type": "text", "text": "北京晴, 20 度。"}]
    assert reply.message is stored[3]
    assert model.title_calls == 1


async def test_second_turn_replays_tool_protocol() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Weather chat")
    agent = FakeChatAgent(result=_agent_result("北京晴, 20 度。"))
    service = build_service(repository=repository, model=model, agent=agent)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("北京天气?"))
    agent.result = AgentResult(text="后天多云。", trace=[])
    await service.send_message(session_id, text_parts("后天呢?"))

    messages = agent.last_messages
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai", "human"]
    assert visible_message_text(messages[0]) == "北京天气?"

    tool_call_message = messages[1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls == [_TOOL_CALL]

    tool_message = messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "晴 20 度"
    assert tool_message.tool_call_id == "call-1"

    assert visible_message_text(messages[3]) == "北京晴, 20 度。"
    assert visible_message_text(messages[4]) == "后天呢?"


async def test_leading_partial_turn_is_trimmed_before_replay() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    agent = FakeChatAgent(result=AgentResult(text="好的。", trace=[]))
    service = build_service(repository=repository, model=model, agent=agent)
    session_id = uuid4()
    now = datetime.now(UTC)

    repository.messages[session_id] = [
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="assistant",
            parts=[_TOOL_PARTS],
            created_at=now,
        ),
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="tool",
            parts=[_TOOL_RESULT_PARTS],
            created_at=now,
        ),
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="user",
            parts=text_parts("上一轮"),
            created_at=now,
        ),
        ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="assistant",
            parts=text_parts("上一轮回答"),
            created_at=now,
        ),
    ]
    repository.titles[session_id] = "Old chat"

    await service.send_message(session_id, text_parts("继续"))

    messages = agent.last_messages
    assert isinstance(messages[0], HumanMessage)
    assert visible_message_text(messages[0]) == "上一轮"
    assert not any(isinstance(message, ToolMessage) for message in messages)
    assert [message.type for message in messages] == ["human", "ai", "human"]


async def test_stream_message_emits_tool_events_and_persists_trace() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Weather chat")
    agent = FakeChatAgent(
        events=[
            AgentToken(delta="查一下"),
            AgentToolCalls(calls=(_TOOL_CALL,), text="查一下"),
            AgentToolResult(
                tool_call_id="call-1",
                name="get_weather",
                content="晴 20 度",
                is_error=False,
            ),
            AgentToken(delta="北京晴"),
            AgentToken(delta="。"),
        ]
    )
    service = build_service(repository=repository, model=model, agent=agent)
    session_id = uuid4()

    events = [event async for event in service.stream_message(session_id, text_parts("天气?"))]

    assert isinstance(events[0], StreamStart)
    assert events[1] == StreamToken(delta="查一下")
    assert events[2] == StreamToolCalls(calls=(_TOOL_CALL,), text="查一下")
    assert events[3] == StreamToolResult(
        tool_call_id="call-1",
        name="get_weather",
        content="晴 20 度",
        is_error=False,
    )
    assert events[4] == StreamToken(delta="北京晴")
    assert events[5] == StreamToken(delta="。")
    assert isinstance(events[-1], StreamDone)

    assert repository.saved_traces[0] == [
        StoredMessageDraft(
            role="assistant",
            parts=[
                {"type": "text", "text": "查一下"},
                _TOOL_PARTS,
            ],
        ),
        StoredMessageDraft(role="tool", parts=[_TOOL_RESULT_PARTS]),
    ]
    stored = repository.messages[session_id]
    assert [message.role for message in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[-1].parts == [{"type": "text", "text": "北京晴。"}]


async def test_agent_failure_is_reported_as_upstream_error() -> None:
    repository = FakeChatRepository()
    service = build_service(
        repository=repository,
        model=FakeChatModel(),
        agent=FakeChatAgent(fail=True),
    )
    session_id = uuid4()

    with pytest.raises(ChatUpstreamError):
        await service.send_message(session_id, text_parts("hello"))

    assert repository.messages == {}


async def test_stream_agent_failure_is_reported_as_upstream_error() -> None:
    repository = FakeChatRepository()
    service = build_service(
        repository=repository,
        model=FakeChatModel(),
        agent=FakeChatAgent(fail=True),
    )
    session_id = uuid4()

    with pytest.raises(ChatUpstreamError):
        [event async for event in service.stream_message(session_id, text_parts("hello"))]

    assert repository.messages == {}
