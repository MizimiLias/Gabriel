"""Chat API tests for visible tool traces."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.runtime import AgentResult, AgentToken, AgentToolCalls, AgentToolResult
from app.configs.dependencies import get_chat_service
from app.main import app
from app.services.chat_service import ChatService
from tests.support import (
    FakeChatAgent,
    FakeChatModel,
    FakeChatRepository,
    make_settings,
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


def _text_parts(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


@pytest.fixture
async def api_context() -> AsyncIterator[tuple[AsyncClient, FakeChatRepository, FakeChatAgent]]:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Weather")
    agent = FakeChatAgent(
        result=AgentResult(
            text="北京晴, 20 度。",
            trace=[
                AIMessage(content="", tool_calls=[_TOOL_CALL]),
                ToolMessage(content="晴 20 度", tool_call_id="call-1", name="get_weather"),
            ],
        )
    )
    service = ChatService(
        repository=repository,  # type: ignore[arg-type]
        model=model.as_base_chat_model(),
        settings=make_settings(),
        agent=agent,
    )

    async def override_service() -> ChatService:
        return service

    app.dependency_overrides[get_chat_service] = override_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, repository, agent
    app.dependency_overrides.clear()


async def test_history_exposes_tool_trace(
    api_context: tuple[AsyncClient, FakeChatRepository, FakeChatAgent],
) -> None:
    client, _repository, _agent = api_context
    session_id = str(uuid4())

    response = await client.post(
        "/api/chat",
        json={"session_id": session_id, "parts": _text_parts("北京天气?")},
    )
    assert response.status_code == 200

    history = await client.get(f"/api/chat/sessions/{session_id}/messages")

    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["parts"] == [_TOOL_PARTS]
    assert messages[2]["parts"] == [_TOOL_RESULT_PARTS]
    assert messages[3]["parts"] == [{"type": "text", "text": "北京晴, 20 度。"}]


async def test_stream_emits_tool_events(
    api_context: tuple[AsyncClient, FakeChatRepository, FakeChatAgent],
) -> None:
    client, _repository, agent = api_context
    agent.events = [
        AgentToken(delta="查一下"),
        AgentToolCalls(calls=(_TOOL_CALL,), text="查一下"),
        AgentToolResult(
            tool_call_id="call-1",
            name="get_weather",
            content="晴 20 度",
            is_error=False,
        ),
        AgentToken(delta="北京晴"),
    ]

    response = await client.post(
        "/api/chat/stream",
        json={"session_id": str(uuid4()), "parts": _text_parts("北京天气?")},
    )

    assert response.status_code == 200
    body = response.text
    assert body.index("event: start") < body.index("event: token")
    assert body.index("event: token") < body.index("event: tool_call")
    assert body.index("event: tool_call") < body.index("event: tool_result")
    assert body.index("event: tool_result") < body.index("event: done")
    assert '"tool_call_id":"call-1"' in body
    assert '"name":"get_weather"' in body
    assert '"city":"北京"' in body
    assert '"content":"晴 20 度"' in body


async def test_request_rejects_client_supplied_tool_parts(
    api_context: tuple[AsyncClient, FakeChatRepository, FakeChatAgent],
) -> None:
    client, _repository, _agent = api_context

    response = await client.post(
        "/api/chat",
        json={
            "session_id": str(uuid4()),
            "parts": [{"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {}}],
        },
    )

    assert response.status_code == 422
