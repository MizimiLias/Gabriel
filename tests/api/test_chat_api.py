"""Chat API contract tests using in-memory dependencies."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.configs.dependencies import get_chat_service
from app.main import app
from app.services.chat_service import ChatService
from tests.support import FakeChatModel, FakeChatRepository, make_settings


@pytest.fixture
async def api_client() -> AsyncIterator[AsyncClient]:
    repository = FakeChatRepository()
    model = FakeChatModel(response="Stored answer")
    service = ChatService(
        repository=repository,
        model=model.as_base_chat_model(),
        settings=make_settings(),
    )

    async def override_service() -> ChatService:
        return service

    app.dependency_overrides[get_chat_service] = override_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_json_chat_history_and_delete(api_client: AsyncClient) -> None:
    session_id = str(uuid4())

    response = await api_client.post(
        "/api/chat", json={"session_id": session_id, "message": "Remember this"}
    )
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Stored answer"
    assert response.headers["X-Request-ID"]

    history = await api_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert history.status_code == 200
    assert [message["role"] for message in history.json()["messages"]] == ["user", "assistant"]

    deleted = await api_client.delete(f"/api/chat/sessions/{session_id}")
    assert deleted.status_code == 204

    missing = await api_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert missing.status_code == 404
    assert missing.json()["type"].endswith("chat-session-not-found")


async def test_validation_problem_detail(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/api/chat",
        json={"session_id": str(uuid4()), "message": "hello", "extra": True},
    )

    assert response.status_code == 422
    assert response.json()["title"] == "Request validation failed"
    assert response.json()["request_id"] != "unknown"


async def test_stream_event_order(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/api/chat/stream",
        json={"session_id": str(uuid4()), "message": "Stream this"},
    )

    assert response.status_code == 200
    body = response.text
    assert body.index("event: start") < body.index("event: token") < body.index("event: done")
    assert '"delta":"Hello"' in body


async def test_health_live() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
