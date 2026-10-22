"""Chat API upstream and SSE error tests."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.configs.dependencies import get_chat_service
from app.main import app
from app.services.chat_service import ChatService
from tests.support import FakeChatModel, FakeChatRepository, make_settings


@pytest.fixture
async def failing_client() -> AsyncIterator[AsyncClient]:
    service = ChatService(
        repository=FakeChatRepository(),  # type: ignore[arg-type]
        model=FakeChatModel(fail=True).as_base_chat_model(),
        settings=make_settings(),
    )

    async def override_service() -> ChatService:
        return service

    app.dependency_overrides[get_chat_service] = override_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_json_upstream_failure_returns_502(failing_client: AsyncClient) -> None:
    response = await failing_client.post(
        "/api/chat", json={"session_id": str(uuid4()), "message": "hello"}
    )

    assert response.status_code == 502
    assert response.json()["title"] == "Language model unavailable"


async def test_stream_upstream_failure_emits_error_event(failing_client: AsyncClient) -> None:
    response = await failing_client.post(
        "/api/chat/stream",
        json={"session_id": str(uuid4()), "message": "hello"},
    )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: error" in response.text
    assert "LLM_UPSTREAM_ERROR" in response.text
