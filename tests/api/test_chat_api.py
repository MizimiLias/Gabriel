"""Chat API contract tests using in-memory dependencies."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.configs.dependencies import get_chat_service
from app.entities.chat_session import SessionSource
from app.main import app
from app.services.chat_service import ChatService
from tests.support import (
    FakeChatModel,
    FakeChatRepository,
    FakePersonaRepository,
    make_settings,
)


@pytest.fixture
async def api_context() -> AsyncIterator[
    tuple[AsyncClient, FakeChatRepository, FakePersonaRepository]
]:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    model = FakeChatModel(response="Stored answer", title_response="Remembering details")
    service = ChatService(
        repository=repository,  # type: ignore[arg-type]
        model=model.as_base_chat_model(),
        settings=make_settings(),
        persona_repository=personas,  # type: ignore[arg-type]
    )

    async def override_service() -> ChatService:
        return service

    app.dependency_overrides[get_chat_service] = override_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, repository, personas
    app.dependency_overrides.clear()


@pytest.fixture
async def api_client(
    api_context: tuple[AsyncClient, FakeChatRepository, FakePersonaRepository],
) -> AsyncClient:
    return api_context[0]


async def test_json_chat_history_and_delete(api_client: AsyncClient) -> None:
    session_id = str(uuid4())

    response = await api_client.post(
        "/api/chat", json={"session_id": session_id, "message": "Remember this"}
    )
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Stored answer"
    assert response.json()["title"] == "Remembering details"
    assert response.headers["X-Request-ID"]

    sessions = await api_client.get("/api/chat/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["items"][0]["session_id"] == session_id
    assert sessions.json()["items"][0]["message_count"] == 2
    assert sessions.json()["items"][0]["source"] == "web"

    history = await api_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert history.status_code == 200
    assert history.json()["title"] == "Remembering details"
    assert [message["role"] for message in history.json()["messages"]] == ["user", "assistant"]

    renamed = await api_client.patch(
        f"/api/chat/sessions/{session_id}", json={"title": "  A better title  "}
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "A better title"

    deleted = await api_client.delete(f"/api/chat/sessions/{session_id}")
    assert deleted.status_code == 204

    missing = await api_client.get(f"/api/chat/sessions/{session_id}/messages")
    assert missing.status_code == 404
    assert missing.json()["type"].endswith("chat-session-not-found")


async def test_rename_missing_session_returns_problem_detail(api_client: AsyncClient) -> None:
    response = await api_client.patch(f"/api/chat/sessions/{uuid4()}", json={"title": "Missing"})
    assert response.status_code == 404
    assert response.json()["title"] == "Chat session not found"


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
    assert '"title":"Remembering details"' in body


async def test_health_live() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_session_list_hides_qq_sessions_by_default(
    api_context: tuple[AsyncClient, FakeChatRepository, FakePersonaRepository],
) -> None:
    client, repository, _personas = api_context
    web_session_id = uuid4()
    qq_session_id = uuid4()
    await repository.save_exchange(
        session_id=web_session_id,
        user_content="web",
        assistant_content="answer",
        assistant_message_id=uuid4(),
    )
    await repository.save_exchange(
        session_id=qq_session_id,
        user_content="qq",
        assistant_content="answer",
        assistant_message_id=uuid4(),
        source=SessionSource.QQ_GROUP,
        external_key="app:group",
    )

    default_sessions = await client.get("/api/chat/sessions")
    qq_sessions = await client.get("/api/chat/sessions?source=qq_group")

    assert [item["session_id"] for item in default_sessions.json()["items"]] == [
        str(web_session_id)
    ]
    assert [item["session_id"] for item in qq_sessions.json()["items"]] == [str(qq_session_id)]
    assert qq_sessions.json()["items"][0]["source"] == "qq_group"


async def test_chat_persona_id_is_persisted_and_listed(
    api_context: tuple[AsyncClient, FakeChatRepository, FakePersonaRepository],
) -> None:
    client, repository, personas = api_context
    persona = await personas.create(
        name="Ada",
        description=None,
        system_prompt="You are Ada.",
        messages=[],
    )
    repository.persona_names[persona.persona_id] = persona.name
    session_id = uuid4()

    response = await client.post(
        "/api/chat",
        json={
            "session_id": str(session_id),
            "message": "hello",
            "persona_id": str(persona.persona_id),
        },
    )

    assert response.status_code == 200
    assert repository.persona_ids[session_id] == persona.persona_id

    sessions = await client.get("/api/chat/sessions")
    item = next(row for row in sessions.json()["items"] if row["session_id"] == str(session_id))
    assert item["persona_id"] == str(persona.persona_id)
    assert item["persona_name"] == "Ada"


async def test_stream_unknown_persona_emits_error_event(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/api/chat/stream",
        json={
            "session_id": str(uuid4()),
            "message": "hello",
            "persona_id": str(uuid4()),
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "PERSONA_NOT_FOUND" in response.text
