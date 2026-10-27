"""Persona API contract tests using in-memory dependencies."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.configs.dependencies import get_persona_service
from app.main import app
from app.services.persona_service import PersonaService
from tests.support import FakePersonaRepository


@pytest.fixture
async def api_client() -> AsyncIterator[AsyncClient]:
    repository = FakePersonaRepository()
    service = PersonaService(repository=repository)  # type: ignore[arg-type]

    async def override_service() -> PersonaService:
        return service

    app.dependency_overrides[get_persona_service] = override_service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_create_list_get_patch_replace_and_delete(api_client: AsyncClient) -> None:
    created = await api_client.post(
        "/api/personas",
        json={
            "name": "Ada",
            "description": "engineering assistant",
            "system_prompt": "You are Ada.",
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Ada."},
            ],
        },
    )
    assert created.status_code == 201
    created_body = created.json()
    persona_id = created_body["persona_id"]
    assert [message["position"] for message in created_body["messages"]] == [0, 1]

    listed = await api_client.get("/api/personas")
    assert listed.status_code == 200
    summary = next(item for item in listed.json()["items"] if item["persona_id"] == persona_id)
    assert summary["name"] == "Ada"
    assert summary["message_count"] == 2

    detail = await api_client.get(f"/api/personas/{persona_id}")
    assert detail.status_code == 200
    assert detail.json()["system_prompt"] == "You are Ada."

    patched = await api_client.patch(
        f"/api/personas/{persona_id}",
        json={"system_prompt": "You are Ada, briefly."},
    )
    assert patched.status_code == 200
    patched_body = patched.json()
    assert patched_body["name"] == "Ada"
    assert patched_body["system_prompt"] == "You are Ada, briefly."
    assert len(patched_body["messages"]) == 2

    replaced = await api_client.put(
        f"/api/personas/{persona_id}/messages",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert replaced.status_code == 200
    assert [
        (message["position"], message["content"]) for message in replaced.json()["messages"]
    ] == [(0, "Hi")]

    deleted = await api_client.delete(f"/api/personas/{persona_id}")
    assert deleted.status_code == 204

    missing = await api_client.get(f"/api/personas/{persona_id}")
    assert missing.status_code == 404


async def test_duplicate_name_returns_conflict(api_client: AsyncClient) -> None:
    payload = {"name": "Ada", "system_prompt": "You are Ada.", "messages": []}

    first = await api_client.post("/api/personas", json=payload)
    second = await api_client.post("/api/personas", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["type"] == "https://gabriel.local/problems/persona-conflict"


async def test_unknown_persona_returns_not_found(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/personas/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["type"] == "https://gabriel.local/problems/persona-not-found"


async def test_invalid_payload_returns_422(api_client: AsyncClient) -> None:
    extra_field = await api_client.post(
        "/api/personas",
        json={
            "name": "Ada",
            "system_prompt": "You are Ada.",
            "messages": [],
            "unexpected": True,
        },
    )
    empty_message = await api_client.post(
        "/api/personas",
        json={
            "name": "Ada",
            "system_prompt": "You are Ada.",
            "messages": [{"role": "user", "content": ""}],
        },
    )
    too_many = await api_client.post(
        "/api/personas",
        json={
            "name": "Ada",
            "system_prompt": "You are Ada.",
            "messages": [{"role": "user", "content": "x"} for _ in range(21)],
        },
    )

    assert extra_field.status_code == 422
    assert empty_message.status_code == 422
    assert too_many.status_code == 422


async def test_explicit_null_name_is_rejected(api_client: AsyncClient) -> None:
    created = await api_client.post(
        "/api/personas",
        json={"name": "Ada", "system_prompt": "You are Ada.", "messages": []},
    )
    persona_id = created.json()["persona_id"]

    response = await api_client.patch(f"/api/personas/{persona_id}", json={"name": None})

    assert response.status_code == 422


async def test_empty_patch_is_a_noop(api_client: AsyncClient) -> None:
    created = await api_client.post(
        "/api/personas",
        json={"name": "Ada", "system_prompt": "You are Ada.", "messages": []},
    )
    persona_id = created.json()["persona_id"]

    response = await api_client.patch(f"/api/personas/{persona_id}", json={})

    assert response.status_code == 200
    assert response.json()["system_prompt"] == "You are Ada."
