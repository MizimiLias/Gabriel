"""Persona exception handler tests."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.exceptions.chat import (
    PersonaConflictError,
    PersonaNotFoundError,
    PersonaValidationError,
)
from app.exceptions.handlers import register_exception_handlers


def _app() -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)

    @application.get("/missing")
    async def missing() -> None:
        raise PersonaNotFoundError("The requested persona does not exist.")

    @application.get("/conflict")
    async def conflict() -> None:
        raise PersonaConflictError("A persona with this name already exists.")

    @application.get("/invalid")
    async def invalid() -> None:
        raise PersonaValidationError("Too many preset messages.")

    return application


async def test_persona_not_found_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "https://gabriel.local/problems/persona-not-found"
    assert body["title"] == "Persona not found"


async def test_persona_conflict_returns_409() -> None:
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/conflict")

    assert response.status_code == 409
    body = response.json()
    assert body["type"] == "https://gabriel.local/problems/persona-conflict"


async def test_persona_validation_returns_422() -> None:
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/invalid")

    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "https://gabriel.local/problems/persona-validation-error"
