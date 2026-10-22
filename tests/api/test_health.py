"""Health endpoint failure behavior tests."""

from collections.abc import AsyncIterator
from typing import cast

from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.configs.dependencies import get_db
from app.main import app


class FailingSession:
    async def execute(self, _: object) -> None:
        raise SQLAlchemyError("database unavailable")


async def test_readiness_returns_503_when_database_query_fails() -> None:
    async def override_db() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, FailingSession())

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["title"] == "Database unavailable"
