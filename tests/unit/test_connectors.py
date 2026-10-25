"""Connector lifecycle and restart supervision tests."""

import asyncio
from contextlib import suppress

import pytest

from app.connectors.manager import ConnectorManager
from app.integrations.qq.connector import QQBotConnector
from app.integrations.qq.handler import QQMessageHandler


class FakeClient:
    """Minimal botpy client double."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self, appid: str, secret: str, ret_coro: bool = False) -> object:
        self.started = appid == "app" and secret == "secret"
        return object()

    async def close(self) -> None:
        self.closed = True


class BlockingFakeClient(FakeClient):
    """Fake client whose connection remains active until cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.connected = asyncio.Event()

    async def start(self, appid: str, secret: str, ret_coro: bool = False) -> object:
        self.started = appid == "app" and secret == "secret"
        self.connected.set()
        await asyncio.Event().wait()
        return object()


class BlockingConnector:
    """Connector double that blocks until stopped."""

    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def run(self) -> None:
        self.started.set()
        await self.release.wait()

    async def close(self) -> None:
        self.closed = True


class FailingConnector(BlockingConnector):
    """Connector double that fails once before blocking."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def run(self) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("connection failed")
        await super().run()


async def test_manager_starts_and_closes_connector() -> None:
    connector = BlockingConnector()
    manager = ConnectorManager([connector])  # type: ignore[list-item]

    await manager.start()
    await asyncio.wait_for(connector.started.wait(), timeout=1)
    await manager.stop()

    assert connector.closed is True


async def test_manager_retries_failed_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.connectors.manager._INITIAL_RETRY_SECONDS", 0.01)
    connector = FailingConnector()
    manager = ConnectorManager([connector])  # type: ignore[list-item]

    await manager.start()
    await asyncio.wait_for(connector.started.wait(), timeout=1)
    await manager.stop()

    assert connector.attempts == 2
    assert connector.closed is True


async def test_qq_connector_closes_sdk_client() -> None:
    handler = QQMessageHandler(
        app_id="app",
        service_scope=lambda: None,  # type: ignore[arg-type,return-value]
        event_store_scope=lambda: None,  # type: ignore[arg-type,return-value]
    )
    client = FakeClient()
    connector = QQBotConnector(
        app_id="app",
        app_secret="secret",
        handler=handler,
        client_factory=lambda _handler: client,
    )

    await connector.run()

    assert client.started is True
    assert client.closed is True


async def test_qq_connector_closes_sdk_client_on_cancellation() -> None:
    handler = QQMessageHandler(
        app_id="app",
        service_scope=lambda: None,  # type: ignore[arg-type,return-value]
        event_store_scope=lambda: None,  # type: ignore[arg-type,return-value]
    )
    client = BlockingFakeClient()
    connector = QQBotConnector(
        app_id="app",
        app_secret="secret",
        handler=handler,
        client_factory=lambda _handler: client,
    )

    task = asyncio.create_task(connector.run())
    await asyncio.wait_for(client.connected.wait(), timeout=1)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert client.closed is True
