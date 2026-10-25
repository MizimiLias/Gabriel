"""Task supervision for independently restartable connectors."""

import asyncio
from collections.abc import Sequence

import structlog

from app.connectors.base import Connector

logger = structlog.get_logger(__name__)

_INITIAL_RETRY_SECONDS = 1.0
_MAX_RETRY_SECONDS = 60.0


class ConnectorManager:
    """Start connectors in the FastAPI event loop and restart failed transports."""

    def __init__(self, connectors: Sequence[Connector]) -> None:
        self._connectors = list(connectors)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False

    async def start(self) -> None:
        """Start one supervisor task per configured connector."""

        if self._tasks:
            return
        self._stopping = False
        for connector in self._connectors:
            self._tasks.append(
                asyncio.create_task(self._supervise(connector), name=f"connector:{connector.name}")
            )

    async def stop(self) -> None:
        """Cancel supervisors and close every connector."""

        self._stopping = True
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for connector in self._connectors:
            try:
                await connector.close()
            except Exception:
                logger.exception("connector_close_failed", connector=connector.name)

    async def _supervise(self, connector: Connector) -> None:
        retry_seconds = _INITIAL_RETRY_SECONDS
        while not self._stopping:
            try:
                logger.info("connector_starting", connector=connector.name)
                await connector.run()
                if self._stopping:
                    return
                logger.warning("connector_stopped", connector=connector.name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "connector_failed",
                    connector=connector.name,
                    error_type=type(exc).__name__,
                    retry_in_seconds=retry_seconds,
                )

            try:
                await asyncio.sleep(retry_seconds)
            except asyncio.CancelledError:
                raise
            retry_seconds = min(retry_seconds * 2, _MAX_RETRY_SECONDS)
