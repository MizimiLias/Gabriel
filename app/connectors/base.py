"""Connector contracts shared by inbound chat integrations."""

from typing import Protocol


class Connector(Protocol):
    """A long-running integration that owns its own transport connection."""

    name: str

    async def run(self) -> None:
        """Run until cancelled or the transport terminates."""

    async def close(self) -> None:
        """Release transport resources. This method must be idempotent."""
