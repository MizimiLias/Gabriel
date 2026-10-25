"""Extensible inbound connector lifecycle management."""

from app.connectors.base import Connector
from app.connectors.manager import ConnectorManager

__all__ = ["Connector", "ConnectorManager"]
