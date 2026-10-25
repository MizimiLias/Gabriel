"""Build the enabled set of inbound connectors."""

from app.configs.settings import Settings
from app.connectors.manager import ConnectorManager
from app.integrations.qq.factory import build_qq_connector


def build_connector_manager(settings: Settings) -> ConnectorManager:
    """Assemble connectors enabled by runtime configuration."""

    connectors = []
    if settings.qq_bot_enabled:
        connectors.append(build_qq_connector(settings))
    return ConnectorManager(connectors)
