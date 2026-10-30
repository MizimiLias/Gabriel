"""Dependency assembly for the QQ integration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.configs.agent import get_chat_agent
from app.configs.database import SessionFactory
from app.configs.llm import build_chat_model
from app.configs.settings import Settings, get_settings
from app.integrations.qq.connector import QQBotConnector
from app.integrations.qq.handler import QQMessageHandler
from app.repositories.chat_repository import ChatRepository, InboundEventRepository
from app.repositories.persona_repository import PersonaRepository
from app.services.chat_service import ChatService


@asynccontextmanager
async def chat_service_scope() -> AsyncIterator[ChatService]:
    """Create a database-backed chat service for one inbound event."""

    settings = get_settings()
    # QQ is a public channel: operators can disable tool calling for it while
    # keeping MCP tools available to the loopback-only Web API.
    agent = get_chat_agent() if settings.qq_mcp_enabled else None
    async with SessionFactory() as session:
        yield ChatService(
            ChatRepository(session),
            build_chat_model(),
            settings,
            PersonaRepository(session),
            agent=agent,
        )


@asynccontextmanager
async def inbound_event_store_scope() -> AsyncIterator[InboundEventRepository]:
    """Create a short-lived idempotency store for one event operation."""

    async with SessionFactory() as session:
        yield InboundEventRepository(session)


def build_qq_connector(settings: Settings) -> QQBotConnector:
    """Build the QQ connector from validated application settings."""

    if settings.qq_bot_app_id is None or settings.qq_bot_app_secret is None:
        raise ValueError("QQ bot credentials are required when QQ_BOT_ENABLED=true.")

    handler = QQMessageHandler(
        app_id=settings.qq_bot_app_id,
        service_scope=chat_service_scope,
        event_store_scope=inbound_event_store_scope,
    )
    return QQBotConnector(
        app_id=settings.qq_bot_app_id,
        app_secret=settings.qq_bot_app_secret.get_secret_value(),
        handler=handler,
    )
