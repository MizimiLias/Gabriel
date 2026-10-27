"""FastAPI dependency providers."""

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.configs.database import SessionFactory
from app.configs.llm import get_chat_model
from app.configs.settings import get_settings
from app.repositories.chat_repository import ChatRepository
from app.repositories.persona_repository import PersonaRepository
from app.services.chat_service import ChatService
from app.services.persona_service import PersonaService


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield one database session per request."""

    async with SessionFactory() as session:
        yield session


def get_request_id(request: Request) -> str:
    """Return the request ID assigned by middleware."""

    return str(getattr(request.state, "request_id", "unknown"))


async def get_chat_service(
    db: AsyncSession = Depends(get_db),  # noqa: B008
    model: BaseChatModel = Depends(get_chat_model),  # noqa: B008
) -> ChatService:
    """Assemble the request-scoped chat service."""

    return ChatService(ChatRepository(db), model, get_settings(), PersonaRepository(db))


async def get_persona_service(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PersonaService:
    """Assemble the request-scoped persona service."""

    return PersonaService(PersonaRepository(db))
