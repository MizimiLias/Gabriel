"""LangChain chat model construction."""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.configs.settings import get_settings


@lru_cache(maxsize=1)
def build_chat_model() -> BaseChatModel:
    """Build and cache the OpenAI-compatible chat model."""

    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        temperature=settings.chat_temperature,
        timeout=settings.chat_timeout_seconds,
        max_retries=settings.chat_max_retries,
    )


async def get_chat_model() -> BaseChatModel:
    """Provide the cached model through an async FastAPI dependency."""

    return build_chat_model()
