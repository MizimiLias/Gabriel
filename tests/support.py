"""Shared test doubles and settings helpers."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.exceptions.chat import ChatSessionNotFoundError


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "db_host": "localhost",
        "db_password": "test-password",
        "chat_base_url": "https://example.invalid/v1",
        "chat_api_key": "test-key",
        "chat_model": "test-model",
        "chat_context_message_limit": 20,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeChatModel:
    """Minimal async chat model that captures the messages it receives."""

    def __init__(
        self,
        response: str = "Hello from the fake model.",
        *,
        fail: bool = False,
        chunks: list[str] | None = None,
    ) -> None:
        self.response = response
        self.fail = fail
        self.chunks = chunks or ["Hello", " ", "world"]
        self.last_messages: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("provider failed")
        return AIMessage(content=self.response)

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("provider failed")
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)

    def as_base_chat_model(self) -> BaseChatModel:
        return cast(BaseChatModel, self)


class FakeChatRepository:
    """In-memory repository matching the interface used by ChatService."""

    def __init__(self) -> None:
        self.messages: dict[UUID, list[ChatMessage]] = {}
        self.saved_exchanges: list[tuple[UUID, str, str]] = []

    async def get_recent_messages(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        return self.messages.get(session_id, [])[-limit:]

    async def get_history(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        if session_id not in self.messages:
            raise ChatSessionNotFoundError("The requested chat session does not exist.")
        return self.messages[session_id][-limit:]

    async def save_exchange(
        self,
        *,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_message_id: UUID,
    ) -> ChatMessage:
        now = datetime.now(UTC)
        user_message = ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="user",
            content=user_content,
            created_at=now,
        )
        assistant_message = ChatMessage(
            id=assistant_message_id,
            session_id=session_id,
            role="assistant",
            content=assistant_content,
            created_at=now,
        )
        self.messages.setdefault(session_id, []).extend([user_message, assistant_message])
        self.saved_exchanges.append((session_id, user_content, assistant_content))
        return assistant_message

    async def delete_session(self, session_id: UUID) -> None:
        self.messages.pop(session_id, None)
