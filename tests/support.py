"""Shared test doubles and settings helpers."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.entities.chat_session import SessionSource
from app.exceptions.chat import ChatSessionNotFoundError
from app.prompts import TITLE_SYSTEM_PROMPT
from app.repositories.chat_repository import ChatSessionSummaryRow


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "db_host": "localhost",
        "db_password": "test-password",
        "chat_base_url": "https://example.invalid/v1",
        "chat_api_key": "test-key",
        "chat_model": "test-model",
        "chat_context_message_limit": 20,
        "qq_bot_enabled": False,
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
        title_response: str = "Test conversation",
        fail_title: bool = False,
    ) -> None:
        self.response = response
        self.fail = fail
        self.chunks = chunks or ["Hello", " ", "world"]
        self.title_response = title_response
        self.fail_title = fail_title
        self.title_calls = 0
        self.last_messages: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.last_messages = messages
        if messages and messages[0].text == TITLE_SYSTEM_PROMPT:
            self.title_calls += 1
            if self.fail_title:
                raise RuntimeError("title provider failed")
            return AIMessage(content=self.title_response)
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
        self.titles: dict[UUID, str | None] = {}
        self.sources: dict[UUID, SessionSource] = {}
        self.external_keys: dict[UUID, str | None] = {}
        self.saved_exchanges: list[tuple[UUID, str, str]] = []

    async def get_recent_messages(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        return self.messages.get(session_id, [])[-limit:]

    async def get_history(
        self, session_id: UUID, limit: int
    ) -> tuple[str | None, list[ChatMessage]]:
        if session_id not in self.messages:
            raise ChatSessionNotFoundError("The requested chat session does not exist.")
        return self.titles.get(session_id), self.messages[session_id][-limit:]

    async def get_session_title(self, session_id: UUID) -> str | None:
        return self.titles.get(session_id)

    async def session_exists(self, session_id: UUID) -> bool:
        return session_id in self.messages

    async def list_sessions(
        self, source: SessionSource = SessionSource.WEB
    ) -> list[ChatSessionSummaryRow]:
        rows: list[ChatSessionSummaryRow] = []
        for session_id, messages in self.messages.items():
            if self.sources.get(session_id, SessionSource.WEB) != source:
                continue
            last_message = messages[-1]
            rows.append(
                ChatSessionSummaryRow(
                    session_id=session_id,
                    title=self.titles.get(session_id),
                    source=source,
                    created_at=messages[0].created_at,
                    updated_at=last_message.created_at,
                    message_count=len(messages),
                    last_message_preview=last_message.content[:160],
                )
            )
        return sorted(rows, key=lambda row: row.updated_at, reverse=True)

    async def save_exchange(
        self,
        *,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_message_id: UUID,
        title: str | None = None,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
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
        self.titles.setdefault(session_id, title)
        self.sources.setdefault(session_id, source)
        self.external_keys.setdefault(session_id, external_key)
        self.saved_exchanges.append((session_id, user_content, assistant_content))
        return assistant_message

    async def update_session_title(self, session_id: UUID, title: str) -> ChatSessionSummaryRow:
        if session_id not in self.messages:
            raise ChatSessionNotFoundError("The requested chat session does not exist.")
        self.titles[session_id] = title
        rows = await self.list_sessions(self.sources.get(session_id, SessionSource.WEB))
        return next(row for row in rows if row.session_id == session_id)

    async def delete_session(self, session_id: UUID) -> None:
        self.messages.pop(session_id, None)
        self.titles.pop(session_id, None)
