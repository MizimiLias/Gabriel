"""Shared test doubles and settings helpers."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.entities.chat_session import SessionSource
from app.exceptions.chat import (
    ChatSessionNotFoundError,
    PersonaConflictError,
    PersonaNotFoundError,
    PersonaValidationError,
)
from app.prompts import TITLE_SYSTEM_PROMPT
from app.repositories.chat_repository import ChatSessionMeta, ChatSessionSummaryRow
from app.repositories.persona_repository import (
    PersonaMessageDraft,
    PersonaMessageMutation,
    PersonaMessageRecord,
    PersonaRecord,
    PersonaSummaryRow,
)
from app.schemas.content import (
    ContentPart,
    TextPart,
    plain_text_from_parts,
    preview_text_from_parts,
)


def text_parts(text: str) -> list[ContentPart]:
    """Build one text part for service-level tests."""

    return [TextPart(text=text)]


def visible_message_text(message: BaseMessage) -> str:
    """Extract visible text from a LangChain message, excluding reminders."""

    if isinstance(message.content, str):
        return message.content
    return plain_text_from_parts(message.content)


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
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.last_messages = messages
        self.calls.append(messages)
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
        self.calls.append(messages)
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
        self.persona_ids: dict[UUID, UUID | None] = {}
        self.persona_names: dict[UUID, str] = {}
        self.saved_exchanges: list[tuple[UUID, list[dict[str, Any]], list[dict[str, Any]]]] = []

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

    async def get_session_meta(self, session_id: UUID) -> ChatSessionMeta | None:
        if session_id not in self.messages:
            return None
        return ChatSessionMeta(
            title=self.titles.get(session_id),
            source=self.sources.get(session_id, SessionSource.WEB),
            persona_id=self.persona_ids.get(session_id),
        )

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
            persona_id = self.persona_ids.get(session_id)
            rows.append(
                ChatSessionSummaryRow(
                    session_id=session_id,
                    title=self.titles.get(session_id),
                    source=source,
                    created_at=messages[0].created_at,
                    updated_at=last_message.created_at,
                    message_count=len(messages),
                    last_message_preview=preview_text_from_parts(last_message.parts)[:160],
                    persona_id=persona_id,
                    persona_name=(
                        self.persona_names.get(persona_id) if persona_id is not None else None
                    ),
                )
            )
        return sorted(rows, key=lambda row: row.updated_at, reverse=True)

    async def save_exchange(
        self,
        *,
        session_id: UUID,
        user_parts: list[dict[str, Any]],
        assistant_parts: list[dict[str, Any]],
        assistant_message_id: UUID,
        title: str | None = None,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        persona_id: UUID | None = None,
    ) -> ChatMessage:
        now = datetime.now(UTC)
        user_message = ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role="user",
            parts=user_parts,
            created_at=now,
        )
        assistant_message = ChatMessage(
            id=assistant_message_id,
            session_id=session_id,
            role="assistant",
            parts=assistant_parts,
            created_at=now,
        )
        self.messages.setdefault(session_id, []).extend([user_message, assistant_message])
        self.titles.setdefault(session_id, title)
        self.sources.setdefault(session_id, source)
        self.external_keys.setdefault(session_id, external_key)
        self.persona_ids.setdefault(session_id, persona_id)
        self.saved_exchanges.append((session_id, user_parts, assistant_parts))
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


class FakePersonaRepository:
    """In-memory repository matching the interface used by PersonaService."""

    def __init__(self) -> None:
        self.personas: dict[UUID, PersonaRecord] = {}

    async def get(self, persona_id: UUID) -> PersonaRecord | None:
        return self.personas.get(persona_id)

    async def list_all(self) -> list[PersonaSummaryRow]:
        return [
            PersonaSummaryRow(
                persona_id=record.persona_id,
                name=record.name,
                description=record.description,
                message_count=len(record.messages),
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            for record in sorted(self.personas.values(), key=lambda item: item.name)
        ]

    async def create(
        self,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
        messages: list[PersonaMessageDraft],
    ) -> PersonaRecord:
        if any(record.name == name for record in self.personas.values()):
            raise PersonaConflictError("A persona with this name already exists.")
        now = datetime.now(UTC)
        persona_id = uuid4()
        record = PersonaRecord(
            persona_id=persona_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            messages=tuple(
                PersonaMessageRecord(
                    id=uuid4(),
                    position=position,
                    role=draft.role,
                    content=draft.content,
                )
                for position, draft in enumerate(messages)
            ),
            created_at=now,
            updated_at=now,
        )
        self.personas[persona_id] = record
        return record

    async def update_with_messages(
        self,
        persona_id: UUID,
        *,
        name: str,
        description: str | None,
        system_prompt: str,
        messages: list[PersonaMessageMutation],
    ) -> PersonaRecord:
        current = self.personas.get(persona_id)
        if current is None:
            raise PersonaNotFoundError("The requested persona does not exist.")
        if any(
            record.persona_id != persona_id and record.name == name
            for record in self.personas.values()
        ):
            raise PersonaConflictError("A persona with this name already exists.")

        retained_count = 0
        saw_new_message = False
        seen_ids: set[UUID] = set()
        for mutation in messages:
            if mutation.id is None:
                saw_new_message = True
                continue
            if saw_new_message:
                raise PersonaValidationError("New preset messages must be appended at the end.")
            if mutation.id in seen_ids:
                raise PersonaValidationError("A preset message ID cannot appear more than once.")
            if (
                retained_count >= len(current.messages)
                or current.messages[retained_count].id != mutation.id
            ):
                raise PersonaValidationError(
                    "Existing preset messages must keep their original order."
                )
            seen_ids.add(mutation.id)
            retained_count += 1

        updated_messages = [
            PersonaMessageRecord(
                id=current.messages[index].id,
                position=index,
                role=mutation.role,
                content=mutation.content,
            )
            for index, mutation in enumerate(messages[:retained_count])
        ]
        updated_messages.extend(
            PersonaMessageRecord(
                id=uuid4(),
                position=position,
                role=mutation.role,
                content=mutation.content,
            )
            for position, mutation in enumerate(messages[retained_count:], start=retained_count)
        )
        updated = replace(
            current,
            name=name,
            description=description,
            system_prompt=system_prompt,
            messages=tuple(updated_messages),
            updated_at=datetime.now(UTC),
        )
        self.personas[persona_id] = updated
        return updated

    async def delete(self, persona_id: UUID) -> None:
        if persona_id not in self.personas:
            raise PersonaNotFoundError("The requested persona does not exist.")
        del self.personas[persona_id]
