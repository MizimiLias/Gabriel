"""Chat orchestration and LangChain message history handling."""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.entities.chat_session import SessionSource
from app.exceptions.chat import ChatUpstreamError, PersonaNotFoundError
from app.prompts import TITLE_SYSTEM_PROMPT
from app.repositories.chat_repository import (
    ChatRepository,
    ChatSessionMeta,
    ChatSessionSummaryRow,
)
from app.repositories.persona_repository import PersonaRecord, PersonaRepository

logger = structlog.get_logger(__name__)

_MAX_TITLE_LENGTH = 80
_TITLE_CONTEXT_LIMIT = 3_000

_session_locks: dict[UUID, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


async def _get_session_lock(session_id: UUID) -> asyncio.Lock:
    async with _locks_guard:
        return _session_locks.setdefault(session_id, asyncio.Lock())


@dataclass(frozen=True, slots=True)
class StreamStart:
    """Internal stream start marker."""

    session_id: UUID
    message_id: UUID


@dataclass(frozen=True, slots=True)
class StreamToken:
    """Internal generated text fragment."""

    delta: str


@dataclass(frozen=True, slots=True)
class StreamReasoning:
    """Internal reasoning/thinking fragment (never persisted)."""

    delta: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    """Internal stream completion marker."""

    session_id: UUID
    message_id: UUID
    title: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ChatReply:
    """A persisted assistant reply and its session title."""

    message: ChatMessage
    title: str | None


type StreamEvent = StreamStart | StreamToken | StreamReasoning | StreamDone


def _extract_reasoning_delta(chunk: object) -> str:
    """Extract one reasoning_content fragment from a streamed chunk."""

    candidates: list[object] = []
    additional = getattr(chunk, "additional_kwargs", None)
    if isinstance(additional, dict):
        candidates.append(additional.get("reasoning_content"))
        candidates.append(additional.get("reasoning"))
    metadata = getattr(chunk, "response_metadata", None)
    if isinstance(metadata, dict):
        candidates.append(metadata.get("reasoning_content"))
        candidates.append(metadata.get("reasoning"))
    # Some OpenAI-compatible gateways expose it as a direct attribute.
    candidates.append(getattr(chunk, "reasoning_content", None))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, list):
            parts = [part for part in candidate if isinstance(part, str) and part]
            if parts:
                return "".join(parts)
    return ""


def _normalise_title(value: str) -> str:
    first_line = value.strip().splitlines()[0] if value.strip() else ""
    first_line = re.sub(r"^\s*(?:title|标题)\s*[:\uff1a]\s*", "", first_line, flags=re.I)
    first_line = re.sub(r"\s+", " ", first_line).strip(" \t\"'\u201c\u201d\u2018\u2019")
    if len(first_line) <= _MAX_TITLE_LENGTH:
        return first_line
    return f"{first_line[: _MAX_TITLE_LENGTH - 1].rstrip()}…"


def _fallback_title(message: str) -> str:
    return _normalise_title(message) or "New chat"


@dataclass(frozen=True, slots=True)
class _ChatContext:
    """Persona data used to assemble one model request."""

    persona: PersonaRecord | None
    persona_id_for_new_session: UUID | None


class ChatService:
    """Coordinates history loading, model invocation, and persistence."""

    def __init__(
        self,
        repository: ChatRepository,
        model: BaseChatModel,
        settings: Settings,
        persona_repository: PersonaRepository | None = None,
    ) -> None:
        self._repository = repository
        self._model = model
        self._settings = settings
        self._persona_repository = persona_repository

    async def send_message(
        self,
        session_id: UUID,
        message: str,
        *,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        persona_id: UUID | None = None,
    ) -> ChatReply:
        """Generate one complete response and persist the successful exchange."""

        async with await _get_session_lock(session_id):
            session_meta = await self._repository.get_session_meta(session_id)
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            context = await self._resolve_chat_context(
                session_meta=session_meta,
                requested_persona_id=persona_id,
            )
            existing_title = session_meta.title if session_meta is not None else None
            response = await self._invoke(self._build_messages(history, message, context.persona))
            if not response:
                raise ChatUpstreamError("The language model returned an empty response.")

            title = await self._resolve_title(
                source=source,
                existing_title=existing_title,
                user_content=message,
                assistant_content=response,
            )
            stored_message = await self._repository.save_exchange(
                session_id=session_id,
                user_content=message,
                assistant_content=response,
                assistant_message_id=uuid4(),
                title=title,
                source=source,
                external_key=external_key,
                persona_id=context.persona_id_for_new_session,
            )
            return ChatReply(message=stored_message, title=title)

    async def stream_message(
        self,
        session_id: UUID,
        message: str,
        *,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        persona_id: UUID | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Generate one response as text fragments and persist only on completion."""

        async with await _get_session_lock(session_id):
            session_meta = await self._repository.get_session_meta(session_id)
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            context = await self._resolve_chat_context(
                session_meta=session_meta,
                requested_persona_id=persona_id,
            )
            existing_title = session_meta.title if session_meta is not None else None
            messages = self._build_messages(history, message, context.persona)
            assistant_message_id = uuid4()
            fragments: list[str] = []
            yield StreamStart(session_id=session_id, message_id=assistant_message_id)

            try:
                async for chunk in self._model.astream(messages):
                    reasoning = _extract_reasoning_delta(chunk)
                    if reasoning:
                        yield StreamReasoning(delta=reasoning)
                    text = chunk.text
                    if text:
                        fragments.append(text)
                        yield StreamToken(delta=text)
            except asyncio.CancelledError:
                logger.info("chat_stream_cancelled", session_id=str(session_id))
                raise
            except Exception as exc:
                logger.warning(
                    "chat_stream_upstream_failed",
                    session_id=str(session_id),
                    error_type=type(exc).__name__,
                )
                raise ChatUpstreamError(
                    "The language model could not complete the response."
                ) from exc

            content = "".join(fragments)
            if not content:
                raise ChatUpstreamError("The language model returned an empty response.")

            title = await self._resolve_title(
                source=source,
                existing_title=existing_title,
                user_content=message,
                assistant_content=content,
            )
            stored_message = await self._repository.save_exchange(
                session_id=session_id,
                user_content=message,
                assistant_content=content,
                assistant_message_id=assistant_message_id,
                title=title,
                source=source,
                external_key=external_key,
                persona_id=context.persona_id_for_new_session,
            )
            yield StreamDone(
                session_id=session_id,
                message_id=stored_message.id,
                title=title,
                created_at=stored_message.created_at,
            )

    async def get_history(
        self, session_id: UUID, limit: int
    ) -> tuple[str | None, list[ChatMessage]]:
        """Return the title and persisted history for an existing session."""

        return await self._repository.get_history(session_id, limit)

    async def list_sessions(
        self, source: SessionSource = SessionSource.WEB
    ) -> list[ChatSessionSummaryRow]:
        """Return sessions for one source ordered by recent activity."""

        return await self._repository.list_sessions(source)

    async def rename_session(self, session_id: UUID, title: str) -> ChatSessionSummaryRow:
        """Rename a session after validating that it exists."""

        return await self._repository.update_session_title(session_id, _normalise_title(title))

    async def delete_session(self, session_id: UUID) -> None:
        """Delete a session if it exists."""

        await self._repository.delete_session(session_id)

    async def _invoke(self, messages: list[BaseMessage]) -> str:
        try:
            response = await self._model.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("chat_upstream_failed", error_type=type(exc).__name__)
            raise ChatUpstreamError("The language model could not complete the response.") from exc
        return response.text

    async def _resolve_title(
        self,
        *,
        source: SessionSource,
        existing_title: str | None,
        user_content: str,
        assistant_content: str,
    ) -> str | None:
        """Keep existing titles; generate one only for web sessions."""

        if existing_title is not None:
            return existing_title
        # Every non-web channel (QQ, future integrations) stays untitled so no
        # platform-specific identity has to flow into the title generation path.
        if source is not SessionSource.WEB:
            return None
        return await self._generate_title(user_content, assistant_content)

    async def _generate_title(self, user_content: str, assistant_content: str) -> str:
        fallback = _fallback_title(user_content)
        title_messages = [
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Create a title for this first exchange.\n\n"
                    f"User:\n{user_content[:_TITLE_CONTEXT_LIMIT]}\n\n"
                    f"Assistant:\n{assistant_content[:_TITLE_CONTEXT_LIMIT]}"
                )
            ),
        ]
        try:
            response = await self._model.ainvoke(title_messages)
            return _normalise_title(response.text) or fallback
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "chat_title_generation_failed",
                error_type=type(exc).__name__,
            )
            return fallback

    async def _resolve_chat_context(
        self,
        *,
        session_meta: ChatSessionMeta | None,
        requested_persona_id: UUID | None,
    ) -> _ChatContext:
        """Resolve the persona for an existing or new session."""

        if session_meta is not None:
            persona = await self._load_persona(session_meta.persona_id)
            return _ChatContext(persona=persona, persona_id_for_new_session=None)

        if requested_persona_id is None:
            return _ChatContext(persona=None, persona_id_for_new_session=None)

        if self._persona_repository is None:
            raise RuntimeError("A persona repository is required to bind a persona.")
        persona = await self._persona_repository.get(requested_persona_id)
        if persona is None:
            raise PersonaNotFoundError("The requested persona does not exist.")
        return _ChatContext(
            persona=persona,
            persona_id_for_new_session=persona.persona_id,
        )

    async def _load_persona(self, persona_id: UUID | None) -> PersonaRecord | None:
        if persona_id is None or self._persona_repository is None:
            return None
        persona = await self._persona_repository.get(persona_id)
        if persona is None:
            logger.warning("chat_persona_missing", persona_id=str(persona_id))
        return persona

    def _build_messages(
        self,
        history: list[ChatMessage],
        message: str,
        persona: PersonaRecord | None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if persona is not None:
            messages.append(SystemMessage(content=persona.system_prompt))
            for preset in persona.messages:
                if preset.role == "user":
                    messages.append(HumanMessage(content=preset.content))
                else:
                    messages.append(AIMessage(content=preset.content))
        for stored_message in history:
            if stored_message.role == "user":
                messages.append(HumanMessage(content=stored_message.content))
            else:
                messages.append(AIMessage(content=stored_message.content))
        messages.append(HumanMessage(content=message))
        return messages
