"""Chat orchestration and LangChain message history handling."""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from app.agents.runtime import (
    AgentReasoning,
    AgentResult,
    AgentToken,
    AgentToolCalls,
    AgentToolResult,
    ChatAgent,
    extract_reasoning_delta,
)
from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.entities.chat_session import SessionSource
from app.exceptions.chat import ChatUpstreamError, PersonaNotFoundError
from app.prompts import TITLE_SYSTEM_PROMPT
from app.repositories.chat_repository import (
    ChatRepository,
    ChatSessionMeta,
    ChatSessionSummaryRow,
    StoredMessageDraft,
)
from app.repositories.persona_repository import PersonaRecord, PersonaRepository
from app.schemas.content import (
    ContentPart,
    ToolCallPart,
    ToolResultPart,
    append_system_reminder,
    assistant_text_parts,
    content_parts_for_model,
    dump_content_parts,
    parse_stored_parts,
    parts_from_trace_message,
    preview_text_from_parts,
)

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
class StreamToolCalls:
    """Internal tool-call request marker."""

    calls: tuple[ToolCall, ...]
    text: str


@dataclass(frozen=True, slots=True)
class StreamToolResult:
    """Internal completed tool invocation marker."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool


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


type StreamEvent = (
    StreamStart | StreamToken | StreamReasoning | StreamToolCalls | StreamToolResult | StreamDone
)


def _normalise_title(value: str) -> str:
    first_line = value.strip().splitlines()[0] if value.strip() else ""
    first_line = re.sub(r"^\s*(?:title|标题)\s*[:\uff1a]\s*", "", first_line, flags=re.I)
    first_line = re.sub(r"\s+", " ", first_line).strip(" \t\"'\u201c\u201d\u2018\u2019")
    if len(first_line) <= _MAX_TITLE_LENGTH:
        return first_line
    return f"{first_line[: _MAX_TITLE_LENGTH - 1].rstrip()}…"


def _fallback_title(message: str) -> str:
    return _normalise_title(message) or "New chat"


def _has_tool_calls(stored: ChatMessage) -> bool:
    """Return whether a stored assistant message requested tools."""

    return any(isinstance(part, ToolCallPart) for part in parse_stored_parts(stored.parts))


def _message_from_stored(stored: ChatMessage) -> BaseMessage:
    """Rebuild one LangChain message from its persisted content parts."""

    parts = parse_stored_parts(stored.parts)
    if stored.role == "tool":
        result = next((part for part in parts if isinstance(part, ToolResultPart)), None)
        if result is None:
            raise ValueError("Stored tool message has no tool result part.")
        return ToolMessage(
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status="error" if result.is_error else "success",
        )

    if stored.role == "assistant":
        tool_calls: list[ToolCall] = [
            {"id": part.id, "name": part.name, "args": part.args, "type": "tool_call"}
            for part in parts
            if isinstance(part, ToolCallPart)
        ]
        content = content_parts_for_model(parts)
        return AIMessage(content=content if content else "", tool_calls=tool_calls)

    return HumanMessage(content=content_parts_for_model(parts))


def _trace_drafts(messages: list[AIMessage | ToolMessage]) -> list[StoredMessageDraft]:
    """Convert intermediate agent messages into persistable drafts."""

    drafts = [
        StoredMessageDraft(
            role="tool" if isinstance(message, ToolMessage) else "assistant",
            parts=dump_content_parts(parts_from_trace_message(message)),
        )
        for message in messages
    ]
    return [draft for draft in drafts if draft.parts]


def _tool_call_draft(text: str, calls: tuple[ToolCall, ...]) -> StoredMessageDraft:
    """Build the assistant draft for one completed tool-calling model step."""

    message = AIMessage(content=text, tool_calls=list(calls))
    return StoredMessageDraft(
        role="assistant",
        parts=dump_content_parts(parts_from_trace_message(message)),
    )


def _tool_result_draft(event: AgentToolResult) -> StoredMessageDraft:
    """Build the tool draft for one completed tool invocation."""

    return StoredMessageDraft(
        role="tool",
        parts=dump_content_parts(
            [
                ToolResultPart(
                    tool_call_id=event.tool_call_id,
                    name=event.name,
                    content=event.content,
                    is_error=event.is_error,
                )
            ]
        ),
    )


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
        agent: ChatAgent | None = None,
    ) -> None:
        self._repository = repository
        self._model = model
        self._settings = settings
        self._persona_repository = persona_repository
        self._agent = agent

    async def send_message(
        self,
        session_id: UUID,
        parts: list[ContentPart],
        *,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        sender_name: str | None = None,
        sender_id: str | None = None,
        persona_id: UUID | None = None,
    ) -> ChatReply:
        """Generate one complete response and persist the successful exchange."""

        async with await _get_session_lock(session_id):
            user_parts = append_system_reminder(
                parts,
                self._settings.app_timezone,
                sender_name=sender_name,
                sender_id=sender_id,
            )
            session_meta = await self._repository.get_session_meta(session_id)
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            context = await self._resolve_chat_context(
                session_meta=session_meta,
                requested_persona_id=persona_id,
            )
            existing_title = session_meta.title if session_meta is not None else None
            messages = self._build_messages(history, user_parts, context.persona)
            trace: list[StoredMessageDraft] = []
            if self._agent is None:
                response = await self._invoke(messages)
            else:
                result = await self._invoke_agent(messages)
                response = result.text
                trace = _trace_drafts(result.trace)
            if not response:
                raise ChatUpstreamError("The language model returned an empty response.")

            user_text = preview_text_from_parts(user_parts)
            title = await self._resolve_title(
                source=source,
                existing_title=existing_title,
                user_content=user_text,
                assistant_content=response,
            )
            stored_message = await self._repository.save_exchange(
                session_id=session_id,
                user_parts=dump_content_parts(user_parts),
                assistant_parts=dump_content_parts(assistant_text_parts(response)),
                assistant_message_id=uuid4(),
                trace_messages=trace,
                title=title,
                source=source,
                external_key=external_key,
                persona_id=context.persona_id_for_new_session,
            )
            return ChatReply(message=stored_message, title=title)

    async def stream_message(
        self,
        session_id: UUID,
        parts: list[ContentPart],
        *,
        source: SessionSource = SessionSource.WEB,
        external_key: str | None = None,
        sender_name: str | None = None,
        sender_id: str | None = None,
        persona_id: UUID | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Generate one response as text fragments and persist only on completion."""

        async with await _get_session_lock(session_id):
            user_parts = append_system_reminder(
                parts,
                self._settings.app_timezone,
                sender_name=sender_name,
                sender_id=sender_id,
            )
            session_meta = await self._repository.get_session_meta(session_id)
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            context = await self._resolve_chat_context(
                session_meta=session_meta,
                requested_persona_id=persona_id,
            )
            existing_title = session_meta.title if session_meta is not None else None
            messages = self._build_messages(history, user_parts, context.persona)
            assistant_message_id = uuid4()
            fragments: list[str] = []
            trace: list[StoredMessageDraft] = []
            yield StreamStart(session_id=session_id, message_id=assistant_message_id)

            try:
                if self._agent is None:
                    async for chunk in self._model.astream(messages):
                        reasoning = extract_reasoning_delta(chunk)
                        if reasoning:
                            yield StreamReasoning(delta=reasoning)
                        text = chunk.text
                        if text:
                            fragments.append(text)
                            yield StreamToken(delta=text)
                else:
                    async for event in self._agent.stream(messages):
                        if isinstance(event, AgentToken):
                            fragments.append(event.delta)
                            yield StreamToken(delta=event.delta)
                        elif isinstance(event, AgentReasoning):
                            yield StreamReasoning(delta=event.delta)
                        elif isinstance(event, AgentToolCalls):
                            yield StreamToolCalls(calls=event.calls, text=event.text)
                            trace.append(_tool_call_draft(event.text, event.calls))
                            fragments.clear()
                        elif isinstance(event, AgentToolResult):
                            yield StreamToolResult(
                                tool_call_id=event.tool_call_id,
                                name=event.name,
                                content=event.content,
                                is_error=event.is_error,
                            )
                            trace.append(_tool_result_draft(event))
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

            user_text = preview_text_from_parts(user_parts)
            title = await self._resolve_title(
                source=source,
                existing_title=existing_title,
                user_content=user_text,
                assistant_content=content,
            )
            stored_message = await self._repository.save_exchange(
                session_id=session_id,
                user_parts=dump_content_parts(user_parts),
                assistant_parts=dump_content_parts(assistant_text_parts(content)),
                assistant_message_id=assistant_message_id,
                trace_messages=trace,
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

    async def _invoke_agent(self, messages: list[BaseMessage]) -> AgentResult:
        agent = self._agent
        if agent is None:
            raise RuntimeError("No chat agent is configured.")
        try:
            return await agent.invoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("chat_agent_failed", error_type=type(exc).__name__)
            raise ChatUpstreamError("The language model could not complete the response.") from exc

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
        parts: list[ContentPart],
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
        for stored_message in self._trim_history(history):
            messages.append(_message_from_stored(stored_message))
        messages.append(HumanMessage(content=content_parts_for_model(parts)))
        return messages

    @staticmethod
    def _trim_history(history: list[ChatMessage]) -> list[ChatMessage]:
        """Drop leading rows that would break the tool-call protocol.

        A context window can start in the middle of a turn. Leading tool results,
        or an assistant tool-call message whose results were cut off, cannot be
        replayed to an OpenAI-compatible model, so skip them. A plain assistant
        answer is safe to keep as the oldest message.
        """

        for index, message in enumerate(history):
            if message.role == "user":
                return history[index:]
            if message.role == "assistant" and not _has_tool_calls(message):
                return history[index:]
        return []
