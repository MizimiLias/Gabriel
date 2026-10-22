"""Chat orchestration and LangChain message history handling."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.configs.settings import Settings
from app.entities.chat_message import ChatMessage
from app.exceptions.chat import ChatUpstreamError
from app.repositories.chat_repository import ChatRepository

logger = structlog.get_logger(__name__)

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
class StreamDone:
    """Internal stream completion marker."""

    session_id: UUID
    message_id: UUID
    created_at: datetime


type StreamEvent = StreamStart | StreamToken | StreamDone


class ChatService:
    """Coordinates history loading, model invocation, and persistence."""

    def __init__(
        self,
        repository: ChatRepository,
        model: BaseChatModel,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._model = model
        self._settings = settings

    async def send_message(self, session_id: UUID, message: str) -> ChatMessage:
        """Generate one complete response and persist the successful exchange."""

        async with await _get_session_lock(session_id):
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            response = await self._invoke(self._build_messages(history, message))
            if not response:
                raise ChatUpstreamError("The language model returned an empty response.")

            return await self._repository.save_exchange(
                session_id=session_id,
                user_content=message,
                assistant_content=response,
                assistant_message_id=uuid4(),
            )

    async def stream_message(self, session_id: UUID, message: str) -> AsyncIterator[StreamEvent]:
        """Generate one response as text fragments and persist only on completion."""

        async with await _get_session_lock(session_id):
            history = await self._repository.get_recent_messages(
                session_id, self._settings.chat_context_message_limit
            )
            messages = self._build_messages(history, message)
            assistant_message_id = uuid4()
            fragments: list[str] = []
            yield StreamStart(session_id=session_id, message_id=assistant_message_id)

            try:
                async for chunk in self._model.astream(messages):
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

            stored_message = await self._repository.save_exchange(
                session_id=session_id,
                user_content=message,
                assistant_content=content,
                assistant_message_id=assistant_message_id,
            )
            yield StreamDone(
                session_id=session_id,
                message_id=stored_message.id,
                created_at=stored_message.created_at,
            )

    async def get_history(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        """Return persisted history for an existing session."""

        return await self._repository.get_history(session_id, limit)

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

    def _build_messages(self, history: list[ChatMessage], message: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self._settings.chat_system_prompt)]
        for stored_message in history:
            if stored_message.role == "user":
                messages.append(HumanMessage(content=stored_message.content))
            else:
                messages.append(AIMessage(content=stored_message.content))
        messages.append(HumanMessage(content=message))
        return messages
