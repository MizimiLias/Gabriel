"""Channel-independent handling for QQ text message events."""

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import structlog

from app.entities.chat_session import SessionSource
from app.exceptions.chat import ChatUpstreamError
from app.services.chat_service import ChatService

logger = structlog.get_logger(__name__)

_MAX_INPUT_CHARS = 32_000
_MAX_REPLY_CHARS = 1_500
_REPLY_LIMITS = {
    SessionSource.QQ_C2C: 4,
    SessionSource.QQ_GROUP: 5,
}
_UNSUPPORTED_MESSAGE = "目前仅支持文本消息。"
_OVERSIZED_INPUT_MESSAGE = "消息过长，请缩短后再试。"  # noqa: RUF001
_UPSTREAM_ERROR_MESSAGE = "Gabriel 暂时无法回复，请稍后再试。"  # noqa: RUF001
_TRUNCATED_SUFFIX = "\n\n…（内容过长，已截断）"  # noqa: RUF001


class InboundEventStore(Protocol):
    """Idempotency operations required by the QQ handler."""

    async def claim(self, source: SessionSource, external_event_id: str) -> bool: ...

    async def mark_completed(self, source: SessionSource, external_event_id: str) -> None: ...

    async def mark_failed(
        self, source: SessionSource, external_event_id: str, error_type: str
    ) -> None: ...


type ServiceScope = Callable[[], AbstractAsyncContextManager[ChatService]]
type EventStoreScope = Callable[[], AbstractAsyncContextManager[InboundEventStore]]
type ReplyFunction = Callable[[str, int], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class QQInboundMessage:
    """The minimal QQ message data needed outside the SDK adapter."""

    source: SessionSource
    event_id: str
    conversation_id: str
    content: str
    has_attachments: bool
    reply: ReplyFunction


@dataclass(frozen=True, slots=True)
class QQSessionIdentity:
    """Stable internal identity for one QQ conversation."""

    session_id: UUID
    external_key: str


def resolve_qq_session(
    app_id: str, source: SessionSource, conversation_id: str
) -> QQSessionIdentity:
    """Resolve a QQ C2C user or group into a stable Gabriel session."""

    if source not in _REPLY_LIMITS:
        raise ValueError(f"Unsupported QQ session source: {source}")
    external_key = f"{app_id}:{conversation_id}"
    session_id = uuid5(NAMESPACE_URL, f"gabriel:{source.value}:{external_key}")
    return QQSessionIdentity(session_id=session_id, external_key=external_key)


def split_qq_reply(text: str, max_parts: int) -> list[str]:
    """Split text into a bounded number of human-friendly QQ messages."""

    normalized = text.strip()
    if not normalized:
        raise ValueError("A QQ reply cannot be empty.")
    if max_parts < 1:
        raise ValueError("max_parts must be positive.")

    parts: list[str] = []
    remaining = normalized
    while remaining and len(parts) < max_parts:
        if len(remaining) <= _MAX_REPLY_CHARS:
            parts.append(remaining)
            remaining = ""
            break

        cut = remaining.rfind("\n", 0, _MAX_REPLY_CHARS + 1)
        if cut < _MAX_REPLY_CHARS // 2:
            cut = remaining.rfind(" ", 0, _MAX_REPLY_CHARS + 1)
        if cut < _MAX_REPLY_CHARS // 2:
            cut = _MAX_REPLY_CHARS

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        available = _MAX_REPLY_CHARS - len(_TRUNCATED_SUFFIX)
        tail = parts[-1][:available].rstrip() if parts else ""
        suffix = _TRUNCATED_SUFFIX.strip()
        parts[-1] = f"{tail}\n\n{suffix}".strip()

    return parts


class QQMessageHandler:
    """Route QQ messages through Gabriel's channel-neutral chat service."""

    def __init__(
        self,
        *,
        app_id: str,
        service_scope: ServiceScope,
        event_store_scope: EventStoreScope,
    ) -> None:
        self._app_id = app_id
        self._service_scope = service_scope
        self._event_store_scope = event_store_scope

    async def handle(self, message: QQInboundMessage) -> None:
        """Process one event exactly once and send its complete text response."""

        try:
            async with self._event_store_scope() as store:
                claimed = await store.claim(message.source, message.event_id)
        except Exception as exc:
            logger.exception(
                "qq_event_claim_failed",
                source=message.source.value,
                event_id=message.event_id,
                error_type=type(exc).__name__,
            )
            return

        if not claimed:
            logger.info(
                "qq_event_duplicate",
                source=message.source.value,
                event_id=message.event_id,
            )
            return

        identity = resolve_qq_session(self._app_id, message.source, message.conversation_id)
        try:
            content = message.content.strip()
            if not content:
                await self._reply(message, _UNSUPPORTED_MESSAGE)
            elif len(content) > _MAX_INPUT_CHARS:
                await self._reply(message, _OVERSIZED_INPUT_MESSAGE)
            else:
                async with self._service_scope() as service:
                    reply = await service.send_message(
                        identity.session_id,
                        content,
                        source=message.source,
                        external_key=identity.external_key,
                    )
                for sequence, part in enumerate(
                    split_qq_reply(reply.message.content, _REPLY_LIMITS[message.source]),
                    start=1,
                ):
                    await message.reply(part, sequence)
        except ChatUpstreamError as exc:
            logger.warning(
                "qq_chat_upstream_failed",
                source=message.source.value,
                event_id=message.event_id,
                session_id=str(identity.session_id),
                error_type=type(exc).__name__,
            )
            await self._reply_or_log(message, _UPSTREAM_ERROR_MESSAGE)
            await self._mark_failed(message, type(exc).__name__)
            return
        except Exception as exc:
            logger.exception(
                "qq_message_failed",
                source=message.source.value,
                event_id=message.event_id,
                session_id=str(identity.session_id),
                error_type=type(exc).__name__,
            )
            await self._reply_or_log(message, _UPSTREAM_ERROR_MESSAGE)
            await self._mark_failed(message, type(exc).__name__)
            return

        await self._mark_completed(message)
        logger.info(
            "qq_message_processed",
            source=message.source.value,
            event_id=message.event_id,
            session_id=str(identity.session_id),
        )

    async def _reply(self, message: QQInboundMessage, content: str) -> None:
        await message.reply(content, 1)

    async def _reply_or_log(self, message: QQInboundMessage, content: str) -> None:
        try:
            await self._reply(message, content)
        except Exception as exc:
            logger.warning(
                "qq_error_reply_failed",
                source=message.source.value,
                event_id=message.event_id,
                error_type=type(exc).__name__,
            )

    async def _mark_completed(self, message: QQInboundMessage) -> None:
        try:
            async with self._event_store_scope() as store:
                await store.mark_completed(message.source, message.event_id)
        except Exception as exc:
            logger.warning(
                "qq_event_completion_failed",
                source=message.source.value,
                event_id=message.event_id,
                error_type=type(exc).__name__,
            )

    async def _mark_failed(self, message: QQInboundMessage, error_type: str) -> None:
        try:
            async with self._event_store_scope() as store:
                await store.mark_failed(message.source, message.event_id, error_type)
        except Exception as exc:
            logger.warning(
                "qq_event_failure_mark_failed",
                source=message.source.value,
                event_id=message.event_id,
                error_type=type(exc).__name__,
            )
