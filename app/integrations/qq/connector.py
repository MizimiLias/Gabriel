"""QQ official bot WebSocket connector and SDK adapter."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

import botpy
import structlog
from botpy import connection as botpy_connection
from botpy.message import C2CMessage, GroupMessage

from app.entities.chat_session import SessionSource
from app.integrations.qq.handler import QQInboundMessage, QQMessageHandler

logger = structlog.get_logger(__name__)


class QQClient(Protocol):
    """The botpy client lifecycle used by the connector."""

    async def start(self, appid: str, secret: str, ret_coro: bool = False) -> object: ...

    async def close(self) -> None: ...


class _EnrichedGroupMessage(GroupMessage):
    """GroupMessage that retains the sender's QQ display name."""

    __slots__ = ("author_name",)

    author_name: str | None

    def __init__(
        self,
        api: object,
        event_id: str | None,
        data: Mapping[str, object],
    ) -> None:
        author = data.get("author")
        if not isinstance(author, Mapping):
            author = {}
            data = {**data, "author": author}
        super().__init__(api, event_id, data)
        username = author.get("username")
        self.author_name = username if isinstance(username, str) and username.strip() else None


def _install_group_message_patch() -> bool:
    """Keep ``d.author.username`` that botpy's group parser drops.

    ``ConnectionState.parse_group_at_message_create`` resolves ``GroupMessage``
    from the ``botpy.connection`` module at call time, so replacing that name
    is enough. Returns whether this call installed the patch.
    """

    current = getattr(botpy_connection, "GroupMessage", None)
    if current is _EnrichedGroupMessage:
        return False
    if not isinstance(current, type) or not hasattr(current, "_User"):
        logger.warning("qq_group_message_patch_skipped", reason="unexpected_botpy_shape")
        return False
    botpy_connection.GroupMessage = _EnrichedGroupMessage
    return True


class QQBotClient(botpy.Client):
    """Translate botpy events into channel-neutral QQ messages."""

    def __init__(self, handler: QQMessageHandler) -> None:
        _install_group_message_patch()
        super().__init__(intents=botpy.Intents(public_messages=True), bot_log=False)
        self._handler = handler

    async def on_ready(self) -> None:
        robot_name = self.robot.name if self.robot is not None else None
        logger.info("qq_bot_ready", robot_name=robot_name)

    async def on_error(self, event_method: str, *_args: object, **_kwargs: object) -> None:
        logger.exception("qq_sdk_event_failed", event_method=event_method)

    async def on_group_at_message_create(self, message: GroupMessage) -> None:
        await self._handler.handle(
            QQInboundMessage(
                source=SessionSource.QQ_GROUP,
                event_id=message.id,
                conversation_id=message.group_openid,
                content=message.content or "",
                has_attachments=bool(message.attachments),
                reply=self._reply_callback(message),
                sender_name=getattr(message, "author_name", None),
                sender_id=getattr(message.author, "member_openid", None),
            )
        )

    async def on_c2c_message_create(self, message: C2CMessage) -> None:
        await self._handler.handle(
            QQInboundMessage(
                source=SessionSource.QQ_C2C,
                event_id=message.id,
                conversation_id=message.author.user_openid,
                content=message.content or "",
                has_attachments=bool(message.attachments),
                reply=self._reply_callback(message),
            )
        )

    @staticmethod
    def _reply_callback(
        message: GroupMessage | C2CMessage,
    ) -> Callable[[str, int], Awaitable[object]]:
        async def reply(content: str, sequence: int) -> object:
            return await message.reply(content=content, msg_seq=sequence)

        return reply


def _build_client(handler: QQMessageHandler) -> QQClient:
    return QQBotClient(handler)


class QQBotConnector:
    """Own one outbound QQ WebSocket connection."""

    name = "qq"

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        handler: QQMessageHandler,
        client_factory: Callable[[QQMessageHandler], QQClient] = _build_client,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._handler = handler
        self._client_factory = client_factory
        self._client: QQClient | None = None

    async def run(self) -> None:
        """Connect to QQ and wait for the SDK connection to terminate."""

        client = self._client_factory(self._handler)
        self._client = client
        try:
            await client.start(appid=self._app_id, secret=self._app_secret)
        finally:
            await self.close()

    async def close(self) -> None:
        """Close the SDK HTTP client and allow the supervisor to restart."""

        client = self._client
        if client is None:
            return
        self._client = None
        await client.close()
