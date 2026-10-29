"""QQ connector media extraction and Markdown reply tests."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import botpy
import pytest
from botpy.message import C2CMessage, GroupMessage

from app.integrations.qq.connector import QQBotClient


@pytest.mark.asyncio
async def test_reply_callback_prefers_native_markdown() -> None:
    reply = AsyncMock(return_value={"id": "sent"})
    message = cast(GroupMessage, SimpleNamespace(reply=reply))

    result = await QQBotClient._reply_callback(message)("**hello**", 1)

    assert result == {"id": "sent"}
    reply.assert_awaited_once_with(
        markdown={"content": "**hello**"},
        msg_type=2,
        msg_seq=1,
    )


@pytest.mark.asyncio
async def test_reply_callback_falls_back_when_markdown_is_denied() -> None:
    reply = AsyncMock(
        side_effect=[
            botpy.errors.ServerError("不允许发送原生 markdown"),
            {"id": "sent-plain"},
        ]
    )
    message = cast(C2CMessage, SimpleNamespace(reply=reply))

    result = await QQBotClient._reply_callback(message)("**hello**", 2)

    assert result == {"id": "sent-plain"}
    assert reply.await_args_list[0].kwargs == {
        "markdown": {"content": "**hello**"},
        "msg_type": 2,
        "msg_seq": 2,
    }
    assert reply.await_args_list[1].kwargs == {
        "content": "**hello**",
        "msg_type": 0,
        "msg_seq": 2,
    }


@pytest.mark.asyncio
async def test_reply_callback_does_not_hide_other_qq_errors() -> None:
    reply = AsyncMock(side_effect=botpy.errors.ServerError("server unavailable"))
    message = cast(GroupMessage, SimpleNamespace(reply=reply))

    with pytest.raises(botpy.errors.ServerError, match="server unavailable"):
        await QQBotClient._reply_callback(message)("hello", 1)


def test_image_urls_filters_non_image_attachments() -> None:
    attachments = [
        SimpleNamespace(content_type="image/png", url="https://example.com/a.png"),
        SimpleNamespace(content_type="video/mp4", url="https://example.com/a.mp4"),
        SimpleNamespace(content_type="image/jpeg", url=None),
        SimpleNamespace(content_type=None, url="https://example.com/unknown"),
    ]

    assert QQBotClient._image_urls(attachments) == ("https://example.com/a.png",)
