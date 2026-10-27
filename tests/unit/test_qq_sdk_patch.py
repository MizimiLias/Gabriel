"""botpy group message patch tests."""

import pytest
from botpy import connection as botpy_connection
from botpy.message import GroupMessage

from app.integrations.qq.connector import (
    _EnrichedGroupMessage,
    _install_group_message_patch,
)


def test_enriched_group_message_keeps_author_username() -> None:
    message = _EnrichedGroupMessage(
        object(),
        "event-1",
        {
            "id": "event-1",
            "content": "hello",
            "group_openid": "group-1",
            "author": {"member_openid": "member-1", "username": "Alice"},
        },
    )

    assert message.author_name == "Alice"
    assert message.author.member_openid == "member-1"
    assert message.group_openid == "group-1"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"author": None},
        {"author": {"member_openid": "member-1"}},
        {"author": {"username": 3}},
        {"author": {"username": "   "}},
    ],
)
def test_enriched_group_message_without_username(data: dict[str, object]) -> None:
    message = _EnrichedGroupMessage(object(), "event-1", data)

    assert message.author_name is None


def test_install_is_idempotent_and_parser_uses_enriched_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(botpy_connection, "GroupMessage", GroupMessage)
    dispatched: list[tuple[object, ...]] = []

    assert _install_group_message_patch() is True
    assert _install_group_message_patch() is False
    assert botpy_connection.GroupMessage is _EnrichedGroupMessage

    state = botpy_connection.ConnectionState(
        lambda *args: dispatched.append(args),
        object(),
    )
    state.parse_group_at_message_create(
        {
            "id": "event-2",
            "d": {"author": {"member_openid": "member-1", "username": "Alice"}},
        }
    )

    parsed = dispatched[0][1]
    assert isinstance(parsed, _EnrichedGroupMessage)
    assert parsed.author_name == "Alice"


def test_install_skips_unexpected_botpy_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(botpy_connection, "GroupMessage")

    assert _install_group_message_patch() is False
