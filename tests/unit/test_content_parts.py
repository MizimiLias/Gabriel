"""Stored content part tests for tool-call traces."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.schemas.content import (
    TextPart,
    ToolCallPart,
    ToolResultPart,
    content_parts_for_model,
    dump_content_parts,
    parse_content_parts,
    parse_stored_parts,
    parts_from_trace_message,
    plain_text_from_parts,
    preview_text_from_parts,
)


def test_parse_stored_parts_accepts_tool_call_and_result() -> None:
    parts = parse_stored_parts(
        [
            {"type": "text", "text": "let me check"},
            {"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {"city": "北京"}},
            {
                "type": "tool_result",
                "tool_call_id": "call-1",
                "name": "get_weather",
                "content": "晴 20 度",
                "is_error": False,
            },
        ]
    )

    assert [type(part) for part in parts] == [TextPart, ToolCallPart, ToolResultPart]
    assert isinstance(parts[1], ToolCallPart)
    assert parts[1].args == {"city": "北京"}
    assert isinstance(parts[2], ToolResultPart)
    assert parts[2].content == "晴 20 度"


def test_request_content_parts_still_reject_tool_parts() -> None:
    with pytest.raises(ValidationError):
        parse_content_parts(
            [{"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {}}]
        )


def test_parse_stored_parts_rejects_unknown_part_type() -> None:
    with pytest.raises(ValidationError):
        parse_stored_parts([{"type": "audio_url", "audio_url": {"url": "https://x.test/a.wav"}}])


def test_plain_text_ignores_tool_parts() -> None:
    parts = [
        {"type": "text", "text": "hello"},
        {"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {}},
        {
            "type": "tool_result",
            "tool_call_id": "call-1",
            "name": "get_weather",
            "content": "sunny",
        },
    ]

    assert plain_text_from_parts(parts) == "hello"
    assert preview_text_from_parts(parts) == "hello"


def test_content_parts_for_model_ignores_tool_parts() -> None:
    parts = [
        {"type": "text", "text": "hello"},
        {"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {}},
    ]

    assert content_parts_for_model(parts) == [{"type": "text", "text": "hello"}]


def test_parts_from_trace_message_rebuilds_assistant_tool_calls() -> None:
    message = AIMessage(
        content="checking",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"city": "北京"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    parts = parts_from_trace_message(message)

    assert dump_content_parts(parts) == [
        {"type": "text", "text": "checking"},
        {"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {"city": "北京"}},
    ]


def test_parts_from_trace_message_omits_empty_assistant_text() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {}, "id": "call-1", "type": "tool_call"},
        ],
    )

    parts = parts_from_trace_message(message)

    assert dump_content_parts(parts) == [
        {"type": "tool_call", "id": "call-1", "name": "get_weather", "args": {}},
    ]


def test_parts_from_trace_message_converts_tool_result() -> None:
    message = ToolMessage(
        content="晴 20 度",
        tool_call_id="call-1",
        name="get_weather",
        status="error",
    )

    parts = parts_from_trace_message(message)

    assert dump_content_parts(parts) == [
        {
            "type": "tool_result",
            "tool_call_id": "call-1",
            "name": "get_weather",
            "content": "晴 20 度",
            "is_error": True,
        }
    ]
