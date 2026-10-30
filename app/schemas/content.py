"""Multimodal chat content parts shared by the API and persistence layers."""

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

SYSTEM_REMINDER_PREFIX = "<system_reminder>"
SYSTEM_REMINDER_SUFFIX = "</system_reminder>"
_SYSTEM_REMINDER_RE = re.compile(
    rf"^{re.escape(SYSTEM_REMINDER_PREFIX)}[\s\S]*{re.escape(SYSTEM_REMINDER_SUFFIX)}$"
)
_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_MAX_REMINDER_SENDER_NAME_CHARS = 64
_MAX_REMINDER_SENDER_ID_CHARS = 128


class TextPart(BaseModel):
    """A text content block."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=32_000)


class ImageURL(BaseModel):
    """The URL payload accepted by an image content block."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: str = Field(min_length=1, max_length=8_192)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require an absolute HTTP(S) image URL."""

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Image URL must use http or https.")
        return value


class ImageURLPart(BaseModel):
    """An image content block compatible with OpenAI-style chat payloads."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class ToolCallPart(BaseModel):
    """One tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any]


class ToolResultPart(BaseModel):
    """The text result returned by one tool invocation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


type ContentPart = Annotated[TextPart | ImageURLPart, Field(discriminator="type")]
type StoredContentPart = Annotated[
    TextPart | ImageURLPart | ToolCallPart | ToolResultPart,
    Field(discriminator="type"),
]

_CONTENT_PARTS_ADAPTER = TypeAdapter(list[ContentPart])
_STORED_PARTS_ADAPTER = TypeAdapter(list[StoredContentPart])


def parse_content_parts(value: object) -> list[ContentPart]:
    """Validate and normalize API-provided content parts.

    Args:
        value: Raw JSON-compatible content parts.

    Returns:
        Validated content parts.
    """

    return _CONTENT_PARTS_ADAPTER.validate_python(value)


def parse_stored_parts(value: object) -> list[StoredContentPart]:
    """Validate and normalize persisted content parts, including tool traces."""

    return _STORED_PARTS_ADAPTER.validate_python(value)


def dump_content_parts(parts: Sequence[StoredContentPart]) -> list[dict[str, Any]]:
    """Serialize content parts for PostgreSQL JSONB storage."""

    return [part.model_dump(mode="json") for part in parts]


def content_parts_for_model(
    parts: Sequence[StoredContentPart | dict[str, Any]],
) -> list[str | dict[Any, Any]]:
    """Convert stored text/image parts to the content array accepted by LangChain/OpenAI."""

    return [
        part.model_dump(mode="json")
        for part in parse_stored_parts(list(parts))
        if isinstance(part, (TextPart, ImageURLPart))
    ]


def is_system_reminder_text(value: str) -> bool:
    """Return whether a text block is an injected runtime reminder."""

    return bool(_SYSTEM_REMINDER_RE.fullmatch(value.strip()))


def plain_text_from_parts(
    parts: Sequence[StoredContentPart | dict[str, Any]],
    *,
    include_system_reminders: bool = False,
) -> str:
    """Extract text blocks from content parts in their stored order.

    Args:
        parts: Stored or API-provided content parts.
        include_system_reminders: Whether runtime reminder blocks should be included.

    Returns:
        Text blocks joined by blank lines.
    """

    texts: list[str] = []
    for part in parse_stored_parts(list(parts)):
        if not isinstance(part, TextPart):
            continue
        if not include_system_reminders and is_system_reminder_text(part.text):
            continue
        if part.text.strip():
            texts.append(part.text.strip())
    return "\n\n".join(texts)


def preview_text_from_parts(parts: Sequence[StoredContentPart | dict[str, Any]]) -> str:
    """Return a concise text preview, falling back to an image marker."""

    parsed = parse_stored_parts(list(parts))
    text = plain_text_from_parts(parsed)
    if text:
        return text
    if any(isinstance(part, ImageURLPart) for part in parsed):
        return "[Image]"
    return ""


def _sanitize_reminder_field(value: object, *, max_chars: int) -> str:
    """Flatten an untrusted sender attribute into one safe reminder fragment."""

    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    cleaned = "".join(character for character in collapsed if character.isprintable())
    cleaned = cleaned.replace("<", "").replace(">", "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _sender_reminder_line(sender_name: object, sender_id: object) -> str | None:
    """Format the optional sender attribution line for a reminder."""

    name = _sanitize_reminder_field(sender_name, max_chars=_MAX_REMINDER_SENDER_NAME_CHARS)
    identifier = _sanitize_reminder_field(sender_id, max_chars=_MAX_REMINDER_SENDER_ID_CHARS)
    if name and identifier:
        return f"Sender: {name} (openid: {identifier})"
    if name:
        return f"Sender: {name}"
    if identifier:
        return f"Sender: openid {identifier}"
    return None


def append_system_reminder(
    parts: Sequence[ContentPart],
    timezone: str,
    *,
    sender_name: str | None = None,
    sender_id: str | None = None,
) -> list[ContentPart]:
    """Append the persisted per-message runtime clock and sender reminder.

    Args:
        parts: User-provided content parts.
        timezone: IANA timezone used to render the current time.
        sender_name: Optional display name of the message sender.
        sender_id: Optional platform identifier of the message sender.

    Returns:
        A new list containing the original parts followed by one reminder block.
    """

    now = datetime.now(ZoneInfo(timezone))
    lines = [
        f"Current datetime: {now.isoformat(timespec='seconds')} ({timezone}), "
        f"Weekday: {_WEEKDAY_NAMES[now.weekday()]}"
    ]
    sender_line = _sender_reminder_line(sender_name, sender_id)
    if sender_line is not None:
        lines.append(sender_line)

    body = "\n".join(lines)
    reminder = TextPart(text=f"{SYSTEM_REMINDER_PREFIX}{body}{SYSTEM_REMINDER_SUFFIX}")
    return [*parts, reminder]


def assistant_text_parts(text: str) -> list[ContentPart]:
    """Build the normal single-text-part representation of an assistant reply."""

    return [TextPart(text=text)]


def parts_from_trace_message(message: AIMessage | ToolMessage) -> list[StoredContentPart]:
    """Convert one intermediate agent message into persisted content parts."""

    if isinstance(message, ToolMessage):
        return [
            ToolResultPart(
                tool_call_id=message.tool_call_id,
                name=message.name or "",
                content=message.text,
                is_error=getattr(message, "status", None) == "error",
            )
        ]

    parts: list[StoredContentPart] = []
    if message.text:
        parts.append(TextPart(text=message.text))
    parts.extend(
        ToolCallPart(id=call["id"] or "", name=call["name"], args=call["args"])
        for call in message.tool_calls
    )
    return parts
