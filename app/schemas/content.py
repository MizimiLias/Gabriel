"""Multimodal chat content parts shared by the API and persistence layers."""

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

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


type ContentPart = Annotated[TextPart | ImageURLPart, Field(discriminator="type")]

_CONTENT_PARTS_ADAPTER = TypeAdapter(list[ContentPart])


def parse_content_parts(value: object) -> list[ContentPart]:
    """Validate and normalize persisted or API-provided content parts.

    Args:
        value: Raw JSON-compatible content parts.

    Returns:
        Validated content parts.
    """

    return _CONTENT_PARTS_ADAPTER.validate_python(value)


def dump_content_parts(parts: Sequence[ContentPart]) -> list[dict[str, Any]]:
    """Serialize content parts for PostgreSQL JSONB storage."""

    return [part.model_dump(mode="json") for part in parts]


def content_parts_for_model(
    parts: Sequence[ContentPart | dict[str, Any]],
) -> list[str | dict[Any, Any]]:
    """Convert stored parts to the content array accepted by LangChain/OpenAI."""

    return [part.model_dump(mode="json") for part in parse_content_parts(list(parts))]


def is_system_reminder_text(value: str) -> bool:
    """Return whether a text block is an injected runtime reminder."""

    return bool(_SYSTEM_REMINDER_RE.fullmatch(value.strip()))


def plain_text_from_parts(
    parts: Sequence[ContentPart | dict[str, Any]],
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
    for part in parse_content_parts(list(parts)):
        if not isinstance(part, TextPart):
            continue
        if not include_system_reminders and is_system_reminder_text(part.text):
            continue
        if part.text.strip():
            texts.append(part.text.strip())
    return "\n\n".join(texts)


def preview_text_from_parts(parts: Sequence[ContentPart | dict[str, Any]]) -> str:
    """Return a concise text preview, falling back to an image marker."""

    parsed = parse_content_parts(list(parts))
    text = plain_text_from_parts(parsed)
    if text:
        return text
    if any(isinstance(part, ImageURLPart) for part in parsed):
        return "[Image]"
    return ""


def append_system_reminder(parts: Sequence[ContentPart], timezone: str) -> list[ContentPart]:
    """Append the persisted per-message runtime clock reminder.

    Args:
        parts: User-provided content parts.
        timezone: IANA timezone used to render the current time.

    Returns:
        A new list containing the original parts followed by one reminder block.
    """

    now = datetime.now(ZoneInfo(timezone))
    reminder = TextPart(
        text=(
            f"{SYSTEM_REMINDER_PREFIX}Current datetime: "
            f"{now.isoformat(timespec='seconds')} ({timezone}), "
            f"Weekday: {_WEEKDAY_NAMES[now.weekday()]}{SYSTEM_REMINDER_SUFFIX}"
        )
    )
    return [*parts, reminder]


def assistant_text_parts(text: str) -> list[ContentPart]:
    """Build the normal single-text-part representation of an assistant reply."""

    return [TextPart(text=text)]
