"""LangChain tool-calling agent runtime.

The rest of the application only depends on the small event/result contract in this
module, so the LangGraph graph type never leaks into the chat service.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool


@dataclass(frozen=True, slots=True)
class AgentToken:
    """One generated text fragment."""

    delta: str


@dataclass(frozen=True, slots=True)
class AgentReasoning:
    """One reasoning/thinking fragment (never persisted)."""

    delta: str


@dataclass(frozen=True, slots=True)
class AgentToolCalls:
    """A completed model step that requests one or more tools."""

    calls: tuple[ToolCall, ...]
    text: str


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """One completed tool invocation."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool


type AgentEvent = AgentToken | AgentReasoning | AgentToolCalls | AgentToolResult


@dataclass(frozen=True, slots=True)
class AgentResult:
    """A completed non-streaming agent turn."""

    text: str
    trace: list[AIMessage | ToolMessage]


class ChatAgent(Protocol):
    """The narrow agent interface consumed by the chat service."""

    async def invoke(self, messages: list[BaseMessage]) -> AgentResult: ...

    def stream(self, messages: list[BaseMessage]) -> AsyncIterator[AgentEvent]: ...


def extract_reasoning_delta(chunk: object) -> str:
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


class LangChainChatAgent:
    """Adapt a LangGraph tool-calling agent to the chat service contract."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @classmethod
    def from_model(
        cls,
        model: BaseChatModel,
        tools: list[BaseTool],
    ) -> LangChainChatAgent:
        """Build the underlying agent graph from a chat model and MCP tools."""

        return cls(create_agent(model=model, tools=list(tools)))

    async def invoke(self, messages: list[BaseMessage]) -> AgentResult:
        """Run one turn and return the final text plus intermediate messages."""

        result: dict[str, Any] = await self._graph.ainvoke({"messages": messages})
        produced: list[BaseMessage] = result["messages"][len(messages) :]
        if not produced:
            return AgentResult(text="", trace=[])

        final = produced[-1]
        text = final.text if isinstance(final, AIMessage) else ""
        trace = [
            message for message in produced[:-1] if isinstance(message, (AIMessage, ToolMessage))
        ]
        return AgentResult(text=text, trace=trace)

    async def stream(self, messages: list[BaseMessage]) -> AsyncIterator[AgentEvent]:
        """Run one turn and emit text, reasoning, and tool events as they complete."""

        async for item in self._graph.astream(
            {"messages": messages},
            stream_mode=["messages", "updates"],
        ):
            mode, payload = _normalise_stream_item(item)
            if mode == "messages":
                message_event = _message_event(payload)
                if message_event is not None:
                    yield message_event
            elif mode == "updates":
                for update_event in _update_events(payload):
                    yield update_event
            else:
                raise RuntimeError(f"Unexpected agent stream payload: {item!r}")


def _normalise_stream_item(item: object) -> tuple[str, Any]:
    """Accept both the v1 ``(mode, payload)`` tuple and the v2 dict shape."""

    if isinstance(item, tuple) and len(item) == 2:
        mode, payload = item
        if isinstance(mode, str):
            return mode, payload
    if isinstance(item, dict) and "type" in item and "data" in item:
        return str(item["type"]), item["data"]
    raise RuntimeError(f"Unexpected agent stream payload: {item!r}")


def _message_event(payload: Any) -> AgentToken | AgentReasoning | None:
    """Convert a ``messages`` stream payload produced by the model node."""

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError(f"Unexpected agent message payload: {payload!r}")
    chunk, metadata = payload
    if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "model":
        return None

    reasoning = extract_reasoning_delta(chunk)
    if reasoning:
        return AgentReasoning(delta=reasoning)
    text = getattr(chunk, "text", "")
    if text:
        return AgentToken(delta=text)
    return None


def _update_events(payload: Any) -> list[AgentToolCalls | AgentToolResult]:
    """Convert one ``updates`` payload into completed tool-call events."""

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected agent update payload: {payload!r}")

    events: list[AgentToolCalls | AgentToolResult] = []
    for node, update in payload.items():
        if not isinstance(update, dict):
            continue
        for message in update.get("messages", []):
            if node == "model" and isinstance(message, AIMessage) and message.tool_calls:
                events.append(AgentToolCalls(calls=tuple(message.tool_calls), text=message.text))
            elif node == "tools" and isinstance(message, ToolMessage):
                events.append(
                    AgentToolResult(
                        tool_call_id=message.tool_call_id,
                        name=message.name or "",
                        content=message.text,
                        is_error=getattr(message, "status", None) == "error",
                    )
                )
    return events


__all__ = [
    "AgentEvent",
    "AgentReasoning",
    "AgentResult",
    "AgentToken",
    "AgentToolCalls",
    "AgentToolResult",
    "LangChainChatAgent",
    "extract_reasoning_delta",
]
