"""Agent runtime wrapper tests."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.agents.runtime import (
    AgentReasoning,
    AgentResult,
    AgentToken,
    AgentToolCalls,
    AgentToolResult,
    LangChainChatAgent,
)


class FakeGraph:
    """Minimal graph double exposing the two methods the runtime uses."""

    def __init__(self, *, invoke_result: dict[str, Any], items: list[Any]) -> None:
        self.invoke_result = invoke_result
        self.items = items
        self.astream_kwargs: dict[str, Any] = {}

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.invoke_result

    async def astream(
        self,
        payload: dict[str, Any],
        *,
        stream_mode: object = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.astream_kwargs = {"stream_mode": stream_mode, **kwargs}
        for item in self.items:
            yield item


def _tool_call(*, call_id: str = "call-1", name: str = "get_weather") -> dict[str, Any]:
    return {"name": name, "args": {"city": "北京"}, "id": call_id, "type": "tool_call"}


async def test_invoke_returns_final_text_and_intermediate_trace() -> None:
    history = [HumanMessage(content="weather?")]
    tool_call_message = AIMessage(content="", tool_calls=[_tool_call()])
    tool_message = ToolMessage(content="晴 20 度", tool_call_id="call-1", name="get_weather")
    final_message = AIMessage(content="北京晴, 20 度。")
    graph = FakeGraph(
        invoke_result={
            "messages": [*history, tool_call_message, tool_message, final_message],
        },
        items=[],
    )
    agent = LangChainChatAgent(graph)

    result = await agent.invoke(history)

    assert result == AgentResult(
        text="北京晴, 20 度。",
        trace=[tool_call_message, tool_message],
    )


async def test_stream_maps_tokens_and_tool_events() -> None:
    graph = FakeGraph(
        invoke_result={"messages": []},
        items=[
            ("messages", (AIMessageChunk(content="查"), {"langgraph_node": "model"})),
            (
                "messages",
                (
                    AIMessageChunk(content="", additional_kwargs={"reasoning_content": "想想"}),
                    {"langgraph_node": "model"},
                ),
            ),
            (
                "messages",
                (AIMessageChunk(content="hidden"), {"langgraph_node": "tools"}),
            ),
            (
                "updates",
                {"model": {"messages": [AIMessage(content="", tool_calls=[_tool_call()])]}},
            ),
            (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content="晴 20 度",
                                tool_call_id="call-1",
                                name="get_weather",
                                status="error",
                            )
                        ]
                    }
                },
            ),
            ("messages", (AIMessageChunk(content="好"), {"langgraph_node": "model"})),
        ],
    )
    agent = LangChainChatAgent(graph)

    events = [event async for event in agent.stream([HumanMessage(content="weather?")])]

    assert events == [
        AgentToken(delta="查"),
        AgentReasoning(delta="想想"),
        AgentToolCalls(calls=(_tool_call(),), text=""),
        AgentToolResult(
            tool_call_id="call-1",
            name="get_weather",
            content="晴 20 度",
            is_error=True,
        ),
        AgentToken(delta="好"),
    ]
    assert graph.astream_kwargs["stream_mode"] == ["messages", "updates"]


async def test_stream_accepts_v2_dict_payloads() -> None:
    graph = FakeGraph(
        invoke_result={"messages": []},
        items=[
            {
                "type": "messages",
                "data": (AIMessageChunk(content="hello"), {"langgraph_node": "model"}),
            },
            {
                "type": "updates",
                "data": {"model": {"messages": [AIMessage(content="", tool_calls=[_tool_call()])]}},
            },
        ],
    )
    agent = LangChainChatAgent(graph)

    events = [event async for event in agent.stream([HumanMessage(content="hi")])]

    assert events == [
        AgentToken(delta="hello"),
        AgentToolCalls(calls=(_tool_call(),), text=""),
    ]


async def test_stream_rejects_unknown_payload_shape() -> None:
    graph = FakeGraph(invoke_result={"messages": []}, items=[("unknown", None)])
    agent = LangChainChatAgent(graph)

    with pytest.raises(RuntimeError, match="Unexpected agent stream payload"):
        [event async for event in agent.stream([HumanMessage(content="hi")])]
