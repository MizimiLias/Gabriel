"""Opt-in end-to-end test against the local weather MCP server."""

import os

import pytest

from app.mcp.tools import load_mcp_tools
from tests.support import make_settings

RUN_MCP_TESTS = os.getenv("RUN_MCP_TESTS") == "true"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_MCP_TESTS, reason="RUN_MCP_TESTS is not set"),
]


def _result_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(
            str(block.get("text", ""))
            for block in result
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(result)


async def test_weather_tool_round_trip() -> None:
    tools = await load_mcp_tools(make_settings(mcp_enabled=True, mcp_config_path="mcp.json"))

    weather = next((tool for tool in tools if tool.name.endswith("get_weather")), None)
    assert weather is not None, [tool.name for tool in tools]

    result = await weather.ainvoke({"city": "北京", "extensions": "base"})

    text = _result_text(result)
    assert "北京" in text
    assert "气温" in text
