"""Process-wide chat agent assembly tests."""

from typing import Any

import pytest

from app.agents.runtime import LangChainChatAgent
from app.configs import agent as agent_config
from tests.support import make_settings


@pytest.fixture(autouse=True)
def _reset_agent() -> None:
    agent_config.reset_chat_agent()


async def test_missing_tools_disable_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_tools(_settings: object) -> list[Any]:
        return []

    monkeypatch.setattr(agent_config, "load_mcp_tools", no_tools)

    await agent_config.initialize_chat_agent(make_settings())

    assert agent_config.get_chat_agent() is None


async def test_tools_build_one_shared_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    async def one_tool(_settings: object) -> list[Any]:
        return [sentinel]

    monkeypatch.setattr(agent_config, "load_mcp_tools", one_tool)
    monkeypatch.setattr(
        LangChainChatAgent,
        "from_model",
        lambda _model, tools: sentinel if tools == [sentinel] else None,
    )

    await agent_config.initialize_chat_agent(make_settings())

    assert agent_config.get_chat_agent() is sentinel
