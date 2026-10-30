"""Process-wide MCP-backed agent assembly."""

import structlog

from app.agents.runtime import ChatAgent, LangChainChatAgent
from app.configs.llm import build_chat_model
from app.configs.settings import Settings
from app.mcp.tools import load_mcp_tools

logger = structlog.get_logger(__name__)

_agent: ChatAgent | None = None


async def initialize_chat_agent(settings: Settings) -> None:
    """Load MCP tools once and build the shared tool-calling agent."""

    global _agent
    tools = await load_mcp_tools(settings)
    if not tools:
        _agent = None
        logger.info("chat_agent_disabled", mcp_enabled=settings.mcp_enabled)
        return

    _agent = LangChainChatAgent.from_model(build_chat_model(), tools)
    logger.info("chat_agent_ready", tool_count=len(tools))


def get_chat_agent() -> ChatAgent | None:
    """Return the shared agent, or None when no tools are available."""

    return _agent


def reset_chat_agent() -> None:
    """Clear the shared agent (used by tests)."""

    global _agent
    _agent = None
