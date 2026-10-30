"""MCP server discovery adapted into LangChain tools."""

from pathlib import Path

import structlog
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool

from app.configs.settings import Settings
from app.mcp.config import load_mcp_config, to_mcp_adapter_config

logger = structlog.get_logger(__name__)


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Load every tool exposed by the configured MCP servers.

    Configuration errors fail startup; an unreachable server only logs a warning so
    the chat service keeps working without tools.
    """

    if not settings.mcp_enabled:
        return []

    config = load_mcp_config(Path(settings.mcp_config_path))
    if not config.mcp_servers:
        return []

    adapter = MCPAdapter(to_mcp_adapter_config(config))
    try:
        tools: list[BaseTool] = await adapter.list_tools()
    except Exception as exc:
        logger.warning("mcp_tools_unavailable", error_type=type(exc).__name__)
        return []

    tools = _apply_allowlist(tools, settings.mcp_allowed_tools)
    logger.info(
        "mcp_tools_loaded",
        count=len(tools),
        tool_names=[tool.name for tool in tools],
    )
    return tools


def _apply_allowlist(tools: list[BaseTool], allowed: str) -> list[BaseTool]:
    """Keep only explicitly allowed tools, matching server-prefixed names too.

    An empty allowlist keeps every tool, which is the default for a local,
    operator-maintained mcp.json.
    """

    names = {name.strip() for name in allowed.split(",") if name.strip()}
    if not names:
        return tools
    return [
        tool
        for tool in tools
        if tool.name in names or any(tool.name.endswith(f"_{name}") for name in names)
    ]
