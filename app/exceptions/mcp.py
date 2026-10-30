"""MCP integration exceptions."""


class MCPError(Exception):
    """Base class for expected MCP integration failures."""


class MCPConfigError(MCPError):
    """Raised when mcp.json is missing or invalid."""
