"""MCP server configuration loaded from mcp.json."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.exceptions.mcp import MCPConfigError


class MCPServerConfig(BaseModel):
    """One entry under ``mcpServers`` in mcp.json."""

    model_config = ConfigDict(extra="forbid")

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    transport: Literal["http", "streamable-http", "sse"] | None = None

    timeout: int | None = None
    keep_alive: bool | None = None

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        """Require exactly one transport and reject cross-transport fields."""

        if (self.command is None) == (self.url is None):
            raise ValueError("A server requires exactly one of command or url.")
        if self.command is not None:
            if self.transport is not None:
                raise ValueError("transport is only valid for remote servers.")
            if self.headers:
                raise ValueError("headers are only valid for remote servers.")
        elif self.args or self.env or self.cwd is not None or self.keep_alive is not None:
            raise ValueError("stdio fields are only valid for command servers.")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        return self


class MCPFileConfig(BaseModel):
    """The top-level mcp.json document."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    mcp_servers: dict[str, MCPServerConfig] = Field(
        alias="mcpServers",
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_server_names(self) -> MCPFileConfig:
        """Reject blank server names."""

        if any(not name.strip() for name in self.mcp_servers):
            raise ValueError("MCP server names cannot be blank.")
        return self


def load_mcp_config(path: Path) -> MCPFileConfig:
    """Load and validate an mcp.json file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MCPConfigError(f"MCP config not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"MCP config could not be read: {path}") from exc

    try:
        return MCPFileConfig.model_validate(payload)
    except ValidationError as exc:
        raise MCPConfigError(f"MCP config is invalid: {path}: {exc}") from exc


def to_mcp_adapter_config(config: MCPFileConfig) -> dict[str, Any]:
    """Render the validated config back into the dict accepted by MCPAdapter."""

    return {
        "mcpServers": {name: _server_dict(server) for name, server in config.mcp_servers.items()}
    }


def _server_dict(server: MCPServerConfig) -> dict[str, Any]:
    data: dict[str, Any]
    if server.command is not None:
        data = {
            "command": server.command,
            "args": list(server.args),
            "env": dict(server.env),
        }
        if server.cwd is not None:
            data["cwd"] = server.cwd
        if server.keep_alive is not None:
            data["keep_alive"] = server.keep_alive
    else:
        data = {"url": server.url, "headers": dict(server.headers)}
        if server.transport is not None:
            data["transport"] = server.transport
    if server.timeout is not None:
        data["timeout"] = server.timeout
    return data
