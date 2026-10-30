"""MCP configuration and loader tests."""

import json
from pathlib import Path

import pytest

from app.exceptions.mcp import MCPConfigError
from app.mcp import tools as tools_module
from app.mcp.config import load_mcp_config, to_mcp_adapter_config
from app.mcp.tools import load_mcp_tools
from tests.support import make_settings


def _write_config(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_stdio_and_remote_servers(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "mcp.json",
        {
            "mcpServers": {
                "weather": {
                    "command": "node",
                    "args": ["/opt/weather/dist/index.js"],
                    "env": {"AMAP_API_KEY": "secret"},
                    "cwd": "/opt/weather",
                },
                "calendar": {
                    "url": "https://calendar.test/mcp",
                    "transport": "streamable-http",
                    "headers": {"Authorization": "Bearer token"},
                },
            }
        },
    )

    config = load_mcp_config(path)

    assert set(config.mcp_servers) == {"weather", "calendar"}
    assert config.mcp_servers["weather"].env == {"AMAP_API_KEY": "secret"}
    assert to_mcp_adapter_config(config) == {
        "mcpServers": {
            "weather": {
                "command": "node",
                "args": ["/opt/weather/dist/index.js"],
                "env": {"AMAP_API_KEY": "secret"},
                "cwd": "/opt/weather",
            },
            "calendar": {
                "url": "https://calendar.test/mcp",
                "transport": "streamable-http",
                "headers": {"Authorization": "Bearer token"},
            },
        }
    }


def test_stdio_server_can_omit_optional_fields(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "mcp.json",
        {"mcpServers": {"weather": {"command": "node"}}},
    )

    assert to_mcp_adapter_config(load_mcp_config(path)) == {
        "mcpServers": {"weather": {"command": "node", "args": [], "env": {}}}
    }


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(MCPConfigError):
        load_mcp_config(tmp_path / "missing.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(MCPConfigError):
        load_mcp_config(path)


@pytest.mark.parametrize(
    "server",
    [
        {"args": ["x"]},
        {"command": "node", "url": "https://x.test/mcp"},
        {"command": "node", "transport": "sse"},
    ],
    ids=["neither", "both", "stdio-with-remote-transport"],
)
def test_invalid_server_config_raises(tmp_path: Path, server: dict[str, object]) -> None:
    path = _write_config(tmp_path / "mcp.json", {"mcpServers": {"bad": server}})

    with pytest.raises(MCPConfigError):
        load_mcp_config(path)


async def test_disabled_mcp_returns_no_tools() -> None:
    assert await load_mcp_tools(make_settings()) == []


async def test_unreachable_server_degrades_to_no_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(
        tmp_path / "mcp.json",
        {"mcpServers": {"weather": {"command": "node"}}},
    )

    class FailingAdapter:
        def __init__(self, _target: object) -> None:
            pass

        async def list_tools(self) -> list[object]:
            raise RuntimeError("server failed to start")

    monkeypatch.setattr(tools_module, "MCPAdapter", FailingAdapter)

    settings = make_settings(mcp_enabled=True, mcp_config_path=str(path))

    assert await load_mcp_tools(settings) == []


async def test_invalid_config_fails_startup(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "mcp.json", {"mcpServers": {"bad": {"args": []}}})

    settings = make_settings(mcp_enabled=True, mcp_config_path=str(path))

    with pytest.raises(MCPConfigError):
        await load_mcp_tools(settings)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _StaticAdapter:
    def __init__(self, _target: object) -> None:
        pass

    async def list_tools(self) -> list[_FakeTool]:
        return [_FakeTool("get_weather"), _FakeTool("weather_get_weather"), _FakeTool("run_shell")]


async def test_allowlist_filters_tools_including_server_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path / "mcp.json", {"mcpServers": {"weather": {"command": "node"}}})
    monkeypatch.setattr(tools_module, "MCPAdapter", _StaticAdapter)

    settings = make_settings(
        mcp_enabled=True,
        mcp_config_path=str(path),
        mcp_allowed_tools="get_weather",
    )

    tools = await load_mcp_tools(settings)

    assert [tool.name for tool in tools] == ["get_weather", "weather_get_weather"]


async def test_empty_allowlist_keeps_every_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path / "mcp.json", {"mcpServers": {"weather": {"command": "node"}}})
    monkeypatch.setattr(tools_module, "MCPAdapter", _StaticAdapter)

    tools = await load_mcp_tools(make_settings(mcp_enabled=True, mcp_config_path=str(path)))

    assert len(tools) == 3
