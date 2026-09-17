"""[LOCAL PATCH] nanowork FR-3.4：`/v1/plugins` 的服务端协议测试。

覆盖四件事：清单形状（含引擎侧激活状态）、启停往返、错误码分派
（400/401/404/503），以及**工作区不可由请求方指定**——激活会落盘写标记，
让请求体决定写哪儿等于把写路径交给调用方。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.agent import plugins as agent_plugins
from nanobot.agent.plugins import (
    AGENT_PLUGIN_MCP_SCHEMA,
    AGENT_PLUGIN_SCHEMA,
    discover_agent_plugins,
)
from nanobot.api.server import create_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

API_KEY = "secret"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


@pytest.fixture(autouse=True)
def _isolate_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """激活标记落到 tmp，别碰宿主机真实配置目录。"""
    monkeypatch.setattr(
        agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json"
    )


@pytest.fixture(autouse=True)
def _isolate_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """用户级插件目录同样隔离——否则断言会随本机装了什么而变。"""
    home = tmp_path / "user-home"
    monkeypatch.setenv("NANOWORK_HOME", str(home))
    return home / "plugins"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _install(workspace: Path, name: str = "demo", **fields: object) -> Path:
    root = workspace / "plugins" / name
    _write_json(root / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": name, **fields})
    return root


def _make_agent(workspace: Path | None) -> MagicMock:
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="ok")
    agent.aclose = AsyncMock()
    agent.workspace = str(workspace) if workspace is not None else ""
    return agent


async def _client_for(aiohttp_client, workspace: Path):
    app = create_app(_make_agent(workspace), model_name="m", api_key=API_KEY)
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# GET /v1/plugins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_reports_project_and_user_plugins(
    aiohttp_client, tmp_path: Path, _isolate_user_dir: Path
) -> None:
    _install(tmp_path, "alpha", description="project copy")
    _write_json(
        _isolate_user_dir / "beta" / "plugin.json",
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": "beta", "description": "user copy"},
    )

    client = await _client_for(aiohttp_client, tmp_path)
    resp = await client.get("/v1/plugins", headers=AUTH_HEADERS)
    assert resp.status == 200

    payload = await resp.json()
    assert payload["object"] == "list"
    by_name = {item["name"]: item for item in payload["data"]}
    assert set(by_name) == {"alpha", "beta"}
    assert by_name["alpha"]["description"] == "project copy"
    # 没启用过就必须是 False——不能默认乐观。
    assert by_name["alpha"]["enabled"] is False
    assert by_name["beta"]["enabled"] is False
    # displayName 缺省回落到 name，客户端可以直接展示而不必自己兜。
    assert by_name["alpha"]["displayName"] == "alpha"
    assert by_name["alpha"]["mcpServers"] == []


@pytest.mark.asyncio
async def test_list_reports_engine_side_activation(aiohttp_client, tmp_path: Path) -> None:
    """激活状态由引擎给出（指纹绑定的标记），客户端不自己算哈希。"""
    _install(tmp_path, "demo")
    agent_plugins.set_agent_plugin_enabled(tmp_path, "demo", True)

    client = await _client_for(aiohttp_client, tmp_path)
    payload = await (await client.get("/v1/plugins", headers=AUTH_HEADERS)).json()
    assert [item["enabled"] for item in payload["data"]] == [True]


@pytest.mark.asyncio
async def test_list_includes_plugin_mcp_server_names(aiohttp_client, tmp_path: Path) -> None:
    _install(tmp_path, "demo")
    _write_json(
        tmp_path / "plugins" / "demo" / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {"srv": {"type": "stdio", "command": "python"}},
        },
    )

    client = await _client_for(aiohttp_client, tmp_path)
    payload = await (await client.get("/v1/plugins", headers=AUTH_HEADERS)).json()
    assert payload["data"][0]["mcpServers"] == ["srv"]


# ---------------------------------------------------------------------------
# POST /v1/plugins/{name}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enable_then_disable_round_trip(aiohttp_client, tmp_path: Path) -> None:
    _install(tmp_path, "demo")
    client = await _client_for(aiohttp_client, tmp_path)

    enabled = await client.post(
        "/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "enable"}
    )
    assert enabled.status == 200
    assert (await enabled.json())["enabled"] is True
    assert discover_agent_plugins(tmp_path)[0].enabled is True

    disabled = await client.post(
        "/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "disable"}
    )
    assert disabled.status == 200
    assert (await disabled.json())["enabled"] is False
    assert discover_agent_plugins(tmp_path)[0].enabled is False


@pytest.mark.asyncio
async def test_action_is_case_insensitive(aiohttp_client, tmp_path: Path) -> None:
    _install(tmp_path, "demo")
    client = await _client_for(aiohttp_client, tmp_path)

    resp = await client.post(
        "/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "ENABLE"}
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_unknown_plugin_is_404(aiohttp_client, tmp_path: Path) -> None:
    client = await _client_for(aiohttp_client, tmp_path)
    resp = await client.post(
        "/v1/plugins/nope", headers=AUTH_HEADERS, json={"action": "enable"}
    )
    assert resp.status == 404


@pytest.mark.asyncio
async def test_invalid_action_is_400(aiohttp_client, tmp_path: Path) -> None:
    """非法动作不能被当成 disable——那会让用户以为点了启用其实被停了。"""
    _install(tmp_path, "demo")
    agent_plugins.set_agent_plugin_enabled(tmp_path, "demo", True)
    client = await _client_for(aiohttp_client, tmp_path)

    resp = await client.post(
        "/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "install"}
    )
    assert resp.status == 400
    assert discover_agent_plugins(tmp_path)[0].enabled is True


@pytest.mark.asyncio
async def test_missing_action_is_400(aiohttp_client, tmp_path: Path) -> None:
    _install(tmp_path, "demo")
    client = await _client_for(aiohttp_client, tmp_path)

    resp = await client.post("/v1/plugins/demo", headers=AUTH_HEADERS, json={})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_package_changed_during_enable_is_409(
    aiohttp_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """包在启用瞬间被改动 → 引擎拒绝签发标记，这是 409 而不是 500。"""
    _install(tmp_path, "demo")
    client = await _client_for(aiohttp_client, tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Agent Plugin 'demo' changed while it was being enabled")

    # 处理器内部按需 `from ... import`，所以打在模块属性上即可生效。
    monkeypatch.setattr(agent_plugins, "set_agent_plugin_enabled", _boom)

    resp = await client.post(
        "/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "enable"}
    )
    assert resp.status == 409
    assert (await resp.json())["error"]["type"] == "plugin_state_error"


@pytest.mark.asyncio
async def test_plugins_endpoint_requires_auth(aiohttp_client, tmp_path: Path) -> None:
    _install(tmp_path, "demo")
    client = await _client_for(aiohttp_client, tmp_path)

    assert (await client.get("/v1/plugins")).status == 401
    assert (
        await client.post("/v1/plugins/demo", json={"action": "enable"})
    ).status == 401


@pytest.mark.asyncio
async def test_workspace_unavailable_is_503(aiohttp_client, tmp_path: Path) -> None:
    client = await _client_for(aiohttp_client, None)

    assert (await client.get("/v1/plugins", headers=AUTH_HEADERS)).status == 503
    assert (
        await client.post("/v1/plugins/demo", headers=AUTH_HEADERS, json={"action": "enable"})
    ).status == 503


@pytest.mark.asyncio
async def test_body_workspace_cannot_redirect_the_write(
    aiohttp_client, tmp_path: Path, _isolate_user_dir: Path
) -> None:
    """请求体里的 workspace 必须被忽略。

    激活会写 ``plugin-data/`` 标记。若让请求方指定工作区，就等于让任何能调到
    这个接口的人决定往哪个目录写文件。
    """
    _install(tmp_path, "demo")
    elsewhere = tmp_path / "elsewhere"
    _install(elsewhere, "demo")
    _write_json(
        _isolate_user_dir / "demo" / "plugin.json",
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo"},
    )

    client = await _client_for(aiohttp_client, tmp_path)
    resp = await client.post(
        "/v1/plugins/demo",
        headers=AUTH_HEADERS,
        json={"action": "enable", "workspace": str(elsewhere)},
    )
    assert resp.status == 200

    # 标记落在这台服务自己的工作区下（tmp_path），不在请求体指定的那个目录。
    assert discover_agent_plugins(tmp_path)[0].enabled is True
    assert not (elsewhere / "plugin-data").exists()
