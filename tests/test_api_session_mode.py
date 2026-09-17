"""[LOCAL PATCH] sciherd-cloud-smartagent：执行模式的服务端协议测试（M2 / FR-1.4 / FR-1.5）。

覆盖两件事：请求级 `session_mode` 落到会话 scope（非法取值必须 400），以及
Plan 模式的结构化计划经 `kind=plan` 事件注入 SSE。

计划事件与审批请求的区别在契约上是硬的：审批是「点对点」（要等回话），
计划是「只出不进」（裁决走下一轮请求），因此这里只需断言它被下发。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import create_app
from nanobot.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:  # pragma: no cover - 依赖缺失时整体跳过
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


def _make_agent(process_direct) -> MagicMock:
    agent = MagicMock()
    agent.process_direct = process_direct
    agent.aclose = AsyncMock()
    return agent


def _with_resolver(agent: MagicMock, tmp_path) -> None:
    resolver = MagicMock()
    resolver.default_workspace = str(tmp_path)
    resolver.default_restrict_to_workspace = True
    agent.workspace_scopes = resolver


async def _sse_events(body: str) -> list[dict]:
    deltas = [
        json.loads(line[len("data: "):])["choices"][0]["delta"]
        for line in body.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    return [d["nanobot_event"] for d in deltas if "nanobot_event" in d]


# ---------------------------------------------------------------------------
# session_mode -> 会话 scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_mode_reaches_workspace_scope(aiohttp_client, tmp_path) -> None:
    """session_mode=plan 必须变成 metadata 里的 session_mode=plan。"""
    captured: dict = {}

    async def fake_process_direct(*, metadata=None, on_stream=None, on_stream_end=None, **kwargs):
        captured["metadata"] = metadata
        return "ok"

    agent = _make_agent(fake_process_direct)
    _with_resolver(agent, tmp_path)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "session_mode": "plan",
        },
    )
    assert resp.status == 200
    scope = captured["metadata"][WORKSPACE_SCOPE_METADATA_KEY]
    assert scope["session_mode"] == "plan"


@pytest.mark.asyncio
async def test_session_mode_rides_alongside_other_axes(aiohttp_client, tmp_path) -> None:
    """执行模式与工作空间 / 权限 / 审批是同一处承载的四条独立轴。"""
    captured: dict = {}

    async def fake_process_direct(*, metadata=None, on_stream=None, on_stream_end=None, **kwargs):
        captured["metadata"] = metadata
        return "ok"

    agent = _make_agent(fake_process_direct)
    _with_resolver(agent, tmp_path)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "workspace": str(tmp_path),
            "access_mode": "full",
            "approval_mode": "auto",
            "session_mode": "ask",
        },
    )
    assert resp.status == 200
    scope = captured["metadata"][WORKSPACE_SCOPE_METADATA_KEY]
    assert scope == {
        "project_path": str(tmp_path),
        "access_mode": "full",
        "tool_approval": "auto",
        "session_mode": "ask",
    }


@pytest.mark.asyncio
async def test_invalid_session_mode_is_rejected(aiohttp_client, tmp_path) -> None:
    """非法模式不能被静默降级成 craft——那是把「只读」悄悄变成「可写」。"""

    async def fake_process_direct(**kwargs):  # pragma: no cover - 不该被调用
        raise AssertionError("invalid session_mode should short-circuit")

    agent = _make_agent(fake_process_direct)
    _with_resolver(agent, tmp_path)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "session_mode": "readonly",
        },
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_no_session_mode_leaves_scope_clean(aiohttp_client) -> None:
    """不带 session_mode 时不应凭空造出 metadata（老客户端零影响）。"""
    captured: dict = {}

    async def fake_process_direct(*, metadata=None, on_stream=None, on_stream_end=None, **kwargs):
        captured["metadata"] = metadata
        return "ok"

    agent = _make_agent(fake_process_direct)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    assert captured["metadata"] is None


# ---------------------------------------------------------------------------
# plan -> SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_is_emitted_as_sse_event(aiohttp_client, tmp_path) -> None:
    """Plan 模式提交计划后，SSE 里要出现一条 kind=plan 的事件。"""

    async def fake_process_direct(*, on_progress=None, on_stream=None, on_stream_end=None, **kwargs):
        await on_progress(
            "",
            plan={
                "plan_id": "plan-abc123",
                "steps": [
                    {"id": "s1", "text": "读一遍现有实现", "status": "pending"},
                    {"id": "s2", "text": "写下改动方案", "status": "pending"},
                ],
            },
        )
        if on_stream_end:
            await on_stream_end()
        return "plan submitted"

    agent = _make_agent(fake_process_direct)
    _with_resolver(agent, tmp_path)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "计划一下"}],
            "stream": True,
            "session_mode": "plan",
        },
    )
    assert resp.status == 200
    events = await _sse_events(await resp.text())

    plan_events = [e for e in events if e.get("kind") == "plan"]
    assert len(plan_events) == 1
    assert plan_events[0]["plan_id"] == "plan-abc123"
    assert [s["text"] for s in plan_events[0]["steps"]] == [
        "读一遍现有实现",
        "写下改动方案",
    ]
