"""[LOCAL PATCH] sciherd-cloud-smartagent：逐条批准的服务端协议测试。

覆盖三件事：请求级 `approval_mode` 落到会话 scope、`approval_request` 进度
事件被注入 SSE、以及 `POST /v1/tool-approvals/{id}` 能把裁决送回挂起的引擎轮次。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import create_app
from nanobot.security.tool_approval import (
    PENDING_APPROVALS,
    ApprovalDecision,
    ApprovalRequest,
)
from nanobot.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY

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


def _make_agent(process_direct) -> MagicMock:
    agent = MagicMock()
    agent.process_direct = process_direct
    agent.aclose = AsyncMock()
    return agent


@pytest_asyncio.fixture(autouse=True)
async def _clear_pending():
    """注册表是进程级的，用例之间必须清干净，否则会串成假通过。"""
    yield
    for pending in list(PENDING_APPROVALS._pending.values()):
        PENDING_APPROVALS.discard(pending.request.request_id)


# ---------------------------------------------------------------------------
# approval_mode -> 会话 scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_mode_reaches_workspace_scope(aiohttp_client, tmp_path) -> None:
    """approval_mode=ask 必须变成 metadata 里的 tool_approval=ask。"""
    captured: dict = {}

    async def fake_process_direct(*, metadata=None, on_stream=None, on_stream_end=None, **kwargs):
        captured["metadata"] = metadata
        return "ok"

    agent = _make_agent(fake_process_direct)
    resolver = MagicMock()
    resolver.default_workspace = str(tmp_path)
    resolver.default_restrict_to_workspace = True
    agent.workspace_scopes = resolver

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "approval_mode": "ask",
        },
    )
    assert resp.status == 200
    scope = captured["metadata"][WORKSPACE_SCOPE_METADATA_KEY]
    assert scope["tool_approval"] == "ask"


@pytest.mark.asyncio
async def test_invalid_approval_mode_is_rejected(aiohttp_client, tmp_path) -> None:
    """非法取值不能被静默忽略，否则前端会以为开了审批其实没开。"""

    async def fake_process_direct(**kwargs):  # pragma: no cover - 不该被调用
        raise AssertionError("invalid approval_mode should short-circuit")

    agent = _make_agent(fake_process_direct)
    resolver = MagicMock()
    resolver.default_workspace = str(tmp_path)
    resolver.default_restrict_to_workspace = True
    agent.workspace_scopes = resolver

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "approval_mode": "whatever",
        },
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_no_approval_mode_leaves_scope_clean(aiohttp_client) -> None:
    """不带 approval_mode 时不应凭空造出 metadata（老客户端零影响）。"""
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
# approval_request -> SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_request_is_emitted_as_sse_event(aiohttp_client) -> None:
    """引擎挂起等裁决时，SSE 里要出现一条 kind=approval_request 的事件。"""

    async def fake_process_direct(*, on_progress=None, on_stream=None, on_stream_end=None, **kwargs):
        await on_progress(
            "",
            approval_request={
                "id": "appr-abc123",
                "tool": "write_file",
                "args": {"path": "demo.txt"},
            },
        )
        if on_stream_end:
            await on_stream_end()
        return "waiting"

    agent = _make_agent(fake_process_direct)
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "write a file"}], "stream": True},
    )
    assert resp.status == 200
    body = await resp.text()
    deltas = [
        json.loads(line[len("data: "):])["choices"][0]["delta"]
        for line in body.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    events = [d["nanobot_event"] for d in deltas if "nanobot_event" in d]
    approval_events = [e for e in events if e.get("kind") == "approval_request"]
    assert len(approval_events) == 1
    assert approval_events[0]["id"] == "appr-abc123"
    assert approval_events[0]["tool"] == "write_file"
    assert approval_events[0]["args"] == {"path": "demo.txt"}


# ---------------------------------------------------------------------------
# POST /v1/tool-approvals/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allow_verdict_resolves_pending_request(aiohttp_client) -> None:
    """客户端回传 allow 后，引擎侧等待的 future 必须拿到放行裁决。"""
    request = ApprovalRequest(
        request_id="appr-allow1",
        tool="write_file",
        args={"path": "x.txt"},
        session_key="api:s1",
    )
    future = PENDING_APPROVALS.register(request)

    agent = _make_agent(AsyncMock(return_value="ok"))
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/tool-approvals/appr-allow1",
        headers=AUTH_HEADERS,
        json={"decision": "allow"},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["decision"] == "allow"

    decision = await future
    assert isinstance(decision, ApprovalDecision)
    assert decision.allowed is True

    # 已经裁决过的请求不应再能用（防止重复回传把状态机搅乱）。
    again = await client.post(
        "/v1/tool-approvals/appr-allow1",
        headers=AUTH_HEADERS,
        json={"decision": "deny"},
    )
    assert again.status == 404


@pytest.mark.asyncio
async def test_deny_verdict_carries_reason(aiohttp_client) -> None:
    """拒绝要带上原因，引擎会把它回灌给模型让它改道。"""
    request = ApprovalRequest(request_id="appr-deny1", tool="run_command", args={})
    future = PENDING_APPROVALS.register(request)

    agent = _make_agent(AsyncMock(return_value="ok"))
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/tool-approvals/appr-deny1",
        headers=AUTH_HEADERS,
        json={"decision": "deny", "reason": "不要删库"},
    )
    assert resp.status == 200

    decision = await future
    assert decision.allowed is False
    assert decision.reason == "不要删库"


@pytest.mark.asyncio
async def test_unknown_request_id_is_404(aiohttp_client) -> None:
    agent = _make_agent(AsyncMock(return_value="ok"))
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/tool-approvals/appr-nope",
        headers=AUTH_HEADERS,
        json={"decision": "allow"},
    )
    assert resp.status == 404


@pytest.mark.asyncio
async def test_bad_decision_is_400(aiohttp_client) -> None:
    request = ApprovalRequest(request_id="appr-bad1", tool="write_file", args={})
    PENDING_APPROVALS.register(request)

    agent = _make_agent(AsyncMock(return_value="ok"))
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/tool-approvals/appr-bad1",
        headers=AUTH_HEADERS,
        json={"decision": "maybe"},
    )
    assert resp.status == 400
    # 非法裁决不能把请求从注册表里抹掉，否则真实裁决会撞 404。
    assert PENDING_APPROVALS.get("appr-bad1") is not None


@pytest.mark.asyncio
async def test_approval_endpoint_requires_auth(aiohttp_client) -> None:
    request = ApprovalRequest(request_id="appr-auth1", tool="write_file", args={})
    PENDING_APPROVALS.register(request)

    agent = _make_agent(AsyncMock(return_value="ok"))
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/tool-approvals/appr-auth1",
        json={"decision": "allow"},
    )
    assert resp.status == 401
