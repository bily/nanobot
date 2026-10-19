"""Tests for SSE streaming support in /v1/chat/completions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import (
    _SSE_DONE,
    _finish_reason_for,
    _sse_chunk,
    create_app,
)

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

API_KEY = "secret"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ---------------------------------------------------------------------------
# Unit tests for SSE helpers
# ---------------------------------------------------------------------------


def test_sse_chunk_with_delta() -> None:
    raw = _sse_chunk("hello", "test-model", "chatcmpl-abc123")
    line = raw.decode()
    assert line.startswith("data: ")
    payload = json.loads(line[len("data: "):])
    assert payload["id"] == "chatcmpl-abc123"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "test-model"
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_sse_chunk_finish_reason() -> None:
    raw = _sse_chunk("", "m", "id1", finish_reason="stop")
    payload = json.loads(raw.decode().split("data: ", 1)[1])
    assert payload["choices"][0]["delta"] == {}
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_sse_done_format() -> None:
    assert _SSE_DONE == b"data: [DONE]\n\n"


def test_finish_reason_maps_engine_stop_reason() -> None:
    """引擎的 stop_reason 必须如实映射：报错/截断不能报成 stop。

    2026-09-29 断流事故里 finish_reason 是硬编码的 "stop"，客户端因此把一次异常结束
    当成了正常收尾（显示「已完成」且没有任何提示）。
    """
    assert _finish_reason_for("completed") == "stop"
    assert _finish_reason_for("empty_final_response") == "stop"
    assert _finish_reason_for("error") == "error"
    assert _finish_reason_for("tool_error") == "error"
    assert _finish_reason_for("max_iterations") == "length"
    # 缺失 / 未知 / 非字符串：保持历史行为，不改变既有语义
    assert _finish_reason_for(None) == "stop"
    assert _finish_reason_for("") == "stop"
    assert _finish_reason_for(42) == "stop"
    assert _finish_reason_for("something_new") == "stop"


# ---------------------------------------------------------------------------
# Integration tests with aiohttp TestClient
# ---------------------------------------------------------------------------


def _make_streaming_agent(tokens: list[str]) -> MagicMock:
    """Create a mock agent that streams tokens via on_stream callback."""
    agent = MagicMock()
    agent.aclose = AsyncMock()

    async def fake_process_direct(*, content="", media=None, session_key="",
                                  channel="", chat_id="", on_stream=None,
                                  on_stream_end=None, **kwargs):
        if on_stream:
            for token in tokens:
                await on_stream(token)
        if on_stream_end:
            await on_stream_end()
        return " ".join(tokens)

    agent.process_direct = fake_process_direct
    return agent


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


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_true_returns_sse(aiohttp_client) -> None:
    """stream=true should return text/event-stream with SSE chunks."""
    agent = _make_streaming_agent(["Hello", " world"])
    app = create_app(agent, model_name="test-model", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    assert resp.content_type == "text/event-stream"

    body = await resp.text()
    lines = [line for line in body.split("\n") if line.startswith("data: ")]

    # Should have: 2 token chunks + 1 finish chunk + [DONE]
    data_lines = [line[len("data: "):] for line in lines]
    assert data_lines[-1] == "[DONE]"

    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    # Last chunk before [DONE] should have finish_reason=stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_false_returns_json(aiohttp_client) -> None:
    """stream=false should still return regular JSON response."""
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="normal reply")
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "normal reply"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_default_is_false(aiohttp_client) -> None:
    """Omitting stream should behave like stream=false."""
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="default reply")
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "chat.completion"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_sse_chunk_ids_are_consistent(aiohttp_client) -> None:
    """All SSE chunks in a single stream should share the same id."""
    agent = _make_streaming_agent(["A", "B", "C"])
    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "go"}], "stream": True},
    )
    body = await resp.text()
    data_lines = [
        line[len("data: "):]
        for line in body.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    chunks = [json.loads(line) for line in data_lines]

    chunk_ids = {c["id"] for c in chunks}
    assert len(chunk_ids) == 1, f"Expected single chunk id, got {chunk_ids}"
    assert chunk_ids.pop().startswith("chatcmpl-")


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_passes_on_stream_callbacks(aiohttp_client) -> None:
    """process_direct should be called with on_stream and on_stream_end when streaming."""
    captured_kwargs: dict = {}

    async def fake_process_direct(**kwargs):
        captured_kwargs.update(kwargs)
        if kwargs.get("on_stream_end"):
            await kwargs["on_stream_end"]()
        return "done"

    agent = MagicMock()
    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    assert captured_kwargs.get("on_stream") is not None
    assert captured_kwargs.get("on_stream_end") is not None


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_segment_end_does_not_close_sse(aiohttp_client) -> None:
    """Intermediate stream-end callbacks should not terminate the HTTP stream."""
    agent = MagicMock()

    async def fake_process_direct(*, on_stream=None, on_stream_end=None, **kwargs):
        assert on_stream is not None
        assert on_stream_end is not None
        await on_stream("planning")
        await on_stream_end(resuming=True)
        await asyncio.sleep(0)
        await on_stream(" final")
        await on_stream_end(resuming=False)
        return "planning final"

    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "use a tool"}], "stream": True},
    )

    assert resp.status == 200
    body = await resp.text()
    data_lines = [
        line[len("data: "):] for line in body.split("\n") if line.startswith("data: ")
    ]
    assert data_lines[-1] == "[DONE]"

    chunks = [json.loads(line) for line in data_lines[:-1]]
    deltas = [c["choices"][0]["delta"].get("content", "") for c in chunks]
    assert "planning" in deltas
    assert " final" in deltas
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_uses_final_response_when_no_deltas(aiohttp_client) -> None:
    """stream=true should not return an empty stream when the agent returns content."""
    agent = MagicMock()

    async def fake_process_direct(*, on_stream=None, on_stream_end=None, **kwargs):
        assert on_stream is not None
        assert on_stream_end is not None
        await on_stream_end(resuming=False)
        return "plain final"

    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert resp.status == 200
    body = await resp.text()
    data_lines = [
        line[len("data: "):] for line in body.split("\n") if line.startswith("data: ")
    ]
    chunks = [json.loads(line) for line in data_lines[:-1]]
    deltas = [c["choices"][0]["delta"].get("content", "") for c in chunks]

    assert "plain final" in deltas
    assert data_lines[-1] == "[DONE]"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_finish_reason_reflects_engine_stop_reason(aiohttp_client) -> None:
    """引擎透出的 `_stop_reason` 必须变成 SSE 的 finish_reason。

    撞上迭代上限的一轮要报 "length"：客户端只能靠这个值区分「完整收尾」与「被截断」，
    否则失败会继续被渲染成「已完成」。
    """
    agent = MagicMock()

    async def fake_process_direct(*, on_stream=None, on_stream_end=None, **kwargs):
        await on_stream_end(resuming=False)
        return SimpleNamespace(
            content="partial answer",
            metadata={"_stop_reason": "max_iterations"},
        )

    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert resp.status == 200
    body = await resp.text()
    data_lines = [
        line[len("data: "):] for line in body.split("\n") if line.startswith("data: ")
    ]
    chunks = [json.loads(line) for line in data_lines[:-1]]

    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert data_lines[-1] == "[DONE]"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_forwards_expert_group_member_events(aiohttp_client) -> None:
    """FR-2.4 第 3 步：专家团成员生命周期必须真的出现在 SSE 里。

    成员归属字段只存在一条通道（`on_progress` → SSE），所以「引擎有没有把成员事件发出来」
    只能在这一层断言。漏了它，客户端拿不到成员名，只能把整团显示成一轮黑箱——
    界面就会退回「只保存配置、不改变执行结果」那种不可验证的状态。
    """
    agent = MagicMock()

    async def fake_process_direct(
        *, on_progress=None, on_stream=None, on_stream_end=None, **kwargs
    ):
        assert on_progress is not None
        # coordinator 的「先出计划」：载荷里带一个 `plan` 键。
        # 它**不能**被走成既有的 kind=plan 分支（那条分支对 str 做 `**plan` 会抛）。
        await on_progress(
            "",
            member={"id": "m1", "name": "架构师"},
            member_event="group_planned",
            member_extra={"plan": "先看现状再动手"},
        )
        await on_progress(
            "",
            member={"id": "m1", "name": "架构师"},
            member_event="member_finished",
            member_extra={"output": "现状如下……"},
        )
        await on_progress(
            "",
            member={"id": "m2", "name": "审校"},
            member_event="member_failed",
            member_extra={"error": "provider 429"},
        )
        if on_stream:
            await on_stream("汇总")
        if on_stream_end:
            await on_stream_end()
        return "汇总"

    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    body = await resp.text()
    events = [
        json.loads(line[len("data: "):])["choices"][0]["delta"].get("nanobot_event")
        for line in body.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    events = [e for e in events if e is not None]

    kind_plan = [e for e in events if e.get("kind") == "plan"]
    assert kind_plan == [], "成员的 plan 载荷被误当成 Plan 模式事件"

    members = [e for e in events if e.get("kind") == "member"]
    assert [m["member_event"] for m in members] == [
        "group_planned",
        "member_finished",
        "member_failed",
    ]
    assert members[0]["member"] == {"id": "m1", "name": "架构师"}
    assert members[0]["plan"] == "先看现状再动手"
    assert members[1]["output"] == "现状如下……"
    assert members[2]["member"] == {"id": "m2", "name": "审校"}
    assert members[2]["error"] == "provider 429"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_with_session_id(aiohttp_client) -> None:
    """Streaming should respect session_id for session key routing."""
    captured_key: str = ""

    async def fake_process_direct(*, session_key="", on_stream=None, on_stream_end=None, **kwargs):
        nonlocal captured_key
        captured_key = session_key
        if on_stream:
            await on_stream("ok")
        if on_stream_end:
            await on_stream_end()
        return "ok"

    agent = MagicMock()
    agent.process_direct = fake_process_direct
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "session_id": "my-session",
        },
    )
    assert resp.status == 200
    assert captured_key == "api:my-session"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_streaming_backend_failure_does_not_emit_success_terminator(aiohttp_client) -> None:
    """Backend exceptions should not surface as a normal stop+[DONE] stream.

    但必须留下一个可判读的终结帧：客户端不能只看到「流没了」，否则只能按正常收尾处理
    （2026-09-29 断流事故的客户端侧成因）。
    """
    agent = MagicMock()

    async def boom(**kwargs):
        raise RuntimeError("backend blew up")

    agent.process_direct = boom
    agent.aclose = AsyncMock()

    app = create_app(agent, model_name="m", api_key=API_KEY)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert resp.status == 200
    body = await resp.text()
    assert '"finish_reason": "stop"' not in body
    assert '"finish_reason": "error"' in body
    assert "[DONE]" not in body
