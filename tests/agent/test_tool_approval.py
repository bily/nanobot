"""Tests for the per-call tool approval gate (D3 / FR-1.5).

[LOCAL PATCH] sciherd-cloud-smartagent。覆盖四层：
1. 策略表判定（只读白名单 + 豁免/强制覆盖）
2. 待决注册表（登记 / 投递 / 丢弃 / 会话清理）
3. 组合钩子的裁决合并（含 fail-closed：策略钩子抛错必须变成拒绝）
4. 执行点消费（拒绝时返回软错误载荷且工具绝不执行）
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    CompositeHook,
    ToolExecutionDecision,
)
from nanobot.agent.hooks.tool_approval import ToolApprovalHook, _summarize_args
from nanobot.agent.tools.execution import _execute_tool_call
from nanobot.providers.base import ToolCallRequest
from nanobot.security.tool_approval import (
    ApprovalDecision,
    ApprovalRequest,
    PendingApprovals,
    ToolApprovalPolicy,
    new_request_id,
)
from nanobot.security.workspace_access import (
    WorkspaceScopeError,
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


def _call(name: str = "write_file", arguments: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(id="call-1", name=name, arguments=arguments or {})


class _StubTool:
    def __init__(self) -> None:
        self.called = False

    async def execute(self, **kwargs: object) -> str:
        self.called = True
        return "executed"


class _StubRegistry:
    """Minimal ToolRegistry stand-in exposing ``prepare_call``."""

    def __init__(self, tool: _StubTool) -> None:
        self._tool = tool

    def prepare_call(self, name: str, arguments: object) -> tuple[object, dict, None]:
        return self._tool, dict(arguments or {}), None

    async def execute(self, name: str, params: dict) -> str:
        return await self._tool.execute(**params)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_read_only_tools_never_require_approval() -> None:
    policy = ToolApprovalPolicy()
    for name in ("read_file", "list_dir", "grep", "web_fetch", "list_sessions"):
        assert policy.requires_approval(name) is False


def test_side_effecting_and_unknown_tools_require_approval() -> None:
    policy = ToolApprovalPolicy()
    for name in ("write_file", "edit_file", "exec", "apply_patch", "generate_image"):
        assert policy.requires_approval(name) is True
    # 未知工具（含 MCP 与未来新增）默认需审批：宁多问一句，不静默放行。
    assert policy.requires_approval("mcp__ranch__delete_all") is True
    assert policy.requires_approval("brand_new_tool") is True


def test_policy_exempt_and_require_overrides() -> None:
    policy = ToolApprovalPolicy(
        exempt=frozenset({"exec"}),
        require=frozenset({"read_file"}),
    )
    # 豁免优先于默认判定
    assert policy.requires_approval("exec") is False
    # 强制优先于只读白名单
    assert policy.requires_approval("read_file") is True


def test_policy_treats_nameless_call_as_needing_approval() -> None:
    assert ToolApprovalPolicy().requires_approval("") is True


def test_summarize_args_truncates_and_ignores_non_dict() -> None:
    assert _summarize_args(None) == {}
    assert _summarize_args("not-a-dict") == {}

    summarized = _summarize_args({"path": "a.txt", "content": "x" * 900})
    assert summarized["path"] == "a.txt"
    assert len(summarized["content"]) < 900
    assert summarized["content"].endswith("…")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_round_trip_and_unknown_id() -> None:
    registry = PendingApprovals()
    request = ApprovalRequest(request_id=new_request_id(), tool="exec", args={})
    future = registry.register(request)

    assert registry.pending_count == 1
    assert registry.resolve(request.request_id, ApprovalDecision(verdict="deny")) is True

    decision = await future
    assert decision.allowed is False
    assert registry.pending_count == 0
    # 重复投递 / 未知 id 都必须是幂等的失败返回，而不是抛错
    assert registry.resolve(request.request_id, ApprovalDecision(verdict="allow")) is False
    assert registry.resolve("nope", ApprovalDecision(verdict="allow")) is False


@pytest.mark.asyncio
async def test_registry_discard_and_session_cancel() -> None:
    registry = PendingApprovals()
    a = ApprovalRequest(request_id=new_request_id(), tool="exec", args={}, session_key="s1")
    b = ApprovalRequest(request_id=new_request_id(), tool="exec", args={}, session_key="s1")
    c = ApprovalRequest(request_id=new_request_id(), tool="exec", args={}, session_key="s2")
    for request in (a, b, c):
        registry.register(request)

    assert registry.cancel_session("s1") == 2
    assert registry.pending_count == 1
    assert registry.get(c.request_id) is not None

    registry.discard(c.request_id)
    assert registry.pending_count == 0
    # 丢弃后再投递必须失败（超时/取消路径已摘除登记）
    assert registry.resolve(c.request_id, ApprovalDecision(verdict="allow")) is False


# ---------------------------------------------------------------------------
# Composite verdict merging
# ---------------------------------------------------------------------------


class _NoOpinion(AgentHook):
    async def before_execute_tool(self, context, tool_call, tool, params):
        return None


class _Denier(AgentHook):
    def __init__(self, reason: str = "denied") -> None:
        super().__init__()
        self._reason = reason

    async def before_execute_tool(self, context, tool_call, tool, params):
        return ToolExecutionDecision.deny(self._reason)


class _BoomHook(AgentHook):
    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self._seen = seen

    async def before_execute_tool(self, context, tool_call, tool, params):
        raise RuntimeError("policy exploded")


class _RecordingAllow(AgentHook):
    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self._seen = seen

    async def before_execute_tool(self, context, tool_call, tool, params):
        self._seen.append("allow")
        return ToolExecutionDecision.allow()


@pytest.mark.asyncio
async def test_composite_returns_none_when_every_hook_has_no_opinion() -> None:
    hook = CompositeHook([_NoOpinion(), _NoOpinion()])
    assert await hook.before_execute_tool(_ctx(), _call(), None, {}) is None


@pytest.mark.asyncio
async def test_composite_propagates_a_denial() -> None:
    hook = CompositeHook([_NoOpinion(), _Denier("nope")])
    decision = await hook.before_execute_tool(_ctx(), _call(), None, {})
    assert decision is not None and decision.allowed is False
    assert decision.reason == "nope"


@pytest.mark.asyncio
async def test_composite_fails_closed_when_a_hook_raises() -> None:
    """策略钩子自己出错时必须变成拒绝——这是与 _for_each_hook_safe 的分水岭。"""
    seen: list[str] = []
    hook = CompositeHook([_BoomHook(seen), _RecordingAllow(seen)])
    decision = await hook.before_execute_tool(_ctx(), _call(), None, {})

    assert decision is not None and decision.allowed is False
    # 出错也不短路：同层的观察型钩子仍要收到事件（composite 的既有契约）
    assert seen == ["allow"]


@pytest.mark.asyncio
async def test_composite_allow_cannot_override_denial() -> None:
    seen: list[str] = []
    hook = CompositeHook([_RecordingAllow(seen), _Denier("still no")])
    decision = await hook.before_execute_tool(_ctx(), _call(), None, {})
    assert decision is not None and decision.allowed is False


# ---------------------------------------------------------------------------
# Execution consumption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denied_call_returns_soft_error_and_never_executes() -> None:
    tool = _StubTool()
    result, event = await _execute_tool_call(
        _StubRegistry(tool),
        _call("write_file", {"path": "a.txt"}),
        {},
        {},
        CompositeHook([_Denier("the user said no")]),
        _ctx(),
    )

    assert tool.called is False
    assert event["status"] == "error"
    assert event["name"] == "write_file"
    assert "the user said no" in str(result)
    # 明确要求换路子，避免模型原样重试导致反复弹窗
    assert "Do not retry it unchanged" in str(result)


@pytest.mark.asyncio
async def test_allowed_call_still_executes() -> None:
    tool = _StubTool()
    result, event = await _execute_tool_call(
        _StubRegistry(tool),
        _call("write_file", {"path": "a.txt"}),
        {},
        {},
        CompositeHook([_NoOpinion()]),
        _ctx(),
    )

    assert tool.called is True
    assert event["status"] == "ok"
    assert result == "executed"


# ---------------------------------------------------------------------------
# The approval hook itself
# ---------------------------------------------------------------------------


def _scope(tmp_path, tool_approval: str):
    return build_workspace_scope(tmp_path, "full", tool_approval=tool_approval)


@pytest.mark.asyncio
async def test_hook_is_inert_when_scope_says_auto(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "auto"))
    try:
        hook = ToolApprovalHook(on_progress=None)
        assert await hook.before_execute_tool(_ctx(), _call(), None, {}) is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_skips_read_only_tools_even_in_ask_mode(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "ask"))
    try:
        hook = ToolApprovalHook(on_progress=None)
        decision = await hook.before_execute_tool(_ctx(), _call("read_file"), None, {})
        assert decision is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_denies_when_no_approval_channel_exists(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "ask"))
    try:
        async def on_progress(content: str, *, tool_hint: bool = False) -> None:
            return None

        hook = ToolApprovalHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(_ctx(), _call(), None, {"path": "a.txt"})
        assert decision is not None and decision.allowed is False
        assert "no channel" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_emits_request_and_honours_approval(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "ask"))
    try:
        registry = PendingApprovals()
        emitted: list[dict] = []

        async def on_progress(content: str, *, approval_request=None, **_: object) -> None:
            emitted.append(approval_request)

        async def resolver() -> None:
            while not emitted:
                await asyncio.sleep(0)
            registry.resolve(emitted[0]["id"], ApprovalDecision(verdict="allow"))

        resolver_task = asyncio.create_task(resolver())
        hook = ToolApprovalHook(on_progress=on_progress, registry=registry)
        decision = await hook.before_execute_tool(
            _ctx(), _call("write_file", {"path": "a.txt"}), None, {"path": "a.txt"}
        )
        await resolver_task

        assert decision is None  # 获准 → 不干预
        assert emitted[0]["tool"] == "write_file"
        assert emitted[0]["args"] == {"path": "a.txt"}
        assert registry.pending_count == 0  # 登记已摘除
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_denies_on_timeout(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "ask"))
    try:
        async def on_progress(content: str, *, approval_request=None, **_: object) -> None:
            return None

        hook = ToolApprovalHook(
            on_progress=on_progress,
            policy=ToolApprovalPolicy(timeout_seconds=0.01),
            registry=PendingApprovals(),
        )
        decision = await hook.before_execute_tool(_ctx(), _call(), None, {})
        assert decision is not None and decision.allowed is False
        assert "did not respond" in decision.reason
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# Scope plumbing
# ---------------------------------------------------------------------------


def test_tool_approval_defaults_to_auto_and_validates(tmp_path) -> None:
    assert build_workspace_scope(tmp_path, "full").tool_approval == "auto"
    scope = build_workspace_scope(tmp_path, "full", tool_approval="ask")
    assert scope.tool_approval == "ask"
    # 随 scope 一起序列化，供前端回显与持久化
    assert scope.metadata()["tool_approval"] == "ask"
    assert scope.payload()["tool_approval"] == "ask"

    with pytest.raises(WorkspaceScopeError):
        build_workspace_scope(tmp_path, "full", tool_approval="sometimes")


def test_validate_payload_carries_tool_approval(tmp_path) -> None:
    scope = validate_workspace_scope_payload(
        {"project_path": str(tmp_path), "access_mode": "full", "tool_approval": "ask"},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.tool_approval == "ask"

    # 缺省不得悄悄开启拦截
    scope = validate_workspace_scope_payload(
        {"project_path": str(tmp_path)},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.tool_approval == "auto"
