"""Tests for the session execution mode gate (M2 / FR-1.4 / FR-1.5).

[LOCAL PATCH] sciherd-cloud-smartagent。覆盖四层：
1. 纯函数真值表（``mode_blocks_tool``）——权限逻辑最怕「一处写对、一处写漏」
2. 作用域承载（``session_mode`` 的解析、校验与序列化）
3. 模式钩子（ask/plan 拒绝写类工具；plan 放行并转交 submit_plan）
4. 与审批门的交叉语义（被模式拒绝的调用不得再弹审批框）
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.hook import (
    AgentHookContext,
    CompositeHook,
)
from nanobot.agent.hooks.session_mode import (
    SessionModeHook,
    normalize_steps,
)
from nanobot.agent.hooks.tool_approval import ToolApprovalHook
from nanobot.agent.tools.execution import _execute_tool_call
from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanobot.providers.base import ToolCallRequest
from nanobot.security.tool_approval import (
    PLAN_SUBMISSION_TOOLS,
    ApprovalDecision,
    PendingApprovals,
    mode_blocks_tool,
)
from nanobot.security.workspace_access import (
    WorkspaceScopeError,
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)
from nanobot.utils.progress_events import (
    invoke_plan_event,
    on_progress_accepts_plan_events,
)


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


def _call(name: str = "write_file", arguments: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(id="call-1", name=name, arguments=arguments or {})


def _scope(tmp_path, session_mode: str):
    return build_workspace_scope(tmp_path, "full", session_mode=session_mode)


def _plan_call(steps: object = None) -> ToolCallRequest:
    return _call("submit_plan", {"steps": steps if steps is not None else ["do a thing"]})


# ---------------------------------------------------------------------------
# 1. Truth table
# ---------------------------------------------------------------------------


def test_craft_never_blocks() -> None:
    for name in ("write_file", "exec", "apply_patch", "submit_plan", "mcp__x__y"):
        assert mode_blocks_tool("craft", name) is False


def test_unknown_mode_is_left_to_the_validator() -> None:
    """未知模式在原子上放行是安全的：入口 ``_normalize_session_mode`` 已经拒过它。"""
    assert mode_blocks_tool("banana", "write_file") is False
    assert mode_blocks_tool("", "write_file") is False


def test_ask_and_plan_block_every_write_tool() -> None:
    for mode in ("ask", "plan"):
        for name in ("write_file", "edit_file", "exec", "apply_patch", "mcp__ranch__del"):
            assert mode_blocks_tool(mode, name) is True, (mode, name)


def test_ask_and_plan_allow_read_only_tools() -> None:
    for mode in ("ask", "plan"):
        for name in ("read_file", "grep", "list_dir", "web_fetch", "list_sessions"):
            assert mode_blocks_tool(mode, name) is False, (mode, name)


def test_submit_plan_is_plan_only() -> None:
    assert mode_blocks_tool("plan", "submit_plan") is False
    # Ask 下没有「提交计划」这一步：它同样是只读模式，但产出的是解释而不是清单
    assert mode_blocks_tool("ask", "submit_plan") is True


def test_mode_matching_is_case_and_space_insensitive() -> None:
    assert mode_blocks_tool("  PLAN ", "write_file") is True
    assert mode_blocks_tool("Ask", "write_file") is True


def test_plan_submission_set_matches_the_tool_name() -> None:
    assert "submit_plan" in PLAN_SUBMISSION_TOOLS


# ---------------------------------------------------------------------------
# 2. Scope plumbing
# ---------------------------------------------------------------------------


def test_session_mode_defaults_to_craft(tmp_path) -> None:
    scope = build_workspace_scope(tmp_path, "full")
    assert scope.session_mode == "craft"
    assert scope.metadata()["session_mode"] == "craft"
    assert scope.payload()["session_mode"] == "craft"


def test_session_mode_round_trips_through_scope(tmp_path) -> None:
    scope = build_workspace_scope(tmp_path, "full", session_mode="plan")
    assert scope.session_mode == "plan"
    assert scope.metadata()["session_mode"] == "plan"
    assert scope.payload()["session_mode"] == "plan"


def test_unknown_session_mode_fails_loudly(tmp_path) -> None:
    """与 tool_approval 的容错方向相反：退化会把「只读」悄悄变成「可写」。"""
    with pytest.raises(WorkspaceScopeError):
        build_workspace_scope(tmp_path, "full", session_mode="readonly")


def test_validate_payload_carries_session_mode(tmp_path) -> None:
    scope = validate_workspace_scope_payload(
        {"project_path": str(tmp_path), "access_mode": "full", "session_mode": "ask"},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.session_mode == "ask"

    # 缺席（历史会话 / 未启用该能力的客户端）按 craft
    scope = validate_workspace_scope_payload(
        {"project_path": str(tmp_path)},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.session_mode == "craft"


def test_non_string_session_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(WorkspaceScopeError):
        validate_workspace_scope_payload(
            {"project_path": str(tmp_path), "session_mode": 3},
            default_workspace=tmp_path,
            default_restrict_to_workspace=False,
        )


# ---------------------------------------------------------------------------
# 3. steps normalization
# ---------------------------------------------------------------------------


def test_normalize_steps_assigns_ids_and_status() -> None:
    assert normalize_steps(["a", "b"]) == [
        {"id": "s1", "text": "a", "status": "pending"},
        {"id": "s2", "text": "b", "status": "pending"},
    ]


def test_normalize_steps_accepts_aliases_and_keeps_explicit_id() -> None:
    steps = normalize_steps([{"title": "t"}, {"step": "s"}, {"id": "keep", "text": "d"}])
    assert [s["text"] for s in steps] == ["t", "s", "d"]
    assert steps[2]["id"] == "keep"


def test_normalize_steps_drops_junk_without_raising() -> None:
    """畸形计划不能把整轮对话拖垮——丢弃而不是抛错。"""
    assert normalize_steps(None) == []
    assert normalize_steps("not-a-list") == []
    assert normalize_steps([3, None, "", "  ", {}]) == []


def test_normalize_steps_truncates_at_the_cap() -> None:
    steps = normalize_steps([f"step {i}" for i in range(200)])
    assert len(steps) == 40
    assert steps[-1]["text"] == "step 39"


# ---------------------------------------------------------------------------
# 4. The plan channel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_plan_event_reports_missing_channel() -> None:
    async def without_channel(content: str, *, tool_hint: bool = False) -> None:
        return None

    async def with_channel(content: str, *, plan=None, **_: object) -> None:
        return None

    assert on_progress_accepts_plan_events(without_channel) is False
    assert await invoke_plan_event(without_channel, {"steps": []}) is False
    assert on_progress_accepts_plan_events(with_channel) is True
    assert await invoke_plan_event(with_channel, {"steps": []}) is True


@pytest.mark.asyncio
async def test_invoke_plan_event_ignores_empty_payload() -> None:
    seen: list[object] = []

    async def with_channel(content: str, *, plan=None, **_: object) -> None:
        seen.append(plan)

    assert await invoke_plan_event(with_channel, {}) is False
    assert seen == []


# ---------------------------------------------------------------------------
# 5. The mode hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_is_inert_without_a_scope() -> None:
    hook = SessionModeHook(on_progress=None)
    assert await hook.before_execute_tool(_ctx(), _call(), None, {}) is None


@pytest.mark.asyncio
async def test_hook_is_inert_in_craft_mode(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "craft"))
    try:
        hook = SessionModeHook(on_progress=None)
        assert await hook.before_execute_tool(_ctx(), _call(), None, {}) is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "label"),
    [("ask", "Ask"), ("plan", "Plan")],
)
async def test_hook_denies_writes_in_read_only_modes(tmp_path, mode, label) -> None:
    token = bind_workspace_scope(_scope(tmp_path, mode))
    try:
        hook = SessionModeHook(on_progress=None)
        decision = await hook.before_execute_tool(_ctx(), _call("write_file"), None, {})
        assert decision is not None and decision.allowed is False
        assert decision.reason.startswith(label)
        assert "read-only" in decision.reason
        # 拒绝理由要给出路，否则模型会原样重试
        assert "Craft mode" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_allows_read_only_tools_in_ask_mode(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "ask"))
    try:
        hook = SessionModeHook(on_progress=None)
        assert await hook.before_execute_tool(_ctx(), _call("read_file"), None, {}) is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_relays_a_plan_and_lets_the_call_through(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        emitted: list[dict] = []

        async def on_progress(content: str, *, plan=None, **_: object) -> None:
            emitted.append(plan)

        hook = SessionModeHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(
            _ctx(), _plan_call(["read the docs", "draft the change"]), None,
            {"steps": ["read the docs", "draft the change"]},
        )

        assert decision is None  # 计划已交付 → 放行工具（工具只回 ack）
        assert len(emitted) == 1
        payload = emitted[0]
        assert payload["plan_id"].startswith("plan-")
        assert payload["steps"] == [
            {"id": "s1", "text": "read the docs", "status": "pending"},
            {"id": "s2", "text": "draft the change", "status": "pending"},
        ]
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_falls_back_to_tool_call_arguments(tmp_path) -> None:
    """``params`` 缺失时从 tool_call.arguments 兜底，避免计划被当成空计划。"""
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        emitted: list[dict] = []

        async def on_progress(content: str, *, plan=None, **_: object) -> None:
            emitted.append(plan)

        hook = SessionModeHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(
            _ctx(), _plan_call(["from arguments"]), None, None
        )
        assert decision is None
        assert emitted[0]["steps"][0]["text"] == "from arguments"
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_refuses_an_empty_plan(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        async def on_progress(content: str, *, plan=None, **_: object) -> None:
            return None

        hook = SessionModeHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(
            _ctx(), _plan_call([]), None, {"steps": []}
        )
        assert decision is not None and decision.allowed is False
        assert "non-empty" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_refuses_the_plan_when_no_channel_exists(tmp_path) -> None:
    """通道缺失时不能假装已提交——模型会停在一份用户看不见的计划上。"""
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        async def on_progress(content: str, *, tool_hint: bool = False) -> None:
            return None

        hook = SessionModeHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(
            _ctx(), _plan_call(), None, {"steps": ["x"]}
        )
        assert decision is not None and decision.allowed is False
        assert "cannot display" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ask", "craft"])
async def test_hook_rejects_submit_plan_outside_plan_mode(tmp_path, mode) -> None:
    token = bind_workspace_scope(_scope(tmp_path, mode))
    try:
        emitted: list[dict] = []

        async def on_progress(content: str, *, plan=None, **_: object) -> None:
            emitted.append(plan)

        hook = SessionModeHook(on_progress=on_progress)
        decision = await hook.before_execute_tool(
            _ctx(), _plan_call(), None, {"steps": ["x"]}
        )
        assert decision is not None and decision.allowed is False
        assert "only available in Plan mode" in decision.reason
        assert emitted == []
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# 6. Cross-talk with the approval gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_blocked_call_does_not_also_prompt_for_approval(tmp_path) -> None:
    """Ask 模式 + 逐条批准同时打开时，写操作只能被拒一次。

    ``CompositeHook`` 会调用**每一个**钩子再合并裁决，所以审批门必须自己识别
    模式；否则用户会看到一个「批了也执行不了」的弹窗。
    """
    scope = build_workspace_scope(tmp_path, "full", tool_approval="ask", session_mode="ask")
    token = bind_workspace_scope(scope)
    try:
        prompts: list[object] = []

        async def on_progress(content: str, *, approval_request=None, **_: object) -> None:
            prompts.append(approval_request)

        approval = ToolApprovalHook(on_progress=on_progress)
        assert await approval.before_execute_tool(_ctx(), _call("write_file"), None, {}) is None

        mode_hook = SessionModeHook(on_progress=on_progress)
        combined = CompositeHook([approval, mode_hook])
        decision = await combined.before_execute_tool(_ctx(), _call("write_file"), None, {})

        assert prompts == []  # 没有弹审批框
        assert decision is not None and decision.allowed is False
        assert "read-only" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_approval_gate_still_fires_when_the_mode_allows_the_call(tmp_path) -> None:
    """模式放行不等于跳过审批：craft 下审批门照旧工作。"""
    scope = build_workspace_scope(tmp_path, "full", tool_approval="ask", session_mode="craft")
    token = bind_workspace_scope(scope)
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
        approval = ToolApprovalHook(on_progress=on_progress, registry=registry)
        decision = await approval.before_execute_tool(_ctx(), _call("write_file"), None, {})
        await resolver_task

        assert [p["tool"] for p in emitted] == ["write_file"]  # 审批门确实被触发
        assert decision is None  # 获准 → 不干预
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# 7. Execution + baseline wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denied_write_never_reaches_the_tool(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        class _Tool:
            def __init__(self) -> None:
                self.called = False

            async def execute(self, **kwargs: object) -> str:
                self.called = True
                return "executed"

        class _Registry:
            def __init__(self, tool: _Tool) -> None:
                self._tool = tool

            def prepare_call(self, name: str, arguments: object):
                return self._tool, dict(arguments or {}), None

            async def execute(self, name: str, params: dict) -> str:
                return await self._tool.execute(**params)

        tool = _Tool()
        result, event = await _execute_tool_call(
            _Registry(tool),
            _call("write_file", {"path": "a.txt"}),
            {},
            {},
            SessionModeHook(on_progress=None),
            _ctx(),
        )

        assert tool.called is False
        assert event["status"] == "error"
        assert "read-only" in str(result)
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_mode_gate_is_part_of_the_baseline_chain(tmp_path) -> None:
    """ephemeral turn 也不能绕过模式门（与审批门同级）。"""
    token = bind_workspace_scope(_scope(tmp_path, "plan"))
    try:
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(channel="api", chat_id="direct", ephemeral=True)
        )
        decision = await hook.before_execute_tool(_ctx(), _call("write_file"), None, {})
        assert decision is not None and decision.allowed is False
        assert "read-only" in decision.reason
    finally:
        reset_workspace_scope(token)
