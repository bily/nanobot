"""Tests for the per-agent tool whitelist (M2 / FR-2.2).

[LOCAL PATCH] nanowork。覆盖四层，与 ``test_session_mode.py`` 同构：

1. 纯函数真值表（``agent_blocks_tool``）——权限逻辑最怕「一处写对、一处写漏」
2. 作用域承载（``tool_allow`` / ``tool_deny`` 的解析、校验与线上形态）
3. 授权钩子（越权调用被拦下且**从未触达工具**）
4. 基线装配（ephemeral turn 同样绕不过）与优先级（黑名单压白名单）
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.hook import AgentHookContext, CompositeHook
from nanobot.agent.hooks.agent_tools import AgentToolPolicyHook
from nanobot.agent.hooks.tool_approval import ToolApprovalHook
from nanobot.agent.tools.execution import _execute_tool_call
from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    WorkspaceScopeError,
    agent_blocks_tool,
    bind_workspace_scope,
    build_workspace_scope,
    normalize_tool_names,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


def _call(name: str, arguments: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(id="call-1", name=name, arguments=arguments or {})


def _scope(tmp_path, *, allow=None, deny=None, agent_id="agent-1"):
    return build_workspace_scope(
        tmp_path,
        "full",
        tool_allow=allow,
        tool_deny=deny,
        agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# 1. 真值表
# ---------------------------------------------------------------------------


def test_no_declaration_allows_everything(tmp_path) -> None:
    """未声明策略（``None``）必须与既有会话完全一致——否则就是静默破坏兼容。"""
    scope = _scope(tmp_path)
    for name in ("read_file", "write_file", "exec", "delete_file", "mcp__x__y"):
        assert agent_blocks_tool(scope, name) is None


def test_whitelist_denies_everything_outside_it(tmp_path) -> None:
    scope = _scope(tmp_path, allow=["read_file", "grep"])
    assert agent_blocks_tool(scope, "read_file") is None
    assert agent_blocks_tool(scope, "grep") is None
    for name in ("write_file", "exec", "delete_file"):
        assert agent_blocks_tool(scope, name) is not None


def test_empty_whitelist_denies_everything(tmp_path) -> None:
    """空白名单 ≠ 无白名单。把「配成空」读成「不限制」会让一次手误变成完全放开。"""
    scope = _scope(tmp_path, allow=[])
    assert scope.tool_allow == frozenset()
    assert agent_blocks_tool(scope, "read_file") is not None
    assert agent_blocks_tool(scope, "write_file") is not None


def test_deny_wins_over_allow(tmp_path) -> None:
    """两边都列上时以拒绝为准——运维要能在宽白名单上精确挖掉危险工具。"""
    scope = _scope(tmp_path, allow=["read_file", "exec"], deny=["exec"])
    assert agent_blocks_tool(scope, "read_file") is None
    assert agent_blocks_tool(scope, "exec") is not None


def test_deny_alone_keeps_the_rest_available(tmp_path) -> None:
    """只声明黑名单 = 宽授权 + 精确挖洞，不能退化成最小权限。"""
    scope = _scope(tmp_path, deny=["exec"])
    assert agent_blocks_tool(scope, "exec") is not None
    assert agent_blocks_tool(scope, "write_file") is None


def test_scope_none_is_never_a_denial(tmp_path) -> None:
    """没有作用域（引擎内直调、后台任务）时不介入，交由其它防线处理。"""
    assert agent_blocks_tool(None, "write_file") is None


def test_nameless_call_is_denied(tmp_path) -> None:
    scope = _scope(tmp_path, allow=["read_file"])
    assert agent_blocks_tool(scope, "") is not None
    assert agent_blocks_tool(scope, "   ") is not None


def test_denial_reason_names_the_agent_and_tool(tmp_path) -> None:
    """拒绝原因要能让人一眼看出「哪个 Agent 缺哪个能力」。"""
    scope = _scope(tmp_path, allow=["read_file"], agent_id="researcher")
    reason = agent_blocks_tool(scope, "write_file")
    assert reason is not None
    assert "researcher" in reason
    assert "write_file" in reason


def test_predicate_is_pure(tmp_path) -> None:
    scope = _scope(tmp_path, allow=["read_file"])
    assert agent_blocks_tool(scope, "exec") == agent_blocks_tool(scope, "exec")


# ---------------------------------------------------------------------------
# 2. 工具名归一
# ---------------------------------------------------------------------------


def test_normalize_returns_none_for_missing_declaration() -> None:
    assert normalize_tool_names(None, field="tool_allow") is None


def test_normalize_distinguishes_empty_from_missing() -> None:
    assert normalize_tool_names([], field="tool_allow") == frozenset()
    assert normalize_tool_names(None, field="tool_allow") is None


def test_normalize_accepts_comma_separated_string() -> None:
    assert normalize_tool_names(" read_file , grep ", field="tool_allow") == frozenset(
        {"read_file", "grep"}
    )


def test_normalize_drops_blank_entries() -> None:
    assert normalize_tool_names(["", "  ", "grep"], field="tool_allow") == frozenset(
        {"grep"}
    )


def test_normalize_deduplicates() -> None:
    assert normalize_tool_names(
        ["grep", "grep", "grep"], field="tool_allow"
    ) == frozenset({"grep"})


@pytest.mark.parametrize("bad", [123, {"a": 1}, [1, 2], object()])
def test_normalize_rejects_bad_shapes(bad) -> None:
    """白名单是权限边界——坏输入必须报错，不能静默放宽。"""
    with pytest.raises(WorkspaceScopeError):
        normalize_tool_names(bad, field="tool_allow")


# ---------------------------------------------------------------------------
# 3. 作用域承载
# ---------------------------------------------------------------------------


def test_scope_carries_the_policy(tmp_path) -> None:
    scope = _scope(tmp_path, allow=["read_file"], deny=["exec"], agent_id="a1")
    assert scope.tool_allow == frozenset({"read_file"})
    assert scope.tool_deny == frozenset({"exec"})
    assert scope.agent_id == "a1"


def test_scope_defaults_are_unrestrictive(tmp_path) -> None:
    """未声明任何策略时的缺省值：不限制、不禁用、不指名 Agent。"""
    scope = build_workspace_scope(tmp_path, "full")
    assert scope.tool_allow is None
    assert scope.tool_deny == frozenset()
    assert scope.agent_id is None


def test_payload_omits_empty_policy(tmp_path) -> None:
    """线上形态只在真有约束时出现，避免给每个会话塞两个空字段。"""
    payload = _scope(tmp_path).payload()
    assert "tool_allow" not in payload
    assert "tool_deny" not in payload


def test_payload_includes_sorted_policy(tmp_path) -> None:
    payload = _scope(tmp_path, allow=["grep", "read_file"], deny=["exec"]).payload()
    assert payload["tool_allow"] == ["grep", "read_file"]
    assert payload["tool_deny"] == ["exec"]


def test_payload_keeps_an_empty_whitelist_visible(tmp_path) -> None:
    """空白名单必须出现在线上形态里，否则客户端无法把它和「没声明」区分开。"""
    payload = _scope(tmp_path, allow=[]).payload()
    assert payload["tool_allow"] == []


def test_request_payload_is_parsed(tmp_path) -> None:
    scope = validate_workspace_scope_payload(
        {
            "project_path": str(tmp_path),
            "access_mode": "full",
            "tool_allow": ["read_file"],
            "tool_deny": ["exec"],
            "agent_id": "agent-1",
        },
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.tool_allow == frozenset({"read_file"})
    assert scope.tool_deny == frozenset({"exec"})
    assert scope.agent_id == "agent-1"


def test_request_without_policy_stays_unrestricted(tmp_path) -> None:
    """旧客户端不发这两个字段——行为必须与改动前一致。"""
    scope = validate_workspace_scope_payload(
        {"project_path": str(tmp_path), "access_mode": "full"},
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
    )
    assert scope.tool_allow is None
    assert scope.tool_deny == frozenset()


def test_request_with_malformed_policy_is_rejected(tmp_path) -> None:
    """结构不对（不是名字列表）必须报错。注意逗号分隔的**字符串**是合法形态，
    所以这里刻意用真正的坏结构。"""
    with pytest.raises(WorkspaceScopeError):
        validate_workspace_scope_payload(
            {
                "project_path": str(tmp_path),
                "access_mode": "full",
                "tool_allow": {"read_file": True},
            },
            default_workspace=tmp_path,
            default_restrict_to_workspace=False,
        )


def test_request_rejects_non_string_entries(tmp_path) -> None:
    with pytest.raises(WorkspaceScopeError):
        validate_workspace_scope_payload(
            {
                "project_path": str(tmp_path),
                "access_mode": "full",
                "tool_allow": ["read_file", 7],
            },
            default_workspace=tmp_path,
            default_restrict_to_workspace=False,
        )


def test_request_with_non_string_agent_id_is_rejected(tmp_path) -> None:
    with pytest.raises(WorkspaceScopeError):
        validate_workspace_scope_payload(
            {
                "project_path": str(tmp_path),
                "access_mode": "full",
                "agent_id": 7,
            },
            default_workspace=tmp_path,
            default_restrict_to_workspace=False,
        )


# ---------------------------------------------------------------------------
# 4. 授权钩子
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_denies_a_tool_outside_the_whitelist(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, allow=["read_file"]))
    try:
        decision = await AgentToolPolicyHook().before_execute_tool(
            _ctx(), _call("write_file", {"path": "a.txt"}), None, {}
        )
        assert decision is not None and decision.allowed is False
        assert "write_file" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_stays_silent_for_a_granted_tool(tmp_path) -> None:
    token = bind_workspace_scope(_scope(tmp_path, allow=["read_file"]))
    try:
        decision = await AgentToolPolicyHook().before_execute_tool(
            _ctx(), _call("read_file", {"path": "a.txt"}), None, {}
        )
        assert decision is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_is_inert_without_a_scope(tmp_path) -> None:
    """没有作用域（引擎内直调、后台任务）时不介入——交由其它防线处理。"""
    token = bind_workspace_scope(_scope(tmp_path, allow=["read_file"]))
    reset_workspace_scope(token)  # 明确回到「未绑定」
    decision = await AgentToolPolicyHook().before_execute_tool(
        _ctx(), _call("write_file"), None, {}
    )
    assert decision is None


class _RecordingTool:
    def __init__(self) -> None:
        self.called = False

    async def execute(self, **kwargs: object) -> str:
        self.called = True
        return "executed"


class _Registry:
    def __init__(self, tool: _RecordingTool) -> None:
        self._tool = tool

    def prepare_call(self, name: str, arguments: object):
        return self._tool, dict(arguments or {}), None

    async def execute(self, name: str, params: dict) -> str:
        return await self._tool.execute(**params)


@pytest.mark.asyncio
async def test_denied_call_never_reaches_the_tool(tmp_path) -> None:
    """授权门的价值全在「工具没被调用」——只看返回值会把旁路放过去。"""
    token = bind_workspace_scope(_scope(tmp_path, allow=["read_file"]))
    try:
        tool = _RecordingTool()
        result, event = await _execute_tool_call(
            _Registry(tool),
            _call("write_file", {"path": "a.txt"}),
            {},
            {},
            AgentToolPolicyHook(),
            _ctx(),
        )

        assert tool.called is False
        assert event["status"] == "error"
        assert "write_file" in str(result)
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_granted_call_reaches_the_tool(tmp_path) -> None:
    """反向钉住：白名单命中时必须真的执行，别把门做成了砖墙。"""
    token = bind_workspace_scope(_scope(tmp_path, allow=["write_file"]))
    try:
        tool = _RecordingTool()
        _result, event = await _execute_tool_call(
            _Registry(tool),
            _call("write_file", {"path": "a.txt"}),
            {},
            {},
            AgentToolPolicyHook(),
            _ctx(),
        )

        assert tool.called is True
        assert event["status"] == "ok"
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_gate_is_part_of_the_baseline_chain(tmp_path) -> None:
    """ephemeral turn（定时任务等）也不能绕过授权门。"""
    token = bind_workspace_scope(_scope(tmp_path, allow=["read_file"]))
    try:
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(channel="api", chat_id="direct", ephemeral=True)
        )
        decision = await hook.before_execute_tool(_ctx(), _call("exec"), None, {})
        assert decision is not None and decision.allowed is False
        assert "exec" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_unauthorized_call_does_not_also_prompt_for_approval(tmp_path) -> None:
    """授权门拒绝的调用不得再弹审批框——否则用户看到一个「批了也执行不了」的框。

    ``CompositeHook`` 会调用**每一个**钩子再合并裁决，所以审批门必须自己识别
    Agent 白名单，而不能指望授权门先把它挡下来。
    """
    scope = build_workspace_scope(
        tmp_path,
        "full",
        tool_approval="ask",
        tool_allow=["read_file"],
    )
    token = bind_workspace_scope(scope)
    try:
        prompts: list[object] = []

        async def on_progress(
            content: str, *, approval_request=None, **_: object
        ) -> None:
            prompts.append(approval_request)

        approval = ToolApprovalHook(on_progress=on_progress)
        # 审批门单独看也必须不介入——它是「自己判断」，不是被授权门挡住。
        assert await approval.before_execute_tool(_ctx(), _call("write_file"), None, {}) is None

        combined = CompositeHook([approval, AgentToolPolicyHook()])
        decision = await combined.before_execute_tool(
            _ctx(), _call("write_file", {"path": "a.txt"}), None, {}
        )

        assert prompts == []  # 没有弹审批框
        assert decision is not None and decision.allowed is False
        assert "write_file" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_approval_gate_still_fires_for_a_granted_tool(tmp_path) -> None:
    """授权放行不等于跳过审批：白名单内的写工具照旧要用户点头。

    这里必须**真的**把审批裁决回灌，否则审批门会一直等到 300s 超时才返回
    （那既是 5 分钟的测试，也是在断言错误的东西）。
    """
    from nanobot.security.tool_approval import ApprovalDecision, PendingApprovals

    scope = build_workspace_scope(
        tmp_path,
        "full",
        tool_approval="ask",
        tool_allow=["write_file"],
    )
    token = bind_workspace_scope(scope)
    try:
        prompts: list[dict] = []
        registry = PendingApprovals()

        async def on_progress(content: str, *, approval_request=None, **_: object) -> None:
            prompts.append(approval_request)

        async def resolver() -> None:
            # 等到审批请求真的发出来，再替用户点「允许」。
            for _ in range(200):
                if prompts:
                    registry.resolve(prompts[0]["id"], ApprovalDecision(verdict="allow"))
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("审批门没有发出请求——白名单内的写工具被静默放行了")

        approval = ToolApprovalHook(on_progress=on_progress, registry=registry)
        resolver_task = asyncio.create_task(resolver())
        combined = CompositeHook([approval, AgentToolPolicyHook()])
        decision = await combined.before_execute_tool(
            _ctx(), _call("write_file", {"path": "a.txt"}), None, {}
        )
        await resolver_task

        # 审批门确实被触发（而不是被授权门或「白名单」这个事实跳过）。
        assert [p["tool"] for p in prompts] == ["write_file"]
        assert decision is None  # 获准 → 两道门都不干预
    finally:
        reset_workspace_scope(token)
