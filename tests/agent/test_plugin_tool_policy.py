"""技能 / 插件声明的工具白名单（FR-3.12）。

**最要紧的一组是「声明只收紧」的三条**:声明不是授权、多份声明取交集、
以及声明不得让 `shell` 绕过审批。这三条各自对应一种「插件把自己提权」的走法。

另一组是落地时用户拍板的口径(2026-10-16):**落差必须说出来**。声明了、但基线
根本不给的能力,交集之后是不生效的(这是对的),但不能静默 —— 插件作者会对着
「我声明了却什么都没发生」发呆。`policy_gaps` 就是这个「说出来」的可测形式。

⚠️ 夹具范式与 `test_agent_tool_policy.py`(FR-2.2)同构:作用域只能从 contextvar
读,所以每个用例都必须 `bind` / `reset`。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.agent import plugins as agent_plugins
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHookContext, CompositeHook
from nanobot.agent.hooks.plugin_tools import (
    DeclaredToolPolicy,
    DeclaredToolPolicyHook,
    ToolDeclaration,
    combine_declarations,
    create_declared_tool_policy_hook,
    parse_declared_tools,
    policy_gaps,
)
from nanobot.agent.hooks.tool_approval import ToolApprovalHook
from nanobot.agent.plugins import AGENT_PLUGIN_SCHEMA, set_agent_plugin_enabled
from nanobot.agent.tools.execution import _execute_tool_call
from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


def _call(name: str, arguments: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(id="call-1", name=name, arguments=arguments or {})


def _scope(tmp_path: Path, **kwargs: object):
    return build_workspace_scope(tmp_path, "full", **kwargs)


def _policy(*pairs: tuple[str, set[str]]) -> DeclaredToolPolicy:
    return combine_declarations(
        [ToolDeclaration(source, frozenset(tools)) for source, tools in pairs]
    )


# ---------------------------------------------------------------------------
# 1. 声明解析
# ---------------------------------------------------------------------------


def test_single_string_declaration_is_read() -> None:
    """`tools: shell` 是最容易写出来的形态。

    只认列表的写法会把它静默读成空,而空 = **不约束** —— 安全方向上的静默失效:
    作者以为「我只允许 shell」,实际「什么都没限制」。
    """
    assert parse_declared_tools("shell") == ("shell",)


def test_list_declaration_is_deduplicated_and_ordered() -> None:
    assert parse_declared_tools(["b", "a", "b"]) == ("b", "a")


def test_invalid_entries_are_dropped_without_widening() -> None:
    """丢弃非法项只让白名单更严,方向是安全的;但必须留痕(见实现里的 warn)。"""
    assert parse_declared_tools(["read_file", 7, "", "  ", "bad name", "shell"]) == (
        "read_file",
        "shell",
    )


def test_namespaced_mcp_tool_names_survive() -> None:
    """MCP 工具名带命名空间。写死成 `\\w+` 会把它们全判非法而静默丢掉。"""
    assert parse_declared_tools(["plugin--server__tool", "mcp:miora"]) == (
        "plugin--server__tool",
        "mcp:miora",
    )


def test_unsupported_shape_yields_empty() -> None:
    assert parse_declared_tools(123) == ()
    assert parse_declared_tools({"a": 1}) == ()


# ---------------------------------------------------------------------------
# 2. 多份声明叠加 = 交集
# ---------------------------------------------------------------------------


def test_no_declaration_means_no_constraint() -> None:
    policy = combine_declarations([])
    assert policy.allow is None
    assert policy.sources == ()


def test_single_declaration_is_itself() -> None:
    assert _policy(("skill:a", {"read_file"})).allow == frozenset({"read_file"})


def test_two_declarations_intersect_not_union() -> None:
    """⭐ 交集。并集会让**任何一份**声明都能放宽另一份 —— 从后门破掉「只收紧」。"""
    policy = _policy(("skill:a", {"read_file", "shell"}), ("plugin:b", {"shell"}))
    assert policy.allow == frozenset({"shell"})
    assert policy.sources == ("skill:a", "plugin:b")


def test_empty_declaration_stays_empty_not_absent() -> None:
    """`tools: []` 是**显式**空集(全拒),不是「没声明」(不约束)。

    把两者混起来,一次 YAML 手滑就变成「什么都没限制」。
    """
    policy = _policy(("skill:a", set()))
    assert policy.allow == frozenset()
    assert policy.allow is not None


def test_blank_source_is_ignored() -> None:
    assert combine_declarations([ToolDeclaration("", frozenset({"shell"}))]).allow is None


# ---------------------------------------------------------------------------
# 3. 落差（用户 2026-10-16 拍板的「必须说出来」）
# ---------------------------------------------------------------------------


def test_no_gap_when_declaration_fits_the_baseline(tmp_path: Path) -> None:
    policy = _policy(("skill:a", {"read_file"}))
    assert policy_gaps(policy, _scope(tmp_path, tool_allow=["read_file", "shell"])) == ()


def test_gap_lists_what_the_baseline_refuses(tmp_path: Path) -> None:
    policy = _policy(("skill:a", {"read_file", "shell"}))
    gaps = policy_gaps(policy, _scope(tmp_path, tool_allow=["read_file"], agent_id="a1"))
    assert [gap.tool for gap in gaps] == ["shell"]
    assert gaps[0].source == "skill:a"
    # 原因复用 `agent_blocks_tool` 的原文,不自己编一套
    assert "a1" in gaps[0].reason


def test_gap_covers_the_denylist_too(tmp_path: Path) -> None:
    """黑名单挖掉的洞同样是「声明了但拿不到」,漏掉它会让用户以为已经生效。"""
    policy = _policy(("plugin:x", {"shell"}))
    gaps = policy_gaps(policy, _scope(tmp_path, tool_deny=["shell"]))
    assert [gap.tool for gap in gaps] == ["shell"]


def test_no_gap_when_baseline_is_unconstrained(tmp_path: Path) -> None:
    policy = _policy(("skill:a", {"shell"}))
    assert policy_gaps(policy, _scope(tmp_path)) == ()


def test_no_gap_without_any_declaration(tmp_path: Path) -> None:
    assert policy_gaps(combine_declarations([]), _scope(tmp_path, tool_allow=[])) == ()


# ---------------------------------------------------------------------------
# 4. 钩子（验收 ①：未列出的工具被拦）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_denies_a_tool_outside_the_declaration(tmp_path: Path) -> None:
    token = bind_workspace_scope(_scope(tmp_path))
    try:
        decision = await DeclaredToolPolicyHook(_policy(("skill:a", {"read_file"}))).before_execute_tool(
            _ctx(), _call("write_file", {"path": "a.txt"}), None, {}
        )
        assert decision is not None and decision.allowed is False
        assert "write_file" in decision.reason
        assert "skill:a" in decision.reason
        # 文案必须让模型把责任指回**声明**,而不是用户设置 —— 否则它会去劝用户改权限
        assert "declared" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_hook_stays_silent_for_a_declared_tool(tmp_path: Path) -> None:
    token = bind_workspace_scope(_scope(tmp_path))
    try:
        decision = await DeclaredToolPolicyHook(_policy(("skill:a", {"read_file"}))).before_execute_tool(
            _ctx(), _call("read_file", {"path": "a.txt"}), None, {}
        )
        assert decision is None
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_empty_declaration_refuses_everything(tmp_path: Path) -> None:
    token = bind_workspace_scope(_scope(tmp_path))
    try:
        hook = DeclaredToolPolicyHook(_policy(("plugin:x", set())))
        for name in ("read_file", "shell"):
            decision = await hook.before_execute_tool(_ctx(), _call(name), None, {})
            assert decision is not None and decision.allowed is False
    finally:
        reset_workspace_scope(token)


def test_unexpected_policy_shape_is_ignored_not_enforced() -> None:
    """认不出形状的值必须**不约束**，而不是挂上一道空白名单把工具全拒。

    实测来源：单测里 `ContextBuilder` 被 mock 掉时，`declared_tool_policy(...)`
    返回的是 `MagicMock` —— 它 `is not None`，于是门被挂上、白名单是空的、
    `__contains__` 恒为 False，结果**所有工具全被拒**，而拒绝原因里连一个来源名
    都没有。那不是「更安全」，那是整轮不可用且无法解释（`test_hook_composite.py`
    的 `test_agent_loop_no_hooks_backward_compat` 就是这么红的）。
    """

    class _LooksLikeAPolicy:
        declarations: tuple[object, ...] = ()
        allow = frozenset()  # 空白名单：最危险的那种「像但不像」

    assert create_declared_tool_policy_hook(None, _LooksLikeAPolicy()) is None  # type: ignore[arg-type]


def test_factory_returns_nothing_without_a_declaration(tmp_path: Path) -> None:
    """没有声明就不挂空钩子:钩子链与引入本需求之前逐字相同。"""
    context = None  # 工厂在无策略时不会用到它
    assert create_declared_tool_policy_hook(context, None) is None
    assert create_declared_tool_policy_hook(context, combine_declarations([])) is None


@pytest.mark.asyncio
async def test_denied_call_never_reaches_the_tool(tmp_path: Path) -> None:
    """这道门的价值全在「工具没被调用」—— 只看返回值会把旁路放过去。"""

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

    token = bind_workspace_scope(_scope(tmp_path))
    try:
        tool = _RecordingTool()
        result, event = await _execute_tool_call(
            _Registry(tool),
            _call("write_file", {"path": "a.txt"}),
            {},
            {},
            DeclaredToolPolicyHook(_policy(("skill:a", {"read_file"}))),
            _ctx(),
        )
        assert tool.called is False
        assert event["status"] == "error"
        assert "write_file" in str(result)
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# 5. 装配（验收 ②：声明不得使工具绕过审批）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declaration_cannot_bypass_the_approval_gate(tmp_path: Path) -> None:
    """⭐ 声明 `tools: [shell]` **不得**使 `shell` 绕过审批。

    这里必须**真的**把审批裁决回灌,否则审批门会一直等到 300s 超时才返回
    （那既是 5 分钟的测试,也是在断言错误的东西）。

    这条在合并器语义下是**免费**成立的（`_decide_for_each_hook` 丢掉所有 `allow`），
    它钉的是**合并**这一层：声明门的放行不会顶掉审批门的裁决。
    它**证不了装配对了** —— 装配那一层由
    `test_assembly_keeps_the_approval_gate_when_a_declaration_exists` 覆盖
    （这条曾因为「手工拼链」而漏掉整层覆盖，见那条用例的注释）。
    """
    from nanobot.security.tool_approval import ApprovalDecision, PendingApprovals

    scope = _scope(tmp_path, tool_approval="ask")
    token = bind_workspace_scope(scope)
    try:
        prompts: list[dict] = []
        registry = PendingApprovals()

        async def on_progress(content: str, *, approval_request=None, **_: object) -> None:
            prompts.append(approval_request)

        async def resolver() -> None:
            for _ in range(200):
                if prompts:
                    registry.resolve(prompts[0]["id"], ApprovalDecision(verdict="allow"))
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("审批门没有发出请求——声明把审批绕过去了")

        approval = ToolApprovalHook(on_progress=on_progress, registry=registry)
        declared = DeclaredToolPolicyHook(_policy(("plugin:demo", {"shell"})))
        resolver_task = asyncio.create_task(resolver())
        # 链序刻意让声明门**在审批门之后**（与 `build_agent_turn_hook` 一致）
        combined = CompositeHook([approval, declared])
        decision = await combined.before_execute_tool(
            _ctx(), _call("shell", {"command": "echo hi"}), None, {}
        )
        await resolver_task

        assert [p["tool"] for p in prompts] == ["shell"]
        assert decision is None  # 用户点头后才放行
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_assembly_keeps_the_approval_gate_when_a_declaration_exists(tmp_path: Path) -> None:
    """⭐ 走**真实装配**（`build_agent_turn_hook`）：有声明时审批门必须仍在链上。

    这条是被变异逼出来的：上面那条用例自己手工拼
    `CompositeHook([approval, declared])`，于是**装配**这一层完全没被覆盖 ——
    把「有声明就跳过审批门」写进 `build_agent_turn_hook`，它照样全绿。
    手工拼链只能证明「合并器不会让声明门顶掉审批门的裁决」，**证不了装配对了**。

    这里用「说了要审批却没有下发通道」这条路径（`on_progress=None`）让审批门
    立刻返回拒绝，因此不必等 300s 超时。
    """
    scope = _scope(tmp_path, tool_approval="ask")
    token = bind_workspace_scope(scope)
    try:
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(
                channel="api",
                chat_id="direct",
                declared_tools=_policy(("plugin:demo", {"shell"})),
            )
        )
        # `shell` **在声明里**，所以声明门对它是放行的；此时如果有拒绝，
        # 只可能来自审批门 —— 反过来说，一旦装配漏掉审批门，这里会是 None。
        decision = await hook.before_execute_tool(_ctx(), _call("shell"), None, {})
        assert decision is not None and decision.allowed is False
        assert "approval" in decision.reason.lower()
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_rejected_approval_still_denies_a_declared_tool(tmp_path: Path) -> None:
    """审批被拒 → 拒绝原因来自**审批**,不是来自声明门。

    两者都叫「拒绝」,但用户要改的东西完全不同:一个去改权限/审批,一个去改声明。
    """
    scope = _scope(tmp_path, tool_approval="ask")
    token = bind_workspace_scope(scope)
    try:
        # 没有下发通道：审批门按拒绝处理（fail-closed,不退化成静默放行）
        approval = ToolApprovalHook(on_progress=None)
        declared = DeclaredToolPolicyHook(_policy(("plugin:demo", {"shell"})))
        decision = await CompositeHook([approval, declared]).before_execute_tool(
            _ctx(), _call("shell", {"command": "echo hi"}), None, {}
        )
        assert decision is not None and decision.allowed is False
        assert "approval" in decision.reason.lower()
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_declared_gate_is_in_the_baseline_chain(tmp_path: Path) -> None:
    """ephemeral turn（定时任务等）同样绕不过声明门。"""
    token = bind_workspace_scope(_scope(tmp_path))
    try:
        hook = build_agent_turn_hook(
            AgentTurnHookSpec(
                channel="api",
                chat_id="direct",
                ephemeral=True,
                declared_tools=_policy(("skill:a", {"read_file"})),
            )
        )
        decision = await hook.before_execute_tool(_ctx(), _call("shell"), None, {})
        assert decision is not None and decision.allowed is False
        assert "shell" in decision.reason
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_chain_is_unchanged_without_a_declaration(tmp_path: Path) -> None:
    """没有声明时,链上不该多出一道门（否则「没声明」也会开始拦人）。"""
    token = bind_workspace_scope(_scope(tmp_path))
    try:
        hook = build_agent_turn_hook(AgentTurnHookSpec(channel="api", chat_id="direct"))
        assert await hook.before_execute_tool(_ctx(), _call("shell"), None, {}) is None
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# 6. 从磁盘到策略（整链路）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json"
    )


def _always_skill(workspace: Path, name: str, frontmatter: str) -> None:
    root = workspace / "skills" / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo.\nalways: true\n{frontmatter}\n---\n\nbody\n",
        encoding="utf-8",
    )


def _plugin_with_tools(workspace: Path, name: str, tools: object) -> None:
    root = workspace / "plugins" / name
    root.mkdir(parents=True)
    (root / "plugin.json").write_text(
        json.dumps(
            {
                "$schema": AGENT_PLUGIN_SCHEMA,
                "name": name,
                "extensions": {"dev.nanobot": {"tools": tools}},
            }
        ),
        encoding="utf-8",
    )


def test_always_skill_and_plugin_declarations_intersect_on_disk(tmp_path: Path) -> None:
    """整链路:技能 frontmatter 与插件清单两处声明 → **交集**。

    这也是「两处声明」唯一的合理解释:两者都是收窄器,所以「两处都得允许」才放行。
    """
    _always_skill(tmp_path, "reader", "tools: [read_file, shell]")
    _plugin_with_tools(tmp_path, "demo", ["read_file", "write_file"])
    set_agent_plugin_enabled(tmp_path, "demo", True)

    policy = ContextBuilder(tmp_path).declared_tool_policy(tmp_path)

    assert policy.allow == frozenset({"read_file"})
    assert policy.sources == ("skill:reader", "plugin:demo")


def test_skill_without_the_field_does_not_constrain(tmp_path: Path) -> None:
    """「没写 tools」不等于「不许用任何工具」。"""
    _always_skill(tmp_path, "reader", "notes: plain")
    assert ContextBuilder(tmp_path).declared_tool_policy(tmp_path).allow is None


def test_single_string_in_skill_frontmatter_is_read_end_to_end(tmp_path: Path) -> None:
    """技能 frontmatter 交到解析器手里的是**原始值**（`skills.py` 刻意不预先收敛），
    所以「单值也要收」这条规则在技能侧是**唯一**的防线。

    插件清单那条路先经 `plugins.py` 的 `_string_list`，所以它更早被兜住 ——
    两条路各自把这一条钉住，是因为**任一侧漏掉都表现为「声明了却不生效」**。
    """
    _always_skill(tmp_path, "reader", "tools: shell")
    assert ContextBuilder(tmp_path).declared_tool_policy(tmp_path).allow == frozenset({"shell"})


def test_disabled_plugin_declaration_is_not_used(tmp_path: Path) -> None:
    _plugin_with_tools(tmp_path, "demo", ["read_file"])
    assert ContextBuilder(tmp_path).declared_tool_policy(tmp_path).allow is None


def test_single_string_in_manifest_is_read_end_to_end(tmp_path: Path) -> None:
    """`tools: "shell"` 在插件清单里同样要生效(与技能侧同一条解析规则)。"""
    _plugin_with_tools(tmp_path, "demo", "shell")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert ContextBuilder(tmp_path).declared_tool_policy(tmp_path).allow == frozenset({"shell"})


def test_read_failure_degrades_to_no_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读盘失败 → 降级为**不施加约束**,而不是让整轮工具全被拒。

    声明是**作者的自律**,不是用户的安全设置:读不到它不绕过任何用户设置
    （Agent 白名单 / 执行模式 / 审批门照旧由另外三道门守着）。反过来,
    一次读盘抖动就让所有工具调用被拒,那是把「可扩展」做成了「可宕机」。
    """
    builder = ContextBuilder(tmp_path)
    monkeypatch.setattr(
        type(builder.skills),
        "declared_tools",
        lambda self, names: (_ for _ in ()).throw(OSError("disk gone")),
    )
    assert builder.declared_tool_policy(tmp_path).allow is None
