"""FR-2.4：专家团「派发**真的**带上成员绑定」的生产接线测试。

## 为什么要有第二个测试文件

`test_expert_group.py` 测的是**调度语义**（顺序 / 并发上限 / 聚合 / 失败隔离），
`run_member` 是注入的假函数——所以它能证明「编排按预期顺序跑完」，却**证明不了**
「成员的人设 / 工具策略 / 模型真的到了子代理手上」。

这三项恰恰是最容易脱钩的一类，而且脱钩之后：

- 编排**照样跑完**，事件**照样发**，`final_text` **照样有内容**；
- 唯一现象是「成员答得像个通才」「工具策略像没设」「模型像没绑」；

——这三句一条报错都不会触发。FR-2.3 时代「专家 `systemPrompt` 只被解析、从未使用」
正是这个形状：端到端看着是通的，人设却从来没进过 prompt。

## 本文件怎么测

把 `AgentLoop` 的**生产接线**（``_make_member_runner`` / ``_member_workspace_scope`` /
``_member_runtime``）当作被测对象，并且不是只断言「它构造出的 scope 长什么样」，
而是让它驱动**真的 `SubagentManager`** 跑一轮，去断言**子代理的行为**：
工具调用是被放行还是被拒、system message 里有没有那句人设。
少了链路上的任何一环，这里就会红。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from nanobot.agent.expert_group import ExpertMember
from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.security.workspace_access import WorkspaceScope, build_workspace_scope
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: LLMProvider, model: str = "turn-model") -> LLMRuntime:
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, model, context_window_tokens=128_000)


def _provider(*responses: LLMResponse) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "turn-model"
    provider.chat_with_retry = AsyncMock(side_effect=list(responses))
    return provider


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[])


def _call(name: str, **arguments: object) -> LLMResponse:
    return LLMResponse(
        content=f"calling {name}",
        tool_calls=[ToolCallRequest(id="call_1", name=name, arguments=dict(arguments))],
    )


def _subagents(tmp_path: Path) -> SubagentManager:
    return SubagentManager(
        workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16_000
    )


def _loop(subagents: object, *, runtime_resolver: object = None) -> AgentLoop:
    """绕过 `AgentLoop.__init__` 造一个只带被测所需依赖的实例。

    这里要测的是**接线**而不是整个 loop 的启动；用 `__new__` 避开构造，
    既不必拼出十几个协作者，也不会让「构造方式变了」伪装成接线故障。

    ⚠️ `AgentLoop` 上有一批只读 property（`model_presets` 派生自 `runtime_resolver`
    等），无法直接赋值——所以这里只写**可写**的那两个槽位。
    `_member_runtime` 用 `getattr(..., None)` 读它们正是为了这种场合：
    没设就自然走「无 resolver」分支，不必假装配置存在。
    """
    loop = AgentLoop.__new__(AgentLoop)
    loop.subagents = subagents
    loop.runtime_resolver = runtime_resolver
    return loop


def _runner_of(
    loop: AgentLoop, *, scope: WorkspaceScope, runtime: LLMRuntime
) -> object:
    return loop._make_member_runner(
        channel="im",
        chat_id="c1",
        message_id="m1",
        session_key="parent-turn-key",
        base_scope=scope,
        base_runtime=runtime,
    )


def _messages_of_call(provider: MagicMock, index: int) -> list[dict]:
    return provider.chat_with_retry.await_args_list[index].kwargs["messages"]


# ── 人设 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persona_reaches_the_subagent_system_message(tmp_path):
    """成员人设必须真的出现在子代理的 system message 里。

    这一条的存在理由就是项目踩过的那个坑：`ExpertMember.system_prompt` 曾经
    **只被 `parse_expert_group` 解析、从未被派发使用**，而编排看起来完全正常。
    """
    provider = _provider(_text("ok"))
    sm = _subagents(tmp_path)
    loop = _loop(sm)
    member = ExpertMember(
        id="legal", name="法务", system_prompt="你是法务专家，只回答合规问题。"
    )

    await _runner_of(loop, scope=build_workspace_scope(tmp_path, "restricted"),
                     runtime=_runtime(provider))(member, "审一下这份合同")

    system = _messages_of_call(provider, 0)[0]["content"]
    assert "你是法务专家" in system
    # 追加而非覆盖：模板的工作区 / 技能段必须还在（那是约束，不是风格）。
    # 断言「人设在末尾」而不是「路径还在」——模板里本就只写 `## Workspace` 段，
    # 并不拼绝对路径；写路径会让这条断言在无关改动下失效。
    assert "## Workspace" in system and "## Skills" in system
    assert system.rstrip().endswith("你是法务专家，只回答合规问题。")


@pytest.mark.asyncio
async def test_blank_persona_is_passed_as_none(tmp_path):
    """没人设就别传空串——`persona=''` 与不传在 `_build_subagent_prompt` 里等价，
    但保持「无 → None」能让意图显式：**没有声明人设**，而不是「人设是一段空白」。
    """
    subagents = MagicMock()
    subagents.run_inline = AsyncMock(return_value="ok")
    loop = _loop(subagents)

    await _runner_of(loop, scope=build_workspace_scope(tmp_path, "restricted"),
                     runtime=_runtime(_provider()))(
        ExpertMember(id="a", name="A", system_prompt=""), "task"
    )

    assert subagents.run_inline.await_args.kwargs["persona"] is None


# ── 工具策略 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_member_tool_deny_blocks_the_tool_inside_the_subagent(tmp_path):
    """成员黑名单要挡得住子代理**实际的**工具调用，而不只是出现在某个 scope 对象里。

    拒绝文案会带上 `agent_id`，而它在成员作用域里被置为**成员 id** ——
    所以这条同时证明「拒绝能指名道姓是哪个成员的策略」。
    """
    provider = _provider(
        _call("write_file", path="out.txt", content="pwned"),
        _text("could not write"),
    )
    sm = _subagents(tmp_path)
    loop = _loop(sm)
    member = ExpertMember(id="legal", name="法务", tool_deny=("write_file",))

    await _runner_of(loop, scope=build_workspace_scope(tmp_path, "restricted"),
                     runtime=_runtime(provider))(member, "把结论写下来")

    tool_result = _messages_of_call(provider, 1)[-1]["content"]
    assert "denies the 'write_file' tool" in tool_result
    assert "'legal'" in tool_result
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_member_tool_allow_intersects_the_parent_whitelist(tmp_path):
    """父作用域已经限定的能力，成员**不能把它重新放宽**。

    这是 FR-2.4 / FR-2.5 共用的「权限单调递减」约束：成员可以比父更窄，
    不能更宽。取的是**交集**而不是以成员声明为准。
    """
    provider = _provider(
        _call("write_file", path="out.txt", content="escaped"),
        _text("denied"),
    )
    sm = _subagents(tmp_path)
    loop = _loop(sm)
    # 父：只允许 read_file；成员自报产出alist 里含 write_file → 交集只剩 read_file
    base = build_workspace_scope(tmp_path, "restricted", tool_allow=("read_file",))
    member = ExpertMember(id="a", name="A", tool_allow=("read_file", "write_file"))

    await _runner_of(loop, scope=base, runtime=_runtime(provider))(member, "write it")

    tool_result = _messages_of_call(provider, 1)[-1]["content"]
    assert "does not include 'write_file'" in tool_result
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_granted_tool_still_runs_inside_the_subagent(tmp_path):
    """反向保护：交集之内被授予的工具必须真的能跑。

    只看「被拒」的用例无法发现「把授权门装反成了全部拒绝」——FR-2.2 的最小权限
    一旦变成 deny-all，专家团会集体空转且每条都「看起来合规」。
    """
    provider = _provider(
        _call("write_file", path="out.txt", content="hello"),
        _text("wrote it"),
    )
    sm = _subagents(tmp_path)
    loop = _loop(sm)
    base = build_workspace_scope(
        tmp_path, "restricted", tool_allow=("read_file", "write_file")
    )
    member = ExpertMember(id="a", name="A", tool_allow=("write_file",))

    await _runner_of(loop, scope=base, runtime=_runtime(provider))(member, "write it")

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"


def test_member_without_policy_inherits_the_parent_scope_unchanged(tmp_path):
    """没声明任何工具策略的成员，原样继承父作用域（而不是清空）。

    「没声明」≠「不受限」：把它读成后者，等于让一次漏配把最小权限反向放开。
    """
    base = build_workspace_scope(tmp_path, "restricted", tool_allow=("read_file",))
    derived = AgentLoop._member_workspace_scope(
        base, ExpertMember(id="a", name="A")
    )

    assert derived is base


def test_declared_policy_narrows_and_stamps_the_member_id(tmp_path):
    """声明了策略就要同时收窄并打上成员 id（拒绝原因据此指名道姓）。"""
    base = build_workspace_scope(
        tmp_path,
        "restricted",
        tool_allow=("read_file", "write_file"),
        tool_deny=("bash",),
        agent_id="parent-agent",
    )
    derived = AgentLoop._member_workspace_scope(
        base, ExpertMember(id="legal", name="法务", tool_deny=("delete_file",))
    )

    assert derived.tool_allow == base.tool_allow  # 成员没声明白名单 → 继承
    assert derived.tool_deny == frozenset({"bash", "delete_file"})  # 黑名单取并集
    assert derived.agent_id == "legal"


# ── 模型 ─────────────────────────────────────────────────────────────────────


def test_member_model_is_resolved_through_the_resolver(tmp_path):
    """成员模型绑定要走到解析出的 runtime，而不是停留在字符串上。

    命中的是**预设路径**：客户端下发的 model 是预设名（与 `Session.model` 同域），
    未走 resolver 就直接 `with_model` 会把预设名当成裸模型名发给 provider。
    """
    provider = _provider()
    base = _runtime(provider, "turn-model")
    bound = LLMRuntime.capture(provider, "member-model", context_window_tokens=8_000)
    resolver = MagicMock()
    resolver.resolve_preset.return_value = bound
    resolver.model_presets = {"member-preset": object()}
    loop = _loop(MagicMock(), runtime_resolver=resolver)

    derived = loop._member_runtime(base, ExpertMember(id="a", name="A", model="member-preset"))

    assert derived is bound
    resolver.resolve_preset.assert_called_once_with("member-preset")


@pytest.fixture
def warnings_captured():
    """收集 loguru 的 WARNING 输出。

    ⚠️ 项目用 **loguru**，它既不走标准 `logging`（`caplog` 恒为空），也不在乎 Python
    层的 stdout/stderr 重定向（`capsys` 同样拿不到）。所以只能临时挂一个自己的 sink。

    顺带这条 fixture 锁定了一件更难发现的事：loguru 的占位符是 ``{}`` **不是 ``%s``**
    —— 写成 ``logger.warning("... %s", x)`` 的话占位符不会被替换，日志照样出，
    但关键信息（哪个成员、哪个模型）一个都看不到，「留痕」等于没留。
    下面断言**具体取值**出现在日志里，所以改回 ``%s`` 会立刻转红。
    """
    captured: list[str] = []
    handler = logger.add(
        lambda message: captured.append(str(message)), level="WARNING", format="{message}"
    )
    try:
        yield captured
    finally:
        logger.remove(handler)


def test_unresolvable_member_model_falls_back_to_the_turn_runtime(
    tmp_path, warnings_captured
):
    """模型解析不了就退回本轮 runtime **并留痕**。

    与客户端 `resolveTurnModel` 同源的一课：引擎收到未知预设会**直接报错**，所以这里
    必须拦。但退回也不能静默——静默退回会让「成员模型没生效」变成无案可查。
    """
    provider = _provider()
    base = _runtime(provider, "turn-model")
    resolver = MagicMock()
    resolver.resolve_override.side_effect = ValueError("unknown preset")
    loop = _loop(MagicMock(), runtime_resolver=resolver)

    derived = loop._member_runtime(base, ExpertMember(id="ghost-member", name="A", model="ghost"))

    assert derived is base
    joined = "\n".join(warnings_captured)
    # 断言**具体取值**出现在日志里，而不只是「有告警」——占位符没替换时这里会红
    assert "ghost-member" in joined
    assert "ghost" in joined


def test_unknown_member_model_does_not_reach_the_provider(tmp_path, warnings_captured):
    """配了预设目录却不认识的名字 → 退回本轮 runtime，**不把它发给 provider**。

    真机证据（2026-10-07）：透传下去 provider 回 `The model 'x' does not exist`，
    于是「一个成员的模型名写错」会把**整个专家团**打成一轮失败，而不是连带留下
    另外两名成员的产出。这正是这条防护的存在理由。
    """
    provider = _provider()
    base = _runtime(provider, "turn-model")
    resolver = MagicMock()
    resolver.model_presets = {"known-preset": object()}
    resolver.resolve_override.return_value = _runtime(provider, "not-a-preset")
    loop = _loop(MagicMock(), runtime_resolver=resolver)

    derived = loop._member_runtime(
        base, ExpertMember(id="typo-member", name="A", model="not-a-preset")
    )

    assert derived is base
    resolver.resolve_override.assert_not_called()
    joined = "\n".join(warnings_captured)
    assert "typo-member" in joined and "not-a-preset" in joined


def test_no_preset_catalogue_still_allows_a_custom_model(tmp_path):
    """反向保护：**没有**配预设时必须允许透传。

    只在有目录时才校验——把「没配预设」读成「模型名全部失效」会让成员的自定义
    模型永远不生效（而这在开发者自用配置里是常态）。
    """
    provider = _provider()
    base = _runtime(provider, "turn-model")
    custom = _runtime(provider, "custom-model")
    resolver = MagicMock()
    resolver.model_presets = {}
    resolver.resolve_override.return_value = custom
    loop = _loop(MagicMock(), runtime_resolver=resolver)

    assert loop._member_runtime(
        base, ExpertMember(id="a", name="A", model="custom-model")
    ) is custom


def test_member_without_model_keeps_the_turn_runtime(tmp_path):
    provider = _provider()
    base = _runtime(provider)
    loop = _loop(MagicMock())

    assert loop._member_runtime(base, ExpertMember(id="a", name="A")) is base


# ── 取消与归属 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_member_runs_under_the_parent_session_key(tmp_path):
    """成员必须跑在**父会话 key** 上——这是「停止本轮」能连带取消在飞成员的前提。

    一旦成员用了自己的 key，并行模式下用户点停止只会砍掉 coordinator，
    成员继续烧 token，`ProcessView` 上还会留下永远走不完的步骤。
    """
    subagents = MagicMock()
    subagents.run_inline = AsyncMock(return_value="ok")
    loop = _loop(subagents)

    await _runner_of(loop, scope=build_workspace_scope(tmp_path, "restricted"),
                     runtime=_runtime(_provider()))(
        ExpertMember(id="a", name="A"), "task"
    )

    kwargs = subagents.run_inline.await_args.kwargs
    assert kwargs["session_key"] == "parent-turn-key"
    assert kwargs["origin_message_id"] == "m1"
    assert subagents.run_inline.await_args.kwargs["workspace_scope"] is not None
