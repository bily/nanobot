"""Declarative tool policy declared by skills and plugins (FR-3.12).

[LOCAL PATCH] nanowork：技能 / 插件声明的工具白名单的引擎侧强制点。

## 这条需求要解决什么

技能与插件的作者最清楚「我这个能力只需要哪些工具」。但在此之前，作者能表达的
只有提示词（「请只使用 read_file」），而提示词按 10.2 的分类只是**软约束**——
一次注入或一次幻觉就绕过去了。这里给作者一个**硬**表达：frontmatter / 清单里的
`tools:`，由引擎在工具执行唯一入口上强制。

## 三条硬纪律

**① 只收紧，绝不放宽。** 声明是一份**白名单**（「我只需要这些」），所以它只能
让可用工具变少。这条**不是**靠这里小心实现出来的，而是 `CompositeHook` 固有的
性质：`_decide_for_each_hook` 只收集 `allowed=False` 的决策，`allow()` 决策被
直接丢弃（见 `agent/hook.py` 的合并循环）。因此本钩子**只能往里加 deny**，
永远盖不掉审批门的 deny —— 「声明 `tools: [shell]` 不得使 `shell` 绕过审批」
是免费成立的。

**② 多份声明取交集。** 技能与插件都可能声明，每一份都是收窄器，所以「两处都得
允许」才放行（`combine_declarations`）。把两份声明并起来会让**任何一份**都能
放宽另一份，等于把①从后门破掉。

**③ 落差必须说出来（用户 2026-10-16 拍板的口径）。** 声明了、但当前 Agent 基线
根本不给的能力（例：技能写了 `tools: [shell]`，而 Agent 白名单里没有 shell），
交集之后是**不生效**的。不生效本身是对的（否则插件就能自我提权），但**不能静默**：
`policy_gaps()` 把「谁声明了、哪个工具、基线为什么给不了」算出来，本模块在每轮
第一次工具调用时记一条 warn。少了这一步，插件作者会对着一个「我声明了却什么都没
发生」的现象发呆 —— 而这个项目已经反复栽在同类问题上（`_run_member` 闭包导致
人设只被解析从未使用；`Composer` 的可选 prop 当开关导致桥接整体为 null）。

## 刻意不做的两件事

* **不设「框架内部工具」豁免名单。** 声明白名单只写 `read_file` 时，`submit_plan`
  之类也会被拦。这与 FR-2.2 的 Agent 白名单**行为一致**（那里同样是严格白名单），
  而加一份隐藏豁免名单恰恰是后来最容易咬人的那种「静默规则」。作者要什么就写什么。
* **不做「声明为空 = 不约束」。** `tools: []` 是显式声明，按**空集**处理（全部拒绝）
  并告警。把它读成「没声明」会让一个 YAML 手滑变成「什么都没限制」——安全方向的
  静默失效，正是 `tools: "shell"` 那类坑的另一面。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentTurnHookContext,
    ToolExecutionDecision,
)
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    WorkspaceScope,
    agent_blocks_tool,
    current_workspace_scope,
)

#: 工具名字符集。**刻意比「字母数字下划线」宽**：MCP 工具是带命名空间的
#: （`plugin--server__tool`），写死成 `\w+` 会把它们全判成非法名而静默丢掉
#: —— 而丢掉的是白名单条目，方向上是「更严」，不会放宽；但会让作者以为
#: 自己声明成功了。故这里只拦真正会出问题的字符（空白 / 控制符 / 超长）。
_TOOL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def parse_declared_tools(value: object) -> tuple[str, ...]:
    """把清单 / frontmatter 里的 `tools:` 收敛成工具名元组。

    **单值与列表都要收**：`tools: shell` 是最容易写出来的形态，用「只认列表」的
    写法会静默读成空 —— 而空 = 不约束，是**安全方向上的静默失效**。

    非法项（非字符串 / 空串 / 含空白或控制符 / 超长）**丢弃并告警**：
    丢弃只让白名单更严，不会放宽；但要说一声，否则作者以为声明成功了。
    返回空元组有两种含义，由调用方区分：`tools: []` 是**显式空集**（全拒），
    字段缺失才是「没声明」。
    """
    if isinstance(value, str):
        items: list[object] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        logger.warning(
            "Ignoring tools declaration of unsupported type: {}", type(value).__name__
        )
        return ()

    names: list[str] = []
    for item in items:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string entry in tools declaration: {!r}", item)
            continue
        name = item.strip()
        if _TOOL_NAME.fullmatch(name) is None:
            logger.warning("Ignoring invalid tool name in tools declaration: {!r}", item)
            continue
        names.append(name)
    # 去重但保序：重复项会让 deny 原因里出现两个一样的名字，读起来像两处声明
    return tuple(dict.fromkeys(names))


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """一处声明：谁声明的（`skill:foo` / `plugin:bar`）+ 声明了哪些工具。"""

    source: str
    tools: frozenset[str]


@dataclass(frozen=True, slots=True)
class DeclaredToolPolicy:
    """把多份声明叠加成一份策略。

    ``allow is None`` = **没有任何声明** = 不施加约束（与「声明了空集」是两件事：
    后者 `allow == frozenset()`，语义是全部拒绝）。
    """

    declarations: tuple[ToolDeclaration, ...]
    allow: frozenset[str] | None

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(declaration.source for declaration in self.declarations)


def combine_declarations(declarations: Sequence[ToolDeclaration]) -> DeclaredToolPolicy:
    """多份声明 → 一份策略，叠加规则 = **交集**（每份都是收窄器，见模块 docstring ②）。"""
    kept = tuple(declaration for declaration in declarations if declaration.source)
    if not kept:
        return DeclaredToolPolicy(declarations=(), allow=None)
    for declaration in kept:
        if not declaration.tools:
            # 显式空集是**合法**声明（作者说「我什么工具都不要」），后果是这一轮
            # 所有调用都被拒。所以要大声说一遍：这种「什么都不发生」的成因
            # 从现象上看是「插件坏了」，从日志上看才是「你自己写了 tools: []」。
            logger.warning(
                "{} declares an empty tools whitelist, so every tool call in this "
                "turn will be refused. Remove the field to stop constraining, or list "
                "the tools it actually needs.",
                declaration.source,
            )
    allow = set(kept[0].tools)
    for declaration in kept[1:]:
        allow &= declaration.tools
    return DeclaredToolPolicy(declarations=kept, allow=frozenset(allow))


@dataclass(frozen=True, slots=True)
class ToolGap:
    """声明了、但当前基线**拿不到**的一个工具（FR-3.12 纪律③）。"""

    source: str
    tool: str
    #: 基线拒绝它的原因（`agent_blocks_tool` 给的原文）
    reason: str


def policy_gaps(
    policy: DeclaredToolPolicy,
    scope: WorkspaceScope | None,
) -> tuple[ToolGap, ...]:
    """算出「声明了但用不上」的工具。

    「用不上」的判据**复用 `agent_blocks_tool`**（FR-2.2 的既有纯函数）：
    自己再推一套「基线允许集」就是两份真相，而这里错的后果是**给用户看一个错的
    落差**——比不报更糟，因为它会把人引去改一个本来没问题的地方。
    """
    if policy.allow is None:
        return ()
    gaps: list[ToolGap] = []
    for declaration in policy.declarations:
        for tool in sorted(declaration.tools):
            reason = agent_blocks_tool(scope, tool)
            if reason is not None:
                gaps.append(ToolGap(source=declaration.source, tool=tool, reason=reason))
    return tuple(gaps)


def _report_gaps(policy: DeclaredToolPolicy, scope: WorkspaceScope | None) -> None:
    """把落差写成一条可搜的 warn（每轮至多一次，由钩子实例记住）。

    刻意**不抛错、不拒绝**：声明拿不到的能力是「作者写多了」，不是用户做错了什么。
    真正改变行为的只有 `allow`，落差只影响可观测性。
    """
    gaps = policy_gaps(policy, scope)
    if not gaps:
        return
    detail = "; ".join(f"{gap.source} -> {gap.tool}" for gap in gaps)
    logger.warning(
        "Declared tool policy cannot take effect for {} entr{}: {}. "
        "The declaration only narrows what the turn may use, so these tools stay "
        "unavailable; widen the agent's tool whitelist if the plugin really needs them.",
        len(gaps),
        "y" if len(gaps) == 1 else "ies",
        detail,
    )


class DeclaredToolPolicyHook(AgentHook):
    """拒绝「不在技能 / 插件声明白名单里」的工具调用。

    与逐条批准门、执行模式门、Agent 授权门**同源**：策略住在构造时给进来的
    不可变对象里（不是工具参数），因此模型改写不了它。
    """

    def __init__(self, policy: DeclaredToolPolicy) -> None:
        self._policy = policy
        #: 落差每轮只报一次：每调一次工具就刷一条 warn 会把日志淹掉
        self._gaps_reported = False

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> ToolExecutionDecision | None:
        allow = self._policy.allow
        if allow is None:
            return None  # 没有任何声明 = 不施加约束
        if not self._gaps_reported:
            self._gaps_reported = True
            # 作用域只能从 contextvar 读（与其它三道门同一条通路）
            _report_gaps(self._policy, current_workspace_scope())
        name = (tool_call.name or "").strip()
        if name in allow:
            return None
        sources = ", ".join(self._policy.sources)
        return ToolExecutionDecision.deny(
            f"The tool '{name}' is outside the tool whitelist declared by {sources}. "
            "That declaration can only narrow what this turn may use, so this tool is "
            "not available now. Do not retry it: complete the task with the declared "
            "tools, or tell the user the declaration (not their permission settings) "
            "is what blocks this call."
        )


def create_declared_tool_policy_hook(
    context: AgentTurnHookContext,
    policy: DeclaredToolPolicy | None,
) -> AgentHook | None:
    """按装配期算好的声明构造这道门。

    无声明时返回 ``None``（不挂空钩子）——这样「没有插件 / 技能声明 `tools:`」
    的会话，钩子链与引入本需求之前**逐字相同**。

    ⚠️ **必须用 `isinstance` 认形状，不能只判 `is not None`**：`declared_tools` 是
    装配方（`loop.py`）喂进来的，而任何「长得像但不是」的值都会造成最坏的一种故障
    —— 门挂上了、白名单是空的、于是**所有工具全被拒**，而拒绝原因里连一个来源名
    都没有（`sources` 为空）。那不是「更安全」，那是整轮不可用且无法解释。
    实测来源：单测里 `ContextBuilder` 被 mock 掉时，`declared_tool_policy(...)`
    返回的是 `MagicMock`，它的 `__contains__` 恒为 False。所以**认不出形状就当没声明**
    （＝不约束），并留一条 warn。
    """
    if policy is None:
        return None
    if not isinstance(policy, DeclaredToolPolicy):
        logger.warning(
            "Ignoring a declared-tool policy of unexpected type {}; no tool "
            "whitelist will be enforced for this turn.",
            type(policy).__name__,
        )
        return None
    if policy.allow is None:
        return None
    return DeclaredToolPolicyHook(policy)
