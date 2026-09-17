"""Agent hook enforcing the per-agent tool whitelist (FR-2.2).

[LOCAL PATCH] nanowork：Agent 工具白名单的引擎侧强制点。

每个 Agent 可以声明「允许哪些工具」（``tool_allow``）与「禁止哪些工具」
（``tool_deny``）。默认最小权限——一旦声明了白名单，未列入的工具就**不可用**。

为什么必须收口在这里：Agent 定义住在客户端（``src/types/agent.ts``），能表达
约束的只有提示词；而提示词按 10.2 的分类只是**软约束**——一次提示词注入或
模型幻觉就能绕过。真正让越权调用做不了的是 ``before_execute_tool``：与逐条
批准门、执行模式门一样，它是引擎的工具执行唯一入口，客户端拦不住也绕不过。

三层门的分工（同一入口，职责不重叠）：

* ``SessionModeHook``：**模式**层面（ask/plan 只读）——「这个会话不许写」。
* ``ToolApprovalHook``：**裁决**层面（要不要问）——「这个调用要用户点头」。
* 本钩子：**授权**层面（这个 Agent 有没有这个能力）——「这个 Agent 不该有这个工具」。

三者都不依赖提示词。授权检查排在审批之前：连问都不该问的能力，弹窗只是噪音。
"""

from __future__ import annotations

from typing import Any

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentTurnHookContext,
    ToolExecutionDecision,
)
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    agent_blocks_tool,
    current_workspace_scope,
)


class AgentToolPolicyHook(AgentHook):
    """Refuse tool calls the current agent was not granted."""

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> ToolExecutionDecision | None:
        # 与审批门 / 模式门同源：策略住在会话作用域里，只能从 contextvar 读
        # （它不出现在工具参数中，也就无法被模型改写）。
        scope = current_workspace_scope()
        reason = agent_blocks_tool(scope, tool_call.name)
        if reason is None:
            return None
        return ToolExecutionDecision.deny(reason)


def create_agent_tool_policy_hook(context: AgentTurnHookContext) -> AgentHook | None:
    """Build the authorization gate for one turn.

    Always constructed (like the other two gates): whether it intervenes depends
    on the session scope read at call time, not on this factory's timing.
    """
    return AgentToolPolicyHook()
