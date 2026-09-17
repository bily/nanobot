"""Agent hook that gates side-effecting tool calls on an explicit user verdict.

[LOCAL PATCH] nanowork：逐条批准工具调用的引擎侧实现。

工具真正执行在引擎进程内，客户端只收到 SSE 单向事件流——它看不见工具调用，
更拦不住。所以「唯一收口」只能落在这里：``before_execute_tool`` 里发一条审批
请求出去，阻塞等裁决，再决定放行还是拒绝。
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from loguru import logger

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentTurnHookContext,
    ToolExecutionDecision,
)
from nanobot.providers.base import ToolCallRequest
from nanobot.security.tool_approval import (
    PENDING_APPROVALS,
    ApprovalRequest,
    PendingApprovals,
    ToolApprovalPolicy,
    mode_blocks_tool,
    new_request_id,
)
from nanobot.security.workspace_access import (
    agent_blocks_tool,
    current_workspace_scope,
)
from nanobot.utils.progress_events import (
    invoke_approval_request,
    on_progress_accepts_approval_requests,
)

#: 单条参数值的展示上限。审批弹窗只需要「这是要动哪个文件 / 跑什么命令」，
#: 而 write_file 的 args 里可能带着整个文件内容——不截断会把 SSE 帧撑爆。
_MAX_ARG_CHARS = 400
_MAX_ARGS = 12


def _summarize_args(raw: Any) -> dict[str, Any]:
    """Render tool arguments into a small, JSON-safe dict for the prompt."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in list(cast(dict[str, Any], raw).items())[:_MAX_ARGS]:
        if value is None or isinstance(value, (int, float, bool)):
            out[str(key)] = value
            continue
        if isinstance(value, str):
            text = value
        else:
            try:
                text = str(value)
            except Exception:
                text = f"<{type(value).__name__}>"
        if len(text) > _MAX_ARG_CHARS:
            text = text[:_MAX_ARG_CHARS] + "…"
        out[str(key)] = text
    return out


class ToolApprovalHook(AgentHook):
    """Ask the user before running a side-effecting tool; refuse on silence."""

    def __init__(
        self,
        *,
        on_progress: Any = None,
        policy: ToolApprovalPolicy | None = None,
        registry: PendingApprovals | None = None,
    ) -> None:
        super().__init__()
        self._on_progress = on_progress
        self._policy = policy or ToolApprovalPolicy()
        self._registry = registry or PENDING_APPROVALS

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> ToolExecutionDecision | None:
        # 会话级开关优先从 contextvar 读：工具执行就在绑定作用域内，且与
        # shell.py 的 workspace 守卫用的是同一条通路。
        scope = current_workspace_scope()
        if scope is None or scope.tool_approval != "ask":
            return None
        # [LOCAL PATCH] 被执行模式禁止的调用由 SessionModeHook 直接拒绝。这里
        # 必须提前返回而不是继续走审批：`_decide_for_each_hook` 会**调用每一个**
        # 钩子再合并裁决，所以少了这一步，Ask 模式下的写操作就会既被拒、
        # 又弹出一个「批准了也执行不了」的对话框。
        if mode_blocks_tool(scope.session_mode, tool_call.name):
            return None
        # [LOCAL PATCH] FR-2.2：被 Agent 白名单拒绝的调用同理——它的授权归
        # AgentToolPolicyHook 判，这里不能替他弹一个「批了也执行不了」的框。
        # （`_decide_for_each_hook` 调用每一个钩子，两道门必须各自判断不介入。）
        if agent_blocks_tool(scope, tool_call.name) is not None:
            return None
        if not self._policy.requires_approval(tool_call.name):
            return None

        if self._on_progress is None or not on_progress_accepts_approval_requests(
            self._on_progress
        ):
            # 说了要审批却没有下发通道：按拒绝处理，不能退化成静默放行。
            return ToolExecutionDecision.deny(
                "This session asked for tool approval but has no channel to "
                "request it, so the call was refused."
            )

        request = ApprovalRequest(
            request_id=new_request_id(),
            tool=tool_call.name,
            args=_summarize_args(
                params if isinstance(params, dict) else getattr(tool_call, "arguments", None)
            ),
            session_key=context.session_key,
            call_id=getattr(tool_call, "id", None) or None,
        )
        future = self._registry.register(request)
        try:
            await invoke_approval_request(self._on_progress, request.payload())
            decision = await asyncio.wait_for(
                future, timeout=self._policy.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.info(
                "Tool approval for {} timed out after {}s; refusing",
                request.tool,
                self._policy.timeout_seconds,
            )
            return ToolExecutionDecision.deny(
                f"The user did not respond within {int(self._policy.timeout_seconds)}s, "
                f"so {request.tool} was refused."
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Tool approval for {} failed; refusing", request.tool)
            return ToolExecutionDecision.deny(
                "The approval request could not be delivered, so the call was refused."
            )
        finally:
            # 无论走哪条路都要摘掉登记，否则注册表会随会话堆积。
            self._registry.discard(request.request_id)

        if decision.allowed:
            return None
        return ToolExecutionDecision.deny(
            decision.reason or "The user declined this tool call."
        )


def create_tool_approval_hook(context: AgentTurnHookContext) -> AgentHook | None:
    """Build the approval gate for one turn.

    Always constructed: whether it intervenes depends on the session scope read
    at call time, not on this factory's timing.
    """
    return ToolApprovalHook(on_progress=context.on_progress)
