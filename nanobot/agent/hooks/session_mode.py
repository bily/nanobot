"""Agent hook enforcing the session execution mode (FR-1.4 / FR-1.5).

[LOCAL PATCH] nanowork：执行模式的引擎侧强制点。

Ask / Plan 是**只读**模式：写类工具不是「问一句再执行」，而是「根本做不了」。
提示词只负责让模型不该做（软约束），真正让它做不了的是这里——与逐条批准门
一样，收口必须在引擎的工具执行前，客户端拦不住（见 10.3.1）。

同一个钩子还负责 Plan 模式的交付物：模型调用 ``submit_plan`` 时把结构化计划
经 SSE 下发给客户端（``kind: plan``），由用户勾选批准。计划提交本身**放行执行**
——工具会返回一条 ack，模型据此收尾本轮；真正的「批准」走下一轮 craft 请求。
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentTurnHookContext,
    ToolExecutionDecision,
)
from nanobot.providers.base import ToolCallRequest
from nanobot.security.tool_approval import PLAN_SUBMISSION_TOOLS, mode_blocks_tool
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.utils.progress_events import invoke_plan_event

#: 计划的步骤数上限。计划是给人看的清单，不是待办管理系统；超过这个量级
#: 说明模型把「实现细节」也拆成了条目，反而不可读。
_MAX_STEPS = 40


def _step_text(raw: Any) -> str:
    text = raw if isinstance(raw, str) else None
    if text is None and isinstance(raw, dict):
        item = cast(dict[str, Any], raw)
        for key in ("text", "title", "step", "description"):
            value = item.get(key)
            if isinstance(value, str):
                text = value
                break
    return (text or "").strip()


def normalize_steps(raw: Any) -> list[dict[str, str]]:
    """Coerce the model's ``steps`` argument into the wire shape.

    Deliberately permissive about *input* shapes (a bare string, ``text``,
    ``title``…) and strict about *output*: the client renders a checklist and
    needs stable ids plus a status on every row. Anything unparseable is
    dropped rather than raising — a malformed plan must not crash the turn.
    """
    if not isinstance(raw, list):
        return []
    steps: list[dict[str, str]] = []
    for idx, item in enumerate(cast(list[Any], raw)):
        text = _step_text(item)
        if not text:
            continue
        raw_id = item.get("id") if isinstance(item, dict) else None
        step_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else f"s{idx + 1}"
        steps.append({"id": step_id, "text": text, "status": "pending"})
        if len(steps) >= _MAX_STEPS:
            break
    return steps


def _steps_argument(params: Any, tool_call: ToolCallRequest) -> Any:
    source = params if isinstance(params, dict) else getattr(tool_call, "arguments", None)
    if isinstance(source, dict):
        return cast(dict[str, Any], source).get("steps")
    return None


class SessionModeHook(AgentHook):
    """Enforce ask/plan read-only semantics; relay plans to the client."""

    def __init__(self, *, on_progress: Any = None) -> None:
        super().__init__()
        self._on_progress = on_progress

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> ToolExecutionDecision | None:
        # 与审批门 / shell 的 workspace 守卫同源：工具执行就在绑定作用域内，
        # 模式只能从这个 contextvar 读（它不在工具参数里）。
        scope = current_workspace_scope()
        if scope is None:
            return None

        name = (tool_call.name or "").strip()
        mode = scope.session_mode

        # Plan 模式：提交计划是正当动作，先把它下发给用户，再放行执行
        # （工具本身只回一条 ack，没有副作用）。其它模式下它没有意义——放行
        # 只会让模型以为「已提交」，实际什么都没发生，所以显式拒绝。
        if name in PLAN_SUBMISSION_TOOLS:
            if mode == "plan":
                return await self._relay_plan(params, tool_call)
            return ToolExecutionDecision.deny(
                "submit_plan is only available in Plan mode. In Craft mode carry "
                "the work out directly; in Ask mode describe the plan in your reply."
            )

        if not mode_blocks_tool(mode, name):
            return None

        label = "Ask" if mode == "ask" else "Plan"
        return ToolExecutionDecision.deny(
            f"{label} mode is read-only, so {name} is not available. "
            "Explain what you would do instead, or ask the user to switch to "
            "Craft mode to make changes."
        )

    async def _relay_plan(
        self,
        params: Any,
        tool_call: ToolCallRequest,
    ) -> ToolExecutionDecision | None:
        """Send the plan upstream; refuse the call if it cannot be delivered."""
        steps = normalize_steps(_steps_argument(params, tool_call))
        if not steps:
            # 空计划没有可勾选的内容，与其下发一张空清单，不如让模型重来一次。
            return ToolExecutionDecision.deny(
                "submit_plan requires a non-empty `steps` array; each step needs "
                "a short description of one action."
            )

        payload = {
            "plan_id": f"plan-{uuid.uuid4().hex[:12]}",
            "steps": steps,
        }
        delivered = await invoke_plan_event(self._on_progress, payload)
        if not delivered:
            # 通道不存在时不能假装已提交——模型会就此停在一份用户看不见的计划上。
            return ToolExecutionDecision.deny(
                "This client cannot display a reviewable plan, so the plan was "
                "not submitted. Describe the plan in your reply instead."
            )
        return None


def create_session_mode_hook(context: AgentTurnHookContext) -> AgentHook | None:
    """Build the mode gate for one turn.

    Always constructed (like the approval gate): whether it intervenes depends on
    the session scope read at call time, not on this factory's timing.
    """
    return SessionModeHook(on_progress=context.on_progress)
