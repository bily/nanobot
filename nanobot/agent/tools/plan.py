"""``submit_plan`` — hand a structured plan to the user for approval (FR-1.5).

[LOCAL PATCH] nanowork：Plan 模式的结构化交付物。

计划不是靠解析助手正文得到的（那对输出格式的微小漂移极其敏感），而是靠这次
工具调用提交的：模型在 Plan 模式下调用本工具，引擎的 :class:`SessionModeHook`
截获参数、经 SSE 下发 ``kind: plan`` 事件，再由客户端渲染成可勾选清单。
用户批准后，客户端切到 Craft 模式并以同一份计划开下一轮（见 10.2）。

本工具的 ``execute`` 只回一条 ack —— 真正的「裁决」发生在客户端，不走工具返回值。
工具在非 Plan 模式下由模式钩子直接拒绝（Ask 模式也一样），因此它出现在工具
列表里并不会给其它模式开口子。
"""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)


@tool_parameters(
    tool_parameters_schema(
        steps=ArraySchema(
            items=ObjectSchema(
                properties={
                    "text": StringSchema("One sentence describing the action."),
                    "id": StringSchema(
                        "Optional stable id for this step; assigned automatically if omitted."
                    ),
                },
                required=["text"],
            ),
            description=(
                "Ordered list of the actions you intend to take. One short "
                "sentence each; do not include implementation minutiae."
            ),
            min_items=1,
        ),
        required=["steps"],
    )
)
class SubmitPlanTool(Tool):
    """Submit a plan for user review instead of executing it."""

    @property
    def name(self) -> str:
        return "submit_plan"

    @property
    def description(self) -> str:
        return (
            "Submit a structured plan for the user to review and approve. "
            "Use this in Plan mode once you have finished investigating: it ends "
            "your turn and shows the user a checklist. After approval the work is "
            "carried out in Craft mode with the approved steps. "
            "This tool is only accepted in Plan mode; elsewhere the call is refused."
        )

    async def execute(self, steps: Any = None, **kwargs: Any) -> str:
        """Acknowledge the plan; the user's verdict arrives on the next turn."""
        count = len(steps) if isinstance(steps, list) else 0
        return (
            f"Plan submitted for review ({count} step(s)). "
            "Stop here and wait for the user's decision; do not make any changes."
        )
