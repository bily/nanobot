"""子代理进度通道（FR-2.5 第 4 步）。

**为什么需要这个 contextvar**：子代理是由 `spawn` 工具在**工具执行期**拉起的，而
`ToolContext` 是循环注册期构建一次、整进程复用的——它拿不到「本轮」的进度回调。
专家团那条链是靠闭包直接把 `ctx.on_progress` 传下去的，子代理没有那个闭包。

与 `security/workspace_access.py` 的 `bind_workspace_scope` 同构：**由循环在每轮
绑定、工具侧读取**。这样「本轮的进度出口」也跟着 contextvar 走，子代理无论在
前台（`run_inline`）还是后台（`spawn`）任务里，都能回到同一个出口。

⚠️ 绑定必须与工具执行**同一协程上下文**：后台任务会复制当时的 context，
所以出口函数本身要能安全地在子任务里 await（它捕获的是本轮的 SSE 队列）。
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from typing import Any

# 进度出口签名：与 RunContext.on_progress 同形（位置参数 + 关键字附加载荷）。
# 关键字里的 ``subagent`` / ``subagent_event`` / ``subagent_extra`` 是契约：
# 服务端据此把它们渲染成 SSE 的 `kind:"subagent"` 事件，客户端再按子代理分组。
SubagentProgressSink = Callable[..., Awaitable[None]]

_CURRENT_SUBAGENT_PROGRESS: contextvars.ContextVar[SubagentProgressSink | None] = (
    contextvars.ContextVar("nanobot_subagent_progress", default=None)
)


def bind_subagent_progress(sink: SubagentProgressSink | None) -> contextvars.Token:
    """绑定本轮的子代理进度出口，返回 token 供 `reset_subagent_progress` 还原。"""
    return _CURRENT_SUBAGENT_PROGRESS.set(sink)


def reset_subagent_progress(token: contextvars.Token) -> None:
    _CURRENT_SUBAGENT_PROGRESS.reset(token)


def current_subagent_progress() -> SubagentProgressSink | None:
    return _CURRENT_SUBAGENT_PROGRESS.get()


async def emit_subagent_event(
    kind: str,
    *,
    subagent: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """发一条子代理进度事件；没有出口（CLI / 离线调用）时静默跳过。

    **失败不许影响子代理执行**：进度是展示层的事，让它把任务带崩是最差结果。
    """
    sink = current_subagent_progress()
    if sink is None:
        return
    try:
        await sink(
            "",
            subagent=subagent,
            subagent_event=kind,
            subagent_extra=dict(extra or {}),
        )
    except Exception:  # noqa: BLE001 - 展示层异常一律吞掉
        return