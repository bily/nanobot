"""子代理进度事件（FR-2.5 第 4 步：客户端子代理分组的上游供给）。

要钉住三件事：
1. `emit_subagent_event` 打到**本轮**绑定的出口，且带齐归属三件套
   （``subagent`` / ``subagent_event`` / ``subagent_extra``）——少一个客户端就渲染不出来；
2. 没有出口（CLI / 离线调用）时**静默跳过**，绝不能把子代理带崩；
3. 出口自己抛错时同样不能影响子代理执行（进度是展示层的事）。

变异验证目标：
- 去掉 ``subagent_event=kind`` → 用例 1 变红；
- 去掉 ``except Exception: return`` → 用例 3 变红。
"""

import pytest

from nanobot.agent.subagent_progress import (
    bind_subagent_progress,
    current_subagent_progress,
    emit_subagent_event,
    reset_subagent_progress,
)


@pytest.mark.asyncio
async def test_emits_to_bound_sink_with_attribution_triplet() -> None:
    seen: list[tuple[tuple, dict]] = []

    async def sink(text: str = "", **kwargs: object) -> None:
        seen.append((text, kwargs))

    token = bind_subagent_progress(sink)
    try:
        await emit_subagent_event(
            "tool",
            subagent={"id": "task-1", "label": "查资料"},
            extra={"tool": "web_search", "iteration": 2},
        )
    finally:
        reset_subagent_progress(token)

    assert len(seen) == 1
    text, kwargs = seen[0]
    assert text == ""
    assert kwargs["subagent"] == {"id": "task-1", "label": "查资料"}
    assert kwargs["subagent_event"] == "tool"
    assert kwargs["subagent_extra"] == {"tool": "web_search", "iteration": 2}


@pytest.mark.asyncio
async def test_extra_is_copied_not_shared() -> None:
    """载荷必须拷贝：调用方后续改自己的 dict，不该改掉已发出去的事件。"""
    captured: list[dict] = []

    async def sink(text: str = "", **kwargs: object) -> None:
        captured.append(kwargs)

    extra = {"tool": "exec"}
    token = bind_subagent_progress(sink)
    try:
        await emit_subagent_event("tool", subagent={"id": "x", "label": "y"}, extra=extra)
    finally:
        reset_subagent_progress(token)

    extra["tool"] = "mutated"
    assert captured[0]["subagent_extra"]["tool"] == "exec"


@pytest.mark.asyncio
async def test_no_sink_is_silent_noop() -> None:
    assert current_subagent_progress() is None
    await emit_subagent_event("started", subagent={"id": "a", "label": "b"})


@pytest.mark.asyncio
async def test_sink_failure_never_breaks_the_subagent() -> None:
    async def broken(text: str = "", **kwargs: object) -> None:
        raise RuntimeError("SSE 队列已关闭")

    token = bind_subagent_progress(broken)
    try:
        # 不抛 = 子代理照常跑完
        await emit_subagent_event("done", subagent={"id": "a", "label": "b"})
    finally:
        reset_subagent_progress(token)


@pytest.mark.asyncio
async def test_reset_restores_previous_sink() -> None:
    async def outer(text: str = "", **kwargs: object) -> None:
        return None

    async def inner(text: str = "", **kwargs: object) -> None:
        return None

    t1 = bind_subagent_progress(outer)
    t2 = bind_subagent_progress(inner)
    assert current_subagent_progress() is inner
    reset_subagent_progress(t2)
    assert current_subagent_progress() is outer
    reset_subagent_progress(t1)
    assert current_subagent_progress() is None

@pytest.mark.asyncio
async def test_subagent_run_emits_started_and_done(tmp_path) -> None:
    """端到端：子代理真跑一轮，事件必须按 started → done 到达本轮出口。

    这是「客户端子代理分组」能成立的上游保证——只要这两次事件没出去，
    界面上就不会出现子代理行。
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import AgentDefaults
    from nanobot.providers.base import GenerationSettings
    from nanobot.utils.llm_runtime import LLMRuntime

    provider = MagicMock()
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=4096)
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
    )
    manager.runner.run = AsyncMock(return_value=SimpleNamespace(
        stop_reason="done", final_content="ok", error=None, tool_events=[],
    ))
    manager._announce_result = AsyncMock()

    events: list[tuple[str, dict]] = []

    async def sink(text: str = "", **kwargs: object) -> None:
        events.append((str(kwargs.get("subagent_event")), dict(kwargs.get("subagent_extra") or {})))

    token = bind_subagent_progress(sink)
    try:
        await manager.run_inline(
            task="查一下资料",
            label="资料员",
            session_key="test:c1",
            runtime=LLMRuntime.capture(provider, "test-model", context_window_tokens=128_000),
        )
    finally:
        reset_subagent_progress(token)

    kinds = [k for k, _ in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "done"
    # started 必须带任务摘要，否则分组行只有一行标题没有内容
    assert events[0][1]["task"] == "查一下资料"
