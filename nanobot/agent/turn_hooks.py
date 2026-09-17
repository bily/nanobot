"""Turn-scoped hook assembly for agent runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hook import (
    AgentHook,
    AgentTurnHookContext,
    AgentTurnHookFactory,
    CompositeHook,
)
from nanobot.agent.hooks.agent_tools import create_agent_tool_policy_hook
from nanobot.agent.hooks.session_mode import create_session_mode_hook
from nanobot.agent.hooks.tool_approval import create_tool_approval_hook
from nanobot.agent.progress_hook import AgentProgressHook


@dataclass(slots=True)
class AgentTurnHookSpec:
    """Inputs needed to build the hook chain for one agent turn."""

    on_progress: Callable[..., Awaitable[None]] | None = None
    on_stream: Callable[[str], Awaitable[None]] | None = None
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    channel: str = "cli"
    chat_id: str = "direct"
    message_id: str | None = None
    metadata: dict[str, Any] | None = None
    session_key: str | None = None
    workspace: Path | None = None
    tool_hint_max_length: int = 40
    registered_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    turn_hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    registered_hooks: list[AgentHook] = field(default_factory=list)
    turn_hooks: list[AgentHook] = field(default_factory=list)
    ephemeral: bool = False
    run_extra_hooks_for_ephemeral: bool = False
    attributes: dict[str, Any] | None = None


def build_agent_turn_hook(spec: AgentTurnHookSpec) -> AgentHook:
    """Build the hook chain used by ``AgentRunner`` for one turn."""
    progress_hook = AgentProgressHook(
        on_progress=spec.on_progress,
        on_stream=spec.on_stream,
        on_stream_end=spec.on_stream_end,
        session_key=spec.session_key,
        tool_hint_max_length=spec.tool_hint_max_length,
    )
    turn_context = AgentTurnHookContext(
        on_progress=spec.on_progress,
        workspace=spec.workspace,
        channel=spec.channel,
        chat_id=spec.chat_id,
        message_id=spec.message_id,
        session_key=spec.session_key,
        metadata=dict(spec.metadata or {}),
        attributes=dict(spec.attributes or {}),
        ephemeral=spec.ephemeral,
    )

    # [LOCAL PATCH] nanowork：审批门是基线能力，不是可选扩展。
    # 原实现在 ephemeral turn 上直接返回 progress_hook，从而跳过全部工厂——
    # 定时任务与不持久化会话会因此成为审批策略的旁路。故把它提到工厂之前装配。
    approval_hook = create_tool_approval_hook(turn_context)
    baseline_chain: list[AgentHook] = [progress_hook]
    if approval_hook is not None:
        baseline_chain.append(approval_hook)

    # [LOCAL PATCH] nanowork：执行模式（FR-1.4/1.5）与审批门同级。
    # ask/plan 的只读语义同样不能在 ephemeral turn 上被跳过，否则同一份
    # 作用域在不同入口下硬度不一致（见 10.3.1）。
    session_mode_hook = create_session_mode_hook(turn_context)
    if session_mode_hook is not None:
        baseline_chain.append(session_mode_hook)

    # [LOCAL PATCH] nanowork：Agent 工具白名单（FR-2.2）同样是基线能力。
    # 授权门和上面的两道门一样不能被 ephemeral turn 跳过——定时任务里的 Agent
    # 若绕开自己的工具约束，最小权限就只是「交互式会话才生效的礼貌」。
    agent_tool_hook = create_agent_tool_policy_hook(turn_context)
    if agent_tool_hook is not None:
        baseline_chain.append(agent_tool_hook)

    if spec.ephemeral and not spec.run_extra_hooks_for_ephemeral:
        return (
            CompositeHook(baseline_chain)
            if len(baseline_chain) > 1
            else progress_hook
        )

    hook_chain: list[AgentHook] = list(baseline_chain)

    for factory in spec.registered_hook_factories:
        try:
            created_hook = factory(turn_context)
        except Exception:
            logger.exception("Agent turn hook factory failed: {}", factory)
            continue
        if created_hook is not None:
            hook_chain.append(created_hook)

    hook_chain.extend(spec.registered_hooks)

    for factory in spec.turn_hook_factories:
        try:
            created_hook = factory(turn_context)
        except Exception:
            logger.exception("Agent turn hook factory failed: {}", factory)
            continue
        if created_hook is not None:
            hook_chain.append(created_hook)

    hook_chain.extend(spec.turn_hooks)
    return CompositeHook(hook_chain) if len(hook_chain) > 1 else progress_hook
