"""Progress callback helpers for user-visible output.

These helpers convert agent progress callbacks into outbound chat messages.
Runtime state notifications such as turn lifecycle and model changes live in
``nanobot.bus.runtime_events``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent, outbound_message_for_event
from nanobot.bus.queue import MessageBus
from nanobot.security.im_approval import build_approval_card


def build_bus_progress_callback(
    bus: MessageBus,
    msg: InboundMessage,
) -> Callable[..., Awaitable[None]]:
    """Return a callback that publishes progress as outbound messages."""

    async def _publish_approval(approval_request: dict[str, Any]) -> None:
        # [LOCAL PATCH] FR-6.5：审批请求必须以**普通出站消息**下发，不能包成
        # ``ProgressEvent``——后者会被渠道的 ``send_progress`` 静音，而审批一旦
        # 被静音，用户永远看不到卡片，工具只能等到超时被拒。卡片正文带唯一编号，
        # 顺带避开了管理器按 ``message_id`` 对同内容去重（两次相同的 write_file
        # 否则第二张卡会被当成重复丢弃）。
        content, buttons = build_approval_card(approval_request)
        await bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                buttons=buttons,
                metadata=dict(msg.metadata or {}),
            )
        )

    async def _publish_progress(
        content: str,
        *,
        tool_hint: bool = False,
        tool_events: list[dict[str, Any]] | None = None,
        file_edit_events: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        reasoning_end: bool = False,
        approval_request: dict[str, Any] | None = None,
    ) -> None:
        if approval_request:
            await _publish_approval(approval_request)
            return
        await bus.publish_outbound(
            outbound_message_for_event(
                channel=msg.channel,
                chat_id=msg.chat_id,
                event=ProgressEvent(
                    content=content,
                    tool_hint=tool_hint,
                    reasoning_delta=reasoning,
                    reasoning_end=reasoning_end,
                    tool_events=tool_events,
                    file_edit_events=file_edit_events,
                ),
                metadata=msg.metadata,
            )
        )

    async def _bus_progress(
        content: str,
        *,
        tool_hint: bool = False,
        tool_events: list[dict[str, Any]] | None = None,
        file_edit_events: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        reasoning_end: bool = False,
        approval_request: dict[str, Any] | None = None,
    ) -> None:
        await _publish_progress(
            content,
            tool_hint=tool_hint,
            tool_events=tool_events,
            file_edit_events=file_edit_events,
            reasoning=reasoning,
            reasoning_end=reasoning_end,
            approval_request=approval_request,
        )

    return _bus_progress
