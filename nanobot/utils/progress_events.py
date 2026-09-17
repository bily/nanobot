"""Structured progress-event helpers shared by agent runtimes."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from nanobot.agent.hook import AgentHookContext


def on_progress_accepts_tool_events(cb: Callable[..., Any]) -> bool:
    return _on_progress_accepts(cb, "tool_events")


def on_progress_accepts_file_edit_events(cb: Callable[..., Any]) -> bool:
    return _on_progress_accepts(cb, "file_edit_events")


def on_progress_accepts_approval_requests(cb: Callable[..., Any]) -> bool:
    """[LOCAL PATCH] 进度回调是否接受审批请求（``approval_request``）。

    审批是唯一「需要回调回话」的进度事件：下发请求后必须等到用户裁决，
    因此调用方要能提前判断这条通道存不存在，好在缺失时立即按拒绝处理，
    而不是干等到超时。
    """
    return _on_progress_accepts(cb, "approval_request")


def on_progress_accepts_plan_events(cb: Callable[..., Any]) -> bool:
    """[LOCAL PATCH] 进度回调是否接受计划事件（``plan``）。

    计划事件是**只出不进**的：下发后由用户在客户端点选批准/打回，结果走下一轮
    请求回来（FR-1.5）。但通道存在与否仍要能提前判断——调用方靠它区分「已下发」
    与「无处可发」，后者必须拒绝 ``submit_plan``，否则模型会停在一份用户根本
    看不见的计划上。
    """
    return _on_progress_accepts(cb, "plan")


def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


async def invoke_on_progress(
    on_progress: Callable[..., Awaitable[None]],
    content: str,
    *,
    tool_hint: bool = False,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    if tool_events and on_progress_accepts_tool_events(on_progress):
        await on_progress(content, tool_hint=tool_hint, tool_events=tool_events)
        return
    await on_progress(content, tool_hint=tool_hint)


async def invoke_file_edit_progress(
    on_progress: Callable[..., Awaitable[None]],
    file_edit_events: list[dict[str, Any]],
) -> None:
    if not file_edit_events or not on_progress_accepts_file_edit_events(on_progress):
        return
    await on_progress("", file_edit_events=file_edit_events)


async def invoke_approval_request(
    on_progress: Callable[..., Awaitable[None]],
    payload: dict[str, Any],
) -> None:
    """[LOCAL PATCH] 把一条待批准的工具调用推给客户端。

    调用方须先用 :func:`on_progress_accepts_approval_requests` 确认通道存在；
    这里保持与相邻辅助函数一致的静默守卫，便于测试用具桩回调。
    """
    if not payload or not on_progress_accepts_approval_requests(on_progress):
        return
    await on_progress("", approval_request=payload)


async def invoke_plan_event(
    on_progress: Callable[..., Awaitable[None]],
    payload: dict[str, Any],
) -> bool:
    """[LOCAL PATCH] 把 Plan 模式产出的结构化计划推给客户端（FR-1.5）。

    返回 ``bool`` 让调用方能区分「已下发」与「通道缺失」：前者可以放心告诉
    模型「计划已交给用户」，后者则不能——谎报会让模型停在一份用户根本看不到
    的计划上，整轮对话就此卡死。
    """
    if not payload or not on_progress_accepts_plan_events(on_progress):
        return False
    await on_progress("", plan=payload)
    return True


def _tool_event_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = getattr(tool_call, "arguments", {}) or {}
    return cast(dict[str, Any], arguments) if isinstance(arguments, dict) else {}


def build_tool_event_start_payload(tool_call: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "phase": "start",
        "call_id": str(getattr(tool_call, "id", "") or ""),
        "name": getattr(tool_call, "name", ""),
        "arguments": _tool_event_arguments(tool_call),
        "result": None,
        "error": None,
        "files": [],
        "embeds": [],
    }


def tool_event_result_extras(result: Any) -> tuple[list[Any], list[Any]]:
    if not isinstance(result, dict):
        return [], []
    result_data = cast(dict[str, Any], result)
    raw_files = result_data.get("files")
    raw_embeds = result_data.get("embeds")
    files: list[Any] = cast(list[Any], raw_files) if isinstance(raw_files, list) else []
    embeds: list[Any] = cast(list[Any], raw_embeds) if isinstance(raw_embeds, list) else []
    return files, embeds


def build_tool_event_finish_payloads(context: AgentHookContext) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    count = min(len(context.tool_calls), len(context.tool_results), len(context.tool_events))
    for idx in range(count):
        tool_call = context.tool_calls[idx]
        result = context.tool_results[idx]
        event = context.tool_events[idx]
        status = event.get("status")
        phase = "end" if status == "ok" else "error"
        files, embeds = tool_event_result_extras(result)
        payload = {
            "version": 1,
            "phase": phase,
            "call_id": str(getattr(tool_call, "id", "") or ""),
            "name": getattr(tool_call, "name", ""),
            "arguments": _tool_event_arguments(tool_call),
            "result": result if phase == "end" else None,
            "error": None,
            "files": files,
            "embeds": embeds,
        }
        if phase == "error":
            if isinstance(result, str) and result.strip():
                payload["error"] = result.strip()
            else:
                payload["error"] = str(event.get("detail") or "Tool execution failed")
        payloads.append(payload)
    return payloads
