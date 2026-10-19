"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json as _json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from aiohttp import web
from loguru import logger

from nanobot.agent.expert_group import (
    EXPERT_GROUP_METADATA_KEY,
    ExpertGroupError,
    parse_expert_group,
)
from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.config.paths import get_media_dir
from nanobot.providers.base import LLMUsage
from nanobot.security.tool_approval import (
    PENDING_APPROVALS,
    ApprovalDecision,
)
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScopeError,
    validate_workspace_scope_payload,
)
from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
)
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
)
from nanobot.utils.media_decode import (
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
    "handle_plugin_action",
    "handle_plugins",
    "handle_tool_approval",
    "handle_usage",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
_AGENT_LOOP_KEY = web.AppKey[Any]("agent_loop")
_MODEL_NAME_KEY = web.AppKey[str]("model_name")
# [LOCAL PATCH] nanowork：热加载要知道配置**文件**在哪——运行中的 loop 持有的是启动时
# 那个 Config 对象，不从磁盘重读就永远看不到客户端刚写进去的模型。
_CONFIG_PATH_KEY = web.AppKey[str | None]("config_path")
_REQUEST_TIMEOUT_KEY = web.AppKey[float]("request_timeout")
_SESSION_LOCKS_KEY = web.AppKey[dict[str, asyncio.Lock]]("session_locks")
_PREPARE_AGENT_KEY = web.AppKey[Callable[[], Awaitable[None]] | None]("prepare_agent")
_MISSING = object()


class _UsageCaptureHook(AgentHook):
    """Capture the aggregate usage owned by one API run."""

    def __init__(self) -> None:
        super().__init__()
        self.usage: LLMUsage | None = None

    async def after_run(self, context: AgentRunHookContext) -> None:
        self.usage = context.usage


def _app_value(
    app: Any,
    key: web.AppKey[Any],
    legacy_key: str,
    default: Any = _MISSING,
) -> Any:
    """Read typed aiohttp state while accepting lightweight dict test doubles."""
    try:
        return app[key]
    except KeyError:
        if default is _MISSING:
            return app[legacy_key]
        return app.get(legacy_key, default)


async def _prepare_agent(app: Any) -> None:
    prepare: Callable[[], Awaitable[None]] | None = _app_value(
        app,
        _PREPARE_AGENT_KEY,
        "prepare_agent",
        None,
    )
    if prepare is not None:
        await prepare()


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(
    content: str,
    model: str,
    usage: LLMUsage | None = None,
) -> dict[str, Any]:
    prompt = usage.input_tokens if usage else 0
    completion = usage.output_tokens if usage else 0
    total = usage.total_tokens if usage else 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


def _as_str(value: object) -> str:
    """Return *value* when it is text, otherwise an empty string."""
    return value if isinstance(value, str) else ""


def _require_json_object(value: object, field: str) -> dict[str, Any]:
    """Validate an object-valued field from an untrusted JSON request."""
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_json_string(value: object, field: str) -> str:
    """Validate a string-valued field from an untrusted JSON request."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_delta(
    delta: dict[str, Any],
    model: str,
    chunk_id: str,
    finish_reason: str | None = None,
) -> bytes:
    """Format a single OpenAI-compatible SSE chunk with an arbitrary delta object.

    [LOCAL PATCH] nanowork：进度事件（深度思考 / 工具执行）经
    ``delta.nanobot_event`` 自定义字段下发；标准 OpenAI 客户端会忽略未知
    delta 字段，兼容性不受影响。
    """
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk (plain-text delta)."""
    return _sse_delta({"content": delta} if delta else {}, model, chunk_id, finish_reason)


_SSE_DONE = b"data: [DONE]\n\n"


def _finish_reason_for(stop_reason: object) -> str:
    """把引擎内部的 ``stop_reason`` 映射成 OpenAI 兼容的 ``finish_reason``。

    OpenAI 只认 ``stop`` / ``length`` / ``tool_calls`` / ``function_call`` /
    ``content_filter``。映射口径是**保守但不说谎**：本轮实际报错、或撞上迭代上限时
    都不能再报 ``stop`` —— 客户端会据此判断这一轮是否正常收尾，把失败报成成功正是
    2026-09-29 断流事故被渲染成「已完成」的一半原因。

    Args:
        stop_reason: 引擎 turn 的结束原因（``ctx.stop_reason``，经 outbound metadata
            的 ``_stop_reason`` 透出）。未知值、缺失、非字符串一律按 ``stop`` 处理：
            这是历史行为，也是唯一的「不改变既有语义」的选择。

    Returns:
        OpenAI 兼容的 finish_reason 字符串。
    """
    if not isinstance(stop_reason, str):
        return "stop"
    if stop_reason in {"error", "tool_error"}:
        return "error"
    if stop_reason == "max_iterations":
        # 迭代预算耗尽 = 这一轮没能收尾，语义上等价于「输出没写完」
        return "length"
    return "stop"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict[str, Any]) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages_value = cast(object, body.get("messages"))
    if not isinstance(messages_value, list):
        raise ValueError("Only a single user message is supported")
    messages = cast(list[object], messages_value)
    if len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message_value: object = messages[0]
    if not isinstance(message_value, dict):
        raise ValueError("Only a single user message is supported")
    message = cast(dict[str, Any], message_value)
    if message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part_value in cast(list[object], user_content):
            if not isinstance(part_value, dict):
                continue
            part = cast(dict[str, Any], part_value)
            if part.get("type") == "text":
                text_parts.append(
                    _require_json_string(
                        cast(object, part.get("text", "")),
                        "messages[0].content[].text",
                    )
                )
            elif part.get("type") == "image_url":
                image_url = _require_json_object(
                    cast(object, part.get("image_url", {})),
                    "messages[0].content[].image_url",
                )
                url = _require_json_string(
                    cast(object, image_url.get("url", "")),
                    "messages[0].content[].image_url.url",
                )
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(
    request: web.Request,
) -> tuple[
    str,
    list[str],
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """Parse multipart/form-data.

    Returns (text, media_paths, session_id, model, workspace, access_mode,
    approval_mode, session_mode, agent_id, tool_allow, tool_deny, expert_group).

    ``expert_group`` 是 JSON 串（multipart 无嵌套对象），由调用方 ``json.loads``
    还原成 dict；空/缺省为 ``None``。
    """
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    workspace = None
    access_mode = None
    approval_mode = None
    session_mode = None
    # [LOCAL PATCH] FR-2.2：Agent 工具策略。multipart 没有原生数组，改用
    # 逗号分隔串（与 ``normalize_tool_names`` 的字符串形态一致）。
    agent_id = None
    tool_allow = None
    tool_deny = None
    # [LOCAL PATCH] FR-2.4：专家团清单以 JSON 串承载（multipart 无嵌套对象）。
    expert_group_raw: str | None = None
    media_paths: list[str] = []

    while True:
        part: Any = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "workspace":
            workspace = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "access_mode":
            access_mode = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "approval_mode":
            approval_mode = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "session_mode":
            session_mode = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "agent_id":
            agent_id = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "tool_allow":
            # 字段**出现**即为「已声明」：空串 = 声明了一个空白名单（一个工具
            # 都不许），而不是「没声明」。方向取 fail-closed——静默放开是看不见
            # 的事故，全部拒绝是立刻可见的。
            tool_allow = (await part.read()).decode("utf-8").strip()
        elif part.name == "tool_deny":
            tool_deny = (await part.read()).decode("utf-8").strip()
        elif part.name == "expert_group":
            # [LOCAL PATCH] FR-2.4：JSON 串，形状校验交给后续的 parse_expert_group。
            expert_group_raw = (await part.read()).decode("utf-8").strip() or None
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return (
        text,
        media_paths,
        session_id,
        model,
        workspace,
        access_mode,
        approval_mode,
        session_mode,
        agent_id,
        tool_allow,
        tool_deny,
        expert_group_raw,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response | web.StreamResponse:
    """POST /v1/chat/completions — supports JSON and multipart/form-data."""
    content_type = _as_str(cast(object, request.content_type or ""))

    agent_loop = _app_value(request.app, _AGENT_LOOP_KEY, "agent_loop")
    timeout_s: float = _app_value(
        request.app,
        _REQUEST_TIMEOUT_KEY,
        "request_timeout",
        120.0,
    )
    model_name: str = _app_value(request.app, _MODEL_NAME_KEY, "model_name", "nanobot")

    stream = False
    workspace: str | None = None
    access_mode: str | None = None
    approval_mode: str | None = None
    session_mode: str | None = None
    # [LOCAL PATCH] FR-2.2：Agent 工具策略。``None`` = 未声明（不限制）；
    # 空列表 = 已声明为空白名单（一个工具都不许）。这个区别贯穿到引擎的
    # ``WorkspaceScope.tool_allow``，不能在这里被抹平。
    agent_id: str | None = None
    tool_allow: Any = None
    tool_deny: Any = None
    # [LOCAL PATCH] FR-2.4：专家团成员清单（仅 JSON 请求体支持；multipart 不带）。
    expert_group_payload: Any = None
    try:
        if content_type.startswith("multipart/"):
            (
                text,
                media_paths,
                session_id,
                requested_model,
                workspace,
                access_mode,
                approval_mode,
                session_mode,
                agent_id,
                tool_allow,
                tool_deny,
                expert_group_raw,
            ) = await _parse_multipart(request)
            # [LOCAL PATCH] FR-2.4：multipart 无嵌套对象，专家团清单以 JSON 串传入，
            # 这里还原成 dict；坏 JSON 直接 400（与形状非法同款，不静默降级）。
            if expert_group_raw:
                try:
                    expert_group_payload = _json.loads(expert_group_raw)
                except ValueError:
                    return _error_json(400, "Invalid expert_group JSON")
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            if not isinstance(body, dict):
                return _error_json(400, "Invalid JSON body")
            body = cast(dict[str, Any], body)
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
            # [LOCAL PATCH] 会话级工作空间 / 权限（客户端按会话选择后透传）
            raw_workspace = body.get("workspace")
            if isinstance(raw_workspace, str) and raw_workspace.strip():
                workspace = raw_workspace.strip()
            raw_mode = body.get("access_mode")
            if isinstance(raw_mode, str) and raw_mode.strip():
                access_mode = raw_mode.strip()
            # [LOCAL PATCH] 逐条批准开关，与 access_mode 同源透传。
            raw_approval = body.get("approval_mode")
            if isinstance(raw_approval, str) and raw_approval.strip():
                approval_mode = raw_approval.strip()
            # [LOCAL PATCH] 执行模式（ask / plan / craft，FR-1.4），同源透传。
            raw_session_mode = body.get("session_mode")
            if isinstance(raw_session_mode, str) and raw_session_mode.strip():
                session_mode = raw_session_mode.strip()
            # [LOCAL PATCH] Agent 工具策略（FR-2.2）。数组原样透传，形状校验交给
            # ``normalize_tool_names``——它在 ``validate_workspace_scope_payload``
            # 里，非法形状会让本次请求 400（而不是静默放宽）。
            raw_agent_id = body.get("agent_id")
            if isinstance(raw_agent_id, str) and raw_agent_id.strip():
                agent_id = raw_agent_id.strip()
            if body.get("tool_allow") is not None:
                tool_allow = body.get("tool_allow")
            if body.get("tool_deny") is not None:
                tool_deny = body.get("tool_deny")
            expert_group_payload = body.get("expert_group")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    # [LOCAL PATCH] 请求级 workspace scope 预校验：非法路径直接 400，
    # 合法则注入消息 metadata，由 WorkspaceScopeResolver 在 turn 边界生效。
    request_metadata: dict[str, Any] | None = None
    if (
        workspace
        or access_mode
        or approval_mode
        or session_mode
        or agent_id
        or tool_allow is not None
        or tool_deny is not None
    ):
        resolver = getattr(agent_loop, "workspace_scopes", None)
        if resolver is not None:
            scope_payload: dict[str, Any] = {}
            if workspace:
                scope_payload["project_path"] = workspace
            if access_mode:
                scope_payload["access_mode"] = access_mode
            # 逐条批准：与 access_mode 同处会话级作用域，随 metadata 落到 turn。
            if approval_mode:
                scope_payload["tool_approval"] = approval_mode
            # 执行模式：同一处承载，工具执行点由 contextvar 读回（FR-1.4）。
            if session_mode:
                scope_payload["session_mode"] = session_mode
            # Agent 工具策略（FR-2.2）：同处承载，由 AgentToolPolicyHook 读回。
            if agent_id:
                scope_payload["agent_id"] = agent_id
            # 注意判据是「不是 None」而不是真值：空列表是**已声明的空白名单**，
            # 用真值判断会把它当成「未声明」而静默放开全部工具。
            if tool_allow is not None:
                scope_payload["tool_allow"] = tool_allow
            if tool_deny is not None:
                scope_payload["tool_deny"] = tool_deny
            try:
                validate_workspace_scope_payload(
                    scope_payload,
                    default_workspace=resolver.default_workspace,
                    default_restrict_to_workspace=resolver.default_restrict_to_workspace,
                )
            except WorkspaceScopeError as e:
                return _error_json(400, f"Invalid workspace scope: {e}")
            request_metadata = {WORKSPACE_SCOPE_METADATA_KEY: scope_payload}

    # [LOCAL PATCH] FR-2.4 专家团：形状非法直接 400。静默忽略会让「专家团」悄悄退化成
    # 普通一轮对话——不报错、界面上也看不出来，比直接报错难查得多。
    if expert_group_payload is not None:
        try:
            parse_expert_group(expert_group_payload)
        except ExpertGroupError as e:
            return _error_json(400, f"Invalid expert_group: {e}")
        request_metadata = dict(request_metadata or {})
        request_metadata[EXPERT_GROUP_METADATA_KEY] = expert_group_payload

    # [LOCAL PATCH] 请求级模型解析：preset 名走配置预设，裸模型名走当前 provider 覆盖；
    # 仅作用于本次请求，不改动网关默认选择。空值跟随配置默认。
    runtime = None
    reported_model = model_name
    if requested_model:
        resolver = getattr(agent_loop, "runtime_resolver", None)
        presets = getattr(agent_loop, "model_presets", None) or {}
        if resolver is not None:
            try:
                if requested_model in presets:
                    runtime = resolver.resolve_preset(requested_model)
                else:
                    runtime = resolver.resolve_override(
                        model=requested_model, model_preset=None
                    )
            except (KeyError, ValueError):
                available = ", ".join(presets) or model_name
                return _error_json(
                    400, f"Unknown model '{requested_model}'. Available: {available}"
                )
        elif requested_model != model_name:
            return _error_json(400, f"Only configured model '{model_name}' is available")
        reported_model = runtime.model if runtime is not None else requested_model

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = _app_value(
        request.app,
        _SESSION_LOCKS_KEY,
        "session_locks",
    )
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    # -- streaming path --
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        stream_failed = False
        emitted_content = False

        async def _on_stream(token: str) -> None:
            nonlocal emitted_content
            if token:
                emitted_content = True
            await queue.put(_sse_chunk(token, reported_model, chunk_id))

        async def _on_progress(
            content: str,
            *,
            tool_hint: bool = False,
            tool_events: list[dict[str, Any]] | None = None,
            file_edit_events: list[dict[str, Any]] | None = None,
            reasoning: bool = False,
            reasoning_end: bool = False,
            approval_request: dict[str, Any] | None = None,
            plan: dict[str, Any] | None = None,
            member: dict[str, Any] | None = None,
            member_event: str | None = None,
            member_extra: dict[str, Any] | None = None,
            subagent: dict[str, Any] | None = None,
            subagent_event: str | None = None,
            subagent_extra: dict[str, Any] | None = None,
            **_kw: Any,
        ) -> None:
            # [LOCAL PATCH] 将 Agent 进度事件注入 SSE 流（delta.nanobot_event），
            # 供客户端消息流渲染「深度思考 / 工具执行」过程时间线。
            # tool_hint 文本行由客户端按 tool_events 自行渲染，此处不下发。
            if subagent and subagent_event:
                # [LOCAL PATCH] FR-2.5 第 4 步：子代理生命周期事件。
                # 与专家团成员事件同形，客户端据此渲染「子代理分组」行；否则一次
                # spawn 只能显示成一个黑箱工具调用，用户看不到它到底在干什么。
                # 透传载荷先展开、固定键后覆盖：归属字段是契约，不允许被载荷改写。
                await queue.put(
                    _sse_delta(
                        {
                            "nanobot_event": {
                                **dict(subagent_extra or {}),
                                "kind": "subagent",
                                "subagent": subagent,
                                "subagent_event": subagent_event,
                            }
                        },
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if member and member_event:
                # [LOCAL PATCH] FR-2.4 第 3 步：专家团成员生命周期。
                # 客户端靠这两个字段把过程事件归到「成员 A / 成员 B」名下，
                # 否则一次多成员编排只能显示成一轮黑箱。
                # 透传载荷先展开、固定键后覆盖：归属字段是契约，不允许被载荷里的同名键改写。
                await queue.put(
                    _sse_delta(
                        {
                            "nanobot_event": {
                                **dict(member_extra or {}),
                                "kind": "member",
                                "member": member,
                                "member_event": member_event,
                            }
                        },
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if approval_request:
                # 工具调用此时阻塞在引擎的 before_execute_tool 里等裁决，
                # 客户端拿到 id 后走 POST /v1/tool-approvals/{id} 回传。
                await queue.put(
                    _sse_delta(
                        {
                            "nanobot_event": {
                                "kind": "approval_request",
                                **approval_request,
                            }
                        },
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if plan:
                # [LOCAL PATCH] Plan 模式的结构化交付物（FR-1.5）。只出不进：
                # 客户端渲染成可勾选清单，用户的批准/打回走下一轮请求回来。
                await queue.put(
                    _sse_delta(
                        {"nanobot_event": {"kind": "plan", **plan}},
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if reasoning and content:
                await queue.put(
                    _sse_delta(
                        {"nanobot_event": {"kind": "reasoning", "content": content}},
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if reasoning_end:
                await queue.put(
                    _sse_delta(
                        {"nanobot_event": {"kind": "reasoning_end"}},
                        reported_model,
                        chunk_id,
                    )
                )
                return
            if tool_events:
                for ev in tool_events:
                    await queue.put(
                        _sse_delta(
                            {"nanobot_event": {"kind": "tool", **ev}},
                            reported_model,
                            chunk_id,
                        )
                    )

        async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
            # Agent stream-end callbacks mark generation segment boundaries.
            # Tool-backed requests may continue after a segment ends, so the
            # HTTP SSE stream is closed only when process_direct returns.
            return None

        # 本轮真实的终结原因：由 process_direct 返回的 outbound metadata 透出
        # （见 AgentLoop._prepare_outbound 写入的 `_stop_reason`）。拿不到时退回 "stop"。
        finish_reason = "stop"

        async def _run() -> None:
            nonlocal stream_failed, finish_reason
            try:
                async with session_lock:
                    async with asyncio.timeout(timeout_s):
                        await _prepare_agent(request.app)
                        response = await agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_progress=_on_progress,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                            runtime=runtime,
                            metadata=request_metadata,
                        )
                    finish_reason = _finish_reason_for(
                        (getattr(response, "metadata", None) or {}).get("_stop_reason")
                    )
                    if not emitted_content:
                        response_text = _response_text(response)
                        if response_text.strip():
                            await queue.put(_sse_chunk(response_text, reported_model, chunk_id))
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await resp.write(chunk)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if stream_failed:
            # 中途失败：**不写** [DONE]（客户端据此判定这次是异常结束），但补一个终结帧
            # 把原因说清楚。旧实现是静默关闭 body，客户端只看到「流没了」，与正常收尾
            # 完全同形——2026-09-29 的断流就是这样被渲染成「已完成」的。
            await resp.write(_sse_chunk("", reported_model, chunk_id, finish_reason="error"))
            return resp
        await resp.write(_sse_chunk("", reported_model, chunk_id, finish_reason=finish_reason))
        await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
    usage_capture = _UsageCaptureHook()
    try:
        async with session_lock:
            try:
                async with asyncio.timeout(timeout_s):
                    await _prepare_agent(request.app)
                    response = await agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                        runtime=runtime,
                        metadata=request_metadata,
                        hooks=[usage_capture],
                    )
                response_text = _response_text(response)
                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, using fallback", session_key)
                    response_text = EMPTY_FINAL_RESPONSE_MESSAGE

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(
        # _chat_completion_response(response_text, reported_model, getattr(agent_loop, "_last_usage", None))
        _chat_completion_response(response_text, reported_model, usage_capture.usage)
    )


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models

    [LOCAL PATCH] nanowork：模型列表来源由「单一默认模型」
    扩展为 nanobot 配置中的模型预设目录（modelPresets + 隐式 default），
    供前端模型选择器动态渲染；default 始终位于首位。
    """
    model_name = _app_value(request.app, _MODEL_NAME_KEY, "model_name", "nanobot")
    agent_loop = _app_value(request.app, _AGENT_LOOP_KEY, "agent_loop", None)

    data: list[dict[str, Any]] = []
    presets = getattr(agent_loop, "model_presets", None) if agent_loop is not None else None
    if presets:
        for name, preset in presets.items():
            # default 为隐式预设（无 label 配置），以实际模型名作为展示名
            fallback_label = name if name != "default" else getattr(preset, "model", name)
            label = getattr(preset, "label", None) or fallback_label
            data.append(
                {
                    "id": name,
                    "label": label,
                    "model": getattr(preset, "model", name),
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            )
        # default（agents.defaults 隐式预设）固定置顶
        data.sort(key=lambda item: (item["id"] != "default", item["id"]))
    else:
        data.append(
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nanobot",
            }
        )

    return web.json_response({"object": "list", "data": data})


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


async def handle_tool_approval(request: web.Request) -> web.Response:
    """POST /v1/tool-approvals/{id} — deliver a verdict for a pending tool call.

    [LOCAL PATCH] nanowork：逐条批准的回传端点。

    引擎在 ``before_execute_tool`` 里挂起等待，客户端从 SSE 拿到
    ``approval_request`` 事件后调用这里把裁决送回去。这是 SSE 单向流出设计
    下唯一的「反向通道」，缺失就意味着所有写类工具都拿不到授权。
    """
    request_id = _as_str(request.match_info.get("id") or "")
    if not request_id:
        return _error_json(400, "Missing approval request id")

    try:
        body = await request.json()
    except Exception:
        body = None
    if body is not None and not isinstance(body, dict):
        return _error_json(400, "Invalid JSON body")
    body_data: dict[str, Any] = cast(dict[str, Any], body) if body else {}

    raw_decision = _as_str(body_data.get("decision") or "").strip().lower()
    # ``always`` = 「本会话内不再询问这个工具」。它是一次性放行的**更强**形态，
    # 因此必须出现在协议白名单里，不能靠客户端把它翻译成 ``allow``——那样
    # 引擎永远学不会「别再问了」，用户每轮都要点一遍。
    if raw_decision not in ("allow", "deny", "always"):
        return _error_json(400, "decision must be 'allow', 'deny' or 'always'")
    reason = _as_str(body_data.get("reason") or "").strip()

    pending = PENDING_APPROVALS.get(request_id)
    if pending is None:
        # 已裁决、已超时或会话已取消——客户端多半是晚到了一步。
        return _error_json(404, "Unknown or already-settled approval request")

    delivered = PENDING_APPROVALS.resolve(
        request_id,
        ApprovalDecision(verdict=cast(Any, raw_decision), reason=reason),
    )
    if not delivered:
        return _error_json(409, "Approval request was already settled")

    logger.info("Tool approval {} -> {}", request_id, raw_decision)
    return web.json_response(
        {"ok": True, "id": request_id, "decision": raw_decision}
    )


async def handle_dream_review(request: web.Request) -> web.Response:
    """GET /v1/dream/review — surface the pending Dream candidate for the review UI.

    [LOCAL PATCH] nanowork FR-4.6：客户端审查页从这里拿候选 diff。
    没有候选时返回 ``{"pending": false}``，避免客户端空轮询报错。
    """
    workspace = _plugin_workspace(request)
    if workspace is None:
        return _error_json(503, "Agent workspace is unavailable")
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(workspace)
    if not store.has_dream_candidate():
        return web.json_response({"pending": False, "diff": {}})
    return web.json_response({"pending": True, "diff": store.compute_dream_candidate_diff()})


async def handle_dream_accept(request: web.Request) -> web.Response:
    """POST /v1/dream/accept — write the accepted candidate back to the real memory files.

    [LOCAL PATCH] nanowork FR-4.6：采纳。只在候选内容确实变化时才触碰正式文件
    （``accept_dream_candidate`` 内部已逐字节比对），并留下审计提交。
    """
    workspace = _plugin_workspace(request)
    if workspace is None:
        return _error_json(503, "Agent workspace is unavailable")
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(workspace)
    if not store.has_dream_candidate():
        return _error_json(409, "No Dream candidate is pending review")
    changed = store.accept_dream_candidate()
    return web.json_response({"ok": True, "updated": changed})


async def handle_dream_reject(request: web.Request) -> web.Response:
    """POST /v1/dream/reject — discard the pending Dream candidate.

    [LOCAL PATCH] nanowork FR-4.6：丢弃。``reject_dream_candidate`` 永远不碰正式文件。
    """
    workspace = _plugin_workspace(request)
    if workspace is None:
        return _error_json(503, "Agent workspace is unavailable")
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(workspace)
    if not store.has_dream_candidate():
        return _error_json(409, "No Dream candidate is pending review")
    store.reject_dream_candidate()
    return web.json_response({"ok": True})


# [LOCAL PATCH] nanowork：配置热加载端点（FR「保存即生效」）。
#
# 背景：客户端在设置里加/删模型是**写配置文件**，而运行中的引擎把 modelPresets 装配成
# 启动时的快照（且快照 loader 闭包住那个 Config 对象）。以前唯一的生效手段是重启网关，
# 于是 UI 只能提示「保存后自动重启网关」——用户明确要求配置变更立刻生效。
async def handle_config_reload(request: web.Request) -> web.Response:
    """POST /v1/config/reload — 重新读盘并热替换模型装配，不重启进程。

    配置解析失败一律 400 且**保持现有装配不动**：半套配置比旧配置更糟。
    """
    config_path = _app_value(request.app, _CONFIG_PATH_KEY, "config_path", None)
    if not config_path:
        return _error_json(503, "Gateway was started without a config file path")
    agent_loop = _app_value(request.app, _AGENT_LOOP_KEY, "agent_loop", None)
    if agent_loop is None:
        return _error_json(503, "Agent loop is not available")

    from nanobot.config.loader import load_config

    try:
        config = load_config(Path(config_path))
    except Exception as exc:  # 配置写坏了：明确回错，别让服务崩在半路
        return _error_json(400, f"Failed to load config: {exc}", "invalid_config")

    reload_fn = getattr(agent_loop, "reload_runtime_config", None)
    if reload_fn is None:
        return _error_json(501, "This engine does not support config hot reload")
    try:
        reload_fn(config)
    except Exception as exc:
        return _error_json(400, f"Failed to apply config: {exc}", "invalid_config")

    # `getattr` 兜底：拿不到目录时回空列表而不是 500——这个端点的价值就是「保存后
    # 立刻能看到新列表」，它读不到目录不该让整个保存流程崩掉（客户端仍会重拉
    # `/v1/models`，那边拿的是同一份装配）。
    presets = getattr(agent_loop, "model_presets", None) or {}
    return web.json_response({"ok": True, "models": sorted(presets)})


def _plugin_workspace(request: web.Request) -> Path | None:
    """插件接口的工作区：直接取 AgentLoop 上的那一个。

    刻意不从请求体里接受 workspace——插件激活会落盘写标记（``plugin-data/``），
    让调用方指定工作区就等于让请求方决定往哪个目录写。
    """
    agent_loop = _app_value(request.app, _AGENT_LOOP_KEY, "agent_loop", None)
    workspace = getattr(agent_loop, "workspace", None)
    # 只接受字符串/Path：拿不到工作区时返回 None 让调用方报 503，
    # 别把一个奇怪的对象喂给 Path() 变成 500。
    if isinstance(workspace, (str, Path)) and str(workspace):
        return Path(workspace)
    return None


async def handle_plugins(request: web.Request) -> web.Response:
    """GET /v1/plugins — 已安装插件清单（含引擎侧激活状态）。

    [LOCAL PATCH] nanowork FR-3.4：客户端需要「装了什么、激活没有」的权威答案。
    引擎的插件启停只挂在 WebUI 通道上，渲染层走 ``/v1`` 够不到，所以在
    这里开一条只读入口。**激活状态由引擎给出**（指纹绑定的标记文件），
    客户端不自己算哈希、也不自己存一份真相。
    """
    workspace = _plugin_workspace(request)
    if workspace is None:
        return _error_json(503, "Agent workspace is unavailable")

    from nanobot.agent.plugins import default_user_plugins_dir, discover_agent_plugins

    plugins = await asyncio.to_thread(
        discover_agent_plugins,
        workspace,
        default_user_plugins_dir(),
    )
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "name": plugin.name,
                    "displayName": plugin.display_name,
                    "description": plugin.description,
                    "category": plugin.category,
                    "enabled": plugin.enabled,
                    "mcpServers": list(plugin.mcp_servers),
                    # [LOCAL PATCH] FR-3.8：来源与版本。本地平铺安装两者都是 None。
                    "market": plugin.market,
                    "version": plugin.version,
                }
                for plugin in plugins
            ],
        }
    )


async def handle_plugin_action(request: web.Request) -> web.Response:
    """POST /v1/plugins/{name} — 启用 / 停用已安装插件。

    body: ``{"action": "enable" | "disable"}``

    只做启停，不接受安装/卸载：安装要写文件系统，那是桌面端主进程的事。
    """
    name = _as_str(request.match_info.get("name") or "").strip()
    if not name:
        return _error_json(400, "Missing plugin name")

    try:
        body = await request.json()
    except Exception:
        body = None
    if body is not None and not isinstance(body, dict):
        return _error_json(400, "Invalid JSON body")
    body_data: dict[str, Any] = cast(dict[str, Any], body) if body else {}

    action = _as_str(body_data.get("action") or "").strip().lower()
    if action not in ("enable", "disable"):
        return _error_json(400, "action must be 'enable' or 'disable'")

    workspace = _plugin_workspace(request)
    if workspace is None:
        return _error_json(503, "Agent workspace is unavailable")

    from nanobot.agent.plugins import default_user_plugins_dir, set_agent_plugin_enabled

    try:
        await asyncio.to_thread(
            set_agent_plugin_enabled,
            workspace,
            name,
            action == "enable",
            default_user_plugins_dir(),
        )
    except ValueError:
        return _error_json(404, f"Unknown plugin '{name}'")
    except RuntimeError as exc:
        # 包在启用过程中被改动：引擎拒绝签发激活标记（防的是「先启用、
        # 再把包内容换掉」）。这不是服务端故障，别报 500。
        return _error_json(409, str(exc), err_type="plugin_state_error")

    logger.info("Agent Plugin '{}' {}", name, "enabled" if action == "enable" else "disabled")
    return web.json_response({"ok": True, "name": name, "enabled": action == "enable"})


def _usage_timezone_name() -> str | None:
    """引擎配置里的时区名；读不到就返回 ``None``（store 会回退 UTC）。

    刻意**不缓存**：设置页改完时区，报表应当立刻按新时区分组。
    也刻意走 ``load_config`` 而不是手解 JSON —— 后者会绕过 ``$ENV`` 插值这类
    语义，得到一个「看着像、实际不等价」的时区，而按日分组错一天是查不出来的。
    """
    try:
        from nanobot.config.loader import load_config

        defaults = load_config().agents.defaults
        return getattr(defaults, "timezone", None) or None
    except Exception:
        logger.warning("usage: 读取引擎时区失败，按 UTC 分组", exc_info=True)
        return None


async def handle_usage(request: web.Request) -> web.Response:
    """GET /v1/usage — 本机 LLM 用量聚合（nanowork FR-11.1）。

    [LOCAL PATCH] nanowork：客户端「用量」页只渲染、不重算。

    为什么不把聚合挪到客户端：口径全都写在 ``nanobot.llm_usage`` 里 —— 按
    **本地日**分组、30 天窗口、连续使用天数、缓存命中率的分母（只算「缓存
    真的被观测到」的那些请求）。复制一份到渲染层就是两份真相，引擎这边改一次，
    客户端就会静默偏一次，而且偏的是数字，不是报错。

    因此这里连 ``days`` 都不裁剪：客户端拿到完整窗口（默认 371 天）后自行切片，
    切换时间范围不需要再请求。

    只读语义：**不写入任何调用记录**。但要说清楚 —— 库还不存在时
    ``usage_payload`` 会建立一个空库（它内部第一步 ``_connect`` 就要
    ``CREATE TABLE IF NOT EXISTS``，这是引擎既有行为，不是本路由加的）；
    所以「打开用量页会凭空多出一个空 ``llm_usage.sqlite3``」是预期内的，
    与「有没有真的用过模型」无关。
    """
    from nanobot.llm_usage import llm_usage_payload

    return web.json_response(
        await asyncio.to_thread(
            llm_usage_payload,
            timezone_name=_usage_timezone_name(),
        )
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop: "AgentLoop",
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    api_key: str = "",
    prepare_agent: Callable[[], Awaitable[None]] | None = None,
    config_path: str | None = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
        api_key: Optional API key for Bearer-token authentication on API routes.
        prepare_agent: Optional application-owned readiness callback run before each turn.
        config_path: [LOCAL PATCH] 配置文件路径，供 `POST /v1/config/reload` 热加载用。
    """
    app = web.Application(client_max_size=20 * 1024 * 1024)  # 20MB for base64 images
    app[_AGENT_LOOP_KEY] = agent_loop
    app[_MODEL_NAME_KEY] = model_name
    app[_CONFIG_PATH_KEY] = config_path
    app[_REQUEST_TIMEOUT_KEY] = request_timeout
    app[_SESSION_LOCKS_KEY] = {}  # per-user locks, keyed by session_key
    app[_PREPARE_AGENT_KEY] = prepare_agent

    @web.middleware
    async def auth_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        # Allow unauthenticated health checks.
        if request.path == "/health":
            return await handler(request)
        if not api_key:
            return await handler(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _error_json(401, "Missing Authorization header. Use: Bearer <api_key>")
        if not hmac.compare_digest(auth[len("Bearer "):], api_key):
            return _error_json(401, "Invalid API key")
        return await handler(request)

    app.middlewares.append(auth_middleware)

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    # [LOCAL PATCH] 逐条批准的反向通道（SSE 只出不进，裁决必须单独回传）
    app.router.add_post("/v1/tool-approvals/{id}", handle_tool_approval)
    # [LOCAL PATCH] nanowork FR-3.4：插件清单与启停。
    # 引擎是激活状态的唯一权威（指纹绑定的标记文件），客户端只读不复制哈希。
    app.router.add_get("/v1/plugins", handle_plugins)
    app.router.add_post("/v1/plugins/{name}", handle_plugin_action)
    # [LOCAL PATCH] nanowork FR-11.1：本机 LLM 用量聚合（只读）。
    # 报表的口径留在引擎（llm_usage），客户端只渲染，不重算。
    app.router.add_get("/v1/usage", handle_usage)
    # [LOCAL PATCH] nanowork FR-4.6：Dream 审查闸门客户端入口（候选 diff + 采纳/丢弃）。
    app.router.add_get("/v1/dream/review", handle_dream_review)
    app.router.add_post("/v1/dream/accept", handle_dream_accept)
    app.router.add_post("/v1/dream/reject", handle_dream_reject)
    # [LOCAL PATCH] nanowork：模型增删改后热加载，让「保存」立刻生效而不必重启网关。
    app.router.add_post("/v1/config/reload", handle_config_reload)
    app.router.add_get("/health", handle_health)
    return app
