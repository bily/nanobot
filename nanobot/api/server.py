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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from aiohttp import web
from loguru import logger

from nanobot.config.paths import get_media_dir
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
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
_AGENT_LOOP_KEY = web.AppKey[Any]("agent_loop")
_MODEL_NAME_KEY = web.AppKey[str]("model_name")
_REQUEST_TIMEOUT_KEY = web.AppKey[float]("request_timeout")
_SESSION_LOCKS_KEY = web.AppKey[dict[str, asyncio.Lock]]("session_locks")
_PREPARE_AGENT_KEY = web.AppKey[Callable[[], Awaitable[None]] | None]("prepare_agent")
_MISSING = object()


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
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    prompt = (usage or {}).get("prompt_tokens", 0)
    completion = (usage or {}).get("completion_tokens", 0)
    total = (usage or {}).get("total_tokens", 0) or prompt + completion
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

    [LOCAL PATCH] sciherd-cloud-smartagent：进度事件（深度思考 / 工具执行）经
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
) -> tuple[str, list[str], str | None, str | None, str | None, str | None]:
    """Parse multipart/form-data.

    Returns (text, media_paths, session_id, model, workspace, access_mode).
    """
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    workspace = None
    access_mode = None
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

    return text, media_paths, session_id, model, workspace, access_mode


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
    try:
        if content_type.startswith("multipart/"):
            (
                text,
                media_paths,
                session_id,
                requested_model,
                workspace,
                access_mode,
            ) = await _parse_multipart(request)
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
    if workspace or access_mode:
        resolver = getattr(agent_loop, "workspace_scopes", None)
        if resolver is not None:
            scope_payload: dict[str, Any] = {}
            if workspace:
                scope_payload["project_path"] = workspace
            if access_mode:
                scope_payload["access_mode"] = access_mode
            try:
                validate_workspace_scope_payload(
                    scope_payload,
                    default_workspace=resolver.default_workspace,
                    default_restrict_to_workspace=resolver.default_restrict_to_workspace,
                )
            except WorkspaceScopeError as e:
                return _error_json(400, f"Invalid workspace scope: {e}")
            request_metadata = {WORKSPACE_SCOPE_METADATA_KEY: scope_payload}

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
            **_kw: Any,
        ) -> None:
            # [LOCAL PATCH] 将 Agent 进度事件注入 SSE 流（delta.nanobot_event），
            # 供客户端消息流渲染「深度思考 / 工具执行」过程时间线。
            # tool_hint 文本行由客户端按 tool_events 自行渲染，此处不下发。
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

        async def _run() -> None:
            nonlocal stream_failed
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

        if not stream_failed:
            await resp.write(_sse_chunk("", reported_model, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
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
        _chat_completion_response(response_text, reported_model, getattr(agent_loop, "_last_usage", None))
    )


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models

    [LOCAL PATCH] sciherd-cloud-smartagent：模型列表来源由「单一默认模型」
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop: "AgentLoop",
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    api_key: str = "",
    prepare_agent: Callable[[], Awaitable[None]] | None = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
        api_key: Optional API key for Bearer-token authentication on API routes.
        prepare_agent: Optional application-owned readiness callback run before each turn.
    """
    app = web.Application(client_max_size=20 * 1024 * 1024)  # 20MB for base64 images
    app[_AGENT_LOOP_KEY] = agent_loop
    app[_MODEL_NAME_KEY] = model_name
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
    app.router.add_get("/health", handle_health)
    return app
