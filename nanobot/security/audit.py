"""[LOCAL PATCH] nanowork：工具网关的 append-only 审计留痕（FR-8.2 决策留痕 / FR-8.9）。

每一次工具调用在**执行前**的唯一收口点
（``nanobot/agent/tools/execution.py::_execute_tool_call``）留下一条记录：
哪个会话、何时、调了什么工具（参数摘要 + 参数哈希）、网关给了什么档位、
最终是被执行还是被拒、以及是不是用户逐条批准的。

四条不变量（改这块前必读）：

1. **append-only**：只向 ``<data_dir>/audit/tool-gateway.ndjson`` 追加 JSON 行，
   从不改写、从不删除既有行。审计日志的价值全在「不可篡改」上，一旦允许回写，
   它就从证据退化成缓存。
2. **绝不影响执行**：写日志失败一律吞掉——与 ``security/trash.py`` 的删除留痕
   同一条原则。守卫的职责是保护数据，不是保证日志；为了写日志而让工具调用
   失败是本末倒置。
3. **含命令哈希**：``args_digest`` 是参数规范化 JSON 的 sha256。参数摘要会被
   截断（``write_file`` 的 content 可能很大），哈希则不会——两者配合才能在
   摘要不完整时仍把同一条指令关联起来。
4. **可重定向、可关闭**：``NANOBOT_TOOL_AUDIT_LOG`` 设为路径即改写落点
   （nanowork 客户端把它指到 ``<userData>/audit/``）；显式设为空串即关闭
   （供不需要留痕的嵌入式场景使用）。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

#: 网关档位：与 PRD 11.1 的决策优先级表一一对应。
#: - ``allow``  ：只读调用，无需裁决，直接执行。
#: - ``sandbox``：允许执行，但强制走沙箱路径（删除类 → 系统回收站，绝不真删）。
#: - ``deny``   ：**不自动放行**。要么等用户逐条批准，要么在无审批通道 /
#:   模式禁止 / 超时下被拒。它描述的是「网关没有自行放行」，不是「已拒绝」——
#:   最终结果看 ``outcome``。
AuditTier = Literal["allow", "sandbox", "deny"]

#: 实际结果。
#: - ``executed``：工具真的跑了。
#: - ``refused`` ：被网关拦下，未执行。
#: - ``error``   ：跑了但抛错 / 返回错误结果。
AuditOutcome = Literal["executed", "refused", "error"]

#: 放行来源：用户逐条批准 / 只读档位无需批准 / 无（被拒）。
AuditApprovedBy = Literal["user", "read_only", "policy", "none"]

AUDIT_LOG_ENV = "NANOBOT_TOOL_AUDIT_LOG"

#: 参数摘要的字符上限，与审批弹窗的 ``_MAX_ARG_CHARS`` 同一考虑：
#: 审计行要能被人读，不能被一个文件内容撑爆。
_MAX_VALUE_CHARS = 400
_MAX_ARGS = 20

_resolved_log_path: Path | None | bool = False
"""``False`` = 尚未解析；``None`` = 已解析且关闭。"""


def _default_log_path() -> Path:
    # 延迟导入：``nanobot.config.paths`` 会反向依赖 config loader，模块级
    # 导入容易在启动早期形成环。
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("audit") / "tool-gateway.ndjson"


def audit_log_path(*, env: dict[str, str] | None = None) -> Path | None:
    """Resolve the audit log path, or ``None`` when auditing is disabled.

    Resolved once per process: the answer cannot change mid-run (the client
    sets the env var before spawning ``nanobot serve``), and caching keeps
    ``_execute_tool_call`` free of a directory probe on every tool call.
    """
    global _resolved_log_path
    if _resolved_log_path is not False:
        return cast(Path | None, _resolved_log_path)

    environ = os.environ if env is None else env
    raw = environ.get(AUDIT_LOG_ENV)
    if raw is not None and not raw.strip():
        # Explicitly disabled.
        _resolved_log_path = None
    elif raw:
        _resolved_log_path = Path(raw).expanduser()
    else:
        try:
            _resolved_log_path = _default_log_path()
        except Exception:
            _resolved_log_path = None
    return cast(Path | None, _resolved_log_path)


def reset_audit_log_path_cache() -> None:
    """Drop the cached path (tests that swap ``NANOBOT_TOOL_AUDIT_LOG``)."""
    global _resolved_log_path
    _resolved_log_path = False


def canonical_args(args: Any) -> str:
    """Serialize tool arguments into a stable, comparable string."""
    if not isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps(
        cast(dict[str, Any], args),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def digest_args(args: Any) -> str:
    """Return the sha256 of the canonical argument JSON (the command hash)."""
    return hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()


def summarize_args(args: Any) -> dict[str, Any]:
    """Render tool arguments into a small, JSON-safe dict for the log line."""
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in list(cast(dict[str, Any], args).items())[:_MAX_ARGS]:
        if value is None or isinstance(value, (int, float, bool)):
            out[str(key)] = value
            continue
        text = value if isinstance(value, str) else str(value)
        if len(text) > _MAX_VALUE_CHARS:
            text = text[:_MAX_VALUE_CHARS] + "…"
        out[str(key)] = text
    return out


def _text(value: Any) -> str:
    """Coerce an audit field to text.

    The record goes straight to ``json.dumps``, so **every** field has to be
    JSON-serializable. This is not cosmetic: ``nanobot/tests/agent/test_dream.py``
    驱动执行链时用的是 mock 会话作用域，`session_key` 会是一个 ``MagicMock``
    ——truthy 所以被写进记录，随后 ``json.dumps`` 抛 ``TypeError`` 并**从工具
    网关一路上抛**，把工具的调用结果改成了失败。不变量 2 说的是「审计绝不影响
    执行」，所以**记录构造本身就必须是全函数**，不能只靠写盘那一步兜底。
    """
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def build_audit_record(
    *,
    tool: str,
    tier: AuditTier,
    outcome: AuditOutcome,
    approved_by: AuditApprovedBy = "none",
    session_key: str | None = None,
    channel: str | None = None,
    args: Any = None,
    request_id: str | None = None,
    reason: str = "",
    approval_mode: str | None = None,
    agent_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one audit record. Pure — no IO, so it can be asserted on directly."""
    record: dict[str, Any] = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "tool": _text(tool),
        "tier": tier,
        "outcome": outcome,
        "approved_by": approved_by,
        "args_digest": digest_args(args),
        "args": summarize_args(args),
    }
    if session_key:
        record["session_key"] = _text(session_key)
    if channel:
        record["channel"] = _text(channel)
    if agent_id:
        # FR-8.9 要求回答「对哪个 Agent」——这正是「越权调用」审计的检索键：
        # 没有它，`tier=deny + outcome=refused` 只能说「有人被拦了」，
        # 说不了「是谁被拦了」。
        record["agent_id"] = _text(agent_id)
    if approval_mode:
        # 会话的 ``tool_approval`` 轴。``tier=deny + outcome=executed +
        # approval_mode=ask`` 就是「用户逐条批准了这次调用」的完整信号，
        # 不需要额外再猜。
        record["approval_mode"] = _text(approval_mode)
    if request_id:
        record["request_id"] = _text(request_id)
    if reason:
        record["reason"] = _text(reason)[:400]
    return record


def append_audit_record(
    record: dict[str, Any],
    *,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Append one audit line. Never raises — auditing must not break execution."""
    target = log_path if log_path is not None else audit_log_path(env=env)
    if target is None:
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=False,
                    # 兜底第二道：即使调用方绕过 ``build_audit_record`` 直接塞进
                    # 一个自定义对象，也退化成它的 repr 而不是整行丢失。
                    default=str,
                )
                + "\n"
            )
    except Exception:
        # 刻意写宽而不是只捕 ``OSError``：实测 ``json.dumps`` 的 ``TypeError``
        # 曾从这个函数逃出去，把工具调用打成了失败（见 ``_text`` 的注释）。
        # 契约是「审计失败绝不影响执行」，所以这里的正确答案是吞掉一切并继续。
        return


def record_tool_decision(
    *,
    tool: str,
    tier: AuditTier,
    outcome: AuditOutcome,
    approved_by: AuditApprovedBy = "none",
    session_key: str | None = None,
    channel: str | None = None,
    args: Any = None,
    request_id: str | None = None,
    reason: str = "",
    approval_mode: str | None = None,
    agent_id: str | None = None,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Build and append in one step — the call site used by the execution path."""
    append_audit_record(
        build_audit_record(
            tool=tool,
            tier=tier,
            outcome=outcome,
            approved_by=approved_by,
            session_key=session_key,
            channel=channel,
            args=args,
            request_id=request_id,
            reason=reason,
            approval_mode=approval_mode,
            agent_id=agent_id,
        ),
        log_path=log_path,
        env=env,
    )
