"""Workspace access scope and sandbox capability helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

WorkspaceAccessMode = Literal["restricted", "full"]
#: [LOCAL PATCH] 逐条批准工具调用的会话级开关。
#: - ``auto``：不介入（既有行为，仍有 denylist / workspace / SSRF 兜底）
#: - ``ask``：写类与副作用工具在执行前等待用户裁决，超时按拒绝处理
ToolApprovalMode = Literal["auto", "ask"]
#: [LOCAL PATCH] 执行模式（FR-1.4）。与上面两个开关**正交**：
#: - ``ask``：只读模式，任何写类工具直接拒绝（不是「问」，是「做不了」）
#: - ``plan``：先产出计划交用户审批，写类工具同样拒绝，直到切到 craft
#: - ``craft``：直接执行（仍受 access_mode / tool_approval 约束）
#: 强制点在 ``before_execute_tool``，不依赖提示词——提示词只是软约束。
SessionMode = Literal["ask", "plan", "craft"]
WORKSPACE_SCOPE_METADATA_KEY = "workspace_scope"
_ACCESS_MODES = {"restricted", "full"}
_TOOL_APPROVAL_MODES = {"auto", "ask"}
_SESSION_MODES = {"ask", "plan", "craft"}

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled", ""}
_PROVIDER_LABELS = {
    "none": "None",
    "unknown": "Unknown system sandbox",
    "macos_app_sandbox": "macOS App Sandbox",
    "bwrap": "Bubblewrap",
}

_CURRENT_WORKSPACE_SCOPE: ContextVar["WorkspaceScope | None"] = ContextVar(
    "nanobot_workspace_scope",
    default=None,
)


class WorkspaceScopeError(ValueError):
    """Raised when a requested WebUI workspace scope is invalid."""

    status = 400

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class WorkspaceSandboxStatus:
    """Resolved workspace sandbox state for runtime display and tooling."""

    restrict_to_workspace: bool
    workspace_root: str
    level: str
    enforced: bool
    provider: str
    provider_label: str
    summary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "restrict_to_workspace": self.restrict_to_workspace,
            "workspace_root": self.workspace_root,
            "level": self.level,
            "enforced": self.enforced,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class WorkspaceScope:
    """Effective project root and access mode for one agent turn."""

    project_path: Path
    access_mode: WorkspaceAccessMode
    restrict_to_workspace: bool
    sandbox_status: WorkspaceSandboxStatus
    source_channel: str | None = None
    #: [LOCAL PATCH] 承认「执行策略」也包含「要不要逐条批准」，与访问模式同一处承载，
    #: 随会话元数据持久化并可经 contextvar 在工具执行点读到。
    tool_approval: ToolApprovalMode = "auto"
    #: [LOCAL PATCH] 执行模式（ask / plan / craft）。同样住在这里，理由同上：
    #: 工具执行点只能从 contextvar 拿到 scope，模式约束必须随 scope 一起送达。
    session_mode: SessionMode = "craft"
    #: [LOCAL PATCH] nanowork FR-2.2「Agent 工具白名单」。
    #: ``None`` = 该 Agent 没有声明白名单（不限制）；非空集合 = **只有**这些
    #: 工具可用（最小权限）。空集合与 ``None`` 语义不同：前者是「一个都不许」，
    #: 后者是「没限制」。区分它们很重要——把「配置成空」误读成「不限制」会让
    #: 一次手误变成完全放开。
    tool_allow: frozenset[str] | None = None
    #: 黑名单**永远优先于**白名单：一个工具同时出现在两边时应被拒绝。
    #: 这条优先级让运维可以在一个宽白名单上精确挖掉个别危险工具。
    tool_deny: frozenset[str] = frozenset()
    #: [LOCAL PATCH] FR-2.2：本会话所用的 Agent 标识，仅用于让拒绝原因能指名道姓
    #: （「哪个 Agent 没被授予这个工具」），不参与任何判定。
    agent_id: str | None = None

    @property
    def project_name(self) -> str:
        return self.project_path.name or str(self.project_path)

    def metadata(self) -> dict[str, str]:
        return {
            "project_path": str(self.project_path),
            "access_mode": self.access_mode,
            "tool_approval": self.tool_approval,
            "session_mode": self.session_mode,
        }

    def tool_policy_dict(self) -> dict[str, Any]:
        """[LOCAL PATCH] FR-2.2：工具策略的线上形态（仅在有约束时出现）。"""
        out: dict[str, Any] = {}
        if self.tool_allow is not None:
            out["tool_allow"] = sorted(self.tool_allow)
        if self.tool_deny:
            out["tool_deny"] = sorted(self.tool_deny)
        return out

    def payload(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "project_name": self.project_name,
            "restrict_to_workspace": self.restrict_to_workspace,
            "sandbox_status": self.sandbox_status.as_dict(),
            **self.tool_policy_dict(),
        }


@dataclass(frozen=True)
class ToolWorkspace:
    """Workspace policy resolved for a tool call."""

    project_path: Path | None
    restrict_to_workspace: bool
    scope: WorkspaceScope | None = None

    @property
    def allowed_root(self) -> Path | None:
        if self.restrict_to_workspace and self.project_path is not None:
            return self.project_path
        return None


@dataclass(frozen=True)
class WorkspaceScopeResolver:
    """Resolve the effective workspace scope at an agent turn boundary."""

    default_workspace: str | Path
    default_restrict_to_workspace: bool
    # [LOCAL PATCH] 允许携带会话级 workspace scope 的渠道集合：
    # websocket（WebUI 原有）+ api（本客户端按会话选择工作空间/权限）。
    scoped_channels: frozenset[str] = frozenset({"websocket", "api"})

    @property
    def sandbox_status(self) -> WorkspaceSandboxStatus:
        return self.default().sandbox_status

    def default(self) -> WorkspaceScope:
        return default_workspace_scope(
            self.default_workspace,
            self.default_restrict_to_workspace,
        )

    def for_message(
        self,
        msg: Any,
        session_metadata: Any,
    ) -> WorkspaceScope:
        return self.for_turn(
            channel=getattr(msg, "channel", None),
            message_metadata=getattr(msg, "metadata", None),
            session_metadata=session_metadata,
        )

    def for_turn(
        self,
        *,
        channel: str | None,
        message_metadata: Any,
        session_metadata: Any,
    ) -> WorkspaceScope:
        if channel not in self.scoped_channels:
            return self.default()
        return resolve_effective_workspace_scope(
            message_metadata=message_metadata,
            session_metadata=session_metadata,
            default_workspace=self.default_workspace,
            default_restrict_to_workspace=self.default_restrict_to_workspace,
            source_channel=channel,
        )

    def persist_message_scope(self, session: Any, msg: Any) -> None:
        if getattr(msg, "channel", None) not in self.scoped_channels:
            return
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata_data = cast(dict[str, Any], metadata)
        raw = metadata_data.get(WORKSPACE_SCOPE_METADATA_KEY)
        if isinstance(raw, dict):
            session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = dict(cast(dict[str, Any], raw))


def workspace_sandbox_status(
    *,
    restrict_to_workspace: bool,
    workspace: str | Path,
    environ: dict[str, str] | None = None,
) -> WorkspaceSandboxStatus:
    """Return how workspace restriction is enforced in the current host."""

    workspace_root = str(Path(workspace).expanduser().resolve(strict=False))
    provider = _env_system_provider(environ)
    if not restrict_to_workspace:
        return WorkspaceSandboxStatus(
            restrict_to_workspace=False,
            workspace_root=workspace_root,
            level="off",
            enforced=False,
            provider="none",
            provider_label=_provider_label("none"),
            summary="Workspace restriction is disabled.",
        )

    if provider:
        label = _provider_label(provider)
        return WorkspaceSandboxStatus(
            restrict_to_workspace=True,
            workspace_root=workspace_root,
            level="system",
            enforced=True,
            provider=provider,
            provider_label=label,
            summary=f"Workspace restriction is system-enforced by {label}.",
        )

    return WorkspaceSandboxStatus(
        restrict_to_workspace=True,
        workspace_root=workspace_root,
        level="application",
        enforced=False,
        provider="none",
        provider_label=_provider_label("none"),
        summary="Workspace restriction uses nanobot application-level guards.",
    )


def default_access_mode(restrict_to_workspace: bool) -> WorkspaceAccessMode:
    return "restricted" if restrict_to_workspace else "full"


def _normalize_tool_approval(raw: str) -> ToolApprovalMode:
    """[LOCAL PATCH] 归一审批开关；未知取值一律退回 ``auto``（不静默开启拦截）。"""
    mode = (raw or "").strip().lower()
    if mode in _TOOL_APPROVAL_MODES:
        return cast(ToolApprovalMode, mode)
    if mode:
        raise WorkspaceScopeError(
            f"tool_approval must be one of {sorted(_TOOL_APPROVAL_MODES)}, got {raw!r}"
        )
    return "auto"


def _normalize_session_mode(raw: str) -> SessionMode:
    """[LOCAL PATCH] 归一执行模式。

    与 ``tool_approval`` 的容错方向**相反**：未知取值一律报错而不是退回默认。
    审批开关退化成 ``auto`` 只是少问一句；模式退化成 ``craft`` 则是把一个
    「只读」请求悄悄变成「可写」——这是权限方向上的错误，必须显式失败。
    """
    mode = (raw or "").strip().lower()
    if mode in _SESSION_MODES:
        return cast(SessionMode, mode)
    if mode:
        raise WorkspaceScopeError(
            f"session_mode must be one of {sorted(_SESSION_MODES)}, got {raw!r}"
        )
    return "craft"


def normalize_tool_names(raw: Any, *, field: str) -> frozenset[str] | None:
    """[LOCAL PATCH] FR-2.2：把声明的工具名列表归一成集合。

    ``None`` 表示**没有声明**（不限制），``frozenset()`` 表示**声明为空**
    （一个都不许）。调用方必须保持这个区别——见 ``WorkspaceScope.tool_allow``。

    容错方向与 ``session_mode`` 一致：坏输入报错而不是静默放宽。工具白名单
    是权限边界，把「配置写坏了」读成「不限制」等于把边界拆掉。
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        # 单个字符串是常见手写形态；按逗号切分以免出现「一个名字是 a,b 的工具」。
        raw = [part for part in raw.split(",") if part.strip()]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise WorkspaceScopeError(f"{field} must be a list of tool names")

    names: set[str] = set()
    for item in cast(Iterable[Any], raw):
        if not isinstance(item, str):
            raise WorkspaceScopeError(f"{field} entries must be strings")
        name = item.strip()
        if name:
            names.add(name)
    return frozenset(names)


def agent_blocks_tool(scope: WorkspaceScope | None, tool_name: str) -> str | None:
    """[LOCAL PATCH] FR-2.2：Agent 工具策略是否禁止该调用？禁止则返回原因。

    纯函数、无 IO，因此可以穷举单测——权限逻辑最怕「只在一处写对、另一处写漏」。

    规则：
    1. 黑名单命中 → 拒绝（**黑名单永远优先**，能在宽白名单上精确挖洞）。
    2. 有白名单且未命中 → 拒绝（最小权限，默认拒绝）。
    3. 其余 → 允许（``None`` 白名单 = 该 Agent 没有声明约束）。
    """
    if scope is None:
        return None
    name = (tool_name or "").strip()
    if not name:
        return "Tool name is empty, so the agent tool policy cannot allow it."

    if scope.tool_deny and name in scope.tool_deny:
        return (
            f"The agent '{scope.agent_id or 'current'}' denies the '{name}' tool "
            "by policy. Do not retry it; complete the task with the tools you have, "
            "or tell the user this capability was not granted."
        )
    if scope.tool_allow is None:
        return None
    if name in scope.tool_allow:
        return None
    return (
        f"The agent '{scope.agent_id or 'current'}' is limited to an explicit tool "
        f"whitelist that does not include '{name}'. Do not retry it; work with the "
        "granted tools or ask the user to widen the agent's permissions."
    )


def build_workspace_scope(
    project_path: str | Path,
    access_mode: str,
    *,
    source_channel: str | None = None,
    tool_approval: str = "auto",
    session_mode: str = "craft",
    tool_allow: Any = None,
    tool_deny: Any = None,
    agent_id: str | None = None,
) -> WorkspaceScope:
    mode = _normalize_access_mode(access_mode)
    root = Path(project_path).expanduser().resolve(strict=False)
    restrict = mode == "restricted"
    # [LOCAL PATCH] FR-2.2：白名单缺省是 None（不限制），黑名单缺省是空集。
    # 两者缺省不同不是笔误——见 WorkspaceScope.tool_allow 的说明。
    deny = normalize_tool_names(tool_deny, field="tool_deny") or frozenset()
    return WorkspaceScope(
        project_path=root,
        access_mode=mode,
        restrict_to_workspace=restrict,
        sandbox_status=workspace_sandbox_status(
            restrict_to_workspace=restrict,
            workspace=root,
        ),
        source_channel=source_channel,
        tool_approval=_normalize_tool_approval(tool_approval),
        session_mode=_normalize_session_mode(session_mode),
        tool_allow=normalize_tool_names(tool_allow, field="tool_allow"),
        tool_deny=deny,
        agent_id=(agent_id or None),
    )


def default_workspace_scope(
    workspace: str | Path,
    restrict_to_workspace: bool,
    *,
    source_channel: str | None = None,
) -> WorkspaceScope:
    return build_workspace_scope(
        workspace,
        default_access_mode(restrict_to_workspace),
        source_channel=source_channel,
    )


def validate_workspace_scope_payload(
    raw: Any,
    *,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """Validate a client-requested workspace scope."""
    if raw is None:
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )
    if not isinstance(raw, dict):
        raise WorkspaceScopeError("workspace_scope must be an object")
    scope_data = cast(dict[str, Any], raw)

    raw_path = scope_data.get("project_path") or scope_data.get("path")
    if raw_path is None or raw_path == "":
        raw_path = str(Path(default_workspace).expanduser().resolve(strict=False))
    if not isinstance(raw_path, str):
        raise WorkspaceScopeError("project_path must be a string")
    if "\0" in raw_path:
        raise WorkspaceScopeError("project_path contains invalid characters")

    project = Path(raw_path).expanduser()
    if not project.is_absolute():
        raise WorkspaceScopeError("project_path must be absolute")
    project = project.resolve(strict=False)
    if not project.is_dir():
        raise WorkspaceScopeError("project_path must be an existing directory")

    raw_mode = scope_data.get("access_mode")
    if raw_mode is None:
        raw_mode = default_access_mode(default_restrict_to_workspace)
    if not isinstance(raw_mode, str):
        raise WorkspaceScopeError("access_mode must be a string")

    raw_approval = scope_data.get("tool_approval")
    if raw_approval is None:
        raw_approval = "auto"
    if not isinstance(raw_approval, str):
        raise WorkspaceScopeError("tool_approval must be a string")

    # [LOCAL PATCH] 执行模式：缺席按 craft（历史会话 / 未启用该能力的客户端）。
    raw_session_mode = scope_data.get("session_mode")
    if raw_session_mode is None:
        raw_session_mode = "craft"
    if not isinstance(raw_session_mode, str):
        raise WorkspaceScopeError("session_mode must be a string")

    # [LOCAL PATCH] FR-2.2：Agent 工具策略。缺席 = 该 Agent 没有声明约束，
    # 与既有会话完全兼容（旧客户端不发这两个字段，行为不变）。
    raw_agent_id = scope_data.get("agent_id")
    if raw_agent_id is not None and not isinstance(raw_agent_id, str):
        raise WorkspaceScopeError("agent_id must be a string")

    return build_workspace_scope(
        project,
        raw_mode,
        source_channel=source_channel,
        tool_approval=raw_approval,
        session_mode=raw_session_mode,
        tool_allow=scope_data.get("tool_allow"),
        tool_deny=scope_data.get("tool_deny"),
        agent_id=raw_agent_id,
    )


def workspace_scope_from_metadata(
    metadata: Any,
    *,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    """Resolve persisted metadata, falling back safely for old or stale sessions."""
    if not isinstance(metadata, dict):
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )
    try:
        metadata_data = cast(dict[str, Any], metadata)
        return validate_workspace_scope_payload(
            metadata_data.get(WORKSPACE_SCOPE_METADATA_KEY),
            default_workspace=default_workspace,
            default_restrict_to_workspace=default_restrict_to_workspace,
            source_channel=source_channel,
        )
    except WorkspaceScopeError:
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=source_channel,
        )


def resolve_effective_workspace_scope(
    *,
    message_metadata: Any,
    session_metadata: Any,
    default_workspace: str | Path,
    default_restrict_to_workspace: bool,
    source_channel: str | None = None,
) -> WorkspaceScope:
    if isinstance(message_metadata, dict) and WORKSPACE_SCOPE_METADATA_KEY in message_metadata:
        message_metadata_data = cast(dict[str, Any], message_metadata)
        return workspace_scope_from_metadata(
            message_metadata_data,
            default_workspace=default_workspace,
            default_restrict_to_workspace=default_restrict_to_workspace,
            source_channel=source_channel,
        )
    return workspace_scope_from_metadata(
        session_metadata,
        default_workspace=default_workspace,
        default_restrict_to_workspace=default_restrict_to_workspace,
        source_channel=source_channel,
    )


def bind_workspace_scope(scope: WorkspaceScope) -> Token[WorkspaceScope | None]:
    return _CURRENT_WORKSPACE_SCOPE.set(scope)


def reset_workspace_scope(token: Token[WorkspaceScope | None]) -> None:
    _CURRENT_WORKSPACE_SCOPE.reset(token)


def current_workspace_scope() -> WorkspaceScope | None:
    return _CURRENT_WORKSPACE_SCOPE.get()


def current_tool_workspace(
    default_workspace: str | Path | None,
    *,
    restrict_to_workspace: bool = False,
    sandbox_restricts_workspace: bool = False,
) -> ToolWorkspace:
    """Return the workspace/access policy for the current tool call."""

    scope = current_workspace_scope()
    project_path = (
        scope.project_path
        if scope is not None
        else Path(default_workspace).expanduser() if default_workspace is not None else None
    )
    restrict = (
        scope.restrict_to_workspace
        if scope is not None
        else bool(restrict_to_workspace)
    ) or sandbox_restricts_workspace
    return ToolWorkspace(
        project_path=project_path,
        restrict_to_workspace=restrict,
        scope=scope,
    )


def current_scope_allows_loopback(*, enabled: bool) -> bool:
    """Return True when the current WebUI Full Access turn may touch loopback URLs."""

    scope = current_workspace_scope()
    return bool(
        enabled
        and scope is not None
        and scope.source_channel == "websocket"
        and scope.access_mode == "full"
        and not scope.restrict_to_workspace
    )


def _env_system_provider(environ: dict[str, str] | None = None) -> str | None:
    env = environ if environ is not None else os.environ
    explicit_provider = env.get("NANOBOT_WORKSPACE_SANDBOX_PROVIDER")
    enforced = env.get("NANOBOT_WORKSPACE_SANDBOX_ENFORCED")
    compatibility = env.get("NANOBOT_SANDBOX_ENFORCED")

    marker = enforced if enforced is not None else compatibility
    if marker is None:
        return None

    normalized_marker = marker.strip().lower()
    if normalized_marker in _FALSE_VALUES:
        return None
    if normalized_marker in _TRUE_VALUES:
        return _normalize_provider(explicit_provider)
    return _normalize_provider(marker)


def _normalize_provider(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "unknown"


def _provider_label(provider: str) -> str:
    if provider in _PROVIDER_LABELS:
        return _PROVIDER_LABELS[provider]
    return provider.replace("_", " ").title()


def _normalize_access_mode(value: str) -> WorkspaceAccessMode:
    mode = value.strip().lower().replace("_", "-")
    if mode == "restrict":
        mode = "restricted"
    if mode == "full-access":
        mode = "full"
    if mode not in _ACCESS_MODES:
        raise WorkspaceScopeError("access_mode must be restricted or full")
    return mode  # type: ignore[return-value]
