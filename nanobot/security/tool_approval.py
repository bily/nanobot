"""Tool-call approval policy and the pending-request registry.

Two responsibilities, both deliberately free of aiohttp / agent-loop imports so
they can be unit-tested in isolation:

1. :class:`ToolApprovalPolicy` decides *which* tool calls need a human decision.
2. :class:`PendingApprovals` is the rendezvous between the agent turn that is
   blocked inside ``before_execute_tool`` and the HTTP endpoint that delivers
   the user's verdict.

[LOCAL PATCH] nanowork：逐条批准工具调用所依赖的策略与注册表。
工具真正执行在引擎进程内，客户端只能收 SSE 单向事件流，因此审批必须在这里
完成裁决，客户端仅承担「呈现 + 回传」。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from nanobot.security.audit import AuditTier
from nanobot.security.trash import is_delete_command

#: [LOCAL PATCH] nanowork FR-8.2：网关档位（allow / sandbox / deny）。
#: 直接复用审计记录里的同一个字面量定义，避免两处枚举各自漂移。
ToolDecisionTier = AuditTier

#: Tools that only read state. Everything else — including tools this module has
#: never heard of, e.g. ``mcp__*`` entries and future additions — requires a
#: decision. Failing safe matters more than avoiding an extra prompt here: a
#: denylist would silently wave through every tool added upstream.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "find_files",
        "grep",
        "web_search",
        "web_fetch",
        "search_sessions",
        "read_session",
        "list_sessions",
        "list_exec_sessions",
    }
)

#: [LOCAL PATCH] FR-8.3：删除类工具走 ``sandbox`` 档——放行，但强制进系统回收站，
#: 绝不真删（见 ``nanobot/security/trash.py``）。
SANDBOX_TOOLS: frozenset[str] = frozenset({"delete_file"})

#: 承载 shell 命令的工具名。
SHELL_TOOLS: frozenset[str] = frozenset({"exec"})

#: ``exec`` 的命令参数名（两种写法都出现在工具 schema 与历史调用里）。
_SHELL_COMMAND_KEYS: tuple[str, ...] = ("command", "cmd")

#: [LOCAL PATCH] 提交计划（FR-1.5）。它不是「副作用工具」，而是**把方案交给用户
#: 裁决**的动作——再叠一层逐条批准只会自相矛盾（用户要批准的就是这个调用本身）。
#: 因此它在豁免集里，同时也只在 Plan 模式下被允许（见 :func:`mode_blocks_tool`）。
PLAN_SUBMISSION_TOOLS: frozenset[str] = frozenset({"submit_plan"})

#: Default grace period for a human to answer before the call is refused.
DEFAULT_TIMEOUT_SECONDS = 300.0

ApprovalVerdict = Literal["allow", "deny"]


def mode_blocks_tool(session_mode: str, tool_name: str) -> bool:
    """[LOCAL PATCH] 执行模式是否**禁止**该工具调用（FR-1.4）。

    这是「做不了」而不是「要不要问」：Ask / Plan 是只读模式，写类工具连问都不问，
    直接拒绝并把原因回注给模型（引擎据此重新规划，见 10.2 的「软约束 + 硬兜底」）。

    纯函数、无 IO，因此真值表可以被穷举单测——权限逻辑最怕的就是「只在一处
    写对、在另一处写漏」。
    """
    mode = (session_mode or "").strip().lower()
    if mode not in {"ask", "plan"}:
        # craft（含未知值）：不介入，交给审批 / 工作空间边界等既有防线。
        # 注意未知值在这里放行是安全的——`_normalize_session_mode` 已在入口
        # 把非法值拒掉了，能走到这里的未知值只可能是测试或内部调用。
        return False
    name = (tool_name or "").strip()
    if name in READ_ONLY_TOOLS:
        return False
    if mode == "plan" and name in PLAN_SUBMISSION_TOOLS:
        # Plan 模式下「提交计划」正是该做的事，放行。
        return False
    return True


def _shell_command(arguments: Any) -> str:
    """Pull the command string out of an ``exec`` call's arguments."""
    if not isinstance(arguments, dict):
        return ""
    for key in _SHELL_COMMAND_KEYS:
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return ""


def decide_tier(tool_name: str, arguments: Any = None) -> tuple[ToolDecisionTier, str]:
    """[LOCAL PATCH] 把一次调用归入 ``allow`` / ``sandbox`` / ``deny``（FR-8.2）。

    纯函数、无 IO，因此可以逐条穷举单测——PRD 11.1 明确要求「决策逻辑必须可
    穷举单测，这是安全能力的底线」。

    规则（与 PRD 11.1 的决策优先级表同序）：

    1. 只读工具 → ``allow``：无需裁决，直接执行。
    2. ``delete_file`` → ``sandbox``：放行，但**强制**走系统回收站（FR-8.3）。
    3. ``exec`` 里的删除命令 → ``deny``：引擎侧连钩子都无法改写命令参数
       （``before_execute_tool`` 只能放行或否决），所以「把 ``rm`` 沙箱化成
       ``delete_file``」做不到，只能拦下并把模型指回 ``delete_file``。
    4. 其余（写文件 / 跑命令 / 未知工具 / ``mcp__*``）→ ``deny``。

    关键语义：``deny`` 是「**网关不自行放行**」，不是「已经拒绝」。第 4 类里
    绝大多数最终都会被用户逐条批准后执行（``tier=deny`` + ``outcome=executed``）；
    落在 ``sandbox`` 档的调用同样可能先经用户批准，再走回收站路径。

    Returns:
        (档位, 一句话理由)。理由用于审计记录，不进提示词。
    """
    name = (tool_name or "").strip()
    if not name:
        return "deny", "nameless tool call cannot be classified"
    if name in READ_ONLY_TOOLS:
        return "allow", "read-only tool"
    if name in SANDBOX_TOOLS:
        return "sandbox", "delete-class call is routed to the system recycle bin"
    if name in SHELL_TOOLS and is_delete_command(_shell_command(arguments)):
        return "deny", "shell delete is intercepted; use the delete_file tool"
    return "deny", "side-effecting call requires an explicit verdict"


@dataclass(frozen=True, slots=True)
class ToolApprovalPolicy:
    """Decides whether a tool call needs an explicit human verdict.

    ``exempt`` always wins over ``require`` so an operator can carve a specific
    tool back out of the ask path without having to enumerate everything else.
    """

    exempt: frozenset[str] = frozenset()
    require: frozenset[str] = frozenset()
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def requires_approval(self, tool_name: str) -> bool:
        """Return whether ``tool_name`` must be confirmed before it executes."""
        name = (tool_name or "").strip()
        if not name:
            # A nameless call cannot be classified; refuse to run it unattended.
            return True
        if name in PLAN_SUBMISSION_TOOLS:
            # [LOCAL PATCH] 提交计划是「把方案交给用户裁决」这个动作本身，
            # 再让它走一次逐条批准就成了「先批准我要给你一份待批准的计划」。
            return False
        if name in self.exempt:
            return False
        if name in self.require:
            return True
        return name not in READ_ONLY_TOOLS


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One tool call waiting for a verdict."""

    request_id: str
    tool: str
    args: dict[str, Any]
    session_key: str | None = None
    call_id: str | None = None

    def payload(self) -> dict[str, Any]:
        """Wire form sent to the client over ``delta.nanobot_event``."""
        return {
            "id": self.request_id,
            "tool": self.tool,
            "args": self.args,
            **({"call_id": self.call_id} if self.call_id else {}),
        }


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """A verdict, plus why — the reason is fed back to the model on refusal."""

    verdict: ApprovalVerdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


@dataclass(slots=True)
class _Pending:
    request: ApprovalRequest
    future: asyncio.Future[ApprovalDecision]


@dataclass(slots=True)
class PendingApprovals:
    """Registry of in-flight approval requests.

    The agent turn registers a future and awaits it; the HTTP endpoint resolves
    it from another task on the same event loop. Every exit path — verdict,
    timeout, turn cancellation — must discard the entry, so registration is
    always paired with :meth:`discard` in a ``finally`` block by the caller.
    """

    _pending: dict[str, _Pending] = field(default_factory=dict)

    def register(
        self,
        request: ApprovalRequest,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Future[ApprovalDecision]:
        """Create and store the future the turn will wait on."""
        running_loop = loop or asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = running_loop.create_future()
        self._pending[request.request_id] = _Pending(request=request, future=future)
        return future

    def resolve(
        self,
        request_id: str,
        decision: ApprovalDecision,
    ) -> bool:
        """Deliver a verdict. Returns False if the request is unknown or gone."""
        entry = self._pending.pop(request_id, None)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(decision)
        return True

    def discard(self, request_id: str) -> None:
        """Drop a request without a verdict (timeout / cancellation path)."""
        self._pending.pop(request_id, None)

    def cancel_session(self, session_key: str | None) -> int:
        """Drop every request belonging to a session; returns how many."""
        if not session_key:
            return 0
        doomed = [
            rid
            for rid, entry in self._pending.items()
            if entry.request.session_key == session_key
        ]
        for rid in doomed:
            self._pending.pop(rid, None)
        return len(doomed)

    def get(self, request_id: str) -> ApprovalRequest | None:
        entry = self._pending.get(request_id)
        return entry.request if entry is not None else None

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def new_request_id() -> str:
    return f"appr-{uuid.uuid4().hex[:12]}"


#: Process-wide registry. The serve process has a single event loop, so one
#: registry is enough to connect the agent turn with the HTTP endpoint.
PENDING_APPROVALS = PendingApprovals()
