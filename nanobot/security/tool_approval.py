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

#: [LOCAL PATCH] nanowork：「本会话内始终允许该工具」的裁决词。
#:
#: 它不是「一次性放行」的同义词，也不该被当成 ``allow`` 的别名——客户端的
#: 「允许本次」与「本会话始终允许」是两个按钮，若在协议层合并，UI 就再也
#: 说不出「我这次的意思是要不要继续问我」。
#:
#: 语义边界刻意收在**工具名**这一维，不扩成「全部工具」：一次点击只该豁免
#: 用户看得见的那个工具（``exec`` 不等于 ``write_file``）。跨会话同样不生效
#: ——``workspace_scope`` 是按请求下发的，而豁免集合挂在进程内的会话键上，
#: 下一个会话自然是一张白纸。
ApprovalVerdict = Literal["allow", "deny", "always"]


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

    def requires_approval(
        self,
        tool_name: str,
        *,
        always_allowed: frozenset[str] = frozenset(),
    ) -> bool:
        """Return whether ``tool_name`` must be confirmed before it executes.

        ``always_allowed`` carries the per-session exemptions earned through an
        ``always`` verdict. It sits **after** ``require``: an operator-declared
        ``require`` is a standing "always ask" instruction, so a runtime click
        must not be able to punch through it.
        """
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
            # ``require`` 是运维声明的常设指令，**排在会话豁免之前**：它表达的
            # 是「这个工具必须每次都问」，而用户的一次运行时点击不该能把它
            # 穿透——那会让「强制审批」变成一句可被点击绕过的口号。
            return True
        if name in always_allowed:
            # [LOCAL PATCH] 本会话内用户点过「始终允许」的工具，不再重复打扰。
            return False
        return name not in READ_ONLY_TOOLS


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One tool call waiting for a verdict."""

    request_id: str
    tool: str
    args: dict[str, Any]
    session_key: str | None = None
    call_id: str | None = None
    #: [LOCAL PATCH] FR-6.5：本次裁决的等待上限（秒）。
    #:
    #: 必须与 ``ToolApprovalHook`` 的 ``asyncio.wait_for`` 用**同一个**值：IM 卡片
    #: 会把它写成「N 分钟内回复」，两处各写一个数就会让卡片承诺的时间与实际拒绝
    #: 时间不一致——用户在第 4 分钟点「允许」，却已经被静默拒绝了。
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def payload(self) -> dict[str, Any]:
        """Wire form sent to the client over ``delta.nanobot_event``."""
        return {
            "id": self.request_id,
            "tool": self.tool,
            "args": self.args,
            **({"call_id": self.call_id} if self.call_id else {}),
            # 增量字段：老客户端（桌面）忽略它；IM 卡片靠它显示 TTL。
            "timeout_seconds": int(self.timeout_seconds),
        }


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """A verdict, plus why — the reason is fed back to the model on refusal."""

    verdict: ApprovalVerdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        # ``always`` 也是放行——两者的差别只在「下次还问不问」，而不是「这次
        # 做不做」。漏掉这一支会让「始终允许」当场变成拒绝（用户看到的是
        # 「点了允许，操作却没发生」，比直接不给按钮更糟）。
        return self.verdict in ("allow", "always")


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
    #: [LOCAL PATCH] 会话键 → 已被「始终允许」豁免的工具名。
    #:
    #: 与 ``_pending`` 同住一个对象不是图省事：两者的生命周期**必须**同一处
    #: 收口。豁免集合若单独找地方存，`cancel_session`（会话取消 / 引擎判超时）
    #: 就只会清掉待裁决项，留下一份「上一个会话点过始终允许」的幽灵豁免，
    #: 而它在下一次同名会话里是静默生效的——用户看不见，也查不出来。
    _always: dict[str, set[str]] = field(default_factory=dict)

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
        """Deliver a verdict. Returns False if the request is unknown or gone.

        An ``always`` verdict additionally records a tool-level exemption for the
        request's session **before** waking the turn — if the exemption were
        written after, the turn could race ahead and re-ask for the very tool
        the user just stopped being asked about.
        """
        entry = self._pending.pop(request_id, None)
        if entry is None or entry.future.done():
            return False
        if decision.verdict == "always":
            self.grant_always(entry.request.session_key, entry.request.tool)
        entry.future.set_result(decision)
        return True

    def grant_always(self, session_key: str | None, tool_name: str) -> None:
        """Remember that ``tool_name`` no longer needs a verdict in this session.

        A missing session key is a no-op rather than a key of ``""``: an empty
        bucket would be shared by every keyless (e.g. ephemeral) turn, turning a
        single click into a process-wide grant.
        """
        if not session_key:
            return
        name = (tool_name or "").strip()
        if not name:
            return
        self._always.setdefault(session_key, set()).add(name)

    def always_allowed(self, session_key: str | None) -> frozenset[str]:
        """Tools already exempted for this session (empty for a keyless turn)."""
        if not session_key:
            return frozenset()
        return frozenset(self._always.get(session_key, ()))

    def discard(self, request_id: str) -> None:
        """Drop a request without a verdict (timeout / cancellation path)."""
        self._pending.pop(request_id, None)

    def cancel_session(self, session_key: str | None) -> int:
        """Drop every request belonging to a session; returns how many.

        Also forgets the session's ``always`` grants: "this session" ends here,
        and carrying them over would silently widen the next session's policy.
        """
        if not session_key:
            return 0
        self._always.pop(session_key, None)
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

    def pending_for_session(self, session_key: str | None) -> list[ApprovalRequest]:
        """[LOCAL PATCH] FR-6.5：某会话当前挂着的待裁决请求。

        IM 里用户只会回一句「允许」/「拒绝」，不会（也不该）背下 ``appr-xxxx``
        全编号。因此裁决入口需要按**会话**找回「刚才问的是哪一条」。返回列表
        而不是单条：若同会话真的有多条在等，调用方必须要求用户带编号，不能猜。
        """
        if not session_key:
            return []
        return [
            entry.request
            for entry in self._pending.values()
            if entry.request.session_key == session_key
        ]

    def find_by_token(self, token: str) -> ApprovalRequest | None:
        """[LOCAL PATCH] FR-6.5：按全编号或**唯一后缀**找回请求。

        卡片上只印短编号（``abc123``），用户可能手打也可能点按钮带回。为了不
        把「短编号」做成两套寻址，这里统一支持「全等」与「后缀匹配」；后缀短于
        4 位或命中多条一律视为无法定位（返回 None）——宁可让用户重发，也不能
        把裁决投给猜出来的那条。
        """
        needle = (token or "").strip().lower()
        if not needle:
            return None
        if needle.startswith("appr-"):
            needle = needle[len("appr-") :]
        matches = [
            entry.request
            for request_id, entry in self._pending.items()
            if request_id.lower() == f"appr-{needle}"
            or (len(needle) >= 4 and request_id.lower().endswith(needle))
        ]
        return matches[0] if len(matches) == 1 else None

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def new_request_id() -> str:
    return f"appr-{uuid.uuid4().hex[:12]}"


#: Process-wide registry. The serve process has a single event loop, so one
#: registry is enough to connect the agent turn with the HTTP endpoint.
PENDING_APPROVALS = PendingApprovals()
