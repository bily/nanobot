"""IM（飞书 / 企微 / 钉钉…）内的工具审批桥。

[LOCAL PATCH] nanowork FR-6.5：把「逐条批准」从桌面弹窗延伸到聊天渠道。

**为什么需要这一层**：审批请求原本只经 SSE 推给桌面客户端，IM 会话的进度
回调不接受 ``approval_request``，于是 ``ToolApprovalHook`` 只能 fail-closed
——IM 里任何写类工具都会被判「无通道可问」而直接拒绝。要打开这条能力，缺的
不是策略，而是两段接线：

1. **出**：把审批请求渲染成渠道消息（工具名 + 参数摘要 + TTL）；
2. **入**：把用户的裁决（按钮点击或文本回复）送回 ``PendingApprovals``。

本模块只做「渲染」与「裁决解释」，不碰 aiohttp、不碰具体渠道 SDK，因此可以
在没有网络的情况下穷举单测。

**为什么裁决是进程内直投，而不是回调 ``POST /v1/tool-approvals/{id}``**：
IM 轮次跑在 **gateway 进程**（``gateway_runtime`` 里的 ``agent.run()``），而
该端点只存在于 **serve 进程**（``api/server.py``）。两个进程各有自己的
``PENDING_APPROVALS``，HTTP 调过去只会落到一张空表上——返回 404/409，而
gateway 里真正挂起的 future 继续等到超时。同一进程内直投 ``registry.resolve``
才是唯一能唤醒它的路径；协议面没有新增，仍然复用既有的 ``ApprovalDecision``。

裁决契约（刻意以**文本**为准）：
    允许 / 同意 / approve / allow      → allow
    拒绝 / 不同意 / deny / reject      → deny
    后缀可带短编号（卡片上的「编号」），用于同会话有多条待裁决时消歧。

为什么不把按钮值当唯一契约：``OutboundMessage.buttons`` 在各渠道只是**标签**
（Slack ``value=label``、Telegram ``callback_data=label``），飞书的 ``send``
更是完全不渲染按钮；把裁决绑在按钮 id 上，会让「飞书里根本点不到」。文本回复
在每家渠道都成立，按钮只是能力更强的渠道的加成。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.security.tool_approval import (
    DEFAULT_TIMEOUT_SECONDS,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    PendingApprovals,
)

#: 裁决词。只收「明确的授权/拒绝」，不收 ``yes`` / ``no`` 这类日常词——
#: 没有待裁决时它们会原样放行给模型，没必要把误伤面做大。
_ALLOW_WORDS: frozenset[str] = frozenset({"允许", "同意", "approve", "allow"})
_DENY_WORDS: frozenset[str] = frozenset({"拒绝", "不同意", "deny", "reject"})

#: 编号形状：``appr-`` + 4 位以上十六进制（卡片只印后 8 位）。
#: 不匹配的形状一律**不当编号**处理——否则「允许 一下」会被当成无效编号吞掉。
_TOKEN_RE = re.compile(r"^(?:appr-)?[0-9a-f]{4,}$")

#: 卡片里参数摘要的总长上限。逐条参数已在钩子里截过，这里再兜一层总量，
#: 避免 12 个 400 字的参数把一条 IM 消息撑成几千字。
_MAX_RENDERED_ARGS = 600

_LEADING_PUNCT = "/／"
_TRAILING_PUNCT = "。.!！,，;；:：（）()【】[]"


def _format_ttl(seconds: int) -> str:
    if seconds <= 0:
        return "立即"
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, rest = divmod(seconds, 60)
    if rest == 0:
        return f"{minutes} 分钟"
    return f"{minutes} 分 {rest} 秒"


def _render_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    # 压掉换行：IM 里一条带缩进的多行 JSON 会喧宾夺主，用户只需要认出「动哪个文件」。
    text = " ".join(text.split())
    if len(text) > _MAX_RENDERED_ARGS:
        text = text[:_MAX_RENDERED_ARGS] + "…"
    return text


def build_approval_card(payload: dict[str, Any]) -> tuple[str, list[list[str]]]:
    """把一条审批请求渲染成（正文, 按钮行）。

    正文里**必须**带短编号：同会话并发两条请求时，用户点按钮只能回一个
    「允许」，没有编号就无从判断该投给哪一条。
    """
    tool = str(payload.get("tool") or "unknown tool")
    request_id = str(payload.get("id") or "")
    try:
        ttl = int(payload.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        ttl = int(DEFAULT_TIMEOUT_SECONDS)

    lines = ["🔐 需要你授权一次工具调用", f"工具：{tool}"]
    rendered = _render_args(payload.get("args"))
    if rendered:
        lines.append(f"参数：{rendered}")
    lines.append(f"请在 {_format_ttl(ttl)} 内回复「允许」或「拒绝」。")

    short = request_id.split("-", 1)[-1][-8:] if request_id else ""
    if short:
        lines.append(f"编号：{short}")
    return "\n".join(lines), [["允许", "拒绝"]]


def parse_approval_verdict(text: str) -> tuple[ApprovalVerdict, str | None] | None:
    """把一条入站文本解释成（裁决, 编号）。不是裁决就返回 ``None``。

    只认**第一个词**：IM 用户常常顺手补一句解释（「允许，这个文件可以改」），
    把整段当命令会漏掉裁决；而只认第一个词又不会把普通聊天误判——首词不是
    裁决词时直接放行。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    line = raw.splitlines()[0].strip()
    # 冒号 / 中文逗号 / 顿号统一成分隔符：让「允许：abc123」「允许，这个文件可以改」
    # 与「允许 abc123」走同一条解释路径。漏掉逗号会让首词变成「允许，这个文件
    # 可以改」，整句既不是裁决词、也匹配不到编号。
    for separator in ("：", ":", "，", ",", "、"):
        line = line.replace(separator, " ")
    parts = line.split()
    if not parts:
        return None

    head = parts[0].lstrip(_LEADING_PUNCT).rstrip(_TRAILING_PUNCT).lower()
    if head in _ALLOW_WORDS:
        verdict: ApprovalVerdict = "allow"
    elif head in _DENY_WORDS:
        verdict = "deny"
    else:
        return None

    token: str | None = None
    if len(parts) > 1:
        candidate = parts[1].strip(_TRAILING_PUNCT).lower()
        # 形状像编号才认；「允许 一下」里的「一下」不是编号，按裸裁决处理。
        if candidate and _TOKEN_RE.match(candidate):
            token = candidate
    return verdict, token


def _decision(verdict: ApprovalVerdict) -> ApprovalDecision:
    # 拒绝理由留空：``ToolApprovalHook`` 已有统一的默认文案，两处各写一份
    # 迟早会分叉（模型看到的原因与用户看到的按钮含义对不上）。
    return ApprovalDecision(verdict=verdict)


def _ack_text(verdict: ApprovalVerdict) -> str:
    if verdict == "allow":
        return "已允许，继续执行。"
    return "已拒绝，该工具不会执行。"


class ApprovalInbox:
    """把 IM 入站消息里的裁决投递到待审批注册表。

    ``consume`` 返回 ``True`` 表示「这条消息已被审批消费，不应再进 Agent」。
    这一点必须由调用方尊重：否则裁决会被当成新一轮用户输入，模型会对着
    「拒绝」两个字开始自由发挥。
    """

    def __init__(
        self,
        registry: PendingApprovals,
        *,
        publish: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        duplicate_window_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._publish = publish
        self._duplicate_window = duplicate_window_seconds
        self._clock = clock
        #: 会话键 → (上一次被消费的裸裁决文本, 时间)。
        #:
        #: 只用于「没有待裁决时」判断重复点击：第一次点击把请求摘掉了，第二次
        #: 到达时注册表里已空，若原样放行就会变成一条莫名其妙的用户消息。
        #: 记住**文本**而不是只记时间，是为了不吞掉同一窗口内的普通发言。
        self._last_bare: dict[str, tuple[str, float]] = {}

    async def _reply(self, msg: InboundMessage, content: str) -> None:
        if self._publish is None or not content:
            return
        await self._publish(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=dict(msg.metadata or {}),
            )
        )

    def _remember(self, session_key: str, text: str) -> None:
        if not session_key:
            return
        if len(self._last_bare) >= 256:
            self._last_bare.clear()
        self._last_bare[session_key] = (text.strip(), self._clock())

    def _is_duplicate(self, session_key: str, text: str) -> bool:
        entry = self._last_bare.get(session_key)
        if entry is None:
            return False
        remembered, at = entry
        return remembered == text.strip() and (self._clock() - at) <= self._duplicate_window

    @staticmethod
    def _same_session(request: ApprovalRequest, session_key: str) -> bool:
        # 请求没有会话键 = 无主裁决，任何渠道都不该替它做主。
        return bool(request.session_key) and request.session_key == session_key

    async def consume(self, msg: InboundMessage, *, session_key: str) -> bool:
        parsed = parse_approval_verdict(msg.content)
        if parsed is None:
            return False
        verdict, token = parsed
        text = (msg.content or "").strip()

        if token:
            request = self._registry.find_by_token(token)
            if request is None:
                await self._reply(msg, "这条审批请求不存在或已过期。")
                return True
            if not self._same_session(request, session_key):
                # 别的会话（或别的群成员）想替这条请求拍板：不消耗它，
                # 让真正该裁决的会话仍有机会处理。
                await self._reply(msg, "这条审批请求不属于当前会话，已忽略。")
                return True
            if self._registry.resolve(request.request_id, _decision(verdict)):
                self._remember(session_key, text)
                await self._reply(msg, _ack_text(verdict))
            else:
                await self._reply(msg, "这条审批请求已处理或已过期。")
            return True

        pending = self._registry.pending_for_session(session_key)
        if not pending:
            if self._is_duplicate(session_key, text):
                await self._reply(msg, "这条审批请求已处理或已过期。")
                return True
            # 没有待裁决、也不是重复点击：这只是一句普通发言，放行给模型。
            return False
        if len(pending) > 1:
            await self._reply(
                msg,
                f"当前有 {len(pending)} 条待审批请求，请回复「允许 <编号>」"
                "或「拒绝 <编号>」以指定是哪一条。",
            )
            return True

        request = pending[0]
        if self._registry.resolve(request.request_id, _decision(verdict)):
            self._remember(session_key, text)
            await self._reply(msg, _ack_text(verdict))
        else:
            # 与本次入站并发地超时/取消：注册表已空，按「已处理」回执。
            await self._reply(msg, "这条审批请求已处理或已过期。")
        return True
