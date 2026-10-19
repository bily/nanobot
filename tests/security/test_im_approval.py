"""[LOCAL PATCH] nanowork FR-6.5：IM 内审批的裁决契约。

## 为什么这一层必须单独钉

审批的「出」与「入」是两条完全不同的失败路径，而且**都不会报错**：

* 出：卡片包成 ``ProgressEvent`` 会被渠道的 ``send_progress`` 静音，用户永远
  看不到，工具只能等到超时被拒——日志里只有一条「用户没响应」。
* 入：裁决若原样放行进 Agent，它会变成新一轮用户输入，模型对着「拒绝」两个字
  自由发挥，而挂起的工具继续等到超时。

所以这里逐条钉住：裁决词解释、编号消歧、会话归属、重放/重复点击、以及
「审批卡不经过 ProgressEvent」。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# 先初始化 agent 包：``utils.progress_events`` 在模块级 import ``agent.hook``，
# 若它先于 ``agent.__init__`` 被导入，会触发已知的包初始化环（context → hooks →
# file_edit_activity → progress_events 自引用）。引擎启动时天然是这个顺序，
# 单测里需要显式还原它。
import nanobot.agent  # noqa: F401
from nanobot.agent import context as agent_context
from nanobot.agent.hook import AgentHookContext
from nanobot.agent.hooks.tool_approval import ToolApprovalHook
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import build_bus_progress_callback
from nanobot.providers.base import ToolCallRequest
from nanobot.security.im_approval import (
    ApprovalInbox,
    build_approval_card,
    parse_approval_verdict,
)
from nanobot.security.tool_approval import (
    ApprovalRequest,
    PendingApprovals,
)
from nanobot.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    channel_tool_approval_modes,
    reset_workspace_scope,
)
from nanobot.utils.progress_events import on_progress_accepts_approval_requests

SESSION = "feishu:chat-1"


# --------------------------------------------------------------------------
# 1. 裁决词解释：只认首词，只认明确词
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("允许", ("allow", None)),
        ("同意", ("allow", None)),
        ("拒绝", ("deny", None)),
        ("不同意", ("deny", None)),
        ("allow", ("allow", None)),
        ("approve", ("allow", None)),
        ("deny", ("deny", None)),
        ("/approve appr-abcd1234", ("allow", "appr-abcd1234")),
        ("拒绝：abcd1234", ("deny", "abcd1234")),
        # 首词是裁决，后面的解释不参与解释——「允许 一下」不该被当成无效编号吞掉。
        ("允许 一下", ("allow", None)),
        ("允许，这个文件可以改", ("allow", None)),
        # 只在第一行取命令：后面是聊天内容。
        ("拒绝\n顺便问一下天气", ("deny", None)),
    ],
)
def test_parse_approval_verdict(text: str, expected: tuple[str, str | None]) -> None:
    assert parse_approval_verdict(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "今天天气不错", "yes", "no", "好"])
def test_parse_approval_verdict_ignores_non_verdicts(text: str) -> None:
    """日常词不当裁决：没有待裁决时它们要原样放行给模型。"""
    assert parse_approval_verdict(text) is None


# --------------------------------------------------------------------------
# 2. 卡片渲染：工具、参数、TTL、短编号、按钮
# --------------------------------------------------------------------------


def test_build_approval_card_carries_the_decision_context() -> None:
    content, buttons = build_approval_card(
        {
            "id": "appr-abcdef123456",
            "tool": "write_file",
            "args": {"path": "docs/a.md", "content": "hi"},
            "timeout_seconds": 300,
        }
    )

    assert "write_file" in content
    assert "docs/a.md" in content
    assert "5 分钟" in content
    # 卡片必须能消歧：同会话两条待裁决时，用户点的「允许」只能靠编号对上。
    assert "123456" in content
    assert buttons == [["允许", "拒绝"]]


@pytest.mark.parametrize(
    ("ttl", "expected"),
    [(45, "45 秒"), (60, "1 分钟"), (90, "1 分 30 秒")],
)
def test_build_approval_card_formats_ttl(ttl: int, expected: str) -> None:
    content, _ = build_approval_card({"id": "appr-abcdef123456", "tool": "exec", "timeout_seconds": ttl})
    assert expected in content


def test_approval_payload_carries_the_same_ttl() -> None:
    """卡片承诺的时间必须来自 ``wait_for`` 的同一个值，不能各写一份。"""
    request = ApprovalRequest(
        request_id="appr-abcdef123456",
        tool="exec",
        args={},
        timeout_seconds=120.0,
    )
    assert request.payload()["timeout_seconds"] == 120


# --------------------------------------------------------------------------
# 3. 出站：审批卡是普通出站消息，不是会被静音的 ProgressEvent
# --------------------------------------------------------------------------


class _RecordingBus:
    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)


async def test_progress_callback_emits_a_plain_outbound_approval_card() -> None:
    bus = _RecordingBus()
    inbound = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="hi")
    callback = build_bus_progress_callback(bus, inbound)

    assert on_progress_accepts_approval_requests(callback) is True

    await callback(
        "",
        approval_request={
            "id": "appr-abcdef123456",
            "tool": "write_file",
            "args": {"path": "a.txt"},
            "timeout_seconds": 300,
        },
    )

    assert len(bus.sent) == 1
    card = bus.sent[0]
    assert card.channel == "feishu"
    assert card.chat_id == "c1"
    assert "write_file" in card.content
    assert card.buttons == [["允许", "拒绝"]]
    # 关键：不是 ProgressEvent，否则会被 ``send_progress`` 静默丢掉。
    assert card.event is None


async def test_progress_callback_still_wraps_normal_progress_as_an_event() -> None:
    bus = _RecordingBus()
    inbound = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="hi")
    callback = build_bus_progress_callback(bus, inbound)

    await callback("thinking")

    assert len(bus.sent) == 1
    assert bus.sent[0].event is not None


# --------------------------------------------------------------------------
# 4. 入站裁决：会话归属、消歧、重放
# --------------------------------------------------------------------------


def _request(request_id: str, *, tool: str = "write_file", session: str = SESSION) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id,
        tool=tool,
        args={"path": "a.txt"},
        session_key=session,
    )


def _msg(content: str, *, channel: str = "feishu", chat_id: str = "chat-1") -> InboundMessage:
    return InboundMessage(channel=channel, sender_id="u1", chat_id=chat_id, content=content)


async def _no() -> bool:
    """``handle_runtime_control`` 的替身：这条消息不是运行期控制指令。"""
    return False


async def test_bare_allow_resolves_the_only_pending_request() -> None:
    registry = PendingApprovals()
    future = registry.register(_request("appr-aaaa1111"))
    inbox = ApprovalInbox(registry)

    assert await inbox.consume(_msg("允许"), session_key=SESSION) is True

    assert future.result().verdict == "allow"
    assert registry.pending_count == 0


async def test_bare_deny_resolves_with_a_refusal() -> None:
    registry = PendingApprovals()
    future = registry.register(_request("appr-aaaa1111"))
    inbox = ApprovalInbox(registry)

    assert await inbox.consume(_msg("拒绝"), session_key=SESSION) is True

    decision = future.result()
    assert decision.verdict == "deny"
    assert decision.allowed is False


async def test_two_pending_requests_require_a_number() -> None:
    """同会话两条待裁决：裸裁决不能猜，必须要求带编号。"""
    registry = PendingApprovals()
    first = registry.register(_request("appr-aaaa1111", tool="write_file"))
    second = registry.register(_request("appr-bbbb2222", tool="exec"))
    replies: list[str] = []

    async def _publish(msg: OutboundMessage) -> None:
        replies.append(msg.content)

    inbox = ApprovalInbox(registry, publish=_publish)

    assert await inbox.consume(_msg("允许"), session_key=SESSION) is True

    assert not first.done()
    assert not second.done()
    assert any("编号" in reply for reply in replies)


async def test_numbered_verdict_targets_exactly_one_request() -> None:
    registry = PendingApprovals()
    first = registry.register(_request("appr-aaaa1111", tool="write_file"))
    second = registry.register(_request("appr-bbbb2222", tool="exec"))
    inbox = ApprovalInbox(registry)

    assert await inbox.consume(_msg("拒绝 bbbb2222"), session_key=SESSION) is True

    assert not first.done()
    assert second.result().verdict == "deny"


async def test_verdict_from_another_session_cannot_approve() -> None:
    """别的会话/群成员不能替这条请求拍板——且不消耗它。

    用带编号的消息来测：裸裁决只会按**本会话**找待裁决项，天然看不到别人的
    请求（这本身也是对的，不该向别的会话泄露「这里有东西在等批准」）。
    """
    registry = PendingApprovals()
    future = registry.register(_request("appr-aaaa1111", session="feishu:someone-else"))
    replies: list[str] = []

    async def _publish(msg: OutboundMessage) -> None:
        replies.append(msg.content)

    inbox = ApprovalInbox(registry, publish=_publish)

    assert await inbox.consume(_msg("允许 aaaa1111"), session_key=SESSION) is True

    assert not future.done()
    assert registry.pending_count == 1
    assert any("不属于当前会话" in reply for reply in replies)


async def test_replayed_numbered_verdict_is_not_applied_twice() -> None:
    registry = PendingApprovals()
    future = registry.register(_request("appr-aaaa1111"))
    replies: list[str] = []

    async def _publish(msg: OutboundMessage) -> None:
        replies.append(msg.content)

    inbox = ApprovalInbox(registry, publish=_publish)

    await inbox.consume(_msg("允许 aaaa1111"), session_key=SESSION)
    assert future.result().verdict == "allow"

    # 重放：注册表里已经没有了，不能再裁决一次，也不能漏进 Agent。
    assert await inbox.consume(_msg("允许 aaaa1111"), session_key=SESSION) is True
    assert any("已过期" in reply for reply in replies)


async def test_bare_verdict_without_pending_is_ordinary_chat() -> None:
    """没有待裁决时，「允许」只是一句话，必须放行给模型。"""
    registry = PendingApprovals()
    inbox = ApprovalInbox(registry)

    assert await inbox.consume(_msg("允许"), session_key=SESSION) is False


async def test_repeated_bare_click_is_swallowed_as_a_duplicate() -> None:
    """第一次点击摘掉了请求；第二次若原样放行，就会变成一条莫名其妙的用户消息。"""
    registry = PendingApprovals()
    registry.register(_request("appr-aaaa1111"))
    inbox = ApprovalInbox(registry)

    assert await inbox.consume(_msg("拒绝"), session_key=SESSION) is True
    assert await inbox.consume(_msg("拒绝"), session_key=SESSION) is True


async def test_duplicate_window_expires() -> None:
    """窗口过后同样的两个字恢复成普通发言，不能被永久吞掉。"""
    registry = PendingApprovals()
    registry.register(_request("appr-aaaa1111"))
    now = [1000.0]
    inbox = ApprovalInbox(registry, clock=lambda: now[0], duplicate_window_seconds=120.0)

    assert await inbox.consume(_msg("允许"), session_key=SESSION) is True
    now[0] += 121.0
    assert await inbox.consume(_msg("允许"), session_key=SESSION) is False


# --------------------------------------------------------------------------
# 5. 出 / 入两段必须真的对接上（防「各自都对、拼起来不通」）
# --------------------------------------------------------------------------


async def test_card_and_verdict_paths_meet_at_the_registry() -> None:
    """从卡片正文里的短编号回复，必须能唤醒同一个 future。

    单测两段各自绿、拼起来不通，是这类桥接最典型的坑（卡片印的编号与
    裁决入口认的形状不一致，谁都不会报错）。
    """
    registry = PendingApprovals()
    request = ApprovalRequest(
        request_id="appr-abcdef123456",
        tool="write_file",
        args={"path": "a.txt"},
        session_key=SESSION,
        timeout_seconds=300,
    )
    future = registry.register(request)

    bus = _RecordingBus()
    inbound = InboundMessage(
        channel="feishu", sender_id="u1", chat_id="chat-1", content="改一下 a.txt"
    )
    await build_bus_progress_callback(bus, inbound)(
        "", approval_request=request.payload()
    )

    card = bus.sent[0]
    short = request.request_id.split("-", 1)[-1][-8:]
    assert short in card.content, "卡片必须印出裁决入口认得的编号"

    verdict = _msg(f"允许 {short}")
    assert await ApprovalInbox(registry).consume(verdict, session_key=SESSION) is True
    assert future.result().verdict == "allow"


# --------------------------------------------------------------------------
# 6. 注册表：短编号寻址必须唯一
# --------------------------------------------------------------------------


async def test_find_by_token_accepts_full_id_and_suffix() -> None:
    registry = PendingApprovals()
    registry.register(_request("appr-aaaa1111"))

    assert registry.find_by_token("appr-aaaa1111") is not None
    assert registry.find_by_token("aaaa1111") is not None
    # 太短的后缀不猜。
    assert registry.find_by_token("111") is None


async def test_find_by_token_refuses_an_ambiguous_suffix() -> None:
    registry = PendingApprovals()
    registry.register(_request("appr-aaaa1111"))
    registry.register(_request("appr-bbbb1111"))

    assert registry.find_by_token("1111") is None


# --------------------------------------------------------------------------
# 7. 渠道开关：IM 必须先被切到 ask，审批桥才会被调用
# --------------------------------------------------------------------------


def test_channel_tool_approval_modes_reads_valid_entries_only() -> None:
    """未知取值 / 非对象配置一律忽略，不让一个可选键把网关起崩。"""
    assert channel_tool_approval_modes(
        {
            "feishu": {"tool_approval": "ask"},
            "weixin": {"tool_approval": "AUTO"},
            "telegram": {"tool_approval": "bogus"},
            "discord": "not-a-section",
        }
    ) == {"feishu": "ask", "weixin": "auto"}


def test_channels_config_roundtrip_preserves_tool_approval() -> None:
    """[真实配置往返] tool_approval 必须穿过 ChannelsConfig 的模型化存活。

    真机事故（2026-10-19）：配置写了开关但审批卡从未出现。读取方吃的是
    ``ChannelsConfig.model_extra`` 里的**原始 dict**（extra="allow"），而不是
    渠道自己的声明式配置（WeixinConfig extra=ignore 会把未知键静默丢掉）。
    这条用例锁住该契约：谁把渠道段变成声明式字段、或把 extra 改成 ignore，
    这里就会红，而不是让 IM 审批再次静默变成死代码。

    段级是唯一权威位置（2026-10-19 统一）：feishu 的 inherited 机制曾把段级键
    泄漏进 instances[] 又被 canonical 回写盘，读取端因此加了对实例级的**兜底
    回收**——段级缺失时取实例上的值，存量污染形态不至于静默变回 auto。
    """
    from nanobot.config.schema import ChannelsConfig

    channels = ChannelsConfig.model_validate(
        {
            "weixin": {"enabled": True, "tool_approval": "ask"},
            "feishu": {"instances": [{"id": "feishu", "tool_approval": "ask"}]},
        }
    )
    # weixin 段顶层生效；feishu 段级缺失 → 从实例兜底回收，同为 ask
    assert channel_tool_approval_modes(channels) == {"weixin": "ask", "feishu": "ask"}


def test_channel_tool_approval_modes_section_level_wins_over_instances() -> None:
    """段级与实例级并存时以段级为准（权威位置唯一）。"""
    assert channel_tool_approval_modes(
        {
            "feishu": {
                "tool_approval": "auto",
                "instances": [{"id": "feishu", "tool_approval": "ask"}],
            }
        }
    ) == {"feishu": "auto"}


def test_unscoped_channel_uses_its_own_approval_mode(tmp_path) -> None:
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
        default_tool_approval_by_channel={"feishu": "ask"},
    )

    assert (
        resolver.for_turn(
            channel="feishu", message_metadata=None, session_metadata=None
        ).tool_approval
        == "ask"
    )
    assert (
        resolver.for_turn(
            channel="weixin", message_metadata=None, session_metadata=None
        ).tool_approval
        == "auto"
    )


def test_scoped_channel_ignores_the_per_channel_map(tmp_path) -> None:
    """api/websocket 的审批模式由请求下发，渠道默认值不能越权覆盖。"""
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
        default_tool_approval_by_channel={"api": "ask"},
    )

    scope = resolver.for_turn(channel="api", message_metadata=None, session_metadata=None)

    assert scope.tool_approval == "auto"


async def test_hook_emits_a_card_and_the_im_verdict_unblocks_it(tmp_path) -> None:
    """端到端：渠道切 ask → 钩子发卡 → 用户回复 → 工具放行。

    这条用例把「配置面 → 作用域 → 钩子 → 卡片 → 裁决 → 放行」串成一条线。
    各段单测都绿但拼不起来，是这类桥接最常见的失败方式。
    """
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
        default_tool_approval_by_channel={"feishu": "ask"},
    )
    scope = resolver.for_turn(channel="feishu", message_metadata=None, session_metadata=None)
    token = bind_workspace_scope(scope)
    try:
        registry = PendingApprovals()
        cards: list[dict[str, object]] = []

        async def _on_progress(content: str, *, approval_request=None, **_: object) -> None:
            if approval_request:
                cards.append(approval_request)

        hook = ToolApprovalHook(on_progress=_on_progress, registry=registry)
        context = AgentHookContext(iteration=0, messages=[], session_key=SESSION)
        call = ToolCallRequest(id="call-1", name="write_file", arguments={"path": "a.txt"})

        task = asyncio.create_task(
            hook.before_execute_tool(context, call, None, {"path": "a.txt"})
        )
        for _ in range(50):
            if cards:
                break
            await asyncio.sleep(0)
        assert cards, "ask 模式下钩子必须发出审批请求"

        short = str(cards[0]["id"]).split("-", 1)[-1][-8:]
        verdict = _msg(f"允许 {short}")

        assert await ApprovalInbox(registry).consume(verdict, session_key=SESSION) is True
        # 放行 = 返回 None（不否决），而不是某个 allow 对象。
        assert await asyncio.wait_for(task, timeout=2) is None
    finally:
        reset_workspace_scope(token)


# ---------------------------------------------------------------------------
# 6. fail-closed 的两条「 inferred from code position 」判据
# ---------------------------------------------------------------------------
#
# 下面两条此前只由代码位置/注释保证（``loop.py`` 里 consume 早于命令分发、
# 子代理构造时不传 ``on_progress``），被改坏不会有任何用例变红。这里补上。


async def test_verdict_is_absorbed_before_command_dispatch(
    tmp_path, monkeypatch
) -> None:
    """⛔ 审批裁决必须在命令分发**之前**被吸收。

    把这段吸收下移到命令分发之后，一条裸「允许」会变成新一轮用户输入：模型对着
    这两个字自由发挥，而真正挂起的工具继续等到超时——两种情况都不报错，所以必须
    在这儿钉住。

    做法是**真实驱动** ``AgentLoop.run()`` 一次迭代：bus 吐出一条裁决后永久阻塞，
    再断言「挂起的审批被裁决了」且「用户输入/命令分发一次都没被触达」。
    """
    registry = PendingApprovals()
    future = registry.register(_request("appr-abc12345"))
    published: list[OutboundMessage] = []

    async def _publish(out: OutboundMessage) -> None:
        published.append(out)

    loop = AgentLoop(bus=MagicMock(), provider=MagicMock(), workspace=tmp_path)
    loop._approval_inbox = ApprovalInbox(registry, publish=_publish)

    pending = [_msg("允许")]

    async def _consume_inbound():
        if pending:
            return pending.pop(0)
        await asyncio.sleep(3600)  # 之后再无入站：只给 run() 这一次迭代

    loop.bus.consume_inbound = MagicMock(side_effect=_consume_inbound)

    # 会话键的推导另有单独的用例；这里替换掉是为了让本用例**只**盯住顺序，
    # 不被 key 推导规则的变化误伤。
    monkeypatch.setattr(loop, "_effective_session_key", lambda _msg: SESSION)
    monkeypatch.setattr(agent_context, "handle_runtime_control", lambda *_a, **_k: _no())
    monkeypatch.setattr(loop, "_automation_turn_coordinators", [])

    dispatched: list[str] = []
    user_inputs: list[str] = []

    async def _dispatch(*_a, **_k):
        dispatched.append("yes")

    async def _user_input_accepted(*_a, **_k):
        user_inputs.append("yes")

    monkeypatch.setattr(loop, "_dispatch_command_inline", _dispatch)
    monkeypatch.setattr(
        loop, "runtime_event_publisher", SimpleNamespace(user_input_accepted=_user_input_accepted)
    )

    task = asyncio.create_task(loop.run())
    try:
        for _ in range(300):
            if future.done():
                break
            await asyncio.sleep(0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ① 裁决确实落到挂起的审批上
    assert future.done(), "「允许」没有裁决挂起的审批请求"
    assert future.result().verdict == "allow"
    # ② 但这条消息没有继续往下走
    assert user_inputs == [], "裁决被当成新一轮用户输入了"
    assert dispatched == [], "裁决流到了命令分发"


async def test_subagent_hook_without_channel_refuses(tmp_path) -> None:
    """⛔ 子代理拿不到下发通道时必须拒绝，不能静默放行。

    ``subagent.py`` 构造 ToolApprovalHook 时**不传** ``on_progress``（子代理没有
    自己的进度面）。若这条路径退化成放行，ask 模式下的子代理就等于绕过了审批——
    而这与「子代理权限单调递减」的安全承诺直接冲突。

    把 ``ToolApprovalHook`` 的 None 判据删掉、或把 deny 改成 allow，本用例都会红。
    """
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=False,
        # ⚠️ 必须是**不在** scoped_channels 里的渠道：api/websocket 会忽略逐渠道映射
        default_tool_approval_by_channel={"feishu": "ask"},
    )
    scope = resolver.for_turn(channel="feishu", message_metadata=None, session_metadata=None)
    token = bind_workspace_scope(scope)
    try:
        # 与 subagent.py 的构造方式保持一致：不传 on_progress
        hook = ToolApprovalHook(registry=PendingApprovals())
        context = AgentHookContext(iteration=0, messages=[], session_key=SESSION)
        call = ToolCallRequest(id="call-1", name="write_file", arguments={"path": "a.txt"})

        decision = await hook.before_execute_tool(context, call, None, {"path": "a.txt"})

        assert decision is not None, "无下发通道时必须拦下来，不能返回「无意见」"
        assert decision.allowed is False
        assert "no channel" in (decision.reason or "")
    finally:
        reset_workspace_scope(token)
