"""[LOCAL PATCH] FR-5.6：session-bound 定时任务的结果回推（`payload.delivery`）。

这里只覆盖**引擎侧新增的那一层**——「按配置把结果另发到外部渠道」。
任务本身的执行、绑定校验、run record 都不动，所以用假 agent / 假 bus 就够，
不需要真的起网关或渠道。
"""

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cron import bound_runner
from nanobot.cron.bound_runner import run_bound_cron_job
from nanobot.cron.session_turns import is_bound_cron_job
from nanobot.cron.types import CronDelivery, CronDeliveryTarget, CronJob, CronPayload


class _FakeRunRecorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write_run_record(self, run_id: str, record: dict) -> None:
        self.records.append(record)


class _FakeTools:
    def get(self, name: str):  # noqa: ANN201 — 只要不是 CronTool 就走不到那条分支
        return None


class _FakeAgent:
    """最小 BoundCronAgent：要么回一段正文，要么抛错。"""

    def __init__(self, *, content: str = "", error: Exception | None = None) -> None:
        self.tools = _FakeTools()
        self._content = content
        self._error = error

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        if self._error is not None:
            raise self._error
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._content)


class _RecordingPublisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[OutboundMessage] = []
        self._fail = fail

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        if self._fail:
            raise RuntimeError("channel down")
        self.sent.append(msg)


def _job(*, delivery: CronDelivery | None = None, name: str = "早报") -> CronJob:
    return CronJob(
        id="j1",
        name=name,
        payload=CronPayload(
            message="生成早报",
            session_key="api:cron-j1",
            origin_channel="api",
            origin_chat_id="cron-j1",
            delivery=delivery,
        ),
    )


def _delivery(*targets: tuple[str, str], on: str = "complete") -> CronDelivery:
    return CronDelivery(
        targets=[CronDeliveryTarget(channel=c, chat_id=i) for c, i in targets],
        on=on,
    )


# ─────────────────────────── 解析层 ───────────────────────────


def test_delivery_parses_camel_and_snake_case_and_drops_incomplete_targets() -> None:
    delivery = CronDelivery.from_store_dict(
        {
            "targets": [
                {"channel": "wecom", "chatId": "chat-1"},
                {"channel": "feishu", "chat_id": "chat-2"},
                {"channel": "wecom"},  # 缺 chatId
                {"chatId": "chat-3"},  # 缺 channel
                {"channel": "   ", "chatId": "chat-4"},  # 空白 channel
                "not-a-dict",
            ],
            "on": "error",
        }
    )

    assert delivery is not None
    assert delivery.on == "error"
    assert [(t.channel, t.chat_id) for t in delivery.targets] == [
        ("wecom", "chat-1"),
        ("feishu", "chat-2"),
    ]


def test_delivery_is_absent_when_nothing_can_be_pushed_to() -> None:
    # 「配置了但推不出去」必须读成「没配置」——否则上游会当成投递成功来报
    assert CronDelivery.from_store_dict(None) is None
    assert CronDelivery.from_store_dict("complete") is None
    assert CronDelivery.from_store_dict({"targets": []}) is None
    assert CronDelivery.from_store_dict({"targets": [{"channel": "wecom"}]}) is None


def test_unknown_on_value_is_kept_raw_for_the_runner_to_judge() -> None:
    # 解析层不做语义判断（已实现的是 complete / error / start；"milestone" 仍未实现），
    # 原样带出去让 runner 决定并告警
    delivery = CronDelivery.from_store_dict(
        {"targets": [{"channel": "wecom", "chatId": "c1"}], "on": "milestone"}
    )
    assert delivery is not None
    assert delivery.on == "milestone"


def test_payload_reads_delivery_without_looking_like_legacy_routing_context() -> None:
    payload = CronPayload.from_store_dict(
        {
            "kind": "agent_turn",
            "message": "hi",
            "sessionKey": "api:cron-j1",
            "originChannel": "api",
            "originChatId": "cron-j1",
            "delivery": {
                "targets": [{"channel": "wecom", "chatId": "c1"}],
                "on": "complete",
            },
        }
    )

    assert payload.session_key == "api:cron-j1"
    assert payload.delivery is not None
    assert payload.delivery.targets[0].channel == "wecom"

    # ⭐ 这条是本设计的核心不变量：`deliver` / `channel` / `to` / `channel_meta`
    # 一旦非空，`is_bound_cron_job` 就返回 False，任务会被 `_enforce_agent_binding`
    # 直接禁用。delivery 必须是**独立字段**，不能借用 legacy 的那三个。
    assert payload.deliver is False
    assert payload.channel is None
    assert payload.to is None
    assert payload.channel_meta == {}
    assert is_bound_cron_job(CronJob(id="j1", name="t", payload=payload)) is True


def test_missing_delivery_key_leaves_payload_unchanged() -> None:
    payload = CronPayload.from_store_dict({"kind": "agent_turn", "message": "hi"})
    assert payload.delivery is None


# ─────────────────────────── 投递层 ───────────────────────────


async def test_no_delivery_config_publishes_nothing() -> None:
    publisher = _RecordingPublisher()
    result = await run_bound_cron_job(
        _job(), agent=_FakeAgent(content="结果"), cron=_FakeRunRecorder(), publish_outbound=publisher.publish_outbound
    )

    assert result == "结果"
    assert publisher.sent == []


async def test_missing_publisher_is_tolerated() -> None:
    # 只关心 run record 的调用方（如既有测试）不必提供 bus
    recorder = _FakeRunRecorder()
    result = await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"))),
        agent=_FakeAgent(content="结果"),
        cron=recorder,
    )

    assert result == "结果"
    assert recorder.records[-1]["status"] == "ok"


async def test_complete_mode_pushes_result_with_job_name_prefix() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "chat-1")), name="早报"),
        agent=_FakeAgent(content="今天有 3 条待办"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert len(publisher.sent) == 1
    sent = publisher.sent[0]
    assert (sent.channel, sent.chat_id) == ("wecom", "chat-1")
    # 前缀是必需的：同一个群可能收多条任务的结果，不带名字就不知道是谁发的
    assert sent.content == "早报\n\n今天有 3 条待办"


async def test_complete_mode_also_pushes_failures() -> None:
    publisher = _RecordingPublisher()
    with pytest.raises(RuntimeError, match="boom"):
        await run_bound_cron_job(
            _job(delivery=_delivery(("wecom", "c1"))),
            agent=_FakeAgent(error=RuntimeError("boom")),
            cron=_FakeRunRecorder(),
            publish_outbound=publisher.publish_outbound,
        )

    assert len(publisher.sent) == 1
    assert "执行失败：" in publisher.sent[0].content
    assert "boom" in publisher.sent[0].content


async def test_error_mode_skips_successful_runs() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), on="error")),
        agent=_FakeAgent(content="一切正常"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert publisher.sent == []


async def test_error_mode_pushes_failures() -> None:
    publisher = _RecordingPublisher()
    with pytest.raises(RuntimeError, match="boom"):
        await run_bound_cron_job(
            _job(delivery=_delivery(("wecom", "c1"), on="error")),
            agent=_FakeAgent(error=RuntimeError("boom")),
            cron=_FakeRunRecorder(),
            publish_outbound=publisher.publish_outbound,
        )

    assert len(publisher.sent) == 1
    assert "boom" in publisher.sent[0].content


# ─────────────── `on: start`：结果之外多推一条「开跑」通知 ───────────────


async def test_start_mode_notifies_before_the_turn_and_still_pushes_the_result() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), on="start"), name="早报"),
        agent=_FakeAgent(content="今天有 3 条待办"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert [m.content for m in publisher.sent] == [
        "早报\n\n已开始执行。",
        "早报\n\n今天有 3 条待办",
    ]
    assert [(m.channel, m.chat_id) for m in publisher.sent] == [("wecom", "c1")] * 2


async def test_start_mode_contains_complete_so_failures_push_both_messages() -> None:
    # 「只告诉我开始了」不是有用的档位——用户永远不会知道结果。
    # 所以 start 是**包含** complete 的，不是并列的一档。
    publisher = _RecordingPublisher()
    with pytest.raises(RuntimeError, match="boom"):
        await run_bound_cron_job(
            _job(delivery=_delivery(("wecom", "c1"), on="start")),
            agent=_FakeAgent(error=RuntimeError("boom")),
            cron=_FakeRunRecorder(),
            publish_outbound=publisher.publish_outbound,
        )

    assert len(publisher.sent) == 2
    assert "已开始执行。" in publisher.sent[0].content
    assert "执行失败：" in publisher.sent[1].content


async def test_start_notice_has_no_dangling_blank_line_without_job_name() -> None:
    # 没名字时正文就是一句话，不能拼成 "\n\n已开始执行。"（空前缀 + 空行）
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), on="start"), name=""),
        agent=_FakeAgent(content="结果"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert publisher.sent[0].content == "已开始执行。"


async def test_every_target_receives_the_start_notice() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), ("feishu", "c2"), on="start")),
        agent=_FakeAgent(content="结果"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    # 先各收到一条开跑通知，再各收到一条结果
    assert [(m.channel, m.chat_id) for m in publisher.sent] == [
        ("wecom", "c1"),
        ("feishu", "c2"),
        ("wecom", "c1"),
        ("feishu", "c2"),
    ]


async def test_non_start_modes_never_get_the_start_notice() -> None:
    # ⭐ 加一个档位不能改变既有档位的行为：「我要结果」的用户不该因为
    # 我们支持了 start 就多收一条开跑通知。
    for mode in ("complete", "error"):
        publisher = _RecordingPublisher()
        await run_bound_cron_job(
            _job(delivery=_delivery(("wecom", "c1"), on=mode)),
            agent=_FakeAgent(content="结果"),
            cron=_FakeRunRecorder(),
            publish_outbound=publisher.publish_outbound,
        )
        assert "已开始执行。" not in "".join(
            m.content for m in publisher.sent
        ), f"mode={mode} 不该收到开跑通知"


async def test_start_notice_failure_does_not_abort_the_run() -> None:
    # 通知是尽力而为，而且理由比结果推送更强：一个**即将执行**的任务，
    # 不能因为一条客套话发不出去就被中断。
    recorder = _FakeRunRecorder()
    result = await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), on="start")),
        agent=_FakeAgent(content="结果"),
        cron=recorder,
        publish_outbound=_RecordingPublisher(fail=True).publish_outbound,
    )

    assert result == "结果"
    assert recorder.records[-1]["status"] == "ok"


def test_delivery_modes_register_every_implemented_mode() -> None:
    # `_DELIVERY_MODES` 是「已知档位」的登记表。实现了新档位却忘了登记，
    # 后果**不是**功能坏掉（`_deliver_started` 自己判 `on == "start"`），
    # 而是每次运行都打一条 "unknown delivery.on" 告警——长期噪音，
    # 于是真正该被看见的告警会被淹掉。这类「不报错的漏登记」值得单独钉一条。
    assert set(bound_runner._DELIVERY_MODES) == {"complete", "error", "start"}


async def test_unknown_mode_still_delivers() -> None:
    # "milestone" 仍未实现：退化成 complete（推）好过静默丢弃用户配的推送
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), on="milestone")),
        agent=_FakeAgent(content="结果"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert len(publisher.sent) == 1


async def test_empty_result_is_not_pushed() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"))),
        agent=_FakeAgent(content="   "),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert publisher.sent == []


async def test_every_target_receives_a_copy() -> None:
    publisher = _RecordingPublisher()
    await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"), ("feishu", "c2"))),
        agent=_FakeAgent(content="结果"),
        cron=_FakeRunRecorder(),
        publish_outbound=publisher.publish_outbound,
    )

    assert [(m.channel, m.chat_id) for m in publisher.sent] == [
        ("wecom", "c1"),
        ("feishu", "c2"),
    ]


async def test_publish_failure_does_not_break_the_run() -> None:
    recorder = _FakeRunRecorder()
    result = await run_bound_cron_job(
        _job(delivery=_delivery(("wecom", "c1"))),
        agent=_FakeAgent(content="结果"),
        cron=recorder,
        publish_outbound=_RecordingPublisher(fail=True).publish_outbound,
    )

    # 投递是尽力而为：跑完的任务不因为推送目标不可达而被改写成失败
    assert result == "结果"
    assert recorder.records[-1]["status"] == "ok"
    assert recorder.records[-1]["response"] == "结果"


async def test_failed_run_still_records_error_when_delivery_also_fails() -> None:
    recorder = _FakeRunRecorder()
    with pytest.raises(RuntimeError, match="boom"):
        await run_bound_cron_job(
            _job(delivery=_delivery(("wecom", "c1"))),
            agent=_FakeAgent(error=RuntimeError("boom")),
            cron=recorder,
            publish_outbound=_RecordingPublisher(fail=True).publish_outbound,
        )

    # 原始失败不得被投递失败盖掉
    assert recorder.records[-1]["status"] == "error"
    assert recorder.records[-1]["error"] == "boom"
