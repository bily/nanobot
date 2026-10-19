"""[LOCAL PATCH] FR-5.8：任务级工作目录靠 `originMetadata.workspace_scope` 送达。

本文件钉住的是一条**跨层契约**，而不是某个函数：
客户端把任务的工作目录写进 job 的 `originMetadata.workspace_scope.project_path`
→ `bound_runner` 把 `origin_metadata` 原样搬成 cron turn 的 message metadata
→ 引擎在 turn 边界用 `WorkspaceScopeResolver.for_turn` 解析出**这一轮该在哪个目录里跑**。

最后那条**否定用例**把「结果回推为什么不能借用 origin 渠道」钉进代码：渠道不属于
`scoped_channels` 时，同一份 metadata 会被**静默忽略**并回落默认工作空间——不报错、不告警。
"""

import json
from pathlib import Path

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cron.bound_runner import run_bound_cron_job
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronPayload, CronStore
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScopeResolver,
)


class _FakeRunRecorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write_run_record(self, run_id: str, record: dict) -> None:
        self.records.append(record)


class _FakeTools:
    def get(self, name: str):  # noqa: ANN201 — 非 CronTool 即可
        return None


class _RecordingAgent:
    """记下 InboundMessage，以便断言 metadata 里真的带着 workspace_scope。"""

    def __init__(self, content: str = "done") -> None:
        self.tools = _FakeTools()
        self.messages: list[InboundMessage] = []
        self._content = content

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        self.messages.append(msg)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._content)


def _job(*, workdir: str | None, channel: str = "api") -> CronJob:
    metadata: dict = {"managedBy": "smartagent", "agentId": "a1"}
    if workdir is not None:
        metadata[WORKSPACE_SCOPE_METADATA_KEY] = {"project_path": workdir}
    return CronJob(
        id="j1",
        name="早报",
        payload=CronPayload(
            message="生成早报",
            session_key=f"{channel}:cron-j1",
            origin_channel=channel,
            origin_chat_id="cron-j1",
            origin_metadata=metadata,
        ),
    )


def _resolver(default_dir: Path) -> WorkspaceScopeResolver:
    return WorkspaceScopeResolver(
        default_workspace=default_dir,
        default_restrict_to_workspace=True,
    )


async def test_cron_turn_metadata_carries_the_task_workspace(tmp_path) -> None:
    workspace = tmp_path / "proj"
    workspace.mkdir()
    agent = _RecordingAgent()

    await run_bound_cron_job(_job(workdir=str(workspace)), agent=agent, cron=_FakeRunRecorder())

    assert len(agent.messages) == 1
    assert agent.messages[0].metadata[WORKSPACE_SCOPE_METADATA_KEY] == {
        "project_path": str(workspace)
    }


async def test_engine_resolves_the_task_workspace_for_the_turn(tmp_path) -> None:
    default_dir = tmp_path / "default"
    task_dir = tmp_path / "proj"
    default_dir.mkdir()
    task_dir.mkdir()
    agent = _RecordingAgent()

    await run_bound_cron_job(_job(workdir=str(task_dir)), agent=agent, cron=_FakeRunRecorder())

    msg = agent.messages[0]
    scope = _resolver(default_dir).for_turn(
        channel=msg.channel,
        message_metadata=msg.metadata,
        session_metadata=None,
    )

    assert msg.channel == "api"
    assert scope.project_path == task_dir.resolve()


async def test_non_scoped_channel_silently_ignores_the_task_workspace(tmp_path) -> None:
    """⚠️ 否定用例：这就是「结果回推不能借用 origin 渠道」的原因。

    把 channel 换成外部 IM 名（`scoped_channels` 只认 `{websocket, api}`），同一份
    metadata 里的工作目录会被**静默忽略**、回落默认工作空间 —— 不报错、不告警。
    所以「任务级工作目录」与「让任务跑在 IM 会话里」是互斥的，这也是 FR-5.6 最终另开
    `delivery` 字段、而不是改 origin 渠道的直接原因。
    """
    default_dir = tmp_path / "default"
    task_dir = tmp_path / "proj"
    default_dir.mkdir()
    task_dir.mkdir()
    agent = _RecordingAgent()

    await run_bound_cron_job(
        _job(workdir=str(task_dir), channel="wecom"),
        agent=agent,
        cron=_FakeRunRecorder(),
    )

    msg = agent.messages[0]
    scope = _resolver(default_dir).for_turn(
        channel=msg.channel,
        message_metadata=msg.metadata,
        session_metadata=None,
    )

    assert msg.channel == "wecom"
    assert scope.project_path == default_dir.resolve()  # 任务配的目录被丢了


async def test_no_workspace_configured_sends_no_scope_key(tmp_path) -> None:
    """没配工作目录时**不下发** `workspace_scope`，让引擎走会话/默认作用域。

    不下发而不是下发一个空对象：`workspace_scope` 一旦出现在 metadata 里，引擎就按它
    解析（缺 `project_path` 时回落到默认路径，但 `access_mode` 等也会跟着走那条分支），
    而「不声明」才是真正的「跟随会话」。
    """
    agent = _RecordingAgent()

    await run_bound_cron_job(_job(workdir=None), agent=agent, cron=_FakeRunRecorder())

    assert WORKSPACE_SCOPE_METADATA_KEY not in agent.messages[0].metadata


def test_workspace_scope_survives_the_jobs_json_round_trip(tmp_path) -> None:
    """工作目录住在已有的 `originMetadata` 里 —— 所以 FR-5.8 **不需要引擎新增字段**。

    不过「已有字段」也得证：`_save_store` 是逐个列举字段的手写序列化（漏一个键就是每次
    落盘静默丢弃，见 `test_cron_delivery_persistence.py`），`originMetadata` 在其中。
    """
    store_path = tmp_path / "cron" / "jobs.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": [
                    {
                        "id": "j1",
                        "name": "早报",
                        "enabled": True,
                        "schedule": {"kind": "every", "everyMs": 3_600_000},
                        "payload": {
                            "kind": "agent_turn",
                            "message": "生成早报",
                            "sessionKey": "api:cron-j1",
                            "originChannel": "api",
                            "originChatId": "cron-j1",
                            "originMetadata": {
                                "managedBy": "smartagent",
                                WORKSPACE_SCOPE_METADATA_KEY: {"project_path": "/proj/x"},
                            },
                        },
                        "state": {},
                        "createdAtMs": 0,
                        "updatedAtMs": 0,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded, _version = CronService(store_path)._load_jobs()
    assert loaded is not None
    service = CronService(store_path)
    service._store = CronStore(version=1, jobs=loaded)
    service._save_store()

    saved = json.loads(store_path.read_text(encoding="utf-8"))
    assert saved["jobs"][0]["payload"]["originMetadata"][WORKSPACE_SCOPE_METADATA_KEY] == {
        "project_path": "/proj/x"
    }
