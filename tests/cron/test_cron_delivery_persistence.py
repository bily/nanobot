"""[LOCAL PATCH] FR-5.6：`payload.delivery` 的**落盘往返**。

单独成文件是因为它锁的不是投递行为，而是 `CronService._save_store` 那条**极易漏**的纪律：
那个函数是**逐字段**手写序列化，新增一个声明式字段若忘了在那里列出，引擎每次落盘都会把它
静默丢掉——读的时候一切正常（内存里还在），下一次启动就没了，且没有任何日志。

（本项目对「声明式字段」的既有要求就是 schema + 往返用例两件一起做；这里补齐往返那一半。）
"""

import json

from nanobot.cron.service import CronService
from nanobot.cron.types import CronStore


def _write_store(store_path, *, delivery: dict | None) -> None:
    payload: dict = {
        "kind": "agent_turn",
        "message": "生成早报",
        "sessionKey": "api:cron-j1",
        "originChannel": "api",
        "originChatId": "cron-j1",
    }
    if delivery is not None:
        payload["delivery"] = delivery

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
                        "payload": payload,
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


def _reload(store_path):
    jobs, _version = CronService(store_path)._load_jobs()
    assert jobs is not None
    return jobs


def test_delivery_survives_a_save_and_reload(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    _write_store(
        store_path,
        delivery={
            "targets": [
                {"channel": "wecom", "chatId": "c1"},
                {"channel": "feishu.feishu", "chatId": "ou_1"},
            ],
            "on": "error",
        },
    )

    # ① 读得回来
    loaded = _reload(store_path)
    assert loaded[0].payload.delivery is not None

    # ② 经过一次落盘之后仍然在（这一步才是本文件存在的理由）
    service = CronService(store_path)
    service._store = CronStore(version=1, jobs=loaded)
    service._save_store()

    reloaded = _reload(store_path)
    delivery = reloaded[0].payload.delivery
    assert delivery is not None, "_save_store 漏列了 delivery —— 落盘即静默丢弃"
    assert [(t.channel, t.chat_id) for t in delivery.targets] == [
        ("wecom", "c1"),
        ("feishu.feishu", "ou_1"),
    ]
    assert delivery.on == "error"


def test_absent_delivery_does_not_gain_a_null_key_on_save(tmp_path) -> None:
    """没配推送时**不写键**。

    写 `"delivery": null` 会让前端那次幂等合并的比对在「有键 null」与「没键」之间来回抖，
    于是每次同步都判定为「有变化」。
    """
    store_path = tmp_path / "cron" / "jobs.json"
    _write_store(store_path, delivery=None)

    service = CronService(store_path)
    service._store = CronStore(version=1, jobs=_reload(store_path))
    service._save_store()

    saved = json.loads(store_path.read_text(encoding="utf-8"))
    assert "delivery" not in saved["jobs"][0]["payload"]
