"""[LOCAL PATCH] nanowork FR-11.1：`/v1/usage` 的服务端协议测试。

覆盖五件事：空库下的 payload 形状（客户端 UI 依赖这个形状）、鉴权、
真实写入后的汇总与 source 分组、**只读**（查询不新增记录）、以及
「引擎配置里的时区确实被读出来」（含读不到时不炸）。

时区对「按本地日分组」的影响**刻意不在这里测**：`LLMUsageStore.usage_payload`
已经用可注入的 `now` 把那条口径钉在 `tests/llm_usage/` 里，这里再测一遍只会
因为「用例运行时刻不同」而变成一条时灵时不灵的断言。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import _usage_timezone_name, create_app
from nanobot.llm_usage import empty_usage_payload, get_llm_usage_store
from nanobot.llm_usage.context import LLMUsageSource
from nanobot.llm_usage.models import LLMCallRecord
from nanobot.llm_usage.store import LLMUsageStore
from nanobot.providers.base import LLMUsage

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:  # pragma: no cover - aiohttp 是运行依赖，这里只是防御
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

API_KEY = "secret"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


@pytest.fixture(autouse=True)
def _isolate_usage_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """用量库落到 tmp —— 绝不能碰宿主机上真实的 ``llm_usage.sqlite3``。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr("nanobot.llm_usage.get_data_dir", lambda: data_dir)
    # 模块级按路径索引的单例缓存；不清掉就会拿到上一个用例（甚至上次运行）的库
    monkeypatch.setattr("nanobot.llm_usage._STORES", {})
    return data_dir / "llm_usage.sqlite3"


@pytest.fixture
def engine_timezone(monkeypatch: pytest.MonkeyPatch):
    """控制引擎配置里的时区（``_usage_timezone_name`` 走 ``load_config``）。"""

    def _set(value: str | None) -> None:
        config = SimpleNamespace(
            agents=SimpleNamespace(defaults=SimpleNamespace(timezone=value))
        )
        monkeypatch.setattr("nanobot.config.loader.load_config", lambda *a, **k: config)

    return _set


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.workspace = ""
    agent.aclose = AsyncMock()
    return agent


async def _client_for(aiohttp_client):
    return await aiohttp_client(create_app(_make_agent(), model_name="m", api_key=API_KEY))


def _recent_ms(hours_ago: int = 1) -> int:
    """近期的绝对时间戳 —— 保证落在 payload 的可见窗口内。"""
    return int((datetime.now(timezone.utc) - timedelta(hours=hours_ago)).timestamp() * 1000)


def _seed(
    db_path: Path,
    *,
    source: LLMUsageSource = "user",
    model: str = "gpt-5",
    provider: str = "openai",
    input_tokens: int = 100,
    output_tokens: int = 20,
    hours_ago: int = 1,
) -> None:
    store = LLMUsageStore(db_path)
    try:
        store.record(
            LLMCallRecord(
                started_at_ms=_recent_ms(hours_ago),
                duration_ms=250,
                provider=provider,
                model=model,
                source=source,
                stream=True,
                finish_reason="stop",
                usage=LLMUsage.reported(
                    input_tokens=input_tokens, output_tokens=output_tokens
                ),
            )
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# GET /v1/usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_returns_full_shape_before_any_call(aiohttp_client) -> None:
    """一条记录都没有时也要给出完整形状 —— 客户端 UI 直接读这些键。"""
    client = await _client_for(aiohttp_client)
    resp = await client.get("/v1/usage", headers=AUTH_HEADERS)
    assert resp.status == 200

    payload = await resp.json()
    assert payload["days"] == []
    assert payload["total_tokens"] == 0
    assert payload["providers_30d"] == []
    assert payload["updated_at"] is None
    assert payload["cache_read_rate_30d"] is None
    # 30 天派生指标一个都不能缺：缺了客户端会在渲染时读到 undefined 而显示 NaN
    for key in (
        "total_tokens_30d",
        "total_tokens_365d",
        "reported_tokens_30d",
        "estimated_tokens_30d",
        "cache_read_tokens_30d",
        "peak_day_tokens",
        "current_streak_days",
        "longest_streak_days",
        "active_days_30d",
        "requests_30d",
        "failed_requests_30d",
    ):
        assert key in payload, key


@pytest.mark.asyncio
async def test_usage_aggregates_records_and_groups_by_source(
    aiohttp_client, _isolate_usage_store: Path
) -> None:
    """写入两条（不同 source）→ 汇总与分组都要对得上。"""
    _seed(_isolate_usage_store, source="user", input_tokens=100, output_tokens=20)
    _seed(
        _isolate_usage_store,
        source="cron",
        model="claude-x",
        provider="anthropic",
        input_tokens=10,
        output_tokens=5,
        hours_ago=2,
    )

    client = await _client_for(aiohttp_client)
    payload = await (await client.get("/v1/usage", headers=AUTH_HEADERS)).json()

    assert payload["total_tokens"] == 135
    assert payload["requests_30d"] == 2

    assert len(payload["days"]) == 1, payload["days"]
    day = payload["days"][0]
    assert day["total_tokens"] == 135
    assert day["sources"]["user"]["total_tokens"] == 120
    assert day["sources"]["cron"]["total_tokens"] == 15

    # providers_30d 是 (provider, model) 二维分组，按 token 降序
    models = {(row["provider"], row["model"]): row["total_tokens"] for row in payload["providers_30d"]}
    assert models == {("openai", "gpt-5"): 120, ("anthropic", "claude-x"): 15}


@pytest.mark.asyncio
async def test_usage_is_read_only(aiohttp_client, _isolate_usage_store: Path) -> None:
    """GET 不许写入任何调用记录 —— 查询自己造出数据是最坏的一种假绿。"""
    client = await _client_for(aiohttp_client)

    await client.get("/v1/usage", headers=AUTH_HEADERS)
    assert get_llm_usage_store().count() == 0

    _seed(_isolate_usage_store)
    await client.get("/v1/usage", headers=AUTH_HEADERS)
    # 两次查询之后仍然只有那一条真实记录
    assert get_llm_usage_store().count() == 1


@pytest.mark.asyncio
async def test_usage_requires_api_key(aiohttp_client) -> None:
    client = await _client_for(aiohttp_client)
    # 缺 Authorization / 给错 key 都必须是 401：用量是账户级信息
    assert (await client.get("/v1/usage")).status == 401
    assert (
        await client.get("/v1/usage", headers={"Authorization": "Bearer nope"})
    ).status == 401
    # 对照：带了正确 key 就不是 401（否则上两条在「路由根本不存在」时也会绿）
    assert (await client.get("/v1/usage", headers=AUTH_HEADERS)).status == 200


# ---------------------------------------------------------------------------
# 时区：配置读得到就照传，读不到就交回 store 回退 UTC（不是 500）
# ---------------------------------------------------------------------------


def test_timezone_comes_from_engine_config(engine_timezone) -> None:
    engine_timezone("Asia/Shanghai")
    assert _usage_timezone_name() == "Asia/Shanghai"

    engine_timezone(None)
    assert _usage_timezone_name() is None


def test_timezone_falls_back_when_config_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置损坏 / 缺失时不能把用量页打成 500：退回 UTC 比报错有用。"""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("config is unreadable")

    monkeypatch.setattr("nanobot.config.loader.load_config", _boom)
    assert _usage_timezone_name() is None


@pytest.mark.asyncio
async def test_usage_passes_engine_timezone_down_to_the_store(
    aiohttp_client, engine_timezone, monkeypatch: pytest.MonkeyPatch
) -> None:
    """路由必须把配置时区**传下去** —— 这是「按本地日分组」唯一的上游环节。

    只断言「配置读得出来」是不够的：把 handler 里那个 kwargs 删掉，配置照样
    读得出，报表却按 UTC 分组，而不会有任何用例变红。
    """
    engine_timezone("Asia/Shanghai")
    seen: dict[str, object] = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return empty_usage_payload()

    monkeypatch.setattr("nanobot.llm_usage.llm_usage_payload", _spy)

    client = await _client_for(aiohttp_client)
    assert (await client.get("/v1/usage", headers=AUTH_HEADERS)).status == 200
    assert seen == {"timezone_name": "Asia/Shanghai"}


@pytest.mark.asyncio
async def test_usage_still_responds_when_config_is_broken(
    aiohttp_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("config is unreadable")

    monkeypatch.setattr("nanobot.config.loader.load_config", _boom)
    client = await _client_for(aiohttp_client)
    resp = await client.get("/v1/usage", headers=AUTH_HEADERS)
    assert resp.status == 200
    assert (await resp.json())["total_tokens"] == 0
