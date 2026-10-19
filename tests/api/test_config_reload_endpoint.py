"""配置热加载端点（FR「保存即生效」）。

背景：客户端在设置里增删模型是**写配置文件**，而运行中的引擎把 modelPresets 装配成
启动时的快照（快照 loader 还闭包住那个 Config 对象）。以前唯一生效手段是重启网关，
UI 只能提示「保存后自动重启网关」。这里钉住「写完配置打一枪就生效」。

关键判据：**新预设必须在同一个进程里出现**，而不是重启后才有。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent import model_presets as preset_helpers
from nanobot.api.server import handle_config_reload
from nanobot.config.schema import Config


def _write_config(path: Path, presets: dict) -> Path:
    cfg = {
        "agents": {
            "workspace": str(path.parent / "ws"),
            "defaults": {"model_preset": "default", "model_preset_default": "default"},
        },
        "providers": {"dashscope": {"api_key": "sk-test"}},
        "modelPresets": presets,
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path


def _preset(name: str, model: str | None = None) -> dict:
    return {
        "model": model or name,
        "provider": "dashscope",
        "maxTokens": 4096,
        "contextWindowTokens": 128000,
    }


class _FakeResolver:
    """只记录 preset 目录的最小 resolver：热加载后目录必须是新对象的内容。"""

    def __init__(self, presets):
        self._presets = dict(presets)
        self.admitted = 0

    def invalidate(self) -> None:  # pragma: no cover - 接口占位
        pass

    def admit(self):
        self.admitted += 1
        return SimpleNamespace(model="fake", model_preset=None, provider=object())

    def resolve_preset(self, name):
        return SimpleNamespace(model=name)


class _FakeLoop:
    """够 AgentLoop.reload_runtime_config 与端点使用的最小替身。"""

    def __init__(self, presets):
        from nanobot.agent.loop import AgentLoop

        self.runtime_resolver = _FakeResolver(presets)
        self.dream_model_preset = None
        self.dream_require_review = True
        self.published: list = []
        self.reload_runtime_config = lambda config: AgentLoop.reload_runtime_config(self, config)
        self._publish_runtime_selection = self.published.append

    @property
    def model_presets(self):
        # 与真实 AgentLoop 同形：属性代理到 resolver（热加载后 resolver 会被换成
        # 真正的 ModelRuntimeResolver，所以这里不能只认假 resolver 的私有字段）。
        resolver = self.runtime_resolver
        if hasattr(resolver, "model_presets"):
            return resolver.model_presets
        return resolver._presets


def _make_loop(presets):
    return _FakeLoop(presets)


def _make_request(app_value: Path | None, loop):
    return SimpleNamespace(app={"config_path": app_value, "agent_loop": loop})


@pytest.mark.asyncio
async def test_reload_applies_new_preset_without_restart(tmp_path) -> None:
    """写进配置的新模型，必须在同一个进程里立刻出现在目录中。"""
    config_path = _write_config(
        tmp_path / "nanobot.config.json",
        {"qwen-a": _preset("qwen-a")},
    )
    config = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    loop = _make_loop(preset_helpers.configured_model_presets(config))
    before = set(loop.runtime_resolver._presets)

    # 客户端写入动作：配置里多一个预设
    _write_config(
        config_path,
        {"qwen-a": _preset("qwen-a"), "qwen-b": _preset("qwen-b")},
    )

    response = await handle_config_reload(_make_request(config_path, loop))
    assert response.status == 200
    body = json.loads(response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body)
    assert body["ok"] is True
    assert "qwen-b" in body["models"]

    # 判据：resolver 整个被换掉，新预设立刻在目录里（不是重启后才有）
    assert "qwen-b" in loop.runtime_resolver.model_presets
    assert "qwen-b" not in before


@pytest.mark.asyncio
async def test_reload_swaps_loader_closure_so_later_invalidate_keeps_new_catalog(
    tmp_path,
) -> None:
    """回归点：loader 闭包若还抓着旧 Config，之后任何 invalidate 都会退回旧目录。"""
    config_path = _write_config(tmp_path / "c.json", {"qwen-a": _preset("qwen-a")})
    old_config = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    loop = _make_loop(preset_helpers.configured_model_presets(old_config))

    _write_config(config_path, {"qwen-a": _preset("qwen-a"), "qwen-b": _preset("qwen-b")})
    new_config = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    loop.reload_runtime_config(new_config)

    # 之后再走一次 invalidate + 读目录：必须仍是新目录（闭包已换成新 config）
    loop.runtime_resolver.invalidate()
    assert "qwen-b" in loop.runtime_resolver.model_presets


@pytest.mark.asyncio
async def test_reload_rejects_broken_config_and_keeps_old_catalog(tmp_path) -> None:
    """配置写坏 → 400，且**现有装配不动**（半套配置比旧配置更糟）。"""
    config_path = _write_config(tmp_path / "c.json", {"qwen-a": _preset("qwen-a")})
    config = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    loop = _make_loop(preset_helpers.configured_model_presets(config))
    original_resolver = loop.runtime_resolver

    config_path.write_text("{ this is not json", encoding="utf-8")
    response = await handle_config_reload(_make_request(config_path, loop))

    assert response.status == 400
    assert loop.runtime_resolver is original_resolver


@pytest.mark.asyncio
async def test_reload_without_config_path_reports_503(tmp_path) -> None:
    """没传配置路径（老网关）→ 明确 503，让客户端能降级提示而不是静默失败。"""
    loop = _make_loop({})
    response = await handle_config_reload(_make_request(None, loop))
    assert response.status == 503