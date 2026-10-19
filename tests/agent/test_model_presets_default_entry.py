"""模型目录里的隐式 `default` 条目（nanowork「设为默认即生效」的上游保证）。

回归背景（2026-10-18 用户实测）：把 `qwen3.8-flash` 设为默认后，配置里
`agents.defaults.modelPreset` 确实写成了它、运行时也确实换了模型，但界面上的
「Auto」仍显示旧的 `qwen3.8-2.4t-a95b`。根因是目录条目与运行时**不同源**：

- 运行时：`config.resolve_preset()` → 认 `model_preset` ✅
- 目录标签：`config.resolve_default_preset()` → 只读 `agents.defaults.model` ❌

这里钉住「两者必须同源」，以及悬空指针不许把引擎带崩。
"""

import pytest

from nanobot.agent.model_presets import configured_model_presets
from nanobot.config.schema import Config


def _config(payload: dict) -> Config:
    return Config.model_validate(payload)


def _preset(name: str, model: str | None = None) -> dict:
    return {
        "model": model or name,
        "provider": "dashscope",
        "maxTokens": 4096,
        "contextWindowTokens": 128000,
    }


def test_default_entry_follows_model_preset_not_stale_bare_model() -> None:
    """指向预设时，`default` 条目 = 那个预设（而不是 agents.defaults.model）。"""
    cfg = _config({
        "agents": {
            "workspace": "D:/tmp/ws",
            # model 是历史裸串：设为默认前它管用，设为默认后应当**不再**决定目录标签
            "defaults": {"model": "qwen3.8-2.4t-a95b", "model_preset": "qwen3.8-flash"},
        },
        "providers": {"dashscope": {"api_key": "sk-x"}},
        "modelPresets": {
            "qwen3.8-2.4t-a95b": _preset("qwen3.8-2.4t-a95b"),
            "qwen3.8-flash": _preset("qwen3.8-flash"),
        },
    })

    presets = configured_model_presets(cfg)

    assert presets["default"].model == "qwen3.8-flash"
    # 判据的核心：与运行时解析同源，两个函数必须给出同一个模型
    assert presets["default"].model == cfg.resolve_preset().model


def test_default_entry_falls_back_to_implicit_when_no_preset_pointer() -> None:
    """没设默认预设时，行为与旧实现一致（读 agents.defaults.model）。"""
    cfg = _config({
        "agents": {"workspace": "D:/tmp/ws", "defaults": {"model": "qwen3.8-2.4t-a95b"}},
        "providers": {"dashscope": {"api_key": "sk-x"}},
        "modelPresets": {"qwen3.8-flash": _preset("qwen3.8-flash")},
    })

    presets = configured_model_presets(cfg)

    assert presets["default"].model == "qwen3.8-2.4t-a95b"
    # 配置里所有显式预设仍原样在目录里（改名 / 删除预设是另一条链路）
    assert "qwen3.8-flash" in presets


def test_dangling_preset_pointer_falls_back_instead_of_raising() -> None:
    """指针悬空时回落隐式 default —— 目录是展示层，没资格让引擎起不来。

    注意构造方式：`Config.model_validate` **自己**就会拒绝悬空指针（上游校验器），
    所以这里先合法加载、再在内存里把指针改坏——模拟的是「加载之后预设被删/改名」
    这类防御场景。想验的判据是：即便如此，目录构建也不抛，且回落仍可用。
    """
    cfg = _config({
        "agents": {
            "workspace": "D:/tmp/ws",
            "defaults": {"model": "qwen3.8-2.4t-a95b", "model_preset": "qwen3.8-flash"},
        },
        "providers": {"dashscope": {"api_key": "sk-x"}},
        "modelPresets": {"qwen3.8-flash": _preset("qwen3.8-flash")},
    })
    cfg.agents.defaults.model_preset = "已删除的预设"

    presets = configured_model_presets(cfg)

    assert presets["default"].model == "qwen3.8-2.4t-a95b"


def test_default_entry_is_a_copy_not_the_live_config_object() -> None:
    """目录不得握着 config 里的活对象：就地改目录会污染配置本身。"""
    cfg = _config({
        "agents": {"workspace": "D:/tmp/ws", "defaults": {"model_preset": "qwen3.8-flash"}},
        "providers": {"dashscope": {"api_key": "sk-x"}},
        "modelPresets": {"qwen3.8-flash": _preset("qwen3.8-flash")},
    })

    presets = configured_model_presets(cfg)
    presets["default"].model = "tampered"

    assert cfg.resolve_preset().model == "qwen3.8-flash"