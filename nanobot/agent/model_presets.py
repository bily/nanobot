"""Helpers for runtime model preset selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import ProviderSnapshot, build_provider_snapshot

PresetSnapshotLoader = Callable[[str], ProviderSnapshot]
PresetCatalogLoader = Callable[[], Mapping[str, ModelPresetConfig]]


def default_selection_signature(
    signature: tuple[object, ...] | None,
    model_preset: str | None = None,
) -> tuple[object, ...] | None:
    return (model_preset, *signature[:2]) if signature else None


def configured_model_presets(config: Config) -> dict[str, ModelPresetConfig]:
    """预设目录，外加一个隐式 `default` 条目（模型选择器的「Auto」行）。

    [LOCAL PATCH] nanowork：`default` 条目必须与**运行时同源**，即走
    `config.resolve_preset()`——它认 `agents.defaults.model_preset`。

    原实现用 `resolve_default_preset()`，那只读 `agents.defaults.model`。于是
    「设为默认」写的是 `model_preset`，运行时确实换了模型，**目录标签却纹丝不动**：
    界面显示 `Auto · 旧模型`，实际跑的是新模型，两者对不上且没有任何报错
    （2026-10-18 用户实测：`modelPreset` 已是 qwen3.8-flash，`Auto` 仍显示
    qwen3.8-2.4t-a95b）。

    悬空指针（`model_preset` 指向已被删除的预设）回落隐式 default 而不是抛：
    目录是展示层，渲染层没资格让整个引擎起不来。
    """
    try:
        default = config.resolve_preset()
    except (KeyError, ValueError):
        default = config.resolve_default_preset()
    # 深拷贝：调用方（resolver 的 model_presets 属性）虽会再拷一层，但目录本身
    # 也不该握着 config 里的活对象——一处就地修改会污染配置。
    return {**config.model_presets, "default": default.model_copy(deep=True)}


def load_model_preset_catalog(
    config_path: Path | None = None,
) -> dict[str, ModelPresetConfig]:
    """Load the current preset catalog from the configured file."""
    from nanobot.config.loader import load_config, resolve_config_env_vars

    return configured_model_presets(
        resolve_config_env_vars(
            load_config(config_path),
            config_path=config_path,
        ),
    )


def make_preset_snapshot_loader(
    config: Config,
    provider_snapshot_loader: Callable[..., ProviderSnapshot] | None,
) -> PresetSnapshotLoader:
    if provider_snapshot_loader is not None:
        return lambda name: provider_snapshot_loader(preset_name=name)
    return lambda name: build_provider_snapshot(config, preset_name=name)


def build_static_preset_snapshot(
    provider: LLMProvider,
    name: str,
    preset: ModelPresetConfig,
) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider=provider,
        model=preset.model,
        context_window_tokens=preset.context_window_tokens,
        signature=("model_preset", name, preset.model_dump_json()),
        generation=preset.to_generation_settings(),
        model_preset=name,
    )


def build_runtime_preset_snapshot(
    *,
    name: str,
    presets: dict[str, ModelPresetConfig],
    provider: LLMProvider,
    loader: PresetSnapshotLoader | None,
) -> ProviderSnapshot:
    if loader is not None:
        return replace(loader(name), model_preset=name)
    return build_static_preset_snapshot(provider, name, presets[name])


def normalize_preset_name(name: str | None, presets: dict[str, ModelPresetConfig]) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model_preset must be a non-empty string")
    name = name.strip()
    if name in presets:
        return name
    matches = [candidate for candidate in presets if candidate.casefold() == name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"model_preset {name!r} not found. Available: {', '.join(presets) or '(none)'}")
