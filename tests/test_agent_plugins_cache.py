"""[LOCAL PATCH] nanowork FR-3.8：插件的**版本化缓存**发现与激活。

锁的是三条真实会失效的性质：

1. **市场刷新不破坏活跃现场**：刷新只往 `cache/` 里**追加**新版本、不动 `current`
   指针，解析结果因此不变（否定用例：新增更高版本后，解析仍停在指针那一版）。
2. **指针坏了不静默回退**：`current` 指向的版本不在磁盘上时当作「未安装」，
   而不是取最新版——静默回退会让用户以为升级成功、其实跑的是另一版。
3. **升级后旧会话不失效**：标记文件是**多条目标记**（每个副本一份签发），
   升级到新版并启用后，钉在旧版上的会话仍然是启用状态。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent import plugins as agent_plugins
from nanobot.agent.plugins import (
    AGENT_PLUGIN_SCHEMA,
    _installed_plugins,
    _plugin_data_dir,
    discover_agent_plugins,
    set_agent_plugin_enabled,
)

MARKET = "cb_teams_marketplace"


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """激活标记与用户级目录都隔离到 tmp：断言不能随宿主机装了什么而变。"""
    monkeypatch.setattr(
        agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json"
    )
    monkeypatch.setenv("NANOWORK_HOME", str(tmp_path / "user-home"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _put_version(
    plugins_dir: Path,
    name: str,
    version: str,
    *,
    market: str = MARKET,
    description: str = "",
) -> Path:
    """在 `cache/<market>/<name>/<version>/` 下放一份包（= 一份生效副本）。"""
    root = plugins_dir / "cache" / market / name / version
    _write_json(
        root / "plugin.json",
        {"$schema": AGENT_PLUGIN_SCHEMA, "name": name, "description": description or version},
    )
    return root


def _point_at(plugins_dir: Path, name: str, version: str, *, market: str = MARKET) -> None:
    (plugins_dir / "cache" / market / name / "current").write_text(version, encoding="utf-8")


def _flat(plugins_dir: Path, name: str) -> Path:
    root = plugins_dir / name
    _write_json(root / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": name})
    return root


def _found(plugins_dir: Path) -> dict[str, agent_plugins.AgentPlugin]:
    workspace = plugins_dir.parent / "workspace"
    workspace.mkdir(exist_ok=True)
    return {item.name: item for item in _installed_plugins(workspace, plugins_dir)}


# ---------------------------------------------------------------------------
# 发现：版本化缓存副本
# ---------------------------------------------------------------------------


def test_cache_copy_is_discovered_with_market_and_version(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "user-home" / "plugins"
    _put_version(plugins_dir, "demo", "1.2.3")
    _point_at(plugins_dir, "demo", "1.2.3")

    found = _found(plugins_dir)
    assert "demo" in found
    assert found["demo"].market == MARKET
    assert found["demo"].version == "1.2.3"
    assert found["demo"].root.name == "1.2.3"


def test_pointer_wins_over_a_newer_version(tmp_path: Path) -> None:
    """⭐ 市场刷新（追加新版本）不得改变解析结果——这是 FR-3.8 的核心。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    _put_version(plugins_dir, "demo", "1.2.3")
    _point_at(plugins_dir, "demo", "1.2.3")

    before = _found(plugins_dir)["demo"].version
    # 市场刷新：拉下来 2.0.0 并放进 cache/，但**不**动指针
    _put_version(plugins_dir, "demo", "2.0.0")
    after = _found(plugins_dir)["demo"].version

    assert before == "1.2.3"
    assert after == "1.2.3", "新增版本后解析跳到了新版：活跃会话会被半路换掉插件定义"


def test_without_pointer_the_highest_version_wins(tmp_path: Path) -> None:
    """无指针时取版本号最大的一版，且 0.1.10 必须大于 0.1.9（不是字符串比）。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    _put_version(plugins_dir, "demo", "0.1.9")
    _put_version(plugins_dir, "demo", "0.1.10")
    _put_version(plugins_dir, "demo", "0.1.2")

    assert _found(plugins_dir)["demo"].version == "0.1.10"


def test_broken_pointer_is_not_installed_instead_of_silent_fallback(tmp_path: Path) -> None:
    """指针指向的版本被 GC 掉了 → 未安装，**不**静默回退到最新版。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    _put_version(plugins_dir, "demo", "1.0.0")
    _put_version(plugins_dir, "demo", "2.0.0")
    _point_at(plugins_dir, "demo", "9.9.9")

    assert "demo" not in _found(plugins_dir)


def test_pointer_escaping_the_plugin_dir_is_rejected(tmp_path: Path) -> None:
    """`current` 里的名字要过字符集检查：`../evil` 不能解析到插件目录之外。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    _put_version(plugins_dir, "demo", "1.0.0")
    _put_version(plugins_dir, "evil", "1.0.0")
    _point_at(plugins_dir, "demo", "../evil")

    found = _found(plugins_dir)
    # 遍历被拒绝 → demo 未安装；evil 没有被 demo 顶掉也不该凭空出现在结果里
    assert "demo" not in found


def test_flat_local_install_beats_cached_copy(tmp_path: Path) -> None:
    """平铺目录 = 用户自己放进去的本地包，语义强于市场副本。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    _flat(plugins_dir, "demo")
    _put_version(plugins_dir, "demo", "1.0.0")
    _point_at(plugins_dir, "demo", "1.0.0")

    found = _found(plugins_dir)
    assert found["demo"].market is None
    assert found["demo"].version is None


def test_missing_cache_dir_is_not_an_error(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "user-home" / "plugins"
    _flat(plugins_dir, "alpha")
    assert set(_found(plugins_dir)) == {"alpha"}


def test_marketplaces_dir_is_not_loaded_directly(tmp_path: Path) -> None:
    """源目录（marketplaces/）不是生效副本：只有同步进 cache/ 才算装上了。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    root = plugins_dir / "marketplaces" / MARKET / "demo" / "1.0.0"
    _write_json(root / "plugin.json", {"$schema": AGENT_PLUGIN_SCHEMA, "name": "demo"})

    assert "demo" not in _found(plugins_dir)


# ---------------------------------------------------------------------------
# 激活：一条标记 = 一份副本的签发（多版本并存）
# ---------------------------------------------------------------------------


def test_enabling_a_version_keeps_the_previous_one_enabled(tmp_path: Path) -> None:
    """⭐ 升级到新版并启用后，钉在旧版上的会话仍是启用状态。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    v1 = _put_version(plugins_dir, "demo", "1.0.0")
    v2 = _put_version(plugins_dir, "demo", "2.0.0")

    _point_at(plugins_dir, "demo", "1.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)
    # 升级：指针翻到 2.0.0（旧副本仍在磁盘上）
    _point_at(plugins_dir, "demo", "2.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)

    assert agent_plugins._enabled_package_fingerprint(workspace, _found(plugins_dir)["demo"])
    # 旧副本单独判定：把指针拨回去，它必须还是启用的
    _point_at(plugins_dir, "demo", "1.0.0")
    assert agent_plugins._enabled_package_fingerprint(workspace, _found(plugins_dir)["demo"])
    assert (v1 / "plugin.json").is_file() and (v2 / "plugin.json").is_file()


def test_disabling_by_name_disables_every_version(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _put_version(plugins_dir, "demo", "1.0.0")
    _put_version(plugins_dir, "demo", "2.0.0")
    _point_at(plugins_dir, "demo", "1.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)
    _point_at(plugins_dir, "demo", "2.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)

    set_agent_plugin_enabled(workspace, "demo", False, plugins_dir)

    for version in ("1.0.0", "2.0.0"):
        _point_at(plugins_dir, "demo", version)
        assert agent_plugins._enabled_package_fingerprint(workspace, _found(plugins_dir)["demo"]) is None


def test_modified_copy_loses_its_activation(tmp_path: Path) -> None:
    """安全属性保持：副本内容被就地改掉 → 指纹不匹配 → 不再算启用。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = _put_version(plugins_dir, "demo", "1.0.0")
    _point_at(plugins_dir, "demo", "1.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)
    assert agent_plugins._enabled_package_fingerprint(workspace, _found(plugins_dir)["demo"])

    (root / "extra.txt").write_text("tampered", encoding="utf-8")
    assert agent_plugins._enabled_package_fingerprint(workspace, _found(plugins_dir)["demo"]) is None


def test_legacy_single_entry_marker_is_still_honored(tmp_path: Path) -> None:
    """一次引擎升级不能把用户已启用的包读成未启用：v1 形态必须继续认。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = _put_version(plugins_dir, "demo", "1.0.0")
    _point_at(plugins_dir, "demo", "1.0.0")

    plugin = _found(plugins_dir)["demo"]
    fingerprint = agent_plugins._package_fingerprint(root)
    data = _plugin_data_dir(workspace, "demo", create=True)
    (data / "enabled").write_text(
        json.dumps({"fingerprint": fingerprint, "root": str(root)}), encoding="utf-8"
    )

    assert agent_plugins._enabled_package_fingerprint(workspace, plugin) == fingerprint


def test_gc_prunes_entries_whose_copy_is_gone(tmp_path: Path) -> None:
    """副本被回收后，标记里那一条要在下次写入时被清掉（否则文件无限长）。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stale = _put_version(plugins_dir, "demo", "0.9.0")
    _put_version(plugins_dir, "demo", "1.0.0")
    _point_at(plugins_dir, "demo", "0.9.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)

    import shutil

    shutil.rmtree(stale)
    _point_at(plugins_dir, "demo", "1.0.0")
    set_agent_plugin_enabled(workspace, "demo", True, plugins_dir)

    marker = _plugin_data_dir(workspace, "demo", create=False) / "enabled"
    entries = json.loads(marker.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 1
    assert Path(entries[0]["root"]).name == "1.0.0"


def test_discovery_reports_market_and_version_in_payload(tmp_path: Path) -> None:
    """/v1/plugins 要能把「从哪来、哪一版」交给客户端（界面据此显示来源）。"""
    plugins_dir = tmp_path / "user-home" / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _put_version(plugins_dir, "demo", "3.2.1")
    _point_at(plugins_dir, "demo", "3.2.1")

    found = {item.name: item for item in discover_agent_plugins(workspace, plugins_dir)}
    assert found["demo"].market == MARKET
    assert found["demo"].version == "3.2.1"
