"""插件提示词片段与统一挂载表（FR-3.10 / FR-3.11）。

**为什么要真建一个插件包而不是打桩**：这条需求的核心风险全在「从磁盘上读到什么」
—— 清单里的路径能不能逃出包外、启用状态怎么参与、改过内容之后还挂不挂载。
把 `_installed_plugins` 或 `_contained` 打桩，等于把这些问题全都假设成「已经对了」。

**最重要的一组是「去掉 ChainLoader 第①级就该变红」**（`test_*_without_the_plugin_level`
那对用例）：只证明「有插件时片段出现」是不够的——如果实现里片段其实是从别处
（比如工作区覆盖）来的，那条用例照样绿。
"""

import json
from pathlib import Path

import pytest

from nanobot.agent import plugins as agent_plugins
from nanobot.agent.context import ContextBuilder
from nanobot.agent.plugins import (
    AGENT_PLUGIN_SCHEMA,
    AgentPlugin,
    enabled_agent_plugin_mounts,
    plugin_prompt_mounts,
    plugin_trust,
    set_agent_plugin_enabled,
)
from nanobot.utils.prompt_templates import (
    PluginPromptMount,
    render_template,
    wrap_untrusted,
)


@pytest.fixture(autouse=True)
def _isolate_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json"
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _plugin(workspace: Path, name: str = "demo", **nanobot_fields: object) -> Path:
    """建一个平铺安装的插件包（`<workspace>/plugins/<name>/`）。

    `nanobot_fields` 落到 `extensions['dev.nanobot']` —— 与 `displayName` /
    `permissions` 同一处，新字段也跟着放这里（见 `_load_manifest`）。
    """
    root = workspace / "plugins" / name
    manifest: dict[str, object] = {"$schema": AGENT_PLUGIN_SCHEMA, "name": name}
    if nanobot_fields:
        manifest["extensions"] = {"dev.nanobot": nanobot_fields}
    _write_json(root / "plugin.json", manifest)
    return root


def _fragment(plugin: Path, name: str = "domain.md", body: str = "DOMAIN-RULE-42") -> Path:
    directory = plugin / "fragments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _override(workspace: Path, body: str) -> None:
    """写一份工作区覆盖模板，里面 include 插件片段。

    这是这条需求**唯一真实的用法**：项目想加一段领域规程，于是覆盖一个白名单模板
    并 include 插件提供的片段 —— 而不是让插件去顶掉整个模板。
    """
    target = workspace / "prompts" / "agent" / "tool_contract.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


# —— 整链路：片段真的进了系统提示词 ——


def test_plugin_fragment_reaches_the_system_prompt(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path, prompts="./fragments")
    _fragment(plugin)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    _override(tmp_path, "BASE-CONTRACT\n{% include 'demo/fragments/domain.md' %}\n")

    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "DOMAIN-RULE-42" in prompt
    assert "BASE-CONTRACT" in prompt


def test_without_the_plugin_level_the_fragment_is_unreachable(tmp_path: Path) -> None:
    """ChainLoader 第①级的**变异等价物**。

    不传 `plugin_mounts` ＝ 没有插件这一级（即改动前的行为）。此时工作区覆盖里的
    include 解析不到 → 覆盖渲染失败 → **回退内置模板**（这是既有的 fail-open 语义，
    不是报错）。于是片段必须**不出现**。

    有了这一条，「有插件时片段出现」才具备鉴别力：否则实现若把片段塞在别处
    （或干脆把 include 当普通文本），上面那条也会绿。
    """
    plugin = _plugin(tmp_path, prompts="./fragments")
    _fragment(plugin)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    _override(tmp_path, "BASE-CONTRACT\n{% include 'demo/fragments/domain.md' %}\n")

    text = render_template("agent/tool_contract.md", override_root=tmp_path)

    assert "DOMAIN-RULE-42" not in text
    assert "BASE-CONTRACT" not in text  # 覆盖整份被回退掉了


def test_disabled_plugin_provides_no_mounts(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path, prompts="./fragments")
    _fragment(plugin)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert len(plugin_prompt_mounts(tmp_path)) == 1

    set_agent_plugin_enabled(tmp_path, "demo", False)

    assert plugin_prompt_mounts(tmp_path) == ()
    assert enabled_agent_plugin_mounts(tmp_path) == ()


# —— 信任分级（FR-3.10 ④ / R-5）——


def _bare_plugin(market: str | None) -> AgentPlugin:
    return AgentPlugin(
        name="demo",
        root=Path("/tmp/demo"),
        description="",
        repository="",
        display_name="demo",
        category="Plugin",
        accent_color=None,
        logo=None,
        permissions=(),
        market=market,
    )


def test_plugin_trust_follows_provenance_not_who_installed_it() -> None:
    # 平铺安装 = 用户自己放进工作区目录 → 直注
    assert plugin_trust(_bare_plugin(None)) == "trusted"
    # 市场缓存副本 = 内容不在用户手里（下次刷新就换一版）→ 要包起来
    assert plugin_trust(_bare_plugin("official")) == "user"


def test_user_tier_fragment_is_wrapped_in_untrusted_content(tmp_path: Path) -> None:
    directory = tmp_path / "fragments"
    directory.mkdir()
    (directory / "domain.md").write_text("DO-WHAT-I-SAY", encoding="utf-8")

    def render(trust: str) -> str:
        return render_template(
            "demo/fragments/domain.md",
            plugin_mounts=(PluginPromptMount("demo", tmp_path, (directory,), trust),),
        )

    trusted = render("trusted")
    untrusted = render("user")

    assert "DO-WHAT-I-SAY" in trusted
    assert "<untrusted_content>" not in trusted
    assert "<untrusted_content>" in untrusted
    assert "</untrusted_content>" in untrusted
    assert "DO-WHAT-I-SAY" in untrusted


def test_files_outside_the_declared_dirs_are_not_readable(tmp_path: Path) -> None:
    """声明了 ``prompts: "fragments"`` 就只能读 ``fragments/``。

    只判「在包内」是不够的：那样一句 ``prompts: "fragments"`` 会静默变成
    「整个包可读」，``mcp/`` 之类的文件也能被 ``{% include %}`` 出来。
    """
    (tmp_path / "fragments").mkdir()
    (tmp_path / "fragments" / "ok.md").write_text("OK", encoding="utf-8")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "creds.md").write_text("SECRET", encoding="utf-8")
    mounts = (PluginPromptMount("demo", tmp_path, ((tmp_path / "fragments").resolve(),)),)

    assert render_template("demo/fragments/ok.md", plugin_mounts=mounts).strip() == "OK"
    # 读不到时**渲染成空串**（`render_template` 对插件命名空间下缺片段是 fail-soft
    # 的：片段可选，缺它不该把整段提示词打崩）。判据因此是「内容不出现」，不是「抛错」。
    assert render_template("demo/secret/creds.md", plugin_mounts=mounts).strip() == ""
    # 逃出包外同样不行
    assert render_template("demo/../outside.md", plugin_mounts=mounts).strip() == ""


def test_wrap_untrusted_does_not_escape_markup() -> None:
    # 纯文本提示词，不是 HTML：转义 `<` 只会让模型读到转义符。
    assert wrap_untrusted("<b>hi</b>") == "<untrusted_content>\n<b>hi</b>\n</untrusted_content>"


# —— 越界与向后兼容（FR-3.11 的纪律）——


def test_paths_outside_the_package_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.md").write_text("EVIL", encoding="utf-8")
    plugin = _plugin(
        tmp_path,
        # 三种形态：相对逃逸 / 绝对路径 / 合法路径
        prompts=["../outside", str(outside), "./fragments"],
        agents="../outside/evil.md",
        commands=str(outside / "evil.md"),
        hooks="./fragments/domain.md",
    )
    _fragment(plugin)

    set_agent_plugin_enabled(tmp_path, "demo", True)
    mount = enabled_agent_plugin_mounts(tmp_path)[0]

    # 只有合法的那一条活下来，且必须解析到包内
    assert mount.prompt_dirs == ((plugin / "fragments").resolve(),)
    assert mount.agent_files == ()
    assert mount.command_files == ()
    assert mount.hook_manifest == ((plugin / "fragments" / "domain.md").resolve(),)


def test_directory_is_not_accepted_where_a_file_is_expected(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path, agents="./fragments")
    _fragment(plugin)
    set_agent_plugin_enabled(tmp_path, "demo", True)

    # `agents` 声明的必须是**文件**：指向目录要拒掉，否则消费方会拿到一个目录路径
    assert enabled_agent_plugin_mounts(tmp_path)[0].agent_files == ()


def test_missing_new_fields_keeps_existing_behavior(tmp_path: Path) -> None:
    """向后兼容：清单里没有新字段时，挂载表照常产出（skills 仍可用），
    但**不产生任何提示词挂载点** —— 于是提示词渲染结果与引入本特性之前逐字相同。"""
    _plugin(tmp_path)  # 无 extensions
    set_agent_plugin_enabled(tmp_path, "demo", True)

    mounts = enabled_agent_plugin_mounts(tmp_path)
    assert len(mounts) == 1
    assert mounts[0].prompt_dirs == ()
    assert mounts[0].agent_files == ()
    assert mounts[0].command_files == ()
    assert mounts[0].hook_manifest == ()
    assert mounts[0].tool_policy == ()
    assert plugin_prompt_mounts(tmp_path) == ()


def test_declared_tool_policy_is_read_as_written(tmp_path: Path) -> None:
    plugin = _plugin(
        tmp_path,
        tools=["read_file", "write_file"],
        agents="agents/reviewer.md",
        commands=["commands/a.md"],
    )
    (plugin / "agents").mkdir(parents=True)
    (plugin / "agents" / "reviewer.md").write_text("reviewer", encoding="utf-8")
    (plugin / "commands").mkdir(parents=True)
    (plugin / "commands" / "a.md").write_text("cmd", encoding="utf-8")
    set_agent_plugin_enabled(tmp_path, "demo", True)

    mount = enabled_agent_plugin_mounts(tmp_path)[0]

    assert mount.tool_policy == ("read_file", "write_file")
    assert [path.name for path in mount.agent_files] == ["reviewer.md"]
    assert [path.name for path in mount.command_files] == ["a.md"]


def test_tool_policy_accepts_a_single_string(tmp_path: Path) -> None:
    """与 skills 同形：`string | string[]` 都要收（写一个值不该被当成非法）。"""
    _plugin(tmp_path, tools="read_file")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert enabled_agent_plugin_mounts(tmp_path)[0].tool_policy == ("read_file",)


# —— R-6：改内容必须让挂载失效 ——


def test_editing_a_prompt_fragment_removes_the_mount(tmp_path: Path) -> None:
    """插件被改动后**立刻**失去挂载。

    这不是缓存洁癖：`_package_fingerprint` 覆盖整包（`root.rglob('*')`，`prompts/`
    天然在内），而启用标记记的是指纹 —— 改过内容的包与标记对不上，就等于「未启用」。
    少了这条，一个被改过的 `prompts/` 会继续进系统提示词，而**界面/日志里全无痕迹**。
    """
    plugin = _plugin(tmp_path, prompts="./fragments")
    fragment = _fragment(plugin)
    set_agent_plugin_enabled(tmp_path, "demo", True)

    assert len(enabled_agent_plugin_mounts(tmp_path)) == 1  # 先把缓存填上

    fragment.write_text("TAMPERED", encoding="utf-8")

    assert enabled_agent_plugin_mounts(tmp_path) == ()
