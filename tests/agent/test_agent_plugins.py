import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from nanobot.agent import plugins as agent_plugins
from nanobot.agent.plugins import (
    AGENT_PLUGIN_MCP_SCHEMA,
    AGENT_PLUGIN_SCHEMA,
    agent_plugin_mcp_servers,
    discover_agent_plugins,
    enabled_agent_plugin_skill_dirs,
    enabled_agent_plugin_skills,
    set_agent_plugin_enabled,
)
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)


@pytest.fixture(autouse=True)
def _isolate_plugin_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_plugins, "get_config_path", lambda: tmp_path / "config" / "config.json"
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(name: str, **fields: object) -> dict[str, object]:
    return {"$schema": AGENT_PLUGIN_SCHEMA, "name": name, **fields}


def _plugin(workspace: Path, name: str = "demo", **fields: object) -> Path:
    root = workspace / "plugins" / name
    _write_json(root / "plugin.json", _manifest(name, **fields))
    return root


def _skill(root: Path, name: str, frontmatter: str | None = None, body: str = "") -> Path:
    path = root / name
    path.mkdir(parents=True)
    metadata = frontmatter or f"name: {name}\ndescription: Plugin skill."
    (path / "SKILL.md").write_text(f"---\n{metadata}\n---\n\n{body}\n", encoding="utf-8")
    return path


def _loaded_skills(workspace: Path) -> list[str]:
    return [name for name, _ in enabled_agent_plugin_skills(workspace)]


def test_plugin_skill_lifecycle_and_precedence(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    _skill(
        plugin / "skills",
        "shared",
        "name: shared\ndescription: Plugin version.\nalways: true",
        "Plugin body.",
    )
    _skill(tmp_path / "builtin", "shared", body="Built-in body.")
    workspace_skill = _skill(
        tmp_path / "skills", "shared", "name: shared\ndescription: Workspace version."
    )
    loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

    assert [entry["source"] for entry in loader.list_skills()] == ["workspace"]
    assert "Workspace version" in (loader.load_skill("shared") or "")
    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert [entry["source"] for entry in loader.list_skills()] == ["workspace"]

    shutil.rmtree(workspace_skill)
    assert [entry["source"] for entry in loader.list_skills()] == ["plugin"]
    assert loader.get_explicitly_invoked_skills("Use $shared") == ["shared"]
    assert loader.get_always_skills() == ["shared"]
    assert "Plugin body" in (loader.load_skill("shared") or "")
    summary = loader.build_skills_summary()
    assert "### Agent Plugin skills (`plugins`)" in summary
    assert "`demo/skills/shared/SKILL.md`" in summary
    assert str(tmp_path.resolve()) not in summary

    set_agent_plugin_enabled(tmp_path, "demo", False)
    assert [entry["source"] for entry in loader.list_skills()] == ["builtin"]
    assert "Built-in body" in (loader.load_skill("shared") or "")


def test_plugin_skills_are_direct_valid_and_contained(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    skills = plugin / "skills"
    _skill(skills, "direct")
    _skill(skills / "group", "nested")
    for name, frontmatter in (
        ("wrong-directory", "name: another\ndescription: Mismatch."),
        ("missing-description", "name: missing-description"),
        ("Bad-Name", "name: Bad-Name\ndescription: Invalid name."),
    ):
        _skill(skills, name, frontmatter)
    outside = _skill(tmp_path / "outside", "escaped")
    try:
        (skills / "escaped").symlink_to(outside, target_is_directory=True)
    except OSError:
        pass

    set_agent_plugin_enabled(tmp_path, "demo", True)
    assert _loaded_skills(tmp_path) == ["direct"]


@pytest.mark.parametrize(
    ("manifest", "valid"),
    [
        ({"$schema": "https://agent-plugins.org/schemas/2.0.0/plugin.schema.json", "name": "demo"}, False),
        (_manifest("Bad-Name"), False),
        (_manifest("demo", futureField=True, extensions="invalid but non-fatal"), True),
    ],
)
def test_plugin_manifest_boundary(tmp_path: Path, manifest: object, valid: bool) -> None:
    _write_json(tmp_path / "plugins" / "candidate" / "plugin.json", manifest)
    assert bool(discover_agent_plugins(tmp_path)) is valid


def test_plugin_logo_is_validated_and_contained(tmp_path: Path) -> None:
    extension = {"extensions": {"dev.nanobot": {"logo": "./assets/icon.png"}}}
    plugin = _plugin(tmp_path, "demo", **extension)
    icon = plugin / "assets" / "icon.png"
    icon.parent.mkdir()
    icon.write_bytes(b"\x89PNG\r\n\x1a\nlogo")
    escaped = _plugin(tmp_path, "escaped", **extension)
    (escaped / "assets").mkdir()
    try:
        (escaped / "assets" / "icon.png").symlink_to(icon)
    except OSError:
        pass

    assert {plugin.name: plugin.logo for plugin in discover_agent_plugins(tmp_path)} == {
        "demo": "data:image/png;base64,iVBORw0KGgpsb2dv",
        "escaped": None,
    }


def test_plugin_mcp_requires_explicit_enable(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path, "desktop")
    executable = plugin / "bin" / "server"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "desktop": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "args": ["--data", "${PLUGIN_DATA}/state"],
                    "cwd": "${PLUGIN_ROOT}",
                },
                "public-http": {"type": "streamable-http", "url": "http://example.com/mcp"},
                "escape": {"type": "stdio", "command": "../outside"},
            },
        },
    )

    assert agent_plugin_mcp_servers(tmp_path) == {}
    set_agent_plugin_enabled(tmp_path, "desktop", True)
    server = agent_plugin_mcp_servers(tmp_path)["desktop"]
    assert (server.command, server.cwd, server.env["PLUGIN_ROOT"]) == (
        str(executable),
        str(plugin),
        str(plugin),
    )
    assert server.args[1].endswith("/state")
    set_agent_plugin_enabled(tmp_path, "desktop", False)
    assert agent_plugin_mcp_servers(tmp_path) == {}


def test_plugin_mcp_namespaces_cannot_shadow_plugin_identities(tmp_path: Path) -> None:
    single = _plugin(tmp_path, "foo-bar")
    multi = _plugin(tmp_path, "foo")
    for root, servers in (
        (single, {"main": {"type": "stdio", "command": "echo", "args": ["single"]}}),
        (
            multi,
            {
                "bar": {"type": "stdio", "command": "echo", "args": ["multi"]},
                "other": {"type": "stdio", "command": "echo"},
            },
        ),
    ):
        _write_json(
            root / "mcp.json",
            {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": servers},
        )
    set_agent_plugin_enabled(tmp_path, "foo-bar", True)
    set_agent_plugin_enabled(tmp_path, "foo", True)

    servers = agent_plugin_mcp_servers(tmp_path)

    assert set(servers) == {"foo-bar", "foo--bar", "foo--other"}
    assert servers["foo-bar"].args == ["single"]
    assert servers["foo--bar"].args == ["multi"]


@pytest.mark.asyncio
async def test_restricted_project_can_read_only_enabled_plugin_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    plugin = _plugin(agent_workspace)
    skill = _skill(plugin / "skills", "demo-skill")
    resource = skill / "reference.md"
    resource.write_text("plugin reference", encoding="utf-8")
    ctx = ToolContext(
        config=ToolsConfig(restrict_to_workspace=True),
        workspace=str(agent_workspace),
    )
    read_tool = ReadFileTool.create(ctx)
    write_tool = WriteFileTool.create(ctx)
    set_agent_plugin_enabled(agent_workspace, "demo", True)
    activation_checks = 0
    activation_marker = agent_plugins._activation_marker

    def count_activation_checks(plugin: agent_plugins.AgentPlugin) -> str | None:
        nonlocal activation_checks
        activation_checks += 1
        return activation_marker(plugin)

    monkeypatch.setattr(agent_plugins, "_activation_marker", count_activation_checks)
    scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=agent_workspace,
        default_restrict_to_workspace=True,
    )

    token = bind_workspace_scope(scope)
    try:
        read_result = await read_tool.execute(path=str(resource))
        repeated_read_result = await read_tool.execute(path=str(resource))
        write_result = await write_tool.execute(path=str(resource), content="changed")
        set_agent_plugin_enabled(agent_workspace, "demo", False)
        disabled_result = await read_tool.execute(path=str(resource))
    finally:
        reset_workspace_scope(token)

    assert "plugin reference" in read_result
    assert "File unchanged since last read" in repeated_read_result
    assert activation_checks == 1
    assert "outside allowed directory" in write_result
    assert "outside allowed directory" in disabled_result
    assert resource.read_text(encoding="utf-8") == "plugin reference"


@pytest.mark.asyncio
async def test_restricted_project_revokes_cached_plugin_read_after_replacement(
    tmp_path: Path,
) -> None:
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    plugin = _plugin(agent_workspace)
    skill = _skill(plugin / "skills", "demo-skill")
    resource = skill / "reference.md"
    resource.write_text("trusted", encoding="utf-8")
    read_tool = ReadFileTool.create(
        ToolContext(
            config=ToolsConfig(restrict_to_workspace=True),
            workspace=str(agent_workspace),
        )
    )
    set_agent_plugin_enabled(agent_workspace, "demo", True)
    scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=agent_workspace,
        default_restrict_to_workspace=True,
    )

    token = bind_workspace_scope(scope)
    try:
        trusted_result = await read_tool.execute(path=str(resource))
        original_stat = resource.stat()
        resource.write_text("hostile", encoding="utf-8")
        os.utime(
            resource,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        replacement_result = await read_tool.execute(path=str(resource))
    finally:
        reset_workspace_scope(token)

    assert "trusted" in trusted_result
    assert "outside allowed directory" in replacement_result
    assert discover_agent_plugins(agent_workspace)[0].enabled is False


@pytest.mark.asyncio
async def test_project_reads_do_not_rescan_cached_plugin_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    project_file = project / "notes.md"
    project_file.write_text("project notes", encoding="utf-8")
    plugin = _plugin(agent_workspace)
    skill = _skill(plugin / "skills", "demo-skill")
    resource = skill / "reference.md"
    resource.write_text("plugin reference", encoding="utf-8")
    set_agent_plugin_enabled(agent_workspace, "demo", True)
    assert enabled_agent_plugin_skill_dirs(agent_workspace) == (skill,)

    fingerprint_checks = 0
    package_fingerprint = agent_plugins._package_fingerprint

    def count_fingerprint_checks(root: Path) -> str | None:
        nonlocal fingerprint_checks
        fingerprint_checks += 1
        return package_fingerprint(root)

    monkeypatch.setattr(agent_plugins, "_package_fingerprint", count_fingerprint_checks)
    read_tool = ReadFileTool.create(
        ToolContext(
            config=ToolsConfig(restrict_to_workspace=True),
            workspace=str(agent_workspace),
        )
    )
    scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=agent_workspace,
        default_restrict_to_workspace=True,
    )

    token = bind_workspace_scope(scope)
    try:
        project_result = await read_tool.execute(path=str(project_file))
        assert fingerprint_checks == 0
        plugin_result = await read_tool.execute(path=str(resource))
    finally:
        reset_workspace_scope(token)

    assert "project notes" in project_result
    assert "plugin reference" in plugin_result
    assert fingerprint_checks == 1


def test_plugin_state_symlink_cannot_escape_config_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (config / "plugin-data").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    monkeypatch.setattr(agent_plugins, "get_config_path", lambda: config / "config.json")
    _plugin(tmp_path, "desktop")

    with pytest.raises(RuntimeError, match="escapes its parent"):
        set_agent_plugin_enabled(tmp_path, "desktop", True)


def test_plugin_activation_requires_one_stable_package_identity(tmp_path: Path) -> None:
    roots = [tmp_path / "plugins" / directory for directory in ("first", "second")]
    for root, marker in zip(roots, ("trusted", "replacement"), strict=True):
        _write_json(root / "plugin.json", _manifest("duplicate"))
        _write_json(
            root / "mcp.json",
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "server": {"type": "stdio", "command": "echo", "args": [marker]}
                },
            },
        )

    assert discover_agent_plugins(tmp_path) == []
    with pytest.raises(ValueError, match="unknown Agent Plugin"):
        set_agent_plugin_enabled(tmp_path, "duplicate", True)

    shutil.rmtree(roots[1])
    set_agent_plugin_enabled(tmp_path, "duplicate", True)
    assert discover_agent_plugins(tmp_path)[0].enabled is True

    moved = tmp_path / "plugins" / "moved"
    roots[0].rename(moved)
    assert discover_agent_plugins(tmp_path)[0].enabled is False
    assert agent_plugin_mcp_servers(tmp_path) == {}


def test_legacy_path_activation_is_upgraded_to_package_fingerprint(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    set_agent_plugin_enabled(tmp_path, "demo", True)
    marker = next((tmp_path / "config" / "plugin-data").glob("*/demo/enabled"))
    marker.write_text(str(plugin), encoding="utf-8")

    assert discover_agent_plugins(tmp_path)[0].enabled is True
    assert marker.read_text(encoding="utf-8").startswith('{"fingerprint":')


def test_plugin_activation_does_not_survive_in_place_contract_replacement(
    tmp_path: Path,
) -> None:
    plugin = _plugin(tmp_path, "desktop")
    mcp = plugin / "mcp.json"

    def write_server(marker: str) -> None:
        _write_json(
            mcp,
            {
                "$schema": AGENT_PLUGIN_MCP_SCHEMA,
                "mcpServers": {
                    "server": {"type": "stdio", "command": "echo", "args": [marker]}
                },
            },
        )

    write_server("trusted")
    set_agent_plugin_enabled(tmp_path, "desktop", True)
    assert agent_plugin_mcp_servers(tmp_path)["desktop"].args == ["trusted"]

    write_server("replacement")

    assert discover_agent_plugins(tmp_path)[0].enabled is False
    assert agent_plugin_mcp_servers(tmp_path) == {}


def test_plugin_activation_does_not_survive_in_place_code_replacement(
    tmp_path: Path,
) -> None:
    plugin = _plugin(tmp_path, "desktop")
    _skill(plugin / "skills", "demo")
    executable = plugin / "server.py"
    executable.write_text("print('trusted')\n", encoding="utf-8")
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "server": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["${PLUGIN_ROOT}/server.py"],
                }
            },
        },
    )
    set_agent_plugin_enabled(tmp_path, "desktop", True)
    assert discover_agent_plugins(tmp_path)[0].enabled is True
    assert enabled_agent_plugin_skill_dirs(tmp_path) == (plugin / "skills" / "demo",)

    executable.write_text("print('replacement')\n", encoding="utf-8")

    assert discover_agent_plugins(tmp_path)[0].enabled is False
    assert enabled_agent_plugin_skill_dirs(tmp_path) == ()
    assert agent_plugin_mcp_servers(tmp_path) == {}


# ---------------------------------------------------------------------------
# [LOCAL PATCH] FR-3.4：用户级插件目录（~/.nanowork/plugins）
# ---------------------------------------------------------------------------


def _user_plugin(user_dir: Path, name: str = "demo", **fields: object) -> Path:
    root = user_dir / name
    _write_json(root / "plugin.json", _manifest(name, **fields))
    return root


def _loaded_skills_for(workspace: Path, user_dir: Path) -> list[str]:
    return [name for name, _ in enabled_agent_plugin_skills(workspace, user_dir)]


def test_user_level_plugin_is_discovered_and_toggleable(tmp_path: Path) -> None:
    user_dir = tmp_path / "user-plugins"
    plugin = _user_plugin(user_dir)
    _skill(plugin / "skills", "demo-skill")

    assert discover_agent_plugins(tmp_path, user_dir)[0].enabled is False
    set_agent_plugin_enabled(tmp_path, "demo", True, user_dir)
    assert discover_agent_plugins(tmp_path, user_dir)[0].enabled is True
    assert _loaded_skills_for(tmp_path, user_dir) == ["demo-skill"]
    set_agent_plugin_enabled(tmp_path, "demo", False, user_dir)
    assert discover_agent_plugins(tmp_path, user_dir)[0].enabled is False
    assert _loaded_skills_for(tmp_path, user_dir) == []


def test_user_level_plugins_stay_invisible_without_the_directory(tmp_path: Path) -> None:
    """没注入用户级目录就完全看不见它——单测的隔离性靠这一条守住。

    这是刻意的：``_installed_plugins`` 不自己兜 ``default_user_plugins_dir()``，
    否则每个构造 Agent 的单测都会去读开发者宿主机上真实装了什么。
    """
    user_dir = tmp_path / "user-plugins"
    plugin = _user_plugin(user_dir)
    _skill(plugin / "skills", "demo-skill")
    set_agent_plugin_enabled(tmp_path, "demo", True, user_dir)

    assert enabled_agent_plugin_skills(tmp_path) == []
    assert discover_agent_plugins(tmp_path) == []


def test_project_plugin_shadows_user_plugin_with_same_identity(tmp_path: Path) -> None:
    """同名时项目级赢——扫描顺序是先到先得，所以项目级必须排在前面。"""
    user_dir = tmp_path / "user-plugins"
    user_plugin = _user_plugin(user_dir, "demo", description="user copy")
    _skill(user_plugin / "skills", "user-only")
    project_plugin = _plugin(tmp_path, "demo", description="project copy")
    _skill(project_plugin / "skills", "project-only")

    set_agent_plugin_enabled(tmp_path, "demo", True, user_dir)

    discovered = discover_agent_plugins(tmp_path, user_dir)
    assert [plugin.description for plugin in discovered] == ["project copy"]
    assert discovered[0].root == project_plugin
    assert _loaded_skills_for(tmp_path, user_dir) == ["project-only"]


def test_user_level_plugin_mcp_servers_require_enable(tmp_path: Path) -> None:
    user_dir = tmp_path / "user-plugins"
    plugin = _user_plugin(user_dir)
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {"srv": {"type": "stdio", "command": "python"}},
        },
    )

    assert agent_plugin_mcp_servers(tmp_path, None, user_dir) == {}
    set_agent_plugin_enabled(tmp_path, "demo", True, user_dir)
    assert list(agent_plugin_mcp_servers(tmp_path, None, user_dir)) == ["demo"]


def test_user_level_plugin_packages_with_distinct_names_coexist(tmp_path: Path) -> None:
    user_dir = tmp_path / "user-plugins"
    for name in ("alpha", "beta"):
        _skill(_user_plugin(user_dir, name) / "skills", f"{name}-skill")
        set_agent_plugin_enabled(tmp_path, name, True, user_dir)

    assert sorted(plugin.name for plugin in discover_agent_plugins(tmp_path, user_dir)) == [
        "alpha",
        "beta",
    ]
    assert sorted(_loaded_skills_for(tmp_path, user_dir)) == ["alpha-skill", "beta-skill"]


# ---------------------------------------------------------------------------
# [LOCAL PATCH] FR-3.4：组件位置回退（PRD §12.2 的 .nanowork-plugin / mcp/）
# ---------------------------------------------------------------------------


def test_alternate_component_locations_are_accepted(tmp_path: Path) -> None:
    root = tmp_path / "plugins" / "demo"
    _write_json(root / ".nanowork-plugin" / "plugin.json", _manifest("demo"))
    _skill(root / "skills", "demo-skill")
    _write_json(
        root / "mcp" / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {"srv": {"type": "stdio", "command": "python"}},
        },
    )

    discovered = discover_agent_plugins(tmp_path)
    assert [plugin.name for plugin in discovered] == ["demo"]
    assert discovered[0].mcp_servers == ("srv",)


def test_root_manifest_wins_over_alternate_location(tmp_path: Path) -> None:
    root = tmp_path / "plugins" / "demo"
    _write_json(root / "plugin.json", _manifest("demo", description="root copy"))
    _write_json(
        root / ".nanowork-plugin" / "plugin.json",
        _manifest("demo", description="dotdir copy"),
    )

    assert discover_agent_plugins(tmp_path)[0].description == "root copy"


def test_broken_root_manifest_does_not_silently_fall_back(tmp_path: Path) -> None:
    """高优先级位置存在但内容坏，就是明确失败——不回退到备用位置。

    回退会让一份坏清单被另一份好清单掩盖，出问题时无从解释。
    """
    root = tmp_path / "plugins" / "demo"
    root.mkdir(parents=True)
    (root / "plugin.json").write_text("{ this is not json", encoding="utf-8")
    _write_json(root / ".nanowork-plugin" / "plugin.json", _manifest("demo"))

    assert discover_agent_plugins(tmp_path) == []


# ---------------------------------------------------------------------------
# [LOCAL PATCH] FR-3.4：用户级插件的技能必须既列得出来、也读得进去
# ---------------------------------------------------------------------------


def test_user_plugin_skill_appears_in_summary_with_usable_path(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    user_dir = tmp_path / "user-plugins"
    plugin = _user_plugin(user_dir)
    _skill(plugin / "skills", "demo-skill", "name: demo-skill\ndescription: User plugin skill.")
    set_agent_plugin_enabled(workspace, "demo", True, user_dir)

    summary = SkillsLoader(workspace, user_plugins_dir=user_dir).build_skills_summary()

    assert "demo-skill" in summary
    # 用户级插件不在 <workspace>/plugins 之下：硬算相对路径会抛 ValueError，
    # 算出来也会把模型指向别的地方——所以必须落成绝对路径。
    assert "demo-skill/SKILL.md" in summary


@pytest.mark.asyncio
async def test_user_level_plugin_skill_is_readable_when_restricted(tmp_path: Path) -> None:
    """用户级插件的技能要同时满足"列得出来"和"读得进去"。

    少了 ``ToolContext.user_plugins_dir``，技能会出现在提示词清单里却在
    ``read_file`` 时被判越权——看得见、读不到。这里用**同一个值**同时喂给
    ``ToolsConfig`` 侧和文件工具侧，镜像生产接线（``loop.py`` 用
    ``default_user_plugins_dir()`` 一次性注入两处）。
    """
    agent_workspace = tmp_path / "agent"
    user_dir = tmp_path / "home" / "plugins"
    plugin = _user_plugin(user_dir)
    skill = _skill(plugin / "skills", "demo-skill")
    resource = skill / "reference.md"
    resource.write_text("user plugin reference", encoding="utf-8")
    set_agent_plugin_enabled(agent_workspace, "demo", True, user_dir)

    project = tmp_path / "project"
    project.mkdir()
    scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=agent_workspace,
        default_restrict_to_workspace=True,
    )

    def build_read_tool(**ctx_kwargs: object):
        return ReadFileTool.create(
            ToolContext(
                config=ToolsConfig(restrict_to_workspace=True),
                workspace=str(agent_workspace),
                **ctx_kwargs,  # type: ignore[arg-type]
            )
        )

    async def read_with(tool) -> str:
        token = bind_workspace_scope(scope)
        try:
            return await tool.execute(path=str(resource))
        finally:
            reset_workspace_scope(token)

    # 注入了目录 → 读得到。
    shared = build_read_tool(user_plugins_dir=user_dir)
    assert "user plugin reference" in await read_with(shared)

    # 没注入 → **fail-closed**，明确拒绝（而不是悄悄放行或误报可用）。
    denied_tool = build_read_tool()
    denied = await read_with(denied_tool)
    assert "user plugin reference" not in denied
    assert denied.startswith("Error:")


def test_agent_registers_tool_context_with_the_user_plugins_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """守住生产接线：``Agent`` 必须把它收到的用户级插件目录传进 ``ToolContext``。

    上面那条读闸口用例是**手工注入** ctx 的，所以它测不出「``loop.py`` 忘了接线」。
    而这正是最容易静默失灵的一处：忘了接，技能"列得出来、读不到"，没有任何
    报错，只有模型反复读文件失败。这里用轻量 stub 直接覆盖 ``Agent`` 的真实
    构造路径——比端到端起一个 Agent 便宜得多。
    """
    from types import SimpleNamespace

    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.loader import ToolLoader

    user_dir = tmp_path / "user-plugins"
    captured: list[object] = []

    def fake_load(self_, ctx, registry, scope=None):  # noqa: ANN001, ARG001
        captured.append(ctx)
        return []

    monkeypatch.setattr(ToolLoader, "load", fake_load)

    stub = SimpleNamespace(
        tools_config=ToolsConfig(),
        workspace=tmp_path,
        bus=None,
        subagents=None,
        cron_service=None,
        _exec_session_manager=None,
        sessions=None,
        _image_generation_provider_configs=None,
        context=SimpleNamespace(timezone="UTC"),
        workspace_scopes=SimpleNamespace(sandbox_status=None),
        runtime_events=None,
        tools=SimpleNamespace(),
        user_plugins_dir=user_dir,
    )
    AgentLoop._register_default_tools(stub, provider_snapshot_loader=None)

    assert len(captured) == 1
    ctx = cast(Any, captured[0])
    assert ctx.user_plugins_dir == user_dir
