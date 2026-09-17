"""Workspace prompt overrides (design §13.4 / ADR-007).

Covers the [LOCAL PATCH] on ``render_template``:

- ``override_root=`` prefers ``<root>/prompts/<name>`` over the bundled template
- only whitelisted templates can be overridden
- a broken override degrades to the bundled template instead of raising
- ``initialize_workspace_templates`` seeds defaults without clobbering user edits
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.utils.prompt_templates import (
    OVERRIDABLE_TEMPLATES,
    initialize_workspace_templates,
    load_workspace_template,
    render_template,
    workspace_template_path,
)


def _write_override(root: Path, name: str, text: str) -> Path:
    path = workspace_template_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# render_template: 覆盖优先
# ---------------------------------------------------------------------------


class TestRenderTemplateOverride:
    def test_no_root_uses_bundled_template(self, tmp_path: Path) -> None:
        rendered = render_template("agent/tool_contract.md")
        assert rendered.strip(), "bundled template should render non-empty"

    def test_root_without_override_uses_bundled(self, tmp_path: Path) -> None:
        with_root = render_template("agent/tool_contract.md", override_root=tmp_path)
        assert with_root == render_template("agent/tool_contract.md")

    def test_override_wins_over_bundled(self, tmp_path: Path) -> None:
        _write_override(tmp_path, "agent/tool_contract.md", "CUSTOM CONTRACT")
        assert render_template("agent/tool_contract.md", override_root=tmp_path).strip() == (
            "CUSTOM CONTRACT"
        )

    def test_override_still_renders_template_variables(self, tmp_path: Path) -> None:
        _write_override(
            tmp_path,
            "agent/identity.md",
            "runtime={{ runtime }} workspace={{ workspace_path }}",
        )
        rendered = render_template(
            "agent/identity.md",
            override_root=tmp_path,
            runtime="Windows AMD64",
            workspace_path="/tmp/ws",
            agent_workspace_path="/tmp/agent",
            platform_policy="POLICY",
            channel="",
        )
        assert rendered == "runtime=Windows AMD64 workspace=/tmp/ws"

    def test_empty_override_falls_back(self, tmp_path: Path) -> None:
        _write_override(tmp_path, "agent/tool_contract.md", "   \n\n  ")
        assert render_template(
            "agent/tool_contract.md", override_root=tmp_path
        ) == render_template("agent/tool_contract.md")

    def test_broken_override_falls_back_instead_of_raising(self, tmp_path: Path) -> None:
        # 坏掉的覆盖只该降级，不该让整轮会话起不来
        _write_override(tmp_path, "agent/tool_contract.md", "{% if unclosed %}")
        rendered = render_template("agent/tool_contract.md", override_root=tmp_path)
        assert rendered == render_template("agent/tool_contract.md")

    def test_non_whitelisted_template_ignores_override(self, tmp_path: Path) -> None:
        # 覆盖面是显式白名单：include 片段与 dream/evaluator 不得被顶掉
        assert "agent/_snippets/untrusted_content.md" not in OVERRIDABLE_TEMPLATES
        _write_override(tmp_path, "agent/_snippets/untrusted_content.md", "HIJACKED")
        rendered = render_template(
            "agent/_snippets/untrusted_content.md",
            override_root=tmp_path,
            content="x",
        )
        assert "HIJACKED" not in rendered

    def test_override_root_is_not_shadowed_by_workspace_variable(self, tmp_path: Path) -> None:
        """回归：``override_root`` 曾一度命名为 ``workspace``，而 subagent 的调用方
        把 ``workspace=<项目路径>`` 当**模板变量**传进来——同名会被函数签名吃掉，
        ``{{ workspace }}`` 静默渲染成空串（不报错，只是子代理提示词缺一块）。

        用覆盖文件来验，因为覆盖与内置走的是同一套 kwargs：这条守护同时钉住两条路径。
        """
        _write_override(
            tmp_path,
            "agent/subagent_system.md",
            "proj={{ workspace }} agent={{ agent_workspace }}",
        )
        rendered = render_template(
            "agent/subagent_system.md",
            override_root=tmp_path,
            workspace="/proj",
            agent_workspace="/agent-home",
            history_log="memory/history.jsonl",
            skills_summary="",
        )
        assert rendered == "proj=/proj agent=/agent-home"

    def test_workspace_variable_coexists_with_override_root(self, tmp_path: Path) -> None:
        """``workspace`` 作为模板变量与 ``override_root`` 并存，不得互相顶掉。"""
        _write_override(
            tmp_path,
            "agent/skills_section.md",
            "ws={{ workspace }} sum={{ skills_summary }}",
        )
        rendered = render_template(
            "agent/skills_section.md",
            override_root=tmp_path,
            workspace="/proj",
            skills_summary="S",
        )
        assert rendered == "ws=/proj sum=S"

    def test_bundled_path_accepts_workspace_kwarg(self, tmp_path: Path) -> None:
        """无覆盖时也接受 ``workspace`` kwarg（否则 subagent 调用方会 TypeError）。"""
        rendered = render_template(
            "agent/subagent_system.md",
            override_root=tmp_path,
            workspace="/proj",
            agent_workspace="/agent-home",
            history_log="memory/history.jsonl",
            skills_summary="",
        )
        assert "/agent-home" in rendered

    def test_deleted_override_restores_bundled(self, tmp_path: Path) -> None:
        path = _write_override(tmp_path, "agent/tool_contract.md", "CUSTOM")
        assert render_template("agent/tool_contract.md", override_root=tmp_path).strip() == "CUSTOM"
        path.unlink()
        assert render_template(
            "agent/tool_contract.md", override_root=tmp_path
        ) == render_template("agent/tool_contract.md")


# ---------------------------------------------------------------------------
# load_workspace_template
# ---------------------------------------------------------------------------


class TestLoadWorkspaceTemplate:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_workspace_template(tmp_path, "agent/identity.md") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        _write_override(tmp_path, "agent/identity.md", "\n\n")
        assert load_workspace_template(tmp_path, "agent/identity.md") is None

    def test_whitespace_is_stripped(self, tmp_path: Path) -> None:
        _write_override(tmp_path, "agent/identity.md", "\n  content  \n")
        assert load_workspace_template(tmp_path, "agent/identity.md") == "content"

    def test_directory_instead_of_file_returns_none(self, tmp_path: Path) -> None:
        workspace_template_path(tmp_path, "agent/identity.md").mkdir(parents=True)
        assert load_workspace_template(tmp_path, "agent/identity.md") is None


# ---------------------------------------------------------------------------
# workspace_template_path
# ---------------------------------------------------------------------------


class TestWorkspaceTemplatePath:
    def test_preserves_subdirectories_and_extension(self, tmp_path: Path) -> None:
        assert workspace_template_path(tmp_path, "agent/identity.md") == (
            tmp_path / "prompts" / "agent" / "identity.md"
        )

    def test_expands_user(self) -> None:
        assert "~" not in str(workspace_template_path("~/ws", "agent/identity.md"))


# ---------------------------------------------------------------------------
# initialize_workspace_templates
# ---------------------------------------------------------------------------


class TestInitializeWorkspaceTemplates:
    def test_creates_all_defaults(self, tmp_path: Path) -> None:
        created = initialize_workspace_templates(tmp_path)
        assert set(created) == set(OVERRIDABLE_TEMPLATES)
        for name in OVERRIDABLE_TEMPLATES:
            assert workspace_template_path(tmp_path, name).is_file()

    def test_created_copy_matches_bundled(self, tmp_path: Path) -> None:
        initialize_workspace_templates(tmp_path)
        copied = workspace_template_path(tmp_path, "agent/tool_contract.md").read_text(
            encoding="utf-8"
        )
        from importlib.resources import files as pkg_files

        bundled = (pkg_files("nanobot") / "templates" / "agent" / "tool_contract.md").read_text(
            encoding="utf-8"
        )
        assert copied == bundled

    def test_does_not_overwrite_user_edits(self, tmp_path: Path) -> None:
        _write_override(tmp_path, "agent/tool_contract.md", "MY EDIT")
        created = initialize_workspace_templates(tmp_path)
        assert "agent/tool_contract.md" not in created
        assert workspace_template_path(tmp_path, "agent/tool_contract.md").read_text(
            encoding="utf-8"
        ) == "MY EDIT"

    def test_is_idempotent(self, tmp_path: Path) -> None:
        assert initialize_workspace_templates(tmp_path)
        assert initialize_workspace_templates(tmp_path) == []

    def test_empty_existing_file_is_refilled(self, tmp_path: Path) -> None:
        path = _write_override(tmp_path, "agent/tool_contract.md", "   ")
        created = initialize_workspace_templates(tmp_path, ["agent/tool_contract.md"])
        assert created == ["agent/tool_contract.md"]
        assert path.read_text(encoding="utf-8").strip()

    def test_ignores_non_whitelisted_names(self, tmp_path: Path) -> None:
        created = initialize_workspace_templates(
            tmp_path, ["agent/dream.md", "agent/tool_contract.md"]
        )
        assert created == ["agent/tool_contract.md"]
        assert not workspace_template_path(tmp_path, "agent/dream.md").exists()

    def test_subset_of_names(self, tmp_path: Path) -> None:
        created = initialize_workspace_templates(tmp_path, ["agent/identity.md"])
        assert created == ["agent/identity.md"]


# ---------------------------------------------------------------------------
# sync_workspace_templates 联动：开箱即在 prompts/ 下可见可改
# ---------------------------------------------------------------------------


def test_sync_workspace_templates_seeds_prompt_overrides(tmp_path: Path) -> None:
    from nanobot.utils.helpers import sync_workspace_templates

    sync_workspace_templates(tmp_path, silent=True)

    for name in OVERRIDABLE_TEMPLATES:
        assert workspace_template_path(tmp_path, name).is_file(), f"not seeded: {name}"


def test_sync_workspace_templates_keeps_user_prompt_edits(tmp_path: Path) -> None:
    from nanobot.utils.helpers import sync_workspace_templates

    sync_workspace_templates(tmp_path, silent=True)
    target = workspace_template_path(tmp_path, "agent/identity.md")
    target.write_text("MY IDENTITY", encoding="utf-8")

    sync_workspace_templates(tmp_path, silent=True)

    assert target.read_text(encoding="utf-8") == "MY IDENTITY"


def test_seeded_identity_template_is_skipped_until_customized(tmp_path: Path) -> None:
    """脚手架副本不该被当成用户内容注入（否则白烧 token）。"""
    from nanobot.agent.context import ContextBuilder
    from nanobot.utils.helpers import sync_workspace_templates

    sync_workspace_templates(tmp_path, silent=True)
    assert "## IDENTITY.md" not in ContextBuilder(tmp_path)._load_bootstrap_files()

    (tmp_path / "IDENTITY.md").write_text("I am ReviewBot.", encoding="utf-8")
    assert "## IDENTITY.md" in ContextBuilder(tmp_path)._load_bootstrap_files()


@pytest.mark.parametrize("name", OVERRIDABLE_TEMPLATES)
def test_every_overridable_template_exists_in_bundle(name: str) -> None:
    from importlib.resources import files as pkg_files

    assert (pkg_files("nanobot") / "templates" / name).is_file()
