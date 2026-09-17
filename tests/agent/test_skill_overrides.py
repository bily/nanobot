"""FR-3.1：技能三态覆盖（skillOverrides）与用户级技能目录。"""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.skills import (
    SKILL_OVERRIDE_OFF,
    SKILL_OVERRIDE_ON,
    SKILL_OVERRIDE_USER_INVOCABLE_ONLY,
    SkillsLoader,
    default_user_skills_dir,
    is_skill_invocable,
    is_skill_listed_to_model,
    normalize_skill_override,
)


def _write_skill(base: Path, name: str, *, body: str = "# Skill\n") -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {name} 技能\n---\n\n{body}", encoding="utf-8")
    return path


def _make_loader(tmp_path: Path, **kwargs: object) -> SkillsLoader:
    workspace = tmp_path / "ws"
    (workspace / "skills").mkdir(parents=True)
    builtin = tmp_path / "builtin"
    builtin.mkdir(exist_ok=True)
    return SkillsLoader(workspace, builtin_skills_dir=builtin, **kwargs)  # type: ignore[arg-type]


# —— normalize_skill_override ——


def test_normalize_accepts_three_known_states() -> None:
    assert normalize_skill_override("on") == SKILL_OVERRIDE_ON
    assert normalize_skill_override("user-invocable-only") == SKILL_OVERRIDE_USER_INVOCABLE_ONLY
    assert normalize_skill_override("off") == SKILL_OVERRIDE_OFF


def test_normalize_tolerates_case_and_surrounding_space() -> None:
    assert normalize_skill_override("  OFF ") == SKILL_OVERRIDE_OFF
    assert normalize_skill_override("On") == SKILL_OVERRIDE_ON
    assert normalize_skill_override("USER-INVOCABLE-ONLY") == SKILL_OVERRIDE_USER_INVOCABLE_ONLY


def test_normalize_unknown_value_fails_closed_to_off() -> None:
    """写错状态名时必须偏保守——读成 on 等于悄悄撤销用户的限制。"""
    for bogus in ("disabled", "false", "0", "manual", "ONLY"):
        assert normalize_skill_override(bogus) == SKILL_OVERRIDE_OFF


def test_normalize_non_string_fails_closed_to_off() -> None:
    for bogus in (None, True, False, 1, 0, [], {}, ["on"]):
        assert normalize_skill_override(bogus) == SKILL_OVERRIDE_OFF


def test_state_predicates() -> None:
    assert is_skill_listed_to_model(SKILL_OVERRIDE_ON) is True
    assert is_skill_listed_to_model(SKILL_OVERRIDE_USER_INVOCABLE_ONLY) is False
    assert is_skill_invocable(SKILL_OVERRIDE_ON) is True
    assert is_skill_invocable(SKILL_OVERRIDE_USER_INVOCABLE_ONLY) is True
    assert is_skill_invocable(SKILL_OVERRIDE_OFF) is False


# —— off ——


def test_off_excluded_from_list_skills(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "off"})
    _write_skill(loader.workspace_skills, "alpha")
    _write_skill(loader.workspace_skills, "beta")

    names = {entry["name"] for entry in loader.list_skills(filter_unavailable=False)}
    assert names == {"beta"}


def test_off_cannot_be_invoked_with_dollar_reference(tmp_path: Path) -> None:
    """`off` 是「当它不存在」，不是「只是不列出来」。"""
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "off"})
    _write_skill(loader.workspace_skills, "alpha")

    assert loader.get_explicitly_invoked_skills("请用 $alpha 处理") == []
    assert loader.load_skill("alpha") is None


def test_off_excluded_from_build_skills_summary(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "off"})
    _write_skill(loader.workspace_skills, "alpha")
    _write_skill(loader.workspace_skills, "beta")

    summary = loader.build_skills_summary()
    assert "beta" in summary
    assert "alpha" not in summary


# —— user-invocable-only ——


def test_user_invocable_only_still_listed_but_hidden_from_model(tmp_path: Path) -> None:
    """这一档的全部意义就在这两个断言的**差**上。"""
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "user-invocable-only"})
    _write_skill(loader.workspace_skills, "alpha")

    listed = {entry["name"] for entry in loader.list_skills(filter_unavailable=False)}
    assert listed == {"alpha"}
    assert "alpha" not in loader.build_skills_summary()


def test_user_invocable_only_still_resolvable_via_dollar_reference(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "user-invocable-only"})
    _write_skill(loader.workspace_skills, "alpha", body="# Alpha 正文")

    assert loader.get_explicitly_invoked_skills("$alpha 一下") == ["alpha"]
    content = loader.load_skill("alpha")
    assert content is not None and "Alpha 正文" in content


def test_explicit_runtime_context_still_builds_for_hidden_skill(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "user-invocable-only"})
    _write_skill(loader.workspace_skills, "alpha", body="# Alpha 正文")

    block = loader.build_explicit_skill_runtime_context("$alpha 一下")
    assert block is not None
    assert "Alpha 正文" in block.content


# —— on / 未声明 ——


def test_explicit_on_is_same_as_undeclared(tmp_path: Path) -> None:
    on_loader = _make_loader(tmp_path, skill_overrides={"alpha": "on"})
    _write_skill(on_loader.workspace_skills, "alpha")
    plain_loader = SkillsLoader(
        on_loader.workspace, builtin_skills_dir=on_loader.builtin_skills
    )

    assert on_loader.list_skills(filter_unavailable=False) == plain_loader.list_skills(
        filter_unavailable=False
    )
    assert on_loader.build_skills_summary() == plain_loader.build_skills_summary()


def test_unknown_value_hides_skill_from_model(tmp_path: Path) -> None:
    """未知取值收敛为 off，端到端表现为「模型看不见、`$skill` 也解析不到」。"""
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "disabled"})
    _write_skill(loader.workspace_skills, "alpha")

    assert loader.list_skills(filter_unavailable=False) == []
    assert loader.get_explicitly_invoked_skills("$alpha") == []


def test_get_skill_override_defaults_to_on(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "off"})
    assert loader.get_skill_override("alpha") == SKILL_OVERRIDE_OFF
    assert loader.get_skill_override("never-mentioned") == SKILL_OVERRIDE_ON


def test_empty_override_map_leaves_everything_untouched(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path)
    _write_skill(loader.workspace_skills, "alpha")
    assert {entry["name"] for entry in loader.list_skills(filter_unavailable=False)} == {"alpha"}


# —— disabled_skills 与 skill_overrides 并存的优先级 ——


def test_disabled_skills_still_excluded(tmp_path: Path) -> None:
    """老机制不能因为新机制上线就失效。"""
    loader = _make_loader(tmp_path, disabled_skills={"alpha"})
    _write_skill(loader.workspace_skills, "alpha")
    assert loader.list_skills(filter_unavailable=False) == []


def test_strictest_state_wins_when_both_mechanisms_match(tmp_path: Path) -> None:
    """`disabled_skills` 与 `skill_overrides` 同时命中山一个名字 → 取 off。"""
    loader = _make_loader(
        tmp_path,
        disabled_skills={"alpha"},
        skill_overrides={"alpha": "user-invocable-only"},
    )
    _write_skill(loader.workspace_skills, "alpha")

    assert loader.get_skill_override("alpha") == SKILL_OVERRIDE_OFF
    assert loader.list_skills(filter_unavailable=False) == []


def test_off_beats_user_invocable_only_in_same_map(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, skill_overrides={"alpha": "off", "beta": "user-invocable-only"})
    _write_skill(loader.workspace_skills, "alpha")
    _write_skill(loader.workspace_skills, "beta")

    names = {entry["name"] for entry in loader.list_skills(filter_unavailable=False)}
    assert names == {"beta"}


# —— 用户级技能目录 ——


def test_user_skills_dir_defaults_off(tmp_path: Path) -> None:
    """默认不扫描——否则单测会被宿主机真实的 ~/.nanowork/skills 污染。"""
    loader = _make_loader(tmp_path)
    assert loader.user_skills_dir is None
    assert loader.list_skills(filter_unavailable=False) == []


def test_user_skills_scanned_with_user_source(tmp_path: Path) -> None:
    user_root = tmp_path / "user-skills"
    user_path = _write_skill(user_root, "gamma")
    loader = _make_loader(tmp_path, user_skills_dir=user_root)

    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [{"name": "gamma", "path": str(user_path), "source": "user"}]


def test_missing_user_skills_dir_is_not_an_error(tmp_path: Path) -> None:
    loader = _make_loader(tmp_path, user_skills_dir=tmp_path / "nope")
    assert loader.list_skills(filter_unavailable=False) == []


def test_project_skills_shadow_user_skills(tmp_path: Path) -> None:
    """PRD §12.4 的优先级链：项目级 > 用户级。"""
    user_root = tmp_path / "user-skills"
    _write_skill(user_root, "dup", body="# User")
    loader = _make_loader(tmp_path, user_skills_dir=user_root)
    ws_path = _write_skill(loader.workspace_skills, "dup", body="# Workspace")

    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["source"] == "workspace"
    assert entries[0]["path"] == str(ws_path)


def test_user_skills_shadow_builtin(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "dup", body="# Builtin")
    user_root = tmp_path / "user-skills"
    user_path = _write_skill(user_root, "dup", body="# User")
    workspace = tmp_path / "ws"
    (workspace / "skills").mkdir(parents=True)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, user_skills_dir=user_root)
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["source"] == "user"
    assert entries[0]["path"] == str(user_path)


def test_user_skills_honor_overrides(tmp_path: Path) -> None:
    user_root = tmp_path / "user-skills"
    _write_skill(user_root, "gamma")
    loader = _make_loader(
        tmp_path,
        user_skills_dir=user_root,
        skill_overrides={"gamma": "user-invocable-only"},
    )

    assert "gamma" not in loader.build_skills_summary()
    assert loader.get_explicitly_invoked_skills("$gamma") == ["gamma"]


def test_summary_uses_absolute_root_for_user_skills(tmp_path: Path) -> None:
    """用户级目录在工作区之外，相对路径展示会让模型 read_file 读错地方。"""
    user_root = tmp_path / "user-skills"
    _write_skill(user_root, "gamma")
    loader = _make_loader(tmp_path, user_skills_dir=user_root)

    summary = loader.build_skills_summary()
    assert "### User skills" in summary
    assert str(user_root.resolve()) in summary


def test_summary_keeps_relative_root_for_workspace_skills(tmp_path: Path) -> None:
    """反向护栏：工作区内的技能仍走相对路径（既有行为不能被顺手改掉）。"""
    user_root = tmp_path / "user-skills"
    _write_skill(user_root, "gamma")
    loader = _make_loader(tmp_path, user_skills_dir=user_root)
    _write_skill(loader.workspace_skills, "alpha")

    summary = loader.build_skills_summary()
    assert "### Workspace skills (`skills`)" in summary


def test_default_user_skills_dir_follows_nanowork_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NANOWORK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEBUDDY_HOME", raising=False)
    assert default_user_skills_dir() == tmp_path / "home" / "skills"


def test_default_user_skills_dir_falls_back_to_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NANOWORK_HOME", raising=False)
    monkeypatch.setenv("CODEBUDDY_HOME", str(tmp_path / "cb"))
    assert default_user_skills_dir() == tmp_path / "cb" / "skills"
