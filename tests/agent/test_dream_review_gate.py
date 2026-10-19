"""FR-4.6 Dream review gate: candidate lifecycle, config toggle, and routing.

Hard guarantee under test: when the gate is open (the default), the real
``MEMORY.md`` / SOUL.md / USER.md are byte-for-byte unchanged unless and until
the candidate is explicitly accepted. Accepting writes back; rejecting never
touches the real files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.config.schema import Config, DreamConfig


def _make_store(tmp_path: Path) -> MemoryStore:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "memory").mkdir()
    return MemoryStore(workspace)


def _write_real(store: MemoryStore, *, memory: str = "", soul: str = "", user: str = "") -> None:
    if memory is not None:
        store.memory_file.write_text(memory, encoding="utf-8")
    if soul is not None:
        store.soul_file.write_text(soul, encoding="utf-8")
    if user is not None:
        store.user_file.write_text(user, encoding="utf-8")


# ---------------------------------------------------------------------------
# Config toggle (B3-02 ④): default on, round-trips, can be turned off
# ---------------------------------------------------------------------------


def test_dream_require_review_defaults_on() -> None:
    assert DreamConfig().require_review is True


def test_dream_require_review_can_be_disabled() -> None:
    cfg = DreamConfig(require_review=False)
    assert cfg.require_review is False


def test_dream_require_review_roundtrip_via_dump() -> None:
    dumped = DreamConfig().model_dump()
    assert dumped["require_review"] is True
    back = DreamConfig.model_validate(dumped)
    assert back.require_review is True


def test_dream_require_review_roundtrip_via_full_config() -> None:
    raw = {"agents": {"defaults": {"dream": {"require_review": False}}}}
    cfg = Config.model_validate(raw)
    assert cfg.agents.defaults.dream.require_review is False


# ---------------------------------------------------------------------------
# build_dream_tools routing (B3-02 ②): gate open → candidate, gate closed → real
# ---------------------------------------------------------------------------


def test_build_dream_tools_gate_open_seeds_candidate_copy(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="# Memory\nold content\n", soul="SOUL", user="USER")

    store.build_dream_tools(require_review=True)

    cand = store.dream_candidate_dir
    assert cand.exists()
    assert cand.joinpath("MEMORY.md").read_text(encoding="utf-8") == "# Memory\nold content\n"
    # Real files untouched by seeding.
    assert store.memory_file.read_text(encoding="utf-8") == "# Memory\nold content\n"
    assert store.soul_file.read_text(encoding="utf-8") == "SOUL"
    assert store.user_file.read_text(encoding="utf-8") == "USER"


def test_build_dream_tools_gate_closed_keeps_legacy_path(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="M")

    store.build_dream_tools(require_review=False)

    # Legacy path: no candidate area is created, real file stays the source.
    assert not store.dream_candidate_dir.exists()


def test_build_dream_tools_default_is_fail_closed(tmp_path: Path) -> None:
    """⛔ 默认值必须 fail-closed。

    不传参调用时若走「直落正式文件」，等于「新增调用点忘记传 require_review」
    就能静默绕过 FR-4.6 的审查闸门——而这恰恰是闸门要消灭的那条路径。
    这条用例锁住默认行为：把签名默认值改回 False 必须立刻变红。

    注意：测旧路径的用例必须**显式**传 require_review=False（见 test_dream.py），
    默认值本身不代表「推荐走旧路径」。
    """
    store = _make_store(tmp_path)
    _write_real(store, memory="# Memory\nold content\n", soul="SOUL", user="USER")

    store.build_dream_tools()  # 故意不传参

    # 默认 True ⇒ 闸门开：候选区被种下，写工具指向候选副本而非正式文件
    assert store.dream_candidate_dir.exists()
    assert (
        store.dream_candidate_dir.joinpath("MEMORY.md").read_text(encoding="utf-8")
        == "# Memory\nold content\n"
    )


def test_build_dream_tools_gate_open_seeds_only_existing_files(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    # Only MEMORY.md exists; SOUL.md/USER.md do not.
    _write_real(store, memory="M", soul=None, user=None)

    store.build_dream_tools(require_review=True)

    assert store.dream_candidate_dir.joinpath("MEMORY.md").exists()
    assert not store.dream_candidate_dir.joinpath("SOUL.md").exists()
    assert not store.dream_candidate_dir.joinpath("USER.md").exists()


# ---------------------------------------------------------------------------
# Candidate lifecycle (B3-02 ②): classify, accept, reject
# ---------------------------------------------------------------------------


def test_has_candidate_false_when_no_diff(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="M")
    store.prepare_dream_candidate()
    assert store.has_dream_candidate() is False


def test_compute_diff_classifies_modified(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="old", soul="S", user="U")
    store.prepare_dream_candidate()
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("new", encoding="utf-8")

    diff = store.compute_dream_candidate_diff()
    assert diff["MEMORY.md"]["status"] == "modified"
    assert diff["MEMORY.md"]["old"] == "old"
    assert diff["MEMORY.md"]["new"] == "new"
    # Unchanged files are omitted from the diff.
    assert "SOUL.md" not in diff
    assert "USER.md" not in diff


def test_compute_diff_classifies_new_and_deleted(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    # MEMORY.md does not exist in the workspace yet → candidate creation = "new".
    store.prepare_dream_candidate()
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("created", encoding="utf-8")

    diff = store.compute_dream_candidate_diff()
    assert diff["MEMORY.md"]["status"] == "new"

    # SOUL.md exists; candidate emptied = "deleted".
    _write_real(store, soul="S")
    store.prepare_dream_candidate()
    store.dream_candidate_dir.joinpath("SOUL.md").write_text("", encoding="utf-8")
    diff2 = store.compute_dream_candidate_diff()
    assert diff2["SOUL.md"]["status"] == "deleted"


def test_accept_writes_back_and_clears_candidate(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="old", soul="S", user="U")
    store.prepare_dream_candidate()
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("updated", encoding="utf-8")

    changed = store.accept_dream_candidate()

    assert changed == 1
    assert store.memory_file.read_text(encoding="utf-8") == "updated"
    # Unchanged files are not rewritten.
    assert store.soul_file.read_text(encoding="utf-8") == "S"
    assert not store.dream_candidate_dir.exists()


def test_accept_creates_new_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    # No MEMORY.md in workspace; Dream proposes one.
    store.prepare_dream_candidate()
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("fresh", encoding="utf-8")

    changed = store.accept_dream_candidate()

    assert changed == 1
    assert store.memory_file.read_text(encoding="utf-8") == "fresh"


def test_reject_never_touches_real_files(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="real-content", soul="S", user="U")
    store.build_dream_tools(require_review=True)
    # Simulate a partial Dream proposal.
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("hacked", encoding="utf-8")

    store.reject_dream_candidate()

    # Real files are byte-for-byte unchanged; candidate is gone.
    assert store.memory_file.read_text(encoding="utf-8") == "real-content"
    assert store.soul_file.read_text(encoding="utf-8") == "S"
    assert store.user_file.read_text(encoding="utf-8") == "U"
    assert not store.dream_candidate_dir.exists()


def test_accept_is_noop_when_candidate_equals_real(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _write_real(store, memory="same", soul="S", user="U")
    store.prepare_dream_candidate()

    # Candidate is an exact copy → accept must not change anything.
    before = store.memory_file.read_bytes()
    changed = store.accept_dream_candidate()
    after = store.memory_file.read_bytes()

    assert changed == 0
    assert before == after
    assert not store.dream_candidate_dir.exists()


def test_hard_guarantee_not_accepted_implies_real_unchanged(tmp_path: Path) -> None:
    """The gate's only hard criterion: without explicit accept, the real file
    is byte-for-byte identical before and after a gated Dream run."""
    store = _make_store(tmp_path)
    _write_real(store, memory="immutable", soul="S", user="U")

    # Gate open: Dream seeds a candidate (and might edit it).
    store.build_dream_tools(require_review=True)
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text("proposal", encoding="utf-8")

    # User never accepts (and does not reject either) → real stays put.
    real_before = store.memory_file.read_bytes()
    assert real_before == b"immutable"
    assert store.memory_file.read_bytes() == real_before
