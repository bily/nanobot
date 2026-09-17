"""[LOCAL PATCH] nanowork FR-8.3：删除改道系统回收站的引擎侧单测。

覆盖四条不变量：临时目录例外、fail-closed（回收站不可用绝不降级真删）、
删除留痕、以及 shell 删除命令的识别（含误判面）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from nanobot.security import trash as trash_mod
from nanobot.security.trash import (
    BulkDeleteError,
    TrashError,
    TrashUnavailableError,
    append_delete_report,
    check_bulk_threshold,
    count_entries,
    default_trash_fn,
    delete_command_is_temp_only,
    delete_command_targets,
    delete_path,
    delete_report_line,
    is_delete_command,
    is_temp_path,
    send_to_trash,
    temp_roots,
)


class _RecordingTrash:
    """Fake trash backend that really moves the entry aside."""

    def __init__(self, *, fail: bool = False, noop: bool = False) -> None:
        self.calls: list[Path] = []
        self.fail = fail
        self.noop = noop

    def __call__(self, path: Path) -> None:
        self.calls.append(Path(path))
        if self.fail:
            raise TrashError("backend exploded")
        if self.noop:
            return
        if Path(path).is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            Path(path).unlink(missing_ok=True)


# ——————————————————————————————— 临时目录 ———————————————————————————————


def test_temp_roots_are_absolute_and_unique():
    roots = temp_roots()
    assert roots, "expected at least one temp root"
    assert all(root.is_absolute() for root in roots)
    assert len({os.path.normcase(str(root)) for root in roots}) == len(roots)


def test_is_temp_path_matches_the_os_temp_root(tmp_path):
    assert is_temp_path(tmp_path) is True
    assert is_temp_path(tmp_path / "child" / "file.txt") is True


def test_is_temp_path_rejects_paths_outside_temp(tmp_path):
    assert is_temp_path(tmp_path, roots=[]) is False
    assert is_temp_path(tmp_path / "x.txt", roots=[tmp_path / "somewhere-else"]) is False


def test_is_temp_path_fails_closed_on_bad_input():
    assert is_temp_path(None) is False
    assert is_temp_path(object()) is False


# —————————————————————————————— 回收站后端 ——————————————————————————————


def test_default_trash_fn_is_available_on_supported_platforms():
    if sys.platform in {"win32", "darwin"} or os.name == "posix":
        assert callable(default_trash_fn())


def test_send_to_trash_moves_the_entry(tmp_path):
    target = tmp_path / "doomed.txt"
    target.write_text("bye", encoding="utf-8")
    backend = _RecordingTrash()

    send_to_trash(target, trash=backend)

    assert backend.calls == [target]
    assert not target.exists()


def test_send_to_trash_propagates_backend_failure(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(TrashError):
        send_to_trash(target, trash=_RecordingTrash(fail=True))

    assert target.exists(), "a failed recycle must never remove the entry"


def test_send_to_trash_fails_closed_when_backend_silently_degrades(tmp_path):
    target = tmp_path / "still-here.txt"
    target.write_text("still here", encoding="utf-8")

    with pytest.raises(TrashError, match="left the entry in place"):
        send_to_trash(target, trash=_RecordingTrash(noop=True))

    assert target.exists()


def test_send_to_trash_missing_path_is_an_error(tmp_path):
    with pytest.raises(TrashError, match="does not exist"):
        send_to_trash(tmp_path / "ghost.txt", trash=_RecordingTrash())


def test_no_trash_backend_raises_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(trash_mod, "default_trash_fn", _raise_unavailable)
    target = tmp_path / "x.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(TrashUnavailableError):
        send_to_trash(target)

    assert target.exists()


def _raise_unavailable():
    raise TrashUnavailableError("no backend on this platform")


# ——————————————————————————————— delete_path ———————————————————————————————


def test_delete_path_uses_real_delete_under_temp(tmp_path):
    target = tmp_path / "build" / "out.bin"
    target.parent.mkdir()
    target.write_text("junk", encoding="utf-8")
    backend = _RecordingTrash()

    assert delete_path(target, trash=backend) == "deleted"

    assert not target.exists()
    assert backend.calls == [], "temp entries must not be routed to the bin"


def test_delete_path_removes_temp_directory_tree(tmp_path):
    tree = tmp_path / "workspace-copy"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "a.txt").write_text("a", encoding="utf-8")

    assert delete_path(tree) == "deleted"
    assert not tree.exists()


def test_delete_path_trashes_entries_outside_temp(monkeypatch, tmp_path):
    target = tmp_path / "important.txt"
    target.write_text("important", encoding="utf-8")
    monkeypatch.setattr(trash_mod, "is_temp_path", lambda *a, **k: False)
    backend = _RecordingTrash()

    assert delete_path(target, trash=backend) == "trashed"

    assert backend.calls == [target]
    assert not target.exists()


def test_delete_path_is_fail_closed_outside_temp(monkeypatch, tmp_path):
    target = tmp_path / "precious.txt"
    target.write_text("precious", encoding="utf-8")
    monkeypatch.setattr(trash_mod, "is_temp_path", lambda *a, **k: False)

    with pytest.raises(TrashError):
        delete_path(target, trash=_RecordingTrash(fail=True))

    assert target.read_text(encoding="utf-8") == "precious"


# ——————————————————————————————— 批量守卫 ———————————————————————————————


def test_count_entries_counts_a_tree(tmp_path):
    tree = tmp_path / "many"
    tree.mkdir()
    for index in range(5):
        (tree / f"{index}.txt").write_text("x", encoding="utf-8")
    assert count_entries(tree) == 5
    assert count_entries(tree / "0.txt") == 1


def test_check_bulk_threshold_blocks_and_then_allows_on_confirm(tmp_path):
    tree = tmp_path / "many"
    tree.mkdir()
    for index in range(3):
        (tree / f"{index}.txt").write_text("x", encoding="utf-8")

    with pytest.raises(BulkDeleteError, match="bulk delete threshold"):
        check_bulk_threshold(tree, threshold=2)

    check_bulk_threshold(tree, threshold=2, confirmed=True)
    check_bulk_threshold(tree, threshold=3)
    check_bulk_threshold(tree, threshold=0)


# ——————————————————————————————— 删除留痕 ———————————————————————————————


def test_delete_report_line_shape():
    record = json.loads(
        delete_report_line(operation="trashed", path="/data/a.txt", ok=True)
    )
    assert record["operation"] == "trashed"
    assert record["runtime"] == "engine"
    assert record["path"] == "/data/a.txt"
    assert record["ok"] is True
    assert record["timestamp"]


def test_append_delete_report_writes_json_lines(tmp_path):
    report = tmp_path / "safe-delete-report.ndjson"
    append_delete_report(
        delete_report_line(operation="trashed", path=str(tmp_path / "a.txt")),
        report_path=report,
    )
    append_delete_report(
        delete_report_line(
            operation="failed",
            path=str(tmp_path / "b.txt"),
            ok=False,
            detail="no backend",
        ),
        report_path=report,
    )

    lines = report.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["operation"] == "trashed"
    assert json.loads(lines[1])["detail"] == "no backend"


def test_append_delete_report_is_disabled_without_a_path(tmp_path):
    append_delete_report("{}", report_path=None)
    append_delete_report("{}", report_path="")
    assert list(tmp_path.iterdir()) == []


def test_append_delete_report_never_raises(tmp_path):
    # A directory where a file is expected must not break the caller.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    append_delete_report("{}", report_path=blocked)


# ——————————————————————————— shell 删除命令识别 ———————————————————————————


@pytest.mark.parametrize(
    "command",
    [
        "rm notes.md",
        "rm -rf ./build",
        "rmdir old-dir",
        "unlink link.txt",
        "del report.txt",
        "erase report.txt",
        "Remove-Item -Recurse -Force ./build",
        "Remove-Item -Path ./a.txt",
        "sudo rm ./a.txt",
        "FOO=1 rm ./a.txt",
        "cd /data && rm a.txt",
        "ls | rm b.txt",
    ],
)
def test_is_delete_command_detects_delete_verbs(command):
    assert is_delete_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "npm run build",
        "echo 'rm -rf /'",
        "git rm --cached a.txt",
        "python -m pytest",
        "",
        "cat a.txt > b.txt",
    ],
)
def test_is_delete_command_ignores_non_deletes(command):
    assert is_delete_command(command) is False


def test_delete_command_targets_extracts_paths():
    assert delete_command_targets("rm -rf ./build") == ["./build"]
    assert delete_command_targets("del a.txt b.txt") == ["a.txt", "b.txt"]
    assert delete_command_targets("Remove-Item -Path ./x -Force") == ["./x"]
    assert delete_command_targets("ls -la") == []


def test_delete_command_is_temp_only_for_temp_targets(tmp_path):
    target = tmp_path / "scratch" / "a.txt"
    assert delete_command_is_temp_only(f'rm -rf "{target}"') is True


def test_delete_command_is_temp_only_rejects_non_temp_and_unparsable(tmp_path):
    assert delete_command_is_temp_only("rm -rf ./build") is False
    assert delete_command_is_temp_only("rm") is False
    assert delete_command_is_temp_only("rm *.log") is False
    assert delete_command_is_temp_only("rm $TARGET") is False


def test_delete_command_is_temp_only_rejects_path_variables(tmp_path):
    assert delete_command_is_temp_only(f"rm {tmp_path}/$NAME") is False
