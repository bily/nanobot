"""[LOCAL PATCH] nanowork FR-8.3：delete_file 工具 + exec 删除守卫的单测。

工具侧的四个出口：临时目录真删、非临时走回收站、回收站失败 fail-closed、
批量超阈值需确认。守卫侧：裸 rm/del 被拦下并指回 delete_file，临时目录例外。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.tools.filesystem import SafeDeleteTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.security import trash as trash_mod
from nanobot.security.trash import TrashError


class _RecordingTrash:
    """Fake trash backend that really moves the entry aside."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[Path] = []
        self.fail = fail

    def __call__(self, path: Path) -> None:
        self.calls.append(Path(path))
        if self.fail:
            raise TrashError("no recycle bin on this volume")
        if Path(path).is_dir():
            shutil.rmtree(path)
        else:
            Path(path).unlink(missing_ok=True)


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture()
def tool(workspace):
    return SafeDeleteTool(workspace=workspace, allowed_dir=workspace)


def _force_non_temp(monkeypatch, backend=None):
    """Model "the target is not in the temp directory" for the trash branch."""
    monkeypatch.setattr(trash_mod, "is_temp_path", lambda *a, **k: False)
    fake = backend or _RecordingTrash()
    monkeypatch.setattr(trash_mod, "default_trash_fn", lambda: fake)
    return fake


# ———————————————————————————— delete_file 工具 ————————————————————————————


class TestDeleteFileTool:

    @pytest.mark.asyncio
    async def test_temp_entry_is_deleted_for_real(self, tool, workspace):
        target = workspace / "scratch.txt"
        target.write_text("junk", encoding="utf-8")

        result = await tool.execute(path=str(target))

        assert "Deleted (temp directory" in result
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_temp_directory_tree_is_removed(self, tool, workspace):
        tree = workspace / "build-copy"
        (tree / "nested").mkdir(parents=True)
        (tree / "nested" / "a.o").write_text("obj", encoding="utf-8")

        result = await tool.execute(path=str(tree))

        assert "Deleted (temp directory" in result
        assert not tree.exists()

    @pytest.mark.asyncio
    async def test_non_temp_entry_goes_to_the_recycle_bin(
        self, monkeypatch, tool, workspace
    ):
        target = workspace / "important.txt"
        target.write_text("important", encoding="utf-8")
        backend = _force_non_temp(monkeypatch)

        result = await tool.execute(path=str(target))

        assert "recycle bin" in result
        assert backend.calls == [target]
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_missing_trash_backend_fails_closed(
        self, monkeypatch, tool, workspace
    ):
        target = workspace / "precious.txt"
        target.write_text("precious", encoding="utf-8")
        _force_non_temp(monkeypatch, _RecordingTrash(fail=True))

        result = await tool.execute(path=str(target))

        assert "Error" in result
        assert "fail-closed" in result
        assert target.read_text(encoding="utf-8") == "precious"

    @pytest.mark.asyncio
    async def test_missing_path_is_a_soft_error(self, tool, workspace):
        result = await tool.execute(path=str(workspace / "ghost.txt"))

        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_outside_workspace_is_refused(self, tmp_path, tool):
        outside = tmp_path / "outside.txt"
        outside.write_text("keep", encoding="utf-8")

        result = await tool.execute(path=str(outside))

        assert "Error" in result
        assert outside.exists()

    @pytest.mark.asyncio
    async def test_bulk_delete_requires_confirm(self, workspace):
        guarded = SafeDeleteTool(
            workspace=workspace,
            allowed_dir=workspace,
            safe_delete_bulk_threshold=2,
        )
        tree = workspace / "many"
        tree.mkdir()
        for index in range(4):
            (tree / f"{index}.txt").write_text("x", encoding="utf-8")

        blocked = await guarded.execute(path=str(tree))
        assert "Error" in blocked
        assert "confirm=true" in blocked

        confirmed = await guarded.execute(path=str(tree), confirm=True)
        assert "Deleted" in confirmed
        assert not tree.exists()

    @pytest.mark.asyncio
    async def test_single_entry_below_threshold_needs_no_confirm(self, workspace):
        guarded = SafeDeleteTool(
            workspace=workspace,
            allowed_dir=workspace,
            safe_delete_bulk_threshold=2,
        )
        target = workspace / "one.txt"
        target.write_text("x", encoding="utf-8")

        result = await guarded.execute(path=str(target))

        assert "Deleted" in result
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_is_recorded_in_the_safe_delete_report(self, tmp_path, workspace):
        report = tmp_path / "safe-delete-report.ndjson"
        tool = SafeDeleteTool(
            workspace=workspace,
            allowed_dir=workspace,
            safe_delete_report_path=str(report),
        )
        target = workspace / "tracked.txt"
        target.write_text("x", encoding="utf-8")

        await tool.execute(path=str(target))

        lines = report.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["operation"] == "deleted"
        assert record["runtime"] == "engine"
        assert record["path"] == str(target)
        assert record["ok"] is True

    @pytest.mark.asyncio
    async def test_refusals_are_recorded_as_well(self, tmp_path, workspace):
        report = tmp_path / "safe-delete-report.ndjson"
        guarded = SafeDeleteTool(
            workspace=workspace,
            allowed_dir=workspace,
            safe_delete_bulk_threshold=1,
            safe_delete_report_path=str(report),
        )
        tree = workspace / "many"
        tree.mkdir()
        for index in range(3):
            (tree / f"{index}.txt").write_text("x", encoding="utf-8")

        await guarded.execute(path=str(tree))

        record = json.loads(report.read_text(encoding="utf-8").strip())
        assert record["operation"] == "refused"
        assert record["ok"] is False

    @pytest.mark.asyncio
    async def test_delete_honours_default_deny_write(self, workspace):
        target = workspace / "unlisted.txt"
        target.write_text("x", encoding="utf-8")
        strict = SafeDeleteTool(
            workspace=workspace,
            allowed_dir=None,
            default_deny_write=True,
        )

        result = await strict.execute(path=str(target))

        assert "Error" in result
        assert target.exists()

    @pytest.mark.asyncio
    async def test_no_report_path_means_no_logging(self, tool, workspace):
        target = workspace / "quiet.txt"
        target.write_text("x", encoding="utf-8")

        await tool.execute(path=str(target))

        assert not (workspace / "safe-delete-report.ndjson").exists()


# ——————————————————————————— exec 删除守卫 ———————————————————————————


class TestExecDeleteGuard:

    @pytest.fixture()
    def exec_tool(self, tmp_path):
        return ExecTool(working_dir=str(tmp_path))

    @pytest.mark.parametrize(
        "command",
        ["rm notes.md", "del notes.md", "Remove-Item -Path notes.md", "unlink notes.md"],
    )
    def test_bare_delete_commands_are_blocked(self, exec_tool, tmp_path, command):
        blocked = exec_tool._guard_command(command, str(tmp_path))

        assert blocked is not None
        assert "safe-delete guard" in blocked
        assert "delete_file" in blocked

    def test_temp_target_is_exempt(self, exec_tool, tmp_path):
        scratch = Path(tempfile.gettempdir()) / "nanowork-guard-probe.txt"

        assert exec_tool._guard_command(f'rm "{scratch}"', str(tmp_path)) is None

    @pytest.mark.parametrize("command", ["ls -la", "npm run build", "python -m pytest"])
    def test_non_delete_commands_pass(self, exec_tool, tmp_path, command):
        assert exec_tool._guard_command(command, str(tmp_path)) is None

    def test_allow_patterns_can_exempt_a_delete(self, tmp_path):
        permissive = ExecTool(
            working_dir=str(tmp_path),
            allow_patterns=[r"rm notes\.md"],
        )

        assert permissive._guard_command("rm notes.md", str(tmp_path)) is None

    def test_existing_flag_variants_still_hit_the_deny_filter(self, exec_tool, tmp_path):
        blocked = exec_tool._guard_command("rm -rf build", str(tmp_path))

        assert blocked is not None
        assert "deny pattern filter" in blocked
