"""[LOCAL PATCH] nanowork FR-8.3「运行时硬兜底」。

删除类操作改道**系统回收站**而非真删，且不依赖提示词约束模型。

四条不变量（改这块前必读）：

1. **绝不静默真删**：路径不在 OS 临时目录下时，删除一律走系统回收站。
2. **fail-closed**：回收站后端不可用 / 系统调用失败 → **抛错**，
   **绝不降级为 ``os.remove`` / ``shutil.rmtree``**。调用方应把它转成一条
   软错误回给模型（见 ``nanobot/agent/tools/filesystem.py`` 的 ``delete_file``），
   而不是自己找退路。
3. **临时目录例外**：OS 临时目录下的删除走真删。否则 pytest / 构建产物会把
   用户的回收站塞满（这些文件本来就没有保留价值）。
4. **留痕**：每次删除追加一行 JSON 到 ``safe-delete-report.ndjson``
   （operation / runtime / path / timestamp / ok），对齐 WorkBuddy 的
   ``CODEBUDDY_SAFE_DELETE_REPORT_PATH``。

为什么把强制点放在引擎而不是 Electron：删除动作发生在**引擎进程内**
（``exec`` 工具的 ``rm`` / 文件工具），客户端拦不住——与 FR-8.4 的归属修正
同一逻辑（见 PRD 11.2.1）。
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

#: 回收站后端签名：把 *path* 移入系统回收站，失败抛 ``TrashError``。
TrashFn = Callable[[Path], None]

#: 批量删除阈值：单次删除超过这个数量的条目时要求显式确认（对齐
#: WorkBuddy ``safe-delete-bulk-guard.cjs`` 的默认阈值）。
DEFAULT_BULK_THRESHOLD = 10

TRASH_UNAVAILABLE_NOTE = (
    " (this is a fail-closed guard: the operation was refused instead of "
    "permanently deleting the data. Do not retry with shell delete commands "
    "or another tool; ask the user how to proceed)"
)

DELETE_REDIRECT_NOTE = (
    " (deleting through the shell is intercepted by a fail-closed guard: only "
    "the OS temp directory may be deleted that way. To delete anything else, "
    "call the `delete_file` tool, which routes the data to the system recycle "
    "bin. Do not retry this command with rm/del/Remove-Item or another "
    "indirection.)"
)


class TrashError(RuntimeError):
    """Raised when a path could not be moved to the system trash."""


class TrashUnavailableError(TrashError):
    """Raised when no working trash backend exists on this platform."""


class BulkDeleteError(TrashError):
    """Raised when a single request would delete more entries than allowed."""


# ——————————————————————————————— 临时目录 ———————————————————————————————


def temp_roots() -> tuple[Path, ...]:
    """Return the OS temp roots that are exempt from the trash redirect."""
    candidates: list[Path] = []
    for raw in (
        os.environ.get("TMPDIR"),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        tempfile.gettempdir(),
    ):
        if raw:
            candidates.append(Path(raw))
    if os.name != "nt":
        candidates.extend([Path("/tmp"), Path("/var/tmp")])
    seen: set[str] = set()
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        key = os.path.normcase(os.fspath(resolved))
        if key and key not in seen:
            seen.add(key)
            roots.append(resolved)
    return tuple(roots)


def _is_within(path: Path, root: Path) -> bool:
    """Normcase prefix containment that does not require the path to exist."""
    try:
        path_key = os.path.normcase(os.path.abspath(path))
    except (OSError, RuntimeError, ValueError):
        return False
    root_key = os.path.normcase(os.path.abspath(root))
    if not root_key:
        return False
    if path_key == root_key:
        return True
    separator = "\\" if "\\" in root_key else "/"
    if not root_key.endswith(separator):
        root_key += separator
    return path_key.startswith(root_key)


def is_temp_path(path: str | Path, *, roots: Iterable[str | Path] | None = None) -> bool:
    """Return True when *path* lives under an OS temp directory.

    Passing *roots* **replaces** the discovered temp roots instead of adding to
    them — that is what tests use to model "nothing is temp" without touching
    the real environment.
    """
    if path is None:
        return False
    try:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    candidates = tuple(roots) if roots is not None else temp_roots()
    return any(_is_within(candidate, Path(root)) for root in candidates)


# ——————————————————————————————— 回收站后端 ———————————————————————————————


def _windows_trash(path: Path) -> None:
    """Send *path* to the Recycle Bin via the Shell API (``SHFileOperationW``)."""
    from ctypes import wintypes

    class _ShFileOpStructW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    # Win32 常量按官方头文件写成大写——`N806` 在这里是误报，故整体豁免。
    # noqa: N806
    FO_DELETE = 0x0003  # noqa: N806
    FOF_SILENT = 0x0004  # noqa: N806
    FOF_NOCONFIRMATION = 0x0010  # noqa: N806
    FOF_ALLOWUNDO = 0x0040  # noqa: N806  ← 进回收站而不是真删
    FOF_NOERRORUI = 0x0400  # noqa: N806

    # pFrom must be double-NUL terminated.
    source = ctypes.c_wchar_p(str(path) + "\0\0")
    # ``fAnyOperationsAborted`` is 0-initialized but left NULL afterwards: the
    # caller must pass NULL for this field on input, which ctypes does for us.
    operation = _ShFileOpStructW()
    operation.wFunc = FO_DELETE
    operation.pFrom = ctypes.cast(source, wintypes.LPCWSTR)
    operation.fFlags = (
        FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    )
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0:
        raise TrashError(
            f"SHFileOperationW failed with code {result} for {path}"
        )
    if operation.fAnyOperationsAborted:
        raise TrashError(f"Shell aborted the recycle operation for {path}")


def _run_trash_command(program: str, args: list[str], path: Path) -> None:
    try:
        completed = subprocess.run(
            [program, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError) as exc:
        raise TrashError(f"{program} could not be executed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise TrashError(
            f"{program} exited {completed.returncode} for {path}"
            + (f": {detail}" if detail else "")
        )


def _darwin_trash(path: Path) -> None:
    _run_trash_command(
        "osascript",
        ["-e", 'tell application "Finder" to delete POSIX file "%s"' % path],
        path,
    )


def _linux_trash(path: Path) -> None:
    if shutil.which("gio"):
        _run_trash_command("gio", ["trash", "--", str(path)], path)
        return
    if shutil.which("trash-put"):
        _run_trash_command("trash-put", ["--", str(path)], path)
        return
    if shutil.which("trash"):
        _run_trash_command("trash", ["--", str(path)], path)
        return
    raise TrashUnavailableError(
        "no trash backend found (looked for gio / trash-put / trash)"
    )


def default_trash_fn() -> TrashFn:
    """Return the platform trash backend, or raise ``TrashUnavailableError``."""
    if sys.platform == "win32":
        return _windows_trash
    if sys.platform == "darwin":
        return _darwin_trash
    if os.name == "posix":
        return _linux_trash
    raise TrashUnavailableError(f"no trash backend for platform {sys.platform!r}")


def send_to_trash(path: str | Path, *, trash: TrashFn | None = None) -> Path:
    """Move *path* to the system trash; raise ``TrashError`` on any failure."""
    target = Path(path).expanduser()
    if not target.exists() and not target.is_symlink():
        raise TrashError(f"path does not exist: {target}")
    trash_fn = trash or default_trash_fn()
    trash_fn(target)
    if target.exists() or target.is_symlink():
        # The backend reported success but the entry is still there — this is
        # almost always a code path that silently degraded. Fail closed.
        raise TrashError(f"trash backend left the entry in place: {target}")
    return target


def delete_path(
    path: str | Path,
    *,
    trash: TrashFn | None = None,
) -> str:
    """Delete *path*: real delete under temp, trash everywhere else.

    Returns ``"deleted"`` (temp-dir exception) or ``"trashed"``. Raises
    ``TrashError`` when the entry must be trashed but the backend failed —
    the caller must **not** fall back to a real delete.
    """
    target = Path(path).expanduser()
    if is_temp_path(target):
        # Temp directories are exempt by design: they hold throwaway files and
        # routing them to the recycle bin would flood it (pytest/build output).
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        return "deleted"
    send_to_trash(target, trash=trash)
    return "trashed"


#: ``count_entries`` stops counting past this many entries — the bulk guard only
#: needs to know "more than the threshold", not the exact size of a huge tree.
_COUNT_CAP = 10_000


def count_entries(path: str | Path, *, cap: int = _COUNT_CAP) -> int:
    """Count the filesystem entries a delete of *path* would remove (capped)."""
    target = Path(path).expanduser()
    if not (target.is_dir() and not target.is_symlink()):
        return 1
    total = 0
    for _ in target.rglob("*"):
        total += 1
        if total >= cap:
            break
    return total


def check_bulk_threshold(
    path: str | Path,
    *,
    threshold: int = DEFAULT_BULK_THRESHOLD,
    confirmed: bool = False,
) -> None:
    """Raise ``BulkDeleteError`` when *path* exceeds the bulk delete guard."""
    if confirmed or threshold <= 0:
        return
    count = count_entries(path)
    if count > threshold:
        raise BulkDeleteError(
            f"{count} entries exceed the bulk delete threshold of {threshold}; "
            f"pass confirm=true to proceed"
        )


# ——————————————————————————————— 删除留痕 ———————————————————————————————


def delete_report_line(
    *,
    operation: str,
    path: str | Path,
    ok: bool = True,
    detail: str | None = None,
    runtime: str = "engine",
) -> str:
    """Build one ``safe-delete-report.ndjson`` line."""
    record: dict[str, object] = {
        "operation": operation,
        "runtime": runtime,
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
    }
    if detail:
        record["detail"] = detail
    return json.dumps(record, ensure_ascii=False)


def append_delete_report(
    line: str,
    *,
    report_path: str | Path | None = None,
) -> None:
    """Append one report line; a missing/blank *report_path* disables logging.

    Reporting must never break a delete that already happened, so failures are
    swallowed deliberately (the guard's job is protecting data, not the log).
    """
    if not report_path:
        return
    try:
        target = Path(report_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
    except OSError:
        return


# ——————————————————————————— shell 删除命令识别 ———————————————————————————

#: 首个裸词命中即视为「删除段」。PowerShell 里 ``rm`` / ``del`` / ``erase`` /
#: ``ri`` 都是 ``Remove-Item`` 的别名，一并覆盖。
_DELETE_VERBS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "rd",
        "del",
        "erase",
        "unlink",
        "shred",
        "srm",
        "remove-item",
        "removeitem",
        "ri",
        "trash",
        "trash-put",
    }
)

#: 带值的目标参数（PowerShell 常见写法）。
_PATH_FLAGS: frozenset[str] = frozenset(
    {"-path", "-literalpath", "-filepath", "/path"}
)

_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;&|\n]")


def _segment_tokens(segment: str) -> list[str]:
    return [_strip_quotes(token) for token in segment.split() if token.strip()]


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _looks_like_flag(token: str) -> bool:
    if token.startswith("-"):
        return True
    # cmd.exe style switches are short: `/f`, `/q`, `/s`, `/a:s`. A POSIX
    # absolute path such as `/tmp/x` or `/data` shares the leading slash, so the
    # switch shape must stay strict — otherwise real paths get skipped and the
    # caller can no longer judge whether the target was temp-bound.
    return bool(re.fullmatch(r"/[A-Za-z]{1,2}(?::[A-Za-z]+)?", token))


def is_delete_command(command: str) -> bool:
    """Return True when *command* contains a shell delete invocation."""
    for segment in _SEGMENT_SPLIT.split(command or ""):
        tokens = _segment_tokens(segment)
        while tokens and ("=" in tokens[0] and not tokens[0].startswith("-")):
            tokens.pop(0)  # skip leading VAR=value assignments
        while tokens and tokens[0] in {"sudo", "command", "exec", "nohup", "env"}:
            tokens.pop(0)
        if not tokens:
            continue
        verb = Path(tokens[0]).name.lower()
        if verb in _DELETE_VERBS:
            return True
    return False


def delete_command_targets(command: str) -> list[str]:
    """Best-effort extraction of the paths a delete command would remove.

    Empty result means "could not determine" — callers must treat that as
    **non-temp** (fail-closed), not as "nothing to delete".
    """
    targets: list[str] = []
    for segment in _SEGMENT_SPLIT.split(command or ""):
        tokens = _segment_tokens(segment)
        while tokens and ("=" in tokens[0] and not tokens[0].startswith("-")):
            tokens.pop(0)
        while tokens and tokens[0] in {"sudo", "command", "exec", "nohup", "env"}:
            tokens.pop(0)
        if not tokens:
            continue
        verb = Path(tokens[0]).name.lower()
        if verb not in _DELETE_VERBS:
            continue
        rest = tokens[1:]
        index = 0
        while index < len(rest):
            token = rest[index]
            lowered = token.lower()
            if lowered in _PATH_FLAGS and index + 1 < len(rest):
                targets.append(rest[index + 1])
                index += 2
                continue
            if _looks_like_flag(token):
                index += 1
                continue
            targets.append(token)
            index += 1
    return [target for target in targets if target and target not in {".", ".."}]


def delete_command_is_temp_only(command: str) -> bool:
    """Return True when every extractable target of *command* is temp-dir bound.

    Unparseable / target-less delete commands return False so that the shell
    guard blocks them (fail-closed) rather than letting them through.
    """
    targets = delete_command_targets(command)
    if not targets:
        return False
    base = Path.cwd()
    resolved: list[Path | None] = []
    for raw in targets:
        try:
            candidate = Path(os.path.expandvars(raw)).expanduser()
        except (OSError, RuntimeError, ValueError):
            return False
        if any(ch in raw for ch in "*?$`"):
            # Globs / variable expansions cannot be judged statically.
            return False
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved.append(candidate)
    return bool(resolved) and all(is_temp_path(item) for item in resolved if item)
