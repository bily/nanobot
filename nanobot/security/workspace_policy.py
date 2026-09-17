"""Workspace path boundary helpers.

These helpers are application-level guards.  They make path decisions
consistent across tools, but they are not a replacement for an OS sandbox.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

WORKSPACE_BOUNDARY_NOTE = (
    " (this is a hard policy boundary, not a transient failure; "
    "do not retry with shell tricks or alternative tools, and ask "
    "the user how to proceed if the resource is genuinely required)"
)

#: [LOCAL PATCH] nanowork FR-8.4「凭据保护」。
#: 这些目录下的任何路径都是 ``no_access``——**读写皆禁**，且**不受任何
#: allow-list 影响**（`full` 访问模式也不例外）。理由：凭据读取本身就是
#: 一次泄漏，允许它出现在「用户已授权完全访问」的语境里等于把私钥交给
#: 一次提示词注入。全部锚定在用户主目录下，避免把一个恰好叫 `.docker` 的
#: 项目子目录误判成凭据目录。
NO_ACCESS_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".password-store",
        ".config/gcloud",
        ".config/gh",
    }
)

#: 主目录下的凭据**文件**（直接子项）。同样锚定在 ``~`` 下：`~/.npmrc` 带
#: registry token 必须拦住，而项目里的 `.npmrc` 通常只是包源配置，不该被误伤。
NO_ACCESS_HOME_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".htpasswd",
        ".authinfo",
    }
)

#: 私有密钥文件按**文件名**匹配（不限目录）。SSH 私钥没有扩展名、也可能被
#: 拷到任何地方，只靠目录判定拦不干净；这几个名字无歧义，误伤面可忽略。
NO_ACCESS_FILE_NAMES: frozenset[str] = frozenset(
    {
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)

NO_ACCESS_NOTE = (
    " (this path holds credentials and is a hard no-access region; "
    "reading and writing are both refused, and no access mode or allow-list "
    "can override it. Do not retry with another tool or a shell indirection; "
    "ask the user to supply what is needed another way)"
)


class WorkspaceBoundaryError(PermissionError):
    """Raised when a requested path escapes an allowed workspace boundary."""


def resolve_path(path: str | Path, workspace: str | Path | None = None, *, strict: bool = False) -> Path:
    """Resolve *path*, interpreting relative paths against *workspace* when set."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and workspace is not None:
        candidate = Path(workspace).expanduser() / candidate
    return candidate.resolve(strict=strict)


def _resolve_logical_path(path: str | Path, workspace: str | Path | None = None) -> Path:
    """Return an absolute normalized path without following symlinks."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and workspace is not None:
        candidate = Path(workspace).expanduser() / candidate
    return Path(os.path.abspath(candidate))


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.fspath(path))


def is_path_within(path: str | Path, root: str | Path) -> bool:
    """Return True when *path* resolves to *root* or a descendant of *root*."""
    try:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        resolved_root = Path(root).expanduser().resolve(strict=False)
        resolved_path.relative_to(resolved_root)
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def is_path_allowed(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """Return True when *path* is inside any allowed root."""
    return any(is_path_within(path, root) for root in roots)


def _user_home(home: str | Path | None = None) -> Path | None:
    """Resolve the home directory the credential deny-list is anchored to."""
    if home is not None:
        try:
            return Path(home).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
    try:
        return Path.home().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        # No determinable home directory: callers still get file-name matching.
        return None


def no_access_roots(home: str | Path | None = None) -> list[Path]:
    """Return the absolute credential directories that are always off-limits."""
    base = _user_home(home)
    if base is None:
        return []
    roots: list[Path] = []
    for name in sorted(NO_ACCESS_DIR_NAMES):
        try:
            roots.append((base / name).resolve(strict=False))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    return roots


def no_access_home_files(home: str | Path | None = None) -> list[Path]:
    """Return the credential files that sit directly in the user's home."""
    base = _user_home(home)
    if base is None:
        return []
    return [base / name for name in sorted(NO_ACCESS_HOME_FILE_NAMES)]


def is_no_access_path(
    path: str | Path | None,
    *,
    home: str | Path | None = None,
) -> bool:
    """Return True when *path* is a credential path (FR-8.4, read **and** write).

    Three independent rules, any one of which is sufficient:

    * the path sits inside a credential directory under the user's home
      (`~/.ssh`, `~/.gnupg`, …);
    * the path *is* a credential file directly in the user's home
      (`~/.npmrc`, `~/.netrc`, …);
    * the final path segment is a private-key file name (`id_rsa`, …) anywhere.

    Every rule is deliberately anchored or name-exact so a project file that
    merely *looks* credential-ish (a `credentials.json` fixture, a repo-local
    `.npmrc`) is not collateral damage.

    Fails **closed**: an unparseable path is treated as no-access rather than
    waved through. This is the one guard in this module that cannot be relaxed
    by configuration.
    """
    if path is None:
        return False
    try:
        logical = _resolve_logical_path(path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return True

    # Rule 2 — a private-key file name, wherever it lives.
    try:
        if Path(logical).name in NO_ACCESS_FILE_NAMES:
            return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return True

    # Rule 3 — a credential file directly in the user's home.
    if any(_path_key(logical) == _path_key(item) for item in no_access_home_files(home)):
        return True

    # Rule 1 — inside a credential directory anchored at the user's home.
    # ``is_path_within`` resolves both sides, so a symlink planted elsewhere
    # and pointing into ~/.ssh is caught too.
    roots = no_access_roots(home)
    if not roots:
        return False
    if is_path_allowed(logical, roots):
        return True
    try:
        resolved = Path(logical).resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return True
    return is_path_allowed(resolved, roots)


def require_no_access_free(
    path: str | Path,
    *,
    home: str | Path | None = None,
) -> None:
    """Raise when *path* is a credential path."""
    if is_no_access_path(path, home=home):
        raise WorkspaceBoundaryError(
            f"Path {path} is inside a no-access credential location" + NO_ACCESS_NOTE
        )


def _is_path_exactly_allowed(
    logical_path: Path,
    resolved_path: Path,
    files: Iterable[str | Path],
) -> bool:
    """Return True when *path* resolves exactly to one of the allowed files."""
    logical_key = _path_key(logical_path)
    if _path_key(resolved_path) != logical_key:
        return False
    for file in files:
        try:
            allowed_file = _resolve_logical_path(file)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if _path_key(allowed_file) == logical_key:
            return True
    return False


def require_path_within(
    path: str | Path,
    root: str | Path,
    *,
    message: str | None = None,
) -> Path:
    """Resolve *path* and require it to be inside *root*."""
    resolved = Path(path).expanduser().resolve(strict=False)
    if not is_path_within(resolved, root):
        raise WorkspaceBoundaryError(
            message
            or f"Path {path} is outside allowed directory {Path(root).expanduser()}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    return resolved


def resolve_allowed_path(
    path: str | Path,
    *,
    workspace: str | Path | None = None,
    allowed_root: str | Path | None = None,
    extra_allowed_roots: Iterable[str | Path] | None = None,
    extra_allowed_files: Iterable[str | Path] | None = None,
    write: bool = False,
    deny_write_by_default: bool = False,
    strict: bool = False,
) -> Path:
    """Resolve a path and enforce containment in allowed roots when configured.

    [LOCAL PATCH] nanowork FR-8.4 adds two guards on top of the upstream
    containment check:

    * **Credential no-access** — checked first and unconditionally. It has to
      run *before* the "no allow-list configured" early return below, otherwise
      ``full`` access mode (``allowed_root is None``) would short-circuit
      straight past it and hand the model ``~/.ssh/id_rsa``.
    * **Write default-deny** — when ``write`` and ``deny_write_by_default`` are
      both set, a write with no configured allow-list is refused instead of
      being treated as unrestricted. Off by default so upstream callers keep
      their existing semantics; the nanowork client turns it on.
    """
    resolved = resolve_path(path, workspace, strict=False)
    require_no_access_free(resolved)

    files = list(extra_allowed_files or [])
    if allowed_root is None and not files:
        if write and deny_write_by_default:
            raise WorkspaceBoundaryError(
                f"Refusing to write {path}: no writable allow-list is configured "
                "for this scope and writes default to denied"
                + WORKSPACE_BOUNDARY_NOTE
            )
        return resolve_path(path, workspace, strict=strict) if strict else resolved

    roots: list[str | Path] = []
    if allowed_root is not None:
        roots.append(allowed_root)
    roots.extend(extra_allowed_roots or [])
    exact_allowed = bool(files) and _is_path_exactly_allowed(
        _resolve_logical_path(path, workspace),
        resolved,
        files,
    )
    if not is_path_allowed(resolved, roots) and not exact_allowed:
        boundary = Path(allowed_root).expanduser() if allowed_root is not None else "allowed files"
        raise WorkspaceBoundaryError(
            f"Path {path} is outside allowed directory {boundary}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    if strict:
        return resolve_path(path, workspace, strict=True)
    return resolved
