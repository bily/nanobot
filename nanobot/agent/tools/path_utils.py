"""Shared path helpers for workspace-scoped tools."""

from pathlib import Path

from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_policy import resolve_allowed_path


def resolve_workspace_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
    extra_allowed_files: list[Path] | None = None,
    include_media_dir: bool = True,
    *,
    write: bool = False,
    deny_write_by_default: bool = False,
) -> Path:
    """Resolve path against workspace and enforce allowed directory containment.

    [LOCAL PATCH] nanowork FR-8.4: ``write`` / ``deny_write_by_default`` are
    forwarded to :func:`resolve_allowed_path` so a write with no configured
    allow-list can be refused (default-deny) instead of silently allowed.
    Credential no-access needs no flag — it is enforced unconditionally.
    """
    media_roots = [get_media_dir()] if include_media_dir else []
    extra_roots = [*media_roots, *(extra_allowed_dirs or [])] if allowed_dir else None
    return resolve_allowed_path(
        path,
        workspace=workspace,
        allowed_root=allowed_dir,
        extra_allowed_roots=extra_roots,
        extra_allowed_files=extra_allowed_files,
        write=write,
        deny_write_by_default=deny_write_by_default,
    )
