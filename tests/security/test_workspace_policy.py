from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    is_no_access_path,
    is_path_within,
    resolve_allowed_path,
)


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(completed.stderr.strip() or completed.stdout.strip())
        return

    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def test_resolve_allowed_path_accepts_workspace_relative_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "src" / "main.py"
    target.parent.mkdir()
    target.write_text("print('ok')", encoding="utf-8")

    resolved = resolve_allowed_path("src/main.py", workspace=workspace, allowed_root=workspace)

    assert resolved == target.resolve()


def test_resolve_allowed_path_blocks_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="outside allowed directory"):
        resolve_allowed_path("../secret.txt", workspace=workspace, allowed_root=workspace)


def test_resolve_allowed_path_blocks_traversal_shapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    traversal_shapes: list[str | Path] = [
        "../secret.txt",
        "src/../../secret.txt",
        Path("..") / "secret.txt",
        workspace / "src" / ".." / ".." / "secret.txt",
    ]
    if os.name == "nt":
        traversal_shapes.append("src\\..\\..\\secret.txt")

    for candidate in traversal_shapes:
        with pytest.raises(WorkspaceBoundaryError, match="outside allowed directory"):
            resolve_allowed_path(candidate, workspace=workspace, allowed_root=workspace)


def test_resolve_allowed_path_blocks_prefix_sibling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="outside allowed directory"):
        resolve_allowed_path(secret, workspace=workspace, allowed_root=workspace)


def test_resolve_allowed_path_blocks_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = workspace / "linked-secret.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    assert not is_path_within(link, workspace)
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path("linked-secret.txt", workspace=workspace, allowed_root=workspace)


def test_resolve_allowed_path_allows_extra_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    image = media / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = resolve_allowed_path(
        image,
        workspace=workspace,
        allowed_root=workspace,
        extra_allowed_roots=[media],
    )

    assert resolved == image.resolve()


def test_resolve_allowed_path_allows_extra_file_only_exactly(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    allowed = outside / "allowed.txt"

    resolved = resolve_allowed_path(
        allowed,
        workspace=workspace,
        allowed_root=workspace,
        extra_allowed_files=[allowed],
    )

    assert resolved == allowed.resolve()
    with pytest.raises(WorkspaceBoundaryError, match="outside allowed directory"):
        resolve_allowed_path(
            allowed / "child.txt",
            workspace=workspace,
            allowed_root=workspace,
            extra_allowed_files=[allowed],
        )


def test_resolve_allowed_path_extra_file_blocks_link_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "MEMORY.md"
    outside_target.write_text("secret", encoding="utf-8")

    memory_link = workspace / "memory"
    _make_directory_link(memory_link, outside)
    logical_allowed = memory_link / "MEMORY.md"

    with pytest.raises(WorkspaceBoundaryError, match="outside allowed directory"):
        resolve_allowed_path(
            "memory/MEMORY.md",
            workspace=workspace,
            allowed_root=workspace / "skills",
            extra_allowed_files=[logical_allowed],
        )


# ---------------------------------------------------------------------------
# FR-8.4 — credential no-access (read *and* write, no allow-list can override)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the credential deny-list at an isolated home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/id_rsa",
        ".ssh/known_hosts",
        ".gnupg/secring.gpg",
        ".aws/credentials",
        ".config/gcloud/application_default_credentials.json",
    ],
)
def test_credential_directory_is_no_access_without_allow_root(
    fake_home: Path, relative: str
) -> None:
    """`full` access mode (no allow-root) must still refuse credential dirs.

    This is the regression guard for the early-return bypass: the guard has to
    run before the "no allow-list configured" shortcut, not after it.
    """
    target = fake_home / relative

    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(target, workspace=fake_home, allowed_root=None)


def test_credential_directory_is_no_access_for_reads_too(fake_home: Path) -> None:
    target = fake_home / ".ssh" / "id_rsa"
    assert is_no_access_path(target) is True

    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(
            "id_rsa",
            workspace=fake_home / ".ssh",
            allowed_root=fake_home / ".ssh",
        )


def test_credential_file_name_is_no_access_anywhere(tmp_path: Path) -> None:
    """Private keys match by name even outside a credential directory."""
    stray = tmp_path / "backup" / "id_ed25519"
    assert is_no_access_path(stray) is True

    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(stray, workspace=tmp_path, allowed_root=tmp_path)


def test_home_credential_file_is_no_access(fake_home: Path) -> None:
    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(
            fake_home / ".npmrc",
            workspace=fake_home,
            allowed_root=fake_home,
        )


def test_allow_list_cannot_override_no_access(fake_home: Path) -> None:
    """Credential protection is unconditional: an explicit allow still fails."""
    key = fake_home / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True)
    key.write_text("PRIVATE", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(
            key,
            workspace=fake_home,
            allowed_root=fake_home / ".ssh",
            extra_allowed_files=[key],
        )


def test_no_access_follows_symlink_into_credential_dir(
    fake_home: Path, tmp_path: Path
) -> None:
    secrets = fake_home / ".ssh"
    secrets.mkdir()
    (secrets / "id_rsa").write_text("PRIVATE", encoding="utf-8")
    link = tmp_path / "innocent"
    try:
        link.symlink_to(secrets, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(WorkspaceBoundaryError, match="no-access"):
        resolve_allowed_path(link / "id_rsa", workspace=tmp_path, allowed_root=tmp_path)


def test_non_credential_home_file_stays_reachable(fake_home: Path) -> None:
    """The guard must not spill over onto ordinary product files under ~."""
    memory = fake_home / ".nanowork" / "MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("notes", encoding="utf-8")

    assert is_no_access_path(memory) is False
    resolved = resolve_allowed_path(memory, workspace=fake_home, allowed_root=fake_home)
    assert resolved == memory.resolve()


def test_project_file_named_like_a_credential_is_not_collateral_damage(
    tmp_path: Path,
) -> None:
    """A repo-local `credentials.json` / `.npmrc` fixture is legitimate work."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture = workspace / "testdata" / "credentials.json"
    local_npmrc = workspace / ".npmrc"

    for candidate in (fixture, local_npmrc):
        assert is_no_access_path(candidate) is False
        resolved = resolve_allowed_path(
            candidate, workspace=workspace, allowed_root=workspace
        )
        assert resolved == candidate.resolve()


# ---------------------------------------------------------------------------
# FR-8.4 — write default-deny
# ---------------------------------------------------------------------------


def test_write_without_allow_list_is_denied_when_default_deny_is_on(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(WorkspaceBoundaryError, match="default"):
        resolve_allowed_path(
            workspace / "notes.md",
            workspace=workspace,
            allowed_root=None,
            write=True,
            deny_write_by_default=True,
        )


def test_read_without_allow_list_still_allowed_with_default_deny_on(
    tmp_path: Path,
) -> None:
    """Default-deny is scoped to writes; reads keep upstream semantics."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.md"
    target.write_text("hi", encoding="utf-8")

    resolved = resolve_allowed_path(
        target,
        workspace=workspace,
        allowed_root=None,
        write=False,
        deny_write_by_default=False,
    )
    assert resolved == target.resolve()


def test_write_allowed_within_allow_list_under_default_deny(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.md"

    resolved = resolve_allowed_path(
        target,
        workspace=workspace,
        allowed_root=workspace,
        write=True,
        deny_write_by_default=True,
    )
    assert resolved == target.resolve()


def test_write_default_deny_is_opt_in_for_upstream_callers(tmp_path: Path) -> None:
    """Without the flag the pre-existing permissive behaviour is unchanged."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.md"

    resolved = resolve_allowed_path(
        target,
        workspace=workspace,
        allowed_root=None,
        write=True,
        deny_write_by_default=False,
    )
    assert resolved == target.resolve()
