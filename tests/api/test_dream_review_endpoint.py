"""Tests for the FR-4.6 Dream review REST endpoints (thin wrappers over MemoryStore).

These exercise the handler functions directly with a lightweight fake request,
so they do not need a full aiohttp application or a running gateway.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.api.server import (
    _AGENT_LOOP_KEY,
    handle_dream_accept,
    handle_dream_reject,
    handle_dream_review,
)


def _fake_request(workspace) -> SimpleNamespace:
    loop = SimpleNamespace(workspace=workspace)
    return SimpleNamespace(app={_AGENT_LOOP_KEY: loop})


async def _seed_candidate(ws, *, real: str, candidate: str) -> MemoryStore:
    (ws / "memory").mkdir(parents=True, exist_ok=True)
    store = MemoryStore(ws)
    store.write_memory(real)
    store.build_dream_tools(require_review=True)
    store.dream_candidate_dir.joinpath("MEMORY.md").write_text(candidate, encoding="utf-8")
    return store


async def test_review_no_candidate_reports_pending_false(tmp_path) -> None:
    resp = await handle_dream_review(_fake_request(tmp_path))
    body = resp.body.decode("utf-8")
    assert '"pending": false' in body


async def test_review_surfaces_candidate_diff(tmp_path) -> None:
    await _seed_candidate(tmp_path, real="original", candidate="edited")
    resp = await handle_dream_review(_fake_request(tmp_path))
    body = resp.body.decode("utf-8")
    assert '"pending": true' in body
    assert "modified" in body
    assert "edited" in body


async def test_accept_writes_back_and_clears_candidate(tmp_path) -> None:
    store = await _seed_candidate(tmp_path, real="original", candidate="edited")
    resp = await handle_dream_accept(_fake_request(tmp_path))
    assert '"ok": true' in resp.body.decode("utf-8")
    assert store.memory_file.read_text(encoding="utf-8") == "edited"
    assert not store.dream_candidate_dir.exists()


async def test_reject_keeps_real_and_clears_candidate(tmp_path) -> None:
    store = await _seed_candidate(tmp_path, real="original", candidate="hacked")
    resp = await handle_dream_reject(_fake_request(tmp_path))
    assert '"ok": true' in resp.body.decode("utf-8")
    assert store.memory_file.read_text(encoding="utf-8") == "original"  # untouched
    assert not store.dream_candidate_dir.exists()


async def test_accept_without_candidate_is_409(tmp_path) -> None:
    resp = await handle_dream_accept(_fake_request(tmp_path))
    assert resp.status == 409
