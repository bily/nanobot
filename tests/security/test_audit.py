"""[LOCAL PATCH] nanowork FR-8.2 / FR-8.9：工具网关档位与审计留痕。

两层各测各的，互不依赖：

* :func:`decide_tier` 是纯函数 → 穷举真值表（PRD 11.1 明确要求决策逻辑可穷举单测）。
* ``build_audit_record`` 是纯函数 → 断言字段；
  ``record_tool_decision`` 有 IO → 用 ``log_path=tmp_path`` 断言 append-only 行为。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.security.audit import (
    append_audit_record,
    audit_log_path,
    build_audit_record,
    canonical_args,
    digest_args,
    record_tool_decision,
    reset_audit_log_path_cache,
    summarize_args,
)
from nanobot.security.tool_approval import decide_tier

# --------------------------------------------------------------------------- #
# decide_tier —— allow / sandbox / deny 三档
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tool_name",
    [
        "read_file",
        "list_dir",
        "find_files",
        "grep",
        "web_search",
        "web_fetch",
        "search_sessions",
        "read_session",
        "list_sessions",
        "list_exec_sessions",
    ],
)
def test_read_only_tools_are_allow_tier(tool_name):
    tier, reason = decide_tier(tool_name, {"path": "a.txt"})
    assert tier == "allow"
    assert reason


def test_delete_file_is_sandbox_tier():
    """FR-8.3：删除放行，但强制走回收站——档位必须区别于普通写工具。"""
    tier, reason = decide_tier("delete_file", {"path": "a.txt"})
    assert tier == "sandbox"
    assert "recycle bin" in reason


@pytest.mark.parametrize(
    "tool_name",
    ["write_file", "apply_patch", "exec", "mcp__github__create_issue", "totally_unknown"],
)
def test_side_effecting_tools_need_a_verdict(tool_name):
    """未知工具 / MCP 工具一律 deny（默认不放行），不能因为「没见过」就放过。"""
    tier, _ = decide_tier(tool_name, {})
    assert tier == "deny"


@pytest.mark.parametrize(
    "command",
    ["rm a.txt", "rm -rf build", "del x.txt", "Remove-Item -Recurse dir", "unlink a"],
)
def test_shell_delete_commands_are_deny_tier(command):
    """FR-8.3：引擎侧改不了 shell 参数，只能拦下并把模型指回 delete_file。"""
    tier, reason = decide_tier("exec", {"command": command})
    assert tier == "deny"
    assert "delete_file" in reason


def test_shell_write_command_is_deny_but_not_delete_reason():
    """普通 shell 命令是 deny（要裁决），但不该被贴上「删除改道」的理由。"""
    tier, reason = decide_tier("exec", {"command": "ls -la"})
    assert tier == "deny"
    assert "delete_file" not in reason


def test_shell_command_read_from_either_key():
    assert decide_tier("exec", {"cmd": "rm a.txt"})[0] == "deny"
    assert decide_tier("exec", {"command": "rm a.txt"})[0] == "deny"


def test_nameless_tool_call_is_deny():
    tier, reason = decide_tier("", {})
    assert tier == "deny"
    assert "nameless" in reason


def test_tier_is_pure_no_io():
    """同一输入必须永远给同一答案——档位是审计字段，抖动就等于日志失真。"""
    first = decide_tier("write_file", {"path": "x"})
    second = decide_tier("write_file", {"path": "x"})
    assert first == second


# --------------------------------------------------------------------------- #
# 参数规范化与哈希
# --------------------------------------------------------------------------- #


def test_canonical_args_is_key_order_independent():
    """命令哈希必须与字典插入顺序无关，否则同一指令会散成多条记录。"""
    assert canonical_args({"b": 1, "a": 2}) == canonical_args({"a": 2, "b": 1})


def test_digest_args_stable_and_discriminating():
    assert digest_args({"path": "a.txt"}) == digest_args({"path": "a.txt"})
    assert digest_args({"path": "a.txt"}) != digest_args({"path": "b.txt"})


def test_digest_args_handles_non_dict_and_non_json_values():
    """工具参数里什么都有：None / 数字 / 自定义对象。哈希不能因此抛异常。"""
    assert digest_args(None)
    assert digest_args({"n": 1, "obj": object()})


def test_summarize_args_truncates_long_values_and_caps_keys():
    summary = summarize_args({"content": "x" * 5000, "n": 1, "flag": True})
    assert len(summary["content"]) < 5000
    assert summary["content"].endswith("…")
    assert summary["n"] == 1
    assert summary["flag"] is True

    assert len(summarize_args({str(i): i for i in range(100)})) <= 20


def test_summarize_args_returns_empty_for_non_dict():
    assert summarize_args("not a dict") == {}


# --------------------------------------------------------------------------- #
# 记录构造
# --------------------------------------------------------------------------- #


def test_build_audit_record_carries_the_required_fields():
    record = build_audit_record(
        tool="write_file",
        tier="deny",
        outcome="executed",
        approved_by="user",
        session_key="cli:direct",
        channel="telegram",
        args={"path": "notes.md", "content": "hi"},
        approval_mode="ask",
        reason="user approved",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert record["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert record["tool"] == "write_file"
    assert record["tier"] == "deny"
    assert record["outcome"] == "executed"
    assert record["approved_by"] == "user"
    assert record["session_key"] == "cli:direct"
    assert record["channel"] == "telegram"
    assert record["approval_mode"] == "ask"
    assert record["args"] == {"path": "notes.md", "content": "hi"}
    assert record["args_digest"] == digest_args({"path": "notes.md", "content": "hi"})


def test_build_audit_record_omits_empty_optionals():
    record = build_audit_record(tool="grep", tier="allow", outcome="executed")
    for key in (
        "session_key",
        "channel",
        "request_id",
        "reason",
        "approval_mode",
        "agent_id",
    ):
        assert key not in record


def test_build_audit_record_carries_the_agent_id():
    """FR-8.9 要回答「对哪个 Agent」——这是越权调用审计的检索键。"""
    record = build_audit_record(
        tool="exec",
        tier="deny",
        outcome="refused",
        agent_id="researcher",
    )
    assert record["agent_id"] == "researcher"


def test_record_tool_decision_carries_the_agent_id(tmp_path: Path):
    log = tmp_path / "audit.ndjson"
    record_tool_decision(
        tool="write_file",
        tier="deny",
        outcome="refused",
        agent_id="researcher",
        log_path=log,
    )
    assert json.loads(log.read_text(encoding="utf-8").strip())["agent_id"] == "researcher"


def test_build_audit_record_truncates_reason():
    record = build_audit_record(
        tool="exec",
        tier="deny",
        outcome="refused",
        reason="boom " * 500,
    )
    assert len(record["reason"]) <= 400


def test_build_audit_record_is_pure():
    """同参数、显式时间戳 → 逐字节相同，这样审计行才能被外部重算校验。"""
    kwargs = {
        "tool": "exec",
        "tier": "deny",
        "outcome": "refused",
        "args": {"command": "rm -rf /"},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    assert build_audit_record(**kwargs) == build_audit_record(**kwargs)


# --------------------------------------------------------------------------- #
# append-only 落盘
# --------------------------------------------------------------------------- #


def test_record_tool_decision_appends_ndjson_lines(tmp_path: Path):
    log = tmp_path / "audit" / "tool-gateway.ndjson"
    for i in range(3):
        record_tool_decision(
            tool="write_file",
            tier="deny",
            outcome="executed",
            approved_by="user",
            args={"path": f"f{i}.txt"},
            log_path=log,
        )

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [r["args"]["path"] for r in records] == ["f0.txt", "f1.txt", "f2.txt"]


def test_appending_never_rewrites_existing_lines(tmp_path: Path):
    """append-only 是审计的全部价值所在：既有行必须逐字节不变。"""
    log = tmp_path / "tool-gateway.ndjson"
    record_tool_decision(tool="a", tier="allow", outcome="executed", log_path=log)
    original = log.read_text(encoding="utf-8")

    record_tool_decision(tool="b", tier="deny", outcome="refused", log_path=log)
    after = log.read_text(encoding="utf-8")

    assert after.startswith(original)
    assert after != original


def test_every_line_is_valid_json(tmp_path: Path):
    log = tmp_path / "tool-gateway.ndjson"
    record_tool_decision(
        tool="write_file",
        tier="deny",
        outcome="executed",
        args={"content": "line1\nline2\ttabbed"},
        log_path=log,
    )
    # 内嵌换行不能把一行记录劈成两行——换行进 JSON 字符串时会被转义。
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
    json.loads(log.read_text(encoding="utf-8").strip())


def test_audit_failure_never_raises(tmp_path: Path):
    """审计绝不能反过来影响执行：写不进去就静默放弃。"""
    # 把日志路径指向一个「父级是文件」的非法位置 → mkdir 必失败。
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    record_tool_decision(
        tool="write_file",
        tier="deny",
        outcome="executed",
        log_path=blocker / "sub" / "audit.ndjson",
    )


def test_append_audit_record_accepts_prebuilt_record(tmp_path: Path):
    log = tmp_path / "audit.ndjson"
    append_audit_record(
        {"tool": "grep", "tier": "allow", "outcome": "executed"},
        log_path=log,
    )
    assert json.loads(log.read_text(encoding="utf-8").strip())["tool"] == "grep"


# --------------------------------------------------------------------------- #
# 开关与路径解析
# --------------------------------------------------------------------------- #


def test_audit_log_path_can_be_redirected(monkeypatch):
    reset_audit_log_path_cache()
    try:
        path = audit_log_path(env={"NANOBOT_TOOL_AUDIT_LOG": "~/custom/audit.ndjson"})
        assert path is not None
        assert path.name == "audit.ndjson"
        assert "~" not in str(path)
    finally:
        reset_audit_log_path_cache()


def test_audit_log_path_empty_string_disables(monkeypatch):
    reset_audit_log_path_cache()
    try:
        assert audit_log_path(env={"NANOBOT_TOOL_AUDIT_LOG": "   "}) is None
    finally:
        reset_audit_log_path_cache()


def test_disabled_audit_writes_nothing(tmp_path: Path, monkeypatch):
    reset_audit_log_path_cache()
    log = tmp_path / "unused.ndjson"
    try:
        monkeypatch.setenv("NANOBOT_TOOL_AUDIT_LOG", "")
        record_tool_decision(tool="write_file", tier="deny", outcome="executed")
        assert not log.exists()
    finally:
        reset_audit_log_path_cache()


# --------------------------------------------------------------------------- #
# 不变量 2 的回归：审计失败绝不影响执行
#
# 实测事故（`tests/agent/test_dream.py::TestEphemeralDirect::
# test_completed_response_after_tool_error_is_success`）：该用例用 mock 会话作用域
# 驱动执行链，`session_key` 等字段拿到的是 ``MagicMock``——truthy 所以被写进记录，
# 随后 ``json.dumps`` 抛 ``TypeError``，**从工具网关一路上抛**把工具结果打成了失败。
# 原因是「记录构造不是全函数」+「append 只捕 OSError」。下面两条分别钉住这两半：
# 构造出的记录必须永远可序列化；即便再遇到别的怪东西，写盘也必须吞掉。
# --------------------------------------------------------------------------- #


def test_build_audit_record_coerces_non_text_fields():
    """mock 作用域（字段 truthy 但不是字符串）不得让记录变成不可序列化的。"""
    scope = MagicMock()
    record = build_audit_record(
        tool="write_file",
        tier="deny",
        outcome="refused",
        session_key=scope.session_key,
        channel=scope.channel,
        agent_id=scope.agent_id,
        approval_mode=scope.approval_mode,
        request_id=scope.request_id,
        reason=scope.reason,
    )

    # 不抛异常即为通过：这一步曾经抛 TypeError，而它离写盘还差一层调用
    json.dumps(record, ensure_ascii=False)
    for key in (
        "session_key",
        "channel",
        "agent_id",
        "approval_mode",
        "request_id",
        "reason",
    ):
        assert isinstance(record[key], str)


def test_build_audit_record_reason_is_truncated():
    record = build_audit_record(
        tool="grep",
        tier="allow",
        outcome="executed",
        reason="x" * 1000,
    )
    assert len(record["reason"]) == 400


def test_append_audit_record_swallows_any_serialization_error(tmp_path: Path):
    """连 ``default=str`` 都救不回来的对象也不得上抛——宁可丢一行日志。"""

    class Exploding:
        def __str__(self) -> str:  # pragma: no cover - 只在失败路径被调用
            raise RuntimeError("boom")

    log = tmp_path / "audit.ndjson"
    append_audit_record(
        {"tool": "grep", "tier": "allow", "outcome": "executed", "extra": Exploding()},
        log_path=log,
    )
    # 没有异常逃出去；该行写不进去就写不进去
    assert not log.exists() or log.read_text(encoding="utf-8") == ""

