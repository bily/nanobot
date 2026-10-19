"""路径级权限规则（FR-8.13，引擎侧）。

四条判据各自对应一种「把规则做错」的方式：

1. **三种动作正交** —— `write=deny` 不得顺带禁读（否则一条规则就把目录变成
   完全不可用，而模型随后会绕道）；
2. **最具体优先** —— 细规则的 `write=allow` 要能压过粗规则的 `write=deny`
   （否则这条需求只剩更细的拒绝，永远无法精确放开某处）；
3. **无规则时行为与现状逐字相同** —— 向后兼容（验收③）；
4. **`no_access` 凭据无条件** —— 规则 `allow` 也不能把 `id_rsa` 交出去（验收④）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.config.schema import Config
from nanobot.security.path_rules import (
    as_rule_dicts,
    describe_path_rules,
    normalize_path_rules,
    rule_verdict,
)
from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    is_path_within,
    resolve_allowed_path,
)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "vendor").mkdir()
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def rules_of(*entries: dict) -> tuple:
    return normalize_path_rules(list(entries))


# ---------------------------------------------------------------------------
# 1. 归一化
# ---------------------------------------------------------------------------


def test_accepts_both_action_shapes(ws: Path) -> None:
    """字符串 `read=deny|write=deny` 与 dict 都要收 —— 配置是人手写的。"""
    rules = rules_of(
        {"path": str(ws), "action": "read=deny|write=deny"},
        {"path": str(ws / "src"), "action": {"write": "allow"}},
    )
    assert rules[0].read == "deny"
    assert rules[0].write == "deny"
    assert rules[1].write == "allow"


def test_bad_entries_are_dropped_not_fatal(ws: Path) -> None:
    """一条写错不该让整份规则失效 —— 那会让「改了半行」表现为「规则全没了」。"""
    rules = rules_of(
        {"path": str(ws), "action": {"write": "deny"}},
        {"action": {"read": "deny"}},  # 缺 path
        {"path": "   ", "action": {"read": "deny"}},  # 空 path
        {"path": str(ws), "action": {"read": "dney"}},  # 拼错取值
        {"path": str(ws), "action": {"chmod": "deny"}},  # 未知动作
        {"path": str(ws), "action": "read"},  # 没有 =value
        "not-an-object",
    )
    assert len(rules) == 1
    assert rules[0].write == "deny"


@pytest.mark.parametrize("raw", [None, "", [], {}, "not-a-list", 42])
def test_non_list_means_no_rules(raw: object) -> None:
    """非 list 一律当「没有规则」（行为与现状一致），而不是抛错。"""
    assert normalize_path_rules(raw) == ()


def test_paths_are_not_resolved_during_normalization(ws: Path) -> None:
    """归一化只做形状收敛；路径归一化交给裁决处（否则同一份路径两套口径）。"""
    rules = rules_of({"path": f"{ws}{''}", "action": {"read": "deny"}})
    assert rules[0].path == Path(ws)


# ---------------------------------------------------------------------------
# 2. 三种动作正交（验收①）
# ---------------------------------------------------------------------------


def test_write_deny_does_not_block_read(ws: Path) -> None:
    """⭐ 本条是 FR-8.13 的存在理由：源码目录「别让改、但要能读」。"""
    rules = rules_of({"path": str(ws), "action": {"write": "deny"}})
    assert rule_verdict(rules, ws / "src" / "b.py", "read") is None  # 没表态
    assert rule_verdict(rules, ws / "src" / "b.py", "write") == "deny"


def test_read_deny_does_not_block_write(ws: Path) -> None:
    rules = rules_of({"path": str(ws), "action": {"read": "deny"}})
    assert rule_verdict(rules, ws / "a.txt", "read") == "deny"
    assert rule_verdict(rules, ws / "a.txt", "write") is None


def test_delete_deny_does_not_block_read_or_write(ws: Path) -> None:
    rules = rules_of({"path": str(ws), "action": {"delete": "deny"}})
    assert rule_verdict(rules, ws / "a.txt", "delete") == "deny"
    assert rule_verdict(rules, ws / "a.txt", "read") is None
    assert rule_verdict(rules, ws / "a.txt", "write") is None


def test_narrow_rule_without_an_opinion_does_not_lift_the_broad_one(ws: Path) -> None:
    """细规则只声明 write 时，**不能**解除粗规则的 read=deny。

    所以「最具体优先」的判据是「对该动作有表态的规则里挑最具体」，
    而不是「最具体的规则有没有表态」—— 后者会让细规则意外放宽粗规则。
    """
    rules = rules_of(
        {"path": str(ws), "action": {"read": "deny", "write": "deny"}},
        {"path": str(ws / "src"), "action": {"write": "allow"}},
    )
    assert rule_verdict(rules, ws / "src" / "b.py", "write") == "allow"
    assert rule_verdict(rules, ws / "src" / "b.py", "read") == "deny"


# ---------------------------------------------------------------------------
# 3. 最具体优先 + 路径比较
# ---------------------------------------------------------------------------


def test_specific_rule_wins_over_broad_one(ws: Path) -> None:
    rules = rules_of(
        {"path": str(ws), "action": {"write": "deny"}},
        {"path": str(ws / "src"), "action": {"write": "allow"}},
    )
    assert rule_verdict(rules, ws / "src" / "b.py", "write") == "allow"
    assert rule_verdict(rules, ws / "a.txt", "write") == "deny"


def test_sibling_prefix_is_not_treated_as_a_match(ws: Path) -> None:
    """`/ws/a` 不得覆盖 `/ws/ab` —— 字符串 `startswith` 的经典错。"""
    (ws / "ab").mkdir()
    (ws / "ab" / "c.txt").write_text("z", encoding="utf-8")
    rules = rules_of({"path": str(ws / "a"), "action": {"write": "deny"}})
    assert rule_verdict(rules, ws / "ab" / "c.txt", "write") is None
    assert rule_verdict(rules, ws / "src", "write") is None


def test_exact_file_rule_matches_only_that_file(ws: Path) -> None:
    rules = rules_of({"path": str(ws / "a.txt"), "action": {"read": "deny"}})
    assert rule_verdict(rules, ws / "a.txt", "read") == "deny"
    assert rule_verdict(rules, ws / "src" / "b.py", "read") is None


def test_unrelated_path_has_no_verdict(ws: Path) -> None:
    rules = rules_of({"path": str(ws / "src"), "action": {"write": "deny"}})
    assert rule_verdict(rules, Path("/somewhere/else"), "write") is None


# ---------------------------------------------------------------------------
# 4. 与 resolve_allowed_path 集成（验收③④）
# ---------------------------------------------------------------------------


def test_no_rules_keeps_existing_behaviour(ws: Path) -> None:
    """无规则：读放行、写由 `deny_write_by_default` 决定（与落地前逐字相同）。"""
    assert resolve_allowed_path(ws / "a.txt", workspace=ws).name == "a.txt"
    assert resolve_allowed_path(ws / "a.txt", workspace=ws, write=True).name == "a.txt"
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path(ws / "a.txt", workspace=ws, write=True, deny_write_by_default=True)


def test_rule_allow_overrides_default_deny_write(ws: Path) -> None:
    """规则（具体）优先于 default_deny_write（笼统）—— 否则只能更细地拒绝。"""
    rules = rules_of(
        {"path": str(ws), "action": {"write": "deny"}},
        {"path": str(ws / "src"), "action": {"write": "allow"}},
    )
    assert resolve_allowed_path(
        ws / "src" / "b.py", workspace=ws, write=True, deny_write_by_default=True, rules=rules
    ).name == "b.py"
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path(
            ws / "a.txt", workspace=ws, write=True, deny_write_by_default=True, rules=rules
        )


def test_rule_denies_read_outside_of_write_path(ws: Path) -> None:
    """`read=deny` 必须在**读**路径上生效，而不只是写。"""
    rules = rules_of({"path": str(ws / "vendor"), "action": {"read": "deny"}})
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path(ws / "vendor" / "x.js", workspace=ws, rules=rules)


def test_delete_action_is_judged_as_delete(ws: Path) -> None:
    """删除复用写解析路径，但动作按 delete 裁决（否则 delete=deny 永不生效）。"""
    rules = rules_of(
        {"path": str(ws), "action": {"write": "allow", "delete": "deny"}},
    )
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path(ws / "a.txt", workspace=ws, write=True, kind="delete", rules=rules)
    # 同一个路径的写仍然放行 —— 这正是「三种动作正交」在集成层的证明
    assert resolve_allowed_path(
        ws / "a.txt", workspace=ws, write=True, kind="write", rules=rules
    ).name == "a.txt"


def test_no_access_credential_wins_over_an_allow_rule(ws: Path) -> None:
    """⭐ 验收④：`no_access` 是无条件守卫，规则 `allow` 也不能覆盖它。"""
    key = ws / "id_rsa"  # 私钥文件名，**不限目录**匹配（见 NO_ACCESS_FILE_NAMES）
    key.write_text("PRIVATE", encoding="utf-8")
    rules = rules_of(
        {"path": str(ws), "action": {"read": "allow", "write": "allow", "delete": "allow"}}
    )
    for kind in ("read", "write", "delete"):
        with pytest.raises(WorkspaceBoundaryError):
            resolve_allowed_path(
                key, workspace=ws, write=kind != "read", kind=kind, rules=rules
            )


def test_rules_cannot_widen_the_boundary(ws: Path, tmp_path: Path) -> None:
    """规则只可能在边界**之内**做减法；放宽边界是 access_mode 的事（ADR-021）。"""
    # ⚠️ 必须是 `ws` 的**兄弟**目录：`ws` 就是 `tmp_path` 本身，
    # 写在 `tmp_path` 下面的「外部路径」其实在边界内，断言会假通过。
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("x", encoding="utf-8")
    rules = rules_of({"path": str(outside), "action": {"read": "allow", "write": "allow"}})
    with pytest.raises(WorkspaceBoundaryError):
        resolve_allowed_path(outside, workspace=ws, allowed_root=ws, rules=rules)


def test_block_reason_names_the_rule_and_action(ws: Path) -> None:
    from nanobot.security.path_rules import rule_block_reason

    rules = rules_of({"path": str(ws), "action": {"write": "deny"}})
    reason = rule_block_reason(rules, ws / "a.txt", "write")
    assert reason is not None
    assert "write" in reason
    assert str(ws) in reason
    # 读没有被拦，就不该有原因
    assert rule_block_reason(rules, ws / "a.txt", "read") is None


# ---------------------------------------------------------------------------
# 5. 摘要 / 反向序列化（设置界面保存用）
# ---------------------------------------------------------------------------


def test_describe_and_round_trip(ws: Path) -> None:
    rules = rules_of(
        {"path": str(ws), "action": {"read": "deny", "write": "deny"}},
        {"path": str(ws / "src"), "action": {"write": "allow"}},
    )
    summary = describe_path_rules(rules)
    assert str(ws) in summary and "read=deny" in summary and "write=allow" in summary
    assert describe_path_rules(()) == ""

    dicts = as_rule_dicts(rules)
    assert dicts[0] == {"path": str(ws), "action": {"read": "deny", "write": "deny"}}
    # 反向序列化再归一化，得到同一份规则（设置界面「保存」的往返）
    assert normalize_path_rules(dicts) == rules


def test_config_round_trip(ws: Path) -> None:
    """`tools.file.path_rules` 必须能进 Config 再出来（引擎 Base 会静默丢未声明键）。"""
    payload = {
        "tools": {
            "file": {
                "path_rules": [{"path": str(ws), "action": {"read": "deny", "write": "allow"}}]
            }
        }
    }
    config = Config.model_validate(payload)
    assert config.tools.file.path_rules[0]["path"] == str(ws)

    dumped = config.model_dump(mode="json", by_alias=True)
    restored = Config.model_validate(dumped)
    assert restored.tools.file.path_rules == config.tools.file.path_rules


def test_config_without_rules_stays_empty() -> None:
    """老配置没有这个键 → 空 list（不抛、不猜默认值）。"""
    assert Config.model_validate({}).tools.file.path_rules == []


def test_is_path_within_still_used_for_matching(ws: Path) -> None:
    """规则匹配走引擎既有的包含性判断（验收②），不是自写的字符串前缀。"""
    rules = rules_of({"path": str(ws), "action": {"write": "deny"}})
    assert is_path_within(ws / "src" / "b.py", ws)
    assert rule_verdict(rules, ws / "src" / "b.py", "write") == "deny"
