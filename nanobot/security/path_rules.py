"""Path-level permission rules with compound actions (FR-8.13).

[LOCAL PATCH] nanowork：路径级权限规则，动作是**复合**的（读 / 写 / 删各自授权）。

## 与既有两层策略的优先级（本条的**前置**问题，计划书要求先定）

落地前必须先回答「规则与 FR-8.4 的默认拒绝写入谁优先」，否则两条策略会在同一次
调用上给出互相矛盾的答案。定下来的顺序是**从具体到笼统、从无条件到可配**：

1. **凭据 `no_access`** —— 无条件守卫（读 + 写 + 删全禁，`full` 模式亦不可越过）。
   **规则无法覆盖它**：否则一条 `read=allow` 就能把 `~/.ssh/id_rsa` 交出去，
   那 FR-8.4 的「凭据读取本身就是一次泄漏」就白做了（验收④）。
2. **工作区 / 允许根包含性** —— 规则**不能放宽边界**。规则只可能在边界**之内**
   做减法；想放宽边界要改 `access_mode`，那是另一根轴（ADR-021）。
3. **路径规则（本条）** —— 命中具体路径时，按该路径声明的动作裁决。
4. **`file.default_deny_write`（FR-8.4）** —— 没有规则对**写**表态时的兜底。

所以「规则优先于默认拒绝写入」：一条 `write=allow` 的规则应当能精确放开某个目录，
否则这条需求就只剩「更细的拒绝」，与 `default_deny_write` 重复而没有增量。

## 为什么动作必须是三种而不是一个 allow/deny

`{path, action: "read=deny|write=deny|delete=deny"}`（对齐 WorkBuddy
`tsbx_rules.json`）：真实诉求几乎全是「这个目录**别让改**，但要能读」——
源码目录、配置模板、第三方 vendor 目录都是这一类。一个笼统的 deny 会让模型连读
都做不到，而它随后会绕道（`shell` + `cat`）——那正是把权限边界做成筛子的方式。
三种动作**正交**：声明 `write=deny` 不得顺带禁读（验收①）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

#: 三种动作。顺序固定（规则摘要与界面都按它渲染）。
ACTION_KINDS: tuple[str, ...] = ("read", "write", "delete")

#: 每个动作只允许这两个取值。**未知取值一律丢弃并告警**：把 `deny` 写错成
#: `dney` 时静默忽略等于放开，而这里是权限边界。
ACTION_VALUES: frozenset[str] = frozenset({"allow", "deny"})


@dataclass(frozen=True, slots=True)
class PathRule:
    """一条规则：一个路径 + 三种动作各自的裁决（未声明的动作是 ``None``）。"""

    path: Path
    read: str | None = None
    write: str | None = None
    delete: str | None = None

    def verdict_for(self, kind: str) -> str | None:
        """该动作的裁决；``None`` = 没表态（**不是** deny）。"""
        if kind not in ACTION_KINDS:
            return None
        return getattr(self, kind)


@dataclass(frozen=True, slots=True)
class PathRuleHit:
    """命中的规则 + 查询的动作 + 该动作的裁决。"""

    rule: PathRule
    kind: str
    verdict: str


def _parse_action(raw: object) -> dict[str, str]:
    """把 ``"read=deny|write=deny"`` 解析成 ``{kind: verdict}``。

    容忍两种形态：上面的字符串，以及已经是 dict 的（配置里手写成
    ``{"read": "deny"}`` 很常见）。坏条目丢弃 + 告警，**不抛**：一条写错的规则
    不该让整个文件失效 —— 那会让「改了半行」表现为「规则全没了」。
    """
    pairs: list[tuple[object, object]] = []
    if isinstance(raw, str):
        for chunk in raw.split("|"):
            kind, sep, value = chunk.partition("=")
            pairs.append((kind.strip(), value.strip() if sep else None))
    elif isinstance(raw, dict):
        pairs.extend((key, value) for key, value in raw.items())
    else:
        return {}

    parsed: dict[str, str] = {}
    for kind, value in pairs:
        if not isinstance(kind, str) or kind not in ACTION_KINDS:
            logger.warning("path rule: ignoring unknown action kind %r", kind)
            continue
        if not isinstance(value, str) or value not in ACTION_VALUES:
            logger.warning(
                "path rule: ignoring action %r=%r (expected allow or deny)", kind, value
            )
            continue
        parsed[kind] = value
    return parsed


def normalize_path_rules(raw: object) -> tuple[PathRule, ...]:
    """把配置里的未知形状收敛成规则元组。

    * 非 list / tuple → 空（当作「没有规则」，**行为与现状一致**，验收③）；
    * 每条缺 ``path``、或 ``path`` 不是字符串、或 ``action`` 解析不出任何有效动作
      → 丢弃该条并告警；
    * 路径**不在这里 resolve**：交给裁决处统一走引擎的 ``resolve_path``（验收②），
      否则同一份路径会有两套归一化口径。
    """
    if not isinstance(raw, (list, tuple)):
        if raw not in (None, "", [], {}):
            logger.warning("path rules: expected a list, got %s", type(raw).__name__)
        return ()

    rules: list[PathRule] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("path rules: skipping non-object entry %r", item)
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            logger.warning("path rules: skipping entry without a usable path: %r", item)
            continue
        actions = _parse_action(item.get("action"))
        if not actions:
            logger.warning(
                "path rules: skipping %r — no valid read/write/delete action was declared",
                raw_path,
            )
            continue
        rules.append(
            PathRule(
                path=Path(raw_path.strip()).expanduser(),
                read=actions.get("read"),
                write=actions.get("write"),
                delete=actions.get("delete"),
            )
        )
    return tuple(rules)


def match_path_rule(
    rules: Sequence[PathRule] | None,
    path: Path,
    kind: str,
) -> PathRuleHit | None:
    """找出**对 ``kind`` 有表态的、最具体**地覆盖 ``path`` 的那条规则。

    两个判断都重要，缺一个都会让规则变得不可推理：

    * **最具体优先**（路径最长者赢）：否则一条挂在工作区根上的 ``write=deny``
      会盖住里面某个子目录特意声明的 ``write=allow`` —— 而用户的直觉是
      「我特意写得越细，它越该赢」。
    * **必须对该动作有表态**：一条只声明 ``write`` 的细规则不该影响 ``read``。
      反过来，它**也不该挡住**更粗那条规则对 ``read`` 的表态 —— 所以这里是
      「对该 kind 有表态的规则里挑最具体」，而不是「最具体的规则有没有表态」。
      后者会让「细规则只写了 write」意外解除了粗规则的 read=deny。

    比较走引擎既有的路径归一化（``resolve_path`` + ``is_path_within``）而不是
    字符串前缀：``startswith`` 会把 ``/ws/a`` 误判成覆盖 ``/ws/ab``，
    且 ``..`` / 大小写 / 尾分隔符全都会让结果变得不可推理（验收②）。
    """
    if not rules or kind not in ACTION_KINDS:
        return None
    from nanobot.security.workspace_policy import is_path_within, resolve_path

    try:
        target = resolve_path(path)
    except (OSError, RuntimeError, ValueError):
        return None

    best: tuple[PathRule, str] | None = None
    best_length = -1
    for rule in rules:
        verdict = rule.verdict_for(kind)
        if verdict is None:
            continue
        try:
            root = resolve_path(rule.path)
        except (OSError, RuntimeError, ValueError):
            logger.warning("path rules: cannot resolve %r, skipping", rule.path)
            continue
        if not (target == root or is_path_within(target, root)):
            continue
        length = len(str(root))
        if length > best_length:
            best = (rule, verdict)
            best_length = length
    if best is None:
        return None
    rule, verdict = best
    return PathRuleHit(rule=rule, kind=kind, verdict=verdict)


def rule_verdict(rules: Sequence[PathRule] | None, path: Path, kind: str) -> str | None:
    """某个动作的裁决。``None`` = 没有规则对这个动作表态（交回既有策略）。"""
    hit = match_path_rule(rules, path, kind)
    return hit.verdict if hit is not None else None


def rule_block_reason(rules: Sequence[PathRule] | None, path: Path, kind: str) -> str | None:
    """被规则拒绝时给人话原因；没被拒返回 ``None``。

    文案必须**点名是哪条规则、什么动作**：只说「路径被策略拒绝」会让用户去翻
    访问模式与凭据名册，而真因只是他自己写的一条 `write=deny`。
    """
    hit = match_path_rule(rules, path, kind)
    if hit is None or hit.verdict != "deny":
        return None
    from nanobot.security.workspace_policy import WORKSPACE_BOUNDARY_NOTE

    return (
        f"The path '{path}' is blocked by a path rule that denies {hit.kind} access "
        f"(rule path: {hit.rule.path}). To proceed, ask the user to change or remove "
        "that rule — it is a configured policy, not a transient failure."
        + WORKSPACE_BOUNDARY_NOTE
    )


def describe_path_rules(rules: Sequence[PathRule] | None) -> str:
    """一行摘要（设置界面与日志用）。空规则给空串，不编「无限制」之类的话。"""
    if not rules:
        return ""
    parts: list[str] = []
    for rule in rules:
        actions = " ".join(
            f"{kind}={rule.verdict_for(kind)}"
            for kind in ACTION_KINDS
            if rule.verdict_for(kind) is not None
        )
        parts.append(f"{rule.path} ({actions})")
    return "; ".join(parts)


def iter_action_kinds() -> Iterable[str]:
    """三种动作的稳定迭代顺序（界面与摘要共用一份顺序）。"""
    return ACTION_KINDS


def as_rule_dicts(rules: Sequence[PathRule] | None) -> list[dict[str, Any]]:
    """反向序列化成配置里的形状（设置界面「保存」用）。"""
    result: list[dict[str, Any]] = []
    for rule in rules or []:
        action: dict[str, str] = {}
        for kind in ACTION_KINDS:
            verdict = rule.verdict_for(kind)
            if verdict is not None:
                action[kind] = verdict
        result.append({"path": str(rule.path), "action": action})
    return result
