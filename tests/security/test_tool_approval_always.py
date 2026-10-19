"""[LOCAL PATCH] nanowork：「本会话始终允许」的审批豁免契约。

## 为什么要单独钉这一组

`always` 是一个**比 `allow` 更强**的裁决：它不只放行这一次，还让引擎在后续
调用同一个工具时不再打扰用户。这类「一次点击改变后续策略」的能力有三处最容易
静默出错，而每一处都不会报错，只会让权限边界悄悄变宽或变窄：

1. **`always` 被当成拒绝**：`ApprovalDecision.allowed` 少一支判断，用户点
   「始终允许」当场变成「操作没执行」——比不给按钮更让人费解。
2. **豁免跨会话泄漏**：豁免集合若按「进程」而不是「会话」记账，用户在 A 会话
   放开的工具会在 B 会话静默生效。
3. **豁免在会话结束后残留**：`cancel_session` 只清了待裁决项、忘了清豁免，
   下一次同名会话就继承了一份用户看不见的许可。

三处都在这里逐条钉住。
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.security.tool_approval import (
    ApprovalDecision,
    ApprovalRequest,
    PendingApprovals,
    ToolApprovalPolicy,
)

SESSION = "api:test-session"


def _registry() -> PendingApprovals:
    return PendingApprovals()


# --------------------------------------------------------------------------
# 1. 裁决语义：always 必须也「放行」
# --------------------------------------------------------------------------


def test_always_verdict_allows_the_call() -> None:
    """``always`` 是放行，不是拒绝——少这一支会让按钮「点了没反应」。"""
    assert ApprovalDecision(verdict="always").allowed is True


def test_allow_verdict_allows_the_call() -> None:
    assert ApprovalDecision(verdict="allow").allowed is True


def test_deny_verdict_does_not_allow_the_call() -> None:
    assert ApprovalDecision(verdict="deny").allowed is False


# --------------------------------------------------------------------------
# 2. 豁免记账：只在「本会话 + 本工具」这一格里生效
# --------------------------------------------------------------------------


def test_resolve_always_records_a_session_exemption() -> None:
    registry = _registry()
    request = ApprovalRequest(request_id="r1", tool="exec", args={}, session_key=SESSION)

    async def scenario() -> bool:
        registry.register(request)
        return registry.resolve("r1", ApprovalDecision(verdict="always"))

    assert asyncio.run(scenario()) is True
    assert registry.always_allowed(SESSION) == frozenset({"exec"})


def test_resolve_allow_does_not_record_an_exemption() -> None:
    """「允许本次」的意思就是「就这一次」——记下来会让按钮失去区别。"""
    registry = _registry()
    request = ApprovalRequest(request_id="r1", tool="exec", args={}, session_key=SESSION)

    async def scenario() -> None:
        registry.register(request)
        registry.resolve("r1", ApprovalDecision(verdict="allow"))

    asyncio.run(scenario())

    assert registry.always_allowed(SESSION) == frozenset()


def test_exemption_for_one_tool_does_not_leak_to_another() -> None:
    """批准 ``exec`` 不等于批准 ``write_file``：一次点击只豁免用户看见的那个工具。"""
    registry = _registry()
    registry.grant_always(SESSION, "exec")

    assert registry.always_allowed(SESSION) == frozenset({"exec"})
    assert "write_file" not in registry.always_allowed(SESSION)


def test_exemption_does_not_leak_across_sessions() -> None:
    """按会话分片：A 会话的放开不能在 B 会话静默生效。"""
    registry = _registry()
    registry.grant_always("api:session-a", "exec")

    assert registry.always_allowed("api:session-a") == frozenset({"exec"})
    assert registry.always_allowed("api:session-b") == frozenset()


def test_keyless_session_gets_no_exemption() -> None:
    """没有会话键就什么都不记。

    若把 ``None`` 归一到 ``""`` 这个桶，所有无键轮次（ephemeral turn）会共用
    同一格豁免——一次点击变成进程级放开。
    """
    registry = _registry()
    registry.grant_always(None, "exec")
    registry.grant_always("", "exec")

    assert registry.always_allowed(None) == frozenset()
    assert registry.always_allowed("") == frozenset()


def test_exemption_is_written_before_the_turn_is_woken() -> None:
    """豁免必须在唤醒等待中的 turn **之前**落账。

    反过来的话，被唤醒的那一轮可能立刻再发一次同工具的调用，而那一刻豁免
    还没写进去——用户刚点完「始终允许」，紧接着又被问一次同一个工具。
    """

    async def scenario() -> list[str]:
        registry = _registry()
        request = ApprovalRequest(
            request_id="r1", tool="exec", args={}, session_key=SESSION
        )
        future = registry.register(request)

        registry.resolve("r1", ApprovalDecision(verdict="always"))
        # future 已被 set_result：这里立刻读豁免，必须已经就位
        assert future.done()
        return sorted(registry.always_allowed(SESSION))

    assert asyncio.run(scenario()) == ["exec"]


# --------------------------------------------------------------------------
# 3. 策略判定：豁免只跳过「要不要问」，不越过任何别的门
# --------------------------------------------------------------------------


def test_policy_asks_for_a_side_effecting_tool_by_default() -> None:
    policy = ToolApprovalPolicy()
    assert policy.requires_approval("exec") is True


def test_policy_stops_asking_once_exempted() -> None:
    policy = ToolApprovalPolicy()
    assert policy.requires_approval("exec", always_allowed=frozenset({"exec"})) is False


def test_read_only_tools_are_still_never_asked() -> None:
    policy = ToolApprovalPolicy()
    assert policy.requires_approval("read_file") is False


def test_operator_require_outranks_a_session_grant() -> None:
    """运维声明的 ``require`` 是常设指令，不能被一次运行时点击穿透。"""
    policy = ToolApprovalPolicy(require=frozenset({"exec"}))

    assert policy.requires_approval("exec") is True
    assert policy.requires_approval("exec", always_allowed=frozenset({"exec"})) is True


def test_operator_exempt_outranks_the_ask_path() -> None:
    policy = ToolApprovalPolicy(exempt=frozenset({"exec"}))
    assert policy.requires_approval("exec") is False


def test_submit_plan_is_never_asked_even_when_exempted() -> None:
    """提交计划本身是「把方案交给用户裁决」，不该再套一层批准。"""
    policy = ToolApprovalPolicy()
    assert policy.requires_approval("submit_plan") is False


# --------------------------------------------------------------------------
# 4. 会话取消：豁免必须跟着待裁决项一起清掉
# --------------------------------------------------------------------------


def test_cancel_session_forgets_the_exemptions() -> None:
    """「本会话」在会话被取消时结束——残留的豁免是用户看不见的许可。"""
    registry = _registry()
    registry.grant_always(SESSION, "exec")
    request = ApprovalRequest(request_id="r1", tool="exec", args={}, session_key=SESSION)

    async def scenario() -> int:
        registry.register(request)
        return registry.cancel_session(SESSION)

    assert asyncio.run(scenario()) == 1
    assert registry.always_allowed(SESSION) == frozenset()


def test_cancel_session_leaves_other_sessions_alone() -> None:
    registry = _registry()
    registry.grant_always("api:session-a", "exec")
    registry.grant_always("api:session-b", "write_file")

    registry.cancel_session("api:session-a")

    assert registry.always_allowed("api:session-a") == frozenset()
    assert registry.always_allowed("api:session-b") == frozenset({"write_file"})


def test_grant_ignores_a_blank_tool_name() -> None:
    """空工具名不该占一格豁免：它无法与任何真实调用对上，只会成为噪音。"""
    registry = _registry()
    registry.grant_always(SESSION, "   ")
    assert registry.always_allowed(SESSION) == frozenset()


def test_discard_does_not_touch_exemptions() -> None:
    """超时 / 取消走 `discard`，它只摘待裁决项，不动已经授予的豁免。

    这个区别是刻意的：`discard` 的语义是「这一条请求没了」，不是「本会话
    结束了」。把豁免一起清掉会让一次网络抖动悄悄撤销用户的「始终允许」。
    """
    registry = _registry()
    registry.grant_always(SESSION, "exec")
    request = ApprovalRequest(request_id="r1", tool="exec", args={}, session_key=SESSION)

    async def scenario() -> None:
        registry.register(request)
        registry.discard("r1")

    asyncio.run(scenario())

    assert registry.always_allowed(SESSION) == frozenset({"exec"})
