"""Tests for the FR-2.4 expert-group orchestrator (engine-side scheduling).

These exercise the scheduling semantics (order, concurrency cap, aggregation,
failure isolation) with a scripted ``run_member`` — no model, no AgentLoop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

import pytest

from nanobot.agent.expert_group import (
    MAX_CONCURRENT_LIMIT,
    MAX_MEMBERS,
    ExpertGroupError,
    ExpertGroupOrchestrator,
    ExpertMember,
    parse_expert_group,
)


def _spec(**overrides: Any):
    payload: dict[str, Any] = {
        "mode": "sequential",
        "members": [
            {"id": "a", "name": "Alice", "system_prompt": "you are the legal expert"},
            {"id": "b", "name": "Bob"},
        ],
    }
    payload.update(overrides)
    spec = parse_expert_group(payload)
    assert spec is not None
    return spec


class _Recorder:
    """Scripted ``run_member``: records calls, tracks peak concurrency."""

    def __init__(
        self,
        outputs: dict[str, str] | None = None,
        fail: Iterable[str] = (),
        *,
        delay: float = 0.0,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.outputs = outputs or {}
        self.fail = set(fail)
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def __call__(self, member: ExpertMember, prompt: str) -> str:
        self.calls.append((member.id, prompt))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if member.id in self.fail:
                raise RuntimeError(f"{member.id} exploded")
            return self.outputs.get(member.id, f"output-{member.id}")
        finally:
            self.active -= 1

    @property
    def call_ids(self) -> list[str]:
        return [call[0] for call in self.calls]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_parse_none_returns_none():
    assert parse_expert_group(None) is None


def test_parse_accepts_persona_alias_and_defaults():
    spec = parse_expert_group(
        {
            "mode": "parallel",
            "members": [{"id": "a", "persona": "be terse"}],
        }
    )
    assert spec is not None
    assert spec.mode == "parallel"
    assert spec.aggregate == "lead"
    assert spec.max_concurrent > 0
    member = spec.members[0]
    # `name` 缺省时回退到 id：卡片没名字好过卡片渲染出 "None"
    assert member.name == "a"
    assert member.system_prompt == "be terse"


def test_parse_keeps_empty_allow_distinct_from_absent():
    spec = parse_expert_group(
        {
            "mode": "sequential",
            "members": [
                {"id": "a", "tool_allow": []},
                {"id": "b"},
            ],
        }
    )
    assert spec is not None
    # 空列表 = 一个都不许；缺省 = 不限制。这个区别必须活着穿过解析。
    assert spec.members[0].tool_allow == ()
    assert spec.members[1].tool_allow is None


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "nope", "members": [{"id": "a"}]},
        {"mode": "sequential", "members": []},
        {"mode": "sequential", "members": "a"},
        {"mode": "sequential", "members": [{"id": "a"}, {"id": "a"}]},
        {"mode": "sequential", "members": [{"id": ""}]},
        {"mode": "sequential", "members": [{"id": "a"}], "lead": "zzz"},
        {"mode": "sequential", "members": [{"id": "a"}], "aggregate": "magic"},
        {"mode": "sequential", "members": [{"id": "a"}], "max_concurrent": 0},
        {"mode": "sequential", "members": [{"id": "a"}], "max_concurrent": True},
        {
            "mode": "sequential",
            "members": [{"id": f"m{i}"} for i in range(MAX_MEMBERS + 1)],
        },
    ],
)
def test_parse_rejects_bad_shape(payload: Any):
    with pytest.raises(ExpertGroupError):
        parse_expert_group(payload)


def test_parse_rejects_too_many_concurrent():
    with pytest.raises(ExpertGroupError):
        parse_expert_group(
            {
                "mode": "parallel",
                "members": [{"id": "a"}],
                "max_concurrent": MAX_CONCURRENT_LIMIT + 1,
            }
        )


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


async def test_sequential_runs_in_order_and_feeds_prior_outputs():
    recorder = _Recorder()
    spec = _spec(mode="sequential", aggregate="concat")
    result = await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert recorder.call_ids == ["a", "b"]
    # 串行的定义就是「后者看得见前者」——第一个成员的产出必须进第二个的输入。
    assert "output-a" not in recorder.calls[0][1]
    assert "output-a" in recorder.calls[1][1]
    assert "output-a" in result.final_text
    assert "output-b" in result.final_text


async def test_parallel_does_not_leak_outputs_between_members():
    recorder = _Recorder()
    spec = _spec(
        mode="parallel",
        aggregate="concat",
        members=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
    )
    await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    # 并发成员彼此看不见——这是并行的定义，不是丢上下文。
    assert "output-a" not in recorder.calls[1][1]
    assert "output-b" not in recorder.calls[0][1]


async def test_parallel_respects_max_concurrent():
    recorder = _Recorder(delay=0.02)
    spec = _spec(
        mode="parallel",
        aggregate="concat",
        max_concurrent=2,
        members=[{"id": mid, "name": mid} for mid in "abcd"],
    )
    await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert len(recorder.calls) == 4
    assert recorder.max_active <= 2


async def test_coordinator_plans_first_then_dispatches_without_lead():
    recorder = _Recorder()
    spec = _spec(
        mode="coordinator",
        aggregate="concat",
        lead="lead",
        members=[
            {"id": "lead", "name": "Lead"},
            {"id": "b", "name": "B"},
            {"id": "c", "name": "C"},
        ],
    )
    await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert recorder.call_ids[0] == "lead"  # 先出计划
    # lead 出过计划就不再当执行者，其余成员各自拿到计划
    assert set(recorder.call_ids[1:]) == {"b", "c"}
    for _pid, prompt in recorder.calls[1:]:
        assert "output-lead" in prompt


async def test_coordinator_single_member_executes_its_own_plan():
    recorder = _Recorder()
    spec = _spec(
        mode="coordinator",
        aggregate="concat",
        lead="solo",
        members=[{"id": "solo", "name": "Solo"}],
    )
    await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    # 只有一个成员时不能用「排除 lead」把执行者清空——必须回退到它自己执行。
    assert recorder.call_ids == ["solo", "solo"]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


async def test_concat_aggregation_does_not_call_lead():
    recorder = _Recorder()
    spec = _spec(aggregate="concat", lead="a")
    result = await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert recorder.call_ids == ["a", "b"]
    assert result.final_text.startswith("## Alice")


async def test_lead_aggregation_makes_a_final_call():
    recorder = _Recorder(outputs={"lead": "FINAL ANSWER"})
    spec = _spec(
        aggregate="lead",
        lead="lead",
        members=[{"id": "lead", "name": "Lead"}, {"id": "b", "name": "B"}],
    )
    result = await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert result.final_text == "FINAL ANSWER"
    assert recorder.call_ids[-1] == "lead"  # 最后一次是收口
    assert "output-b" in recorder.calls[-1][1]


async def test_empty_lead_aggregate_falls_back_to_concat():
    async def run_member(member: ExpertMember, prompt: str) -> str:
        return "" if member.id == "lead" else "KEEP-ME"

    spec = _spec(
        aggregate="lead",
        lead="lead",
        members=[{"id": "lead", "name": "Lead"}, {"id": "b", "name": "B"}],
    )
    result = await ExpertGroupOrchestrator(run_member).run(spec, user_request="Q")

    # 空汇总会被落库阶段当空 assistant 丢掉 —— 退化为拼接好过用户什么都看不到。
    assert "KEEP-ME" in result.final_text


# --------------------------------------------------------------------------- #
# failure isolation
# --------------------------------------------------------------------------- #


async def test_single_member_failure_is_isolated():
    recorder = _Recorder(fail={"b"})
    spec = _spec(aggregate="lead", lead="a")
    result = await ExpertGroupOrchestrator(recorder).run(spec, user_request="Q")

    assert [o.member.id for o in result.outcomes if o.ok] == ["a"]
    failed = result.failed
    assert [o.member.id for o in failed] == ["b"]
    assert failed[0].error is not None


async def test_every_member_failing_raises():
    recorder = _Recorder(fail={"a", "b"})
    with pytest.raises(ExpertGroupError):
        await ExpertGroupOrchestrator(recorder).run(
            _spec(aggregate="concat"), user_request="Q"
        )


async def test_member_lifecycle_events_are_emitted_with_attribution():
    events: list[tuple[str, str]] = []

    async def on_event(kind: str, member: ExpertMember, extra: dict[str, Any]) -> None:
        events.append((kind, member.id))

    recorder = _Recorder(fail={"b"})
    await ExpertGroupOrchestrator(recorder, on_event=on_event).run(
        _spec(aggregate="concat"), user_request="Q"
    )

    assert ("member_started", "a") in events
    assert ("member_finished", "a") in events
    assert ("member_started", "b") in events
    assert ("member_failed", "b") in events


async def test_event_listener_failure_does_not_break_scheduling():
    async def on_event(kind: str, member: ExpertMember, extra: dict[str, Any]) -> None:
        raise RuntimeError("ui exploded")

    recorder = _Recorder()
    result = await ExpertGroupOrchestrator(recorder, on_event=on_event).run(
        _spec(aggregate="concat"), user_request="Q"
    )

    # 进度上报是旁路：它炸了不能把整团带走。
    assert "output-a" in result.final_text
