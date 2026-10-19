"""[LOCAL PATCH] FR-2.4 专家团编排：引擎侧调度（路径 2 第 2 步）。

设计见 `docs/design-fr-2-4-expert-orchestration.md` §3 / §7.2。本模块只管一件事：
**拿到成员清单之后怎么调度**。真正「跑一个成员」的动作由调用方以 `run_member` 注入
（生产实现是 `SubagentManager.run_inline`），因此调度语义——顺序、并发上限、聚合方式、
失败策略——可以**完全不依赖模型**地被单测覆盖。

## 落库形态（第 2 步的关键决策）

一次专家团请求**只产出一条 assistant 消息**（最终汇总）。成员输出不是消息，只作为过程
事件。理由见设计 §7.2：否则会话历史被 N 倍膨胀，并污染下一轮上下文。
本模块只产出 `final_text`；「怎么把它写进会话」由 `loop.py` 的 turn 分支负责。

## 为什么严格校验形状

请求侧来自客户端，一个拼错字段名如果被「宽松忽略」，就会退化成「成员数量静默变成 0」
或「人设静默丢失」——两种都不会报错，只会让专家团变成普通一轮对话。所以这里**一律
报错**（`ExpertGroupError`），由调用方转成 400。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: 请求 metadata 里承载专家团清单的键（`/v1` 请求体 → turn metadata）。
EXPERT_GROUP_METADATA_KEY = "expert_group"

MODES: frozenset[str] = frozenset({"sequential", "parallel", "coordinator"})
AGGREGATES: frozenset[str] = frozenset({"lead", "concat"})

#: 一个团最多几个成员。这个上限不是能力问题而是**成本刹车**：每多一个成员就多一份
#: token 账单（设计 §2 第 4 条）。默认 4 个并发是「够快也不至于把 provider 打满」。
MAX_MEMBERS = 8
MAX_CONCURRENT_LIMIT = 8
DEFAULT_MAX_CONCURRENT = 4


class ExpertGroupError(ValueError):
    """专家团请求形状非法，或整团成员全部失败。"""


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpertGroupError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_str_tuple(value: Any, field: str) -> tuple[str, ...] | None:
    """``None``（缺省）与空元组是**不同**的语义，不能抹平（同 WorkspaceScope.tool_allow）。"""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpertGroupError(f"{field} must be an array of strings")
    out: list[str] = []
    for item in value:
        out.append(_require_str(item, field))
    return tuple(out)


@dataclass(frozen=True)
class ExpertMember:
    """团里的一个成员：人设 + 工具策略 + 可选的模型覆盖。"""

    id: str
    name: str
    system_prompt: str = ""
    #: ``None`` = 不限制（继承父作用域）；空元组 = 一个都不许。区别必须保留。
    tool_allow: tuple[str, ...] | None = None
    tool_deny: tuple[str, ...] = ()
    model: str | None = None


@dataclass(frozen=True)
class ExpertGroupSpec:
    mode: str
    members: tuple[ExpertMember, ...]
    lead: str | None = None
    aggregate: str = "lead"
    max_concurrent: int = DEFAULT_MAX_CONCURRENT

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.members)

    def find(self, member_id: str | None) -> ExpertMember | None:
        if member_id is None:
            return None
        return next((m for m in self.members if m.id == member_id), None)

    @property
    def lead_member(self) -> ExpertMember:
        """lead 缺省时用首位成员——不额外报错，因为「谁是 lead」有天然默认值。"""
        return self.find(self.lead) or self.members[0]


def _parse_member(raw: Any, index: int) -> ExpertMember:
    if not isinstance(raw, Mapping):
        raise ExpertGroupError(f"members[{index}] must be an object")
    member_id = _require_str(raw.get("id"), f"members[{index}].id")
    raw_name = raw.get("name")
    name = (
        raw_name.strip()
        if isinstance(raw_name, str) and raw_name.strip()
        else member_id
    )
    # 人设键名兼容 `system_prompt` / `persona`（客户端历史上两种都出现过）。
    raw_prompt = raw.get("system_prompt")
    if raw_prompt is None:
        raw_prompt = raw.get("persona")
    if raw_prompt is not None and not isinstance(raw_prompt, str):
        raise ExpertGroupError(f"members[{index}].system_prompt must be a string")
    raw_model = raw.get("model")
    if raw_model is not None and (not isinstance(raw_model, str) or not raw_model.strip()):
        raise ExpertGroupError(f"members[{index}].model must be a non-empty string")
    return ExpertMember(
        id=member_id,
        name=name,
        system_prompt=raw_prompt or "",
        tool_allow=_optional_str_tuple(raw.get("tool_allow"), f"members[{index}].tool_allow"),
        tool_deny=_optional_str_tuple(raw.get("tool_deny"), f"members[{index}].tool_deny") or (),
        model=raw_model.strip() if isinstance(raw_model, str) else None,
    )


def parse_expert_group(payload: Any) -> ExpertGroupSpec | None:
    """把请求里的 ``expert_group`` 解析成规格。

    ``None`` / 缺省 → 返回 ``None``（普通单 Agent 轮次）。形状非法一律抛
    ``ExpertGroupError``，由调用方转成 400——不静默降级。
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ExpertGroupError("expert_group must be an object")

    mode = payload.get("mode")
    if mode not in MODES:
        raise ExpertGroupError(
            f"expert_group.mode must be one of {', '.join(sorted(MODES))}"
        )

    raw_members = payload.get("members")
    if (
        isinstance(raw_members, (str, bytes))
        or not isinstance(raw_members, Sequence)
        or not raw_members
    ):
        raise ExpertGroupError("expert_group.members must be a non-empty array")
    if len(raw_members) > MAX_MEMBERS:
        raise ExpertGroupError(f"expert_group.members exceeds the limit of {MAX_MEMBERS}")

    members = tuple(_parse_member(raw, i) for i, raw in enumerate(raw_members))
    seen: set[str] = set()
    for member in members:
        if member.id in seen:
            raise ExpertGroupError(f"duplicate member id '{member.id}'")
        seen.add(member.id)

    raw_lead = payload.get("lead")
    lead: str | None = None
    if raw_lead is not None:
        lead = _require_str(raw_lead, "expert_group.lead")
        if lead not in seen:
            raise ExpertGroupError(f"expert_group.lead '{lead}' is not a member id")

    aggregate = payload.get("aggregate", "lead")
    if aggregate not in AGGREGATES:
        raise ExpertGroupError(
            f"expert_group.aggregate must be one of {', '.join(sorted(AGGREGATES))}"
        )

    raw_concurrent = payload.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
    if isinstance(raw_concurrent, bool) or not isinstance(raw_concurrent, int):
        raise ExpertGroupError("expert_group.max_concurrent must be an integer")
    if not 1 <= raw_concurrent <= MAX_CONCURRENT_LIMIT:
        raise ExpertGroupError(
            f"expert_group.max_concurrent must be between 1 and {MAX_CONCURRENT_LIMIT}"
        )

    return ExpertGroupSpec(
        mode=mode,
        members=members,
        lead=lead,
        aggregate=aggregate,
        max_concurrent=raw_concurrent,
    )


@dataclass
class MemberOutcome:
    member: ExpertMember
    ok: bool
    output: str
    error: str | None = None


@dataclass
class ExpertGroupResult:
    final_text: str
    outcomes: tuple[MemberOutcome, ...]
    plan: str | None = None

    @property
    def failed(self) -> tuple[MemberOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.ok)


#: 跑一个成员：``(成员, 该成员的输入) -> 该成员的产出文本``。
RunMember = Callable[[ExpertMember, str], Awaitable[str]]
#: 成员生命周期事件：``(kind, 成员, 附加字段) -> None``。供 UI 做归属展示。
MemberEventListener = Callable[[str, ExpertMember, Mapping[str, Any]], Awaitable[None]]

_PLAN_PROMPT = (
    "You are the coordinator of an expert group. Before any member starts, produce a short,\n"
    "concrete execution plan for the request below: which part each specialist must cover and\n"
    "what the final answer must contain. Output the plan only, not the answer.\n\n"
    "--- REQUEST ---\n{request}\n"
)
_MEMBER_PROMPT = (
    "You are '{name}', one specialist in a team answering a single user request. Contribute\n"
    "only your own part, from your own expertise. Be concrete and self-contained. Do not\n"
    "address the user directly and do not summarise the other members.\n\n"
    "--- REQUEST ---\n{request}\n"
)
_MEMBER_PLAN_SECTION = "\n--- TEAM PLAN ---\n{plan}\n"
_MEMBER_PRIOR_SECTION = "\n--- PRIOR MEMBERS' FINDINGS ---\n{prior}\n"
_AGGREGATE_PROMPT = (
    "You are the lead of an expert group. Below are your members' findings on the same\n"
    "request. Synthesise them into ONE final answer for the user: resolve conflicts, remove\n"
    "duplication and keep the substance.\n\n"
    "--- REQUEST ---\n{request}\n\n--- MEMBER FINDINGS ---\n{findings}\n"
)


class ExpertGroupOrchestrator:
    """按 mode 调度成员，并按 ``aggregate`` 收口。"""

    def __init__(
        self,
        run_member: RunMember,
        *,
        on_event: MemberEventListener | None = None,
    ) -> None:
        self._run_member = run_member
        self._on_event = on_event

    async def run(self, spec: ExpertGroupSpec, *, user_request: str) -> ExpertGroupResult:
        if spec.mode == "parallel":
            outcomes, plan = await self._run_parallel(spec, user_request)
        elif spec.mode == "coordinator":
            outcomes, plan = await self._run_coordinator(spec, user_request)
        else:
            outcomes, plan = await self._run_sequential(spec, user_request)

        if not any(o.ok for o in outcomes):
            raise ExpertGroupError("every member of the expert group failed")

        final_text = await self._aggregate(spec, user_request, outcomes)
        return ExpertGroupResult(final_text=final_text, outcomes=outcomes, plan=plan)

    # -- modes ---------------------------------------------------------------

    async def _run_sequential(
        self, spec: ExpertGroupSpec, user_request: str
    ) -> tuple[tuple[MemberOutcome, ...], str | None]:
        outcomes: list[MemberOutcome] = []
        for member in spec.members:
            prompt = self._compose_member_prompt(user_request, member, prior=outcomes)
            outcomes.append(await self._dispatch(member, prompt))
        return tuple(outcomes), None

    async def _run_parallel(
        self, spec: ExpertGroupSpec, user_request: str
    ) -> tuple[tuple[MemberOutcome, ...], str | None]:
        # 并发成员彼此看不到对方的输出——这是并行的定义，不是缺陷。上下文只给原始请求。
        semaphore = asyncio.Semaphore(spec.max_concurrent)

        async def _one(member: ExpertMember) -> MemberOutcome:
            async with semaphore:
                prompt = self._compose_member_prompt(user_request, member, prior=())
                return await self._dispatch(member, prompt)

        outcomes = await asyncio.gather(*(_one(m) for m in spec.members))
        return tuple(outcomes), None

    async def _run_coordinator(
        self, spec: ExpertGroupSpec, user_request: str
    ) -> tuple[tuple[MemberOutcome, ...], str | None]:
        lead = spec.lead_member
        plan = (await self._run_member(lead, _PLAN_PROMPT.format(request=user_request))).strip()
        await self._emit("group_planned", lead, {"plan": plan})

        # lead 已经出了计划，不再让它当执行者——除非它就是唯一的成员。
        executors = tuple(m for m in spec.members if m.id != lead.id) or (lead,)
        outcomes: list[MemberOutcome] = []
        for member in executors:
            prompt = self._compose_member_prompt(
                user_request, member, prior=outcomes, plan=plan
            )
            outcomes.append(await self._dispatch(member, prompt))
        return tuple(outcomes), plan

    # -- aggregation ---------------------------------------------------------

    async def _aggregate(
        self,
        spec: ExpertGroupSpec,
        user_request: str,
        outcomes: tuple[MemberOutcome, ...],
    ) -> str:
        if spec.aggregate == "concat":
            return _concat_outcomes(outcomes)

        lead = spec.lead_member
        findings = _render_findings(outcomes)
        prompt = _AGGREGATE_PROMPT.format(request=user_request, findings=findings)
        await self._emit("group_aggregating", lead, {})
        text = (await self._run_member(lead, prompt)).strip()
        if not text:
            # lead 收口返回空 = 这一轮没有任何产出落到用户手里，退化为机械拼接
            # 好过返回空串（空 assistant 消息会被落库阶段直接丢弃，用户什么都看不到）。
            logger.warning("expert group lead produced an empty aggregate; falling back to concat")
            return _concat_outcomes(outcomes)
        return text

    # -- helpers -------------------------------------------------------------

    async def _dispatch(self, member: ExpertMember, prompt: str) -> MemberOutcome:
        await self._emit("member_started", member, {})
        try:
            output = await self._run_member(member, prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 单个成员失败不该炸掉整团
            error = str(exc) or exc.__class__.__name__
            # ⚠️ 本模块用的是**标准 logging**（`import logging`），占位符必须是 `%s`。
            # 项目里两套 logger 并存：`loop.py` 用 loguru（占位符 `{}`），本文件用标准
            # logging。写反不会报错——`%s` 进 loguru 只是占位符不替换，而 `{}` 进标准
            # logging 会在 emit 时炸 `TypeError: not all arguments converted`。
            logger.warning("expert group member '%s' failed: %s", member.id, error)
            await self._emit("member_failed", member, {"error": error})
            return MemberOutcome(member=member, ok=False, output="", error=error)
        text = (output or "").strip()
        await self._emit("member_finished", member, {"output": text})
        return MemberOutcome(member=member, ok=True, output=text)

    def _compose_member_prompt(
        self,
        user_request: str,
        member: ExpertMember,
        *,
        prior: Sequence[MemberOutcome],
        plan: str | None = None,
    ) -> str:
        prompt = _MEMBER_PROMPT.format(name=member.name, request=user_request)
        if plan:
            prompt += _MEMBER_PLAN_SECTION.format(plan=plan)
        successful = [o for o in prior if o.ok and o.output]
        if successful:
            prompt += _MEMBER_PRIOR_SECTION.format(prior=_render_findings(tuple(successful)))
        return prompt

    async def _emit(
        self, kind: str, member: ExpertMember, extra: Mapping[str, Any]
    ) -> None:
        if self._on_event is None:
            return
        try:
            await self._on_event(kind, member, extra)
        except Exception:  # noqa: BLE001 - 进度上报不得反向影响调度
            logger.debug("expert group event listener failed for %s", kind, exc_info=True)


def _render_findings(outcomes: Sequence[MemberOutcome]) -> str:
    blocks: list[str] = []
    for outcome in outcomes:
        if outcome.ok and outcome.output:
            blocks.append(f"[{outcome.member.name}]\n{outcome.output}")
        elif not outcome.ok:
            blocks.append(f"[{outcome.member.name}] FAILED: {outcome.error or 'unknown error'}")
    return "\n\n".join(blocks) if blocks else "(no member produced output)"


def _concat_outcomes(outcomes: Sequence[MemberOutcome]) -> str:
    blocks = [
        f"## {o.member.name}\n{o.output}" for o in outcomes if o.ok and o.output
    ]
    failed = [o.member.name for o in outcomes if not o.ok]
    if failed:
        blocks.append("## Failed members\n" + ", ".join(failed))
    return "\n\n".join(blocks)
