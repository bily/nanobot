"""Execution helpers for session-bound cron jobs."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from nanobot.agent.tools.cron import CronTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cron.session_delivery import origin_delivery_context
from nanobot.cron.session_turns import CRON_DEFER_UNTIL_IDLE_META, CRON_TRIGGER_META
from nanobot.cron.types import CronJob
from nanobot.cron.webui_metadata import cron_proactive_delivery_metadata
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry


class BoundCronAgent(Protocol):
    tools: ToolRegistry

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        ...


class CronRunRecorder(Protocol):
    def write_run_record(self, run_id: str, record: dict[str, Any]) -> None:
        ...


#: Outbound sink handed to us as a bare callable (``MessageBus.publish_outbound``),
#: matching how ``CronService`` takes ``on_job``.  Recipients of that message go
#: through ``ChannelManager._dispatch_outbound``, which retries with backoff.
PublishOutbound = Callable[[OutboundMessage], Awaitable[None]]

# [LOCAL PATCH] FR-5.6 result push.  Opt-in per job via ``payload.delivery``
# (see ``nanobot.cron.types.CronDelivery``); jobs without it behave exactly as
# before.  Three modes, deliberately nested rather than disjoint:
#   complete — push every finished run (success **and** failure)
#   error    — push only failures (the narrow filter)
#   start    — also push a one-line notice when the run *begins*, i.e. it
#              contains ``complete``.  "Only tell me it started" is not a
#              useful mode: the user would never learn the outcome.
# "milestone" (push at each key step) is **not** implemented — an unknown mode
# still falls back to "complete" rather than silently dropping the user's push.
_DELIVERY_MODES = ("complete", "error", "start")
_DELIVERY_FAILURE_PREFIX = "执行失败："
_DELIVERY_STARTED_TEXT = "已开始执行。"


def _cron_prompt_ref(prompt: str) -> dict[str, Any]:
    return {
        "id": "cron.agent_turn.reminder",
        "version": 1,
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def _bound_session_delivery_context(
    job: CronJob,
    *,
    turn_seed: str,
    source_label: str | None,
) -> tuple[str, str, dict[str, Any]]:
    channel, chat_id, metadata = origin_delivery_context(job)

    if channel == "websocket":
        metadata["webui"] = True
        metadata.update(
            cron_proactive_delivery_metadata(
                "websocket",
                metadata,
                turn_seed=turn_seed,
                source_label=source_label,
            )
        )

    return channel, chat_id, metadata


async def _deliver_started(
    job: CronJob,
    *,
    publish_outbound: PublishOutbound | None,
) -> None:
    """Push a one-line "started" notice before the turn runs (``on: start``).

    Only ``start`` jobs get this.  ``complete`` / ``error`` users asked for
    *results*; adding a mode must not start waking them up with extra messages.

    Best-effort like :func:`_deliver_result`, and for a stronger reason: a run
    that is about to execute must never be aborted because a courtesy notice
    could not be sent.

    Timing caveat worth knowing: this fires where the run is handed to the
    agent, which under the defer-until-idle path means "queued and dispatched",
    not "first token generated".  The engine has no callback for the latter, and
    pretending otherwise would make the notice lie for busy sessions.
    """
    delivery = job.payload.delivery
    if delivery is None or publish_outbound is None or delivery.on != "start":
        return
    content = (
        f"{job.name}\n\n{_DELIVERY_STARTED_TEXT}" if job.name else _DELIVERY_STARTED_TEXT
    )
    for target in delivery.targets:
        try:
            await publish_outbound(
                OutboundMessage(
                    channel=target.channel,
                    chat_id=target.chat_id,
                    content=content,
                )
            )
            logger.info(
                "Cron: notified start of '{}' to {}:{}",
                job.name,
                target.channel,
                target.chat_id,
            )
        except Exception as exc:  # best-effort — see docstring
            logger.warning(
                "Cron: failed to notify start of '{}' to {}:{}: {}",
                job.name,
                target.channel,
                target.chat_id,
                exc,
            )


async def _deliver_result(
    job: CronJob,
    *,
    publish_outbound: PublishOutbound | None,
    status: str,
    body: str,
) -> None:
    """Push one finished run's result to the job's configured targets (FR-5.6).

    Best-effort by construction: a run that already finished must never be
    re-marked as failed — nor a successful one made to look cancelled — just
    because a push target was unreachable.  Every failure path here only logs.

    Delivery deliberately does **not** append to the target session's history.
    This is a *push*, not a turn in that conversation: persisting it would put
    an assistant message the user never saw into that channel's session, and the
    next real turn there would then carry it as context.
    """
    delivery = job.payload.delivery
    if delivery is None or publish_outbound is None:
        return

    mode = delivery.on if delivery.on in _DELIVERY_MODES else "complete"
    if delivery.on not in _DELIVERY_MODES:
        logger.warning(
            "Cron: job '{}' has unknown delivery.on '{}'; delivering on completion",
            job.name,
            delivery.on,
        )

    # "error" is the narrow filter; "complete" **and** "start" both mean
    # "whenever it finishes" and therefore cover failures too (a silently-failing
    # morning report is the one users most need pushed at them).  "start" only
    # adds the notice emitted by :func:`_deliver_started`; it never removes the
    # result push.
    if mode == "error" and status != "error":
        return

    text = body.strip()
    if not text:
        logger.info("Cron: job '{}' produced no text to deliver", job.name)
        return

    content = f"{job.name}\n\n{text}" if job.name else text
    for target in delivery.targets:
        try:
            await publish_outbound(
                OutboundMessage(
                    channel=target.channel,
                    chat_id=target.chat_id,
                    content=content,
                )
            )
            logger.info(
                "Cron: delivered result of '{}' to {}:{}",
                job.name,
                target.channel,
                target.chat_id,
            )
        except Exception as exc:  # delivery is best-effort — see docstring
            logger.warning(
                "Cron: failed to deliver result of '{}' to {}:{}: {}",
                job.name,
                target.channel,
                target.chat_id,
                exc,
            )


async def run_bound_cron_job(
    job: CronJob,
    *,
    agent: BoundCronAgent,
    cron: CronRunRecorder,
    publish_outbound: PublishOutbound | None = None,
) -> str | None:
    """Execute a session-bound cron job as a normal agent session turn.

    ``publish_outbound`` is optional and only used for FR-5.6 result push:
    without it a job's ``payload.delivery`` is simply not honoured (tests that
    only care about the run record don't have to provide a bus).
    """
    session_key = job.payload.session_key
    if not session_key:
        raise ValueError(f"cron job {job.id} is missing payload.session_key")

    prompt = render_template(
        "agent/cron_reminder.md",
        strip=True,
        message=job.payload.message,
    )
    prompt_ref = _cron_prompt_ref(prompt)
    run_id = f"{job.id}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
    channel, chat_id, metadata = _bound_session_delivery_context(
        job,
        turn_seed=f"cron:{job.id}",
        source_label=job.name,
    )
    metadata[CRON_TRIGGER_META] = {
        "job_id": job.id,
        "job_name": job.name,
        "run_id": run_id,
        "prompt_ref": prompt_ref,
        "persist_content": (
            f"Scheduled cron job triggered: {job.name}\n\n{job.payload.message}"
        ),
    }
    metadata[CRON_DEFER_UNTIL_IDLE_META] = True
    run_record_base: dict[str, Any] = {
        "job_id": job.id,
        "job_name": job.name,
        "session_key": session_key,
        "prompt_ref": prompt_ref,
        "prompt_vars": {"message": job.payload.message},
        "rendered_prompt": prompt,
    }

    cron.write_run_record(
        run_id,
        {
            **run_record_base,
            "status": "queued",
        },
    )

    # ``on: start`` 的开跑通知：放在 queued 记录之后、真跑之前。
    # 排在这里有原因——它是「这次运行已经开始排队/派发」这个事实的出口，
    # 早于任何可能失败的执行；jobs 没配这个档位时它是空操作。
    await _deliver_started(job, publish_outbound=publish_outbound)

    cron_tool = agent.tools.get("cron")
    cron_token = None
    if isinstance(cron_tool, CronTool):
        cron_token = cron_tool.set_cron_context(True)
    try:
        resp = await agent.submit_cron_turn(
            InboundMessage(
                channel=channel,
                sender_id="cron",
                chat_id=chat_id,
                content=prompt,
                metadata=metadata,
                session_key_override=session_key,
            )
        )
    except (Exception, asyncio.CancelledError) as exc:
        error_text = str(exc) or exc.__class__.__name__
        cron.write_run_record(
            run_id,
            {
                **run_record_base,
                "status": "error",
                "error": error_text,
            },
        )
        await _deliver_result(
            job,
            publish_outbound=publish_outbound,
            status="error",
            body=f"{_DELIVERY_FAILURE_PREFIX}{error_text}",
        )
        raise
    finally:
        if isinstance(cron_tool, CronTool) and cron_token is not None:
            cron_tool.reset_cron_context(cron_token)

    response = resp.content if resp else ""
    cron.write_run_record(
        run_id,
        {
            **run_record_base,
            "status": "ok",
            "response": response,
        },
    )
    await _deliver_result(
        job,
        publish_outbound=publish_outbound,
        status="ok",
        body=response,
    )
    return response
