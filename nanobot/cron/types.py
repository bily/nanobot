"""Cron types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast, overload

from nanobot.utils.dict_keys import get_camel_snake


@overload
def _store_int(value: Any, default: Literal[None]) -> int | None: ...


@overload
def _store_int(value: Any, default: int = 0) -> int: ...


def _store_int(value: Any, default: int | None = 0) -> int | None:
    """Coerce JSON numerics to int; treat null/blank like a missing key."""
    if value is None or value == "":
        return default
    return int(value)


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""
    kind: Literal["at", "every", "cron"]
    # For "at": timestamp in ms
    at_ms: int | None = None
    # For "every": interval in ms
    every_ms: int | None = None
    # For "cron": cron expression (e.g. "0 9 * * *")
    expr: str | None = None
    # Timezone for cron expressions
    tz: str | None = None

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronSchedule:
        return cls(
            kind=data["kind"],
            at_ms=_store_int(get_camel_snake(data, "atMs", "at_ms"), None),
            every_ms=_store_int(get_camel_snake(data, "everyMs", "every_ms"), None),
            expr=data.get("expr"),
            tz=data.get("tz"),
        )


@dataclass
class CronDeliveryTarget:
    """One external destination for a bound cron run's result (FR-5.6)."""

    channel: str  # engine channel name, e.g. "wecom"
    chat_id: str  # target conversation id on that channel

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronDeliveryTarget | None:
        channel = str(data.get("channel") or "").strip()
        chat_id = str(get_camel_snake(data, "chatId", "chat_id") or "").strip()
        # A half-filled target can never be delivered to; dropping it here keeps
        # "configured" and "actually pushable" the same thing.
        if not channel or not chat_id:
            return None
        return cls(channel=channel, chat_id=chat_id)

    def to_store_dict(self) -> dict[str, Any]:
        """Back to the camelCase wire shape (see ``CronService._save_store``)."""
        return {"channel": self.channel, "chatId": self.chat_id}


@dataclass
class CronDelivery:
    """[LOCAL PATCH] FR-5.6: where else a bound job's result should be pushed.

    Deliberately **separate** from ``deliver`` / ``channel`` / ``to`` on
    :class:`CronPayload`.  Those three are the *pre*-session-bound routing
    target: the loader migrates them into ``origin_*`` and clears them, and
    :func:`is_bound_cron_job` treats their presence as "this job is not bound",
    which disables the job.  This field is purely additive — the job keeps
    executing in its bound session and a copy of the result is pushed to
    ``targets`` after the turn finishes.
    """

    targets: list[CronDeliveryTarget] = field(default_factory=list)
    # Validated where it is acted on (``bound_runner``) so the warning for an
    # unknown value lives next to the code that has to cope with it.
    # "complete" | "error" | "start" (see ``bound_runner._DELIVERY_MODES``).
    # "milestone" — push at each key step — is intentionally **not** implemented:
    # a bound job is a single agent turn, and inside it there is no step that is
    # both user-meaningful and observable without turning the chat into a log.
    on: str = "complete"

    @classmethod
    def from_store_dict(cls, data: Any) -> CronDelivery | None:
        if not isinstance(data, dict):
            return None
        raw_targets = data.get("targets")
        targets: list[CronDeliveryTarget] = []
        if isinstance(raw_targets, list):
            for item in raw_targets:
                if not isinstance(item, dict):
                    continue
                target = CronDeliveryTarget.from_store_dict(item)
                if target is not None:
                    targets.append(target)
        # No usable target ⇒ no delivery.  A config that would push nowhere must
        # not read back as "configured", or callers would report success.
        if not targets:
            return None
        return cls(targets=targets, on=str(data.get("on") or "complete"))

    def to_store_dict(self) -> dict[str, Any]:
        """Back to the camelCase wire shape (see ``CronService._save_store``)."""
        return {
            "targets": [target.to_store_dict() for target in self.targets],
            "on": self.on,
        }


@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Legacy delivery fields used by pre-session-bound cron jobs.
    deliver: bool = False
    channel: str | None = None  # e.g. "whatsapp"
    to: str | None = None  # e.g. phone number
    channel_meta: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None  # original session key for correct session recording
    origin_channel: str | None = None
    origin_chat_id: str | None = None
    origin_metadata: dict[str, Any] = field(default_factory=dict)
    # [LOCAL PATCH] FR-5.6 result push; see CronDelivery.
    delivery: CronDelivery | None = None

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronPayload:
        return cls(
            kind=data.get("kind", "agent_turn"),
            message=data.get("message", ""),
            deliver=data.get("deliver", False),
            channel=data.get("channel"),
            to=data.get("to"),
            channel_meta=dict(
                get_camel_snake(data, "channelMeta", "channel_meta", {}) or {}
            ),
            session_key=get_camel_snake(data, "sessionKey", "session_key"),
            origin_channel=get_camel_snake(data, "originChannel", "origin_channel"),
            origin_chat_id=get_camel_snake(data, "originChatId", "origin_chat_id"),
            origin_metadata=dict(
                get_camel_snake(data, "originMetadata", "origin_metadata", {}) or {}
            ),
            delivery=CronDelivery.from_store_dict(data.get("delivery")),
        )


@dataclass
class CronRunRecord:
    """A single execution record for a cron job."""
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronRunRecord:
        return cls(
            run_at_ms=_store_int(get_camel_snake(data, "runAtMs", "run_at_ms", 0)),
            status=data["status"],
            duration_ms=_store_int(get_camel_snake(data, "durationMs", "duration_ms", 0)),
            error=data.get("error"),
        )


@dataclass
class CronJobState:
    """Runtime state of a job."""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronJobState:
        history = cast(
            list[object],
            get_camel_snake(data, "runHistory", "run_history", []) or [],
        )
        return cls(
            next_run_at_ms=_store_int(
                get_camel_snake(data, "nextRunAtMs", "next_run_at_ms"), None
            ),
            last_run_at_ms=_store_int(
                get_camel_snake(data, "lastRunAtMs", "last_run_at_ms"), None
            ),
            last_status=get_camel_snake(data, "lastStatus", "last_status"),
            last_error=get_camel_snake(data, "lastError", "last_error"),
            run_history=[
                record
                if isinstance(record, CronRunRecord)
                else CronRunRecord.from_store_dict(cast(dict[str, Any], record))
                for record in history
                if isinstance(record, (dict, CronRunRecord))
            ],
        )


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> CronJob:
        state_kwargs = dict(cast(dict[str, Any], kwargs.get("state", {})))
        state_kwargs["run_history"] = [
            record
            if isinstance(record, CronRunRecord)
            else CronRunRecord(**cast(dict[str, Any], record))
            for record in cast(list[object], state_kwargs.get("run_history", []))
        ]
        kwargs["schedule"] = CronSchedule(
            **cast(dict[str, Any], kwargs.get("schedule", {"kind": "every"}))
        )
        payload_kwargs = dict(cast(dict[str, Any], kwargs.get("payload", {})))
        # `delivery` 走与磁盘路径**同一个**解析器：直接塞进来的裸 dict 会让投递代码
        # 拿到一个没有 `.targets` 的对象，而报错发生在一次已经跑完的任务之后 —— 最难查。
        raw_delivery = payload_kwargs.get("delivery")
        if raw_delivery is not None and not isinstance(raw_delivery, CronDelivery):
            payload_kwargs["delivery"] = CronDelivery.from_store_dict(raw_delivery)
        kwargs["payload"] = CronPayload(**payload_kwargs)
        kwargs["state"] = CronJobState(**state_kwargs)
        return cls(**cast(Any, kwargs))

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronJob:
        """Load a job from jobs.json (camelCase with snake_case fallbacks)."""
        return cls(
            id=data["id"],
            name=data["name"],
            enabled=data.get("enabled", True),
            schedule=CronSchedule.from_store_dict(data["schedule"]),
            payload=CronPayload.from_store_dict(data.get("payload") or {}),
            state=CronJobState.from_store_dict(data.get("state") or {}),
            created_at_ms=_store_int(get_camel_snake(data, "createdAtMs", "created_at_ms", 0)),
            updated_at_ms=_store_int(get_camel_snake(data, "updatedAtMs", "updated_at_ms", 0)),
            delete_after_run=bool(
                get_camel_snake(data, "deleteAfterRun", "delete_after_run", False)
            ),
        )


@dataclass
class CronStore:
    """Persistent store for cron jobs."""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
