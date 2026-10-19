"""Gateway ``/v1/turn``：在**已有的 IM 会话内**跑一轮智能体。

## 为什么要有这个端点

桌面端「助理」视图要在 IM 会话里发言，但不能复用 ``serve`` 的 ``/v1/chat``：
那条路径执行 ``session_key = f"api:{session_id}"``（见 ``api/server.py``），
客户端传什么都会被加 ``api:`` 前缀，落到一个与该 IM 线程**毫无关系**的新会话 ——
IM 那边的智能体看不到这段历史，本地乐观上屏的消息也会在下一次轮询重建会话时被冲掉。

``/v1/turn`` 因此不走 HTTP 聊天接口，而是把文本作为一条**渠道 inbound 消息**
投进 bus，由 ``AgentLoop`` 按正常渠道消息处理。这组用例钉住的就是「它必须落在
那个本来就属于该 IM 线程的会话上」这条不变量 —— 一旦 session key 算错，
症状是「看起来发出去了、IM 那边从没收到过」，且全程零报错。
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session.keys import session_key_for_channel

CHANNEL = "feishu.feishu"
CHAT_ID = "oc_v3_abc123"


def _loop(tmp_path, responses: list[str]) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.chat_with_retry = AsyncMock(
        side_effect=[LLMResponse(content=response, usage=None) for response in responses]
    )
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        cron_service=MagicMock(),
    )


def _desktop_turn(content: str, *, require_existing: bool = True) -> InboundMessage:
    """复刻 ``_handle_turn`` 构造的那条 inbound 消息。"""
    return InboundMessage(
        channel=CHANNEL,
        sender_id="desktop",
        chat_id=CHAT_ID,
        content=content,
        require_existing_session=require_existing,
    )


@pytest.mark.asyncio
async def test_desktop_turn_joins_the_same_session_as_prior_channel_turns(tmp_path) -> None:
    """⭐ 核心不变量：这一轮必须落进 IM 线程**原本**那个会话。

    先用渠道身份起一轮（模拟 IM 里用户说话），再用桌面身份起一轮 —— 断言
    ``chat_with_retry`` 第二次调用时看得见第一次的历史。这正是「助理视图与任务
    视图同款输入框」能成立的前提：两条入口共用一条上下文。
    """
    loop = _loop(tmp_path, ["IM 里的回答", "桌面端的回答"])
    key = session_key_for_channel(CHANNEL, CHAT_ID)

    await loop._process_message(
        InboundMessage(channel=CHANNEL, sender_id="im-user", chat_id=CHAT_ID, content="第一个问题")
    )
    await loop._process_message(_desktop_turn("第二个问题"))

    second_call = loop.provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    rendered = str(second_call)
    assert "第一个问题" in rendered
    assert "IM 里的回答" in rendered
    # 会话 key 必须就是渠道那一个（不是 api: 前缀的新会话）
    assert loop.sessions.get_cached(key) is not None


@pytest.mark.asyncio
async def test_desktop_turn_is_recorded_in_channel_history(tmp_path) -> None:
    """这一轮要落进会话历史 —— 否则桌面端轮询读不到，回复永远不会上屏。"""
    loop = _loop(tmp_path, ["答复"])
    key = session_key_for_channel(CHANNEL, CHAT_ID)
    loop.sessions.get_or_create(key)

    await loop._process_message(_desktop_turn("从桌面端发起"))

    session = loop.sessions.get_cached(key)
    assert session is not None
    # 会话里的消息是 TypedDict（落盘形态），按 key 取而不是属性访问
    roles = [m["role"] for m in session.messages]
    assert "user" in roles and "assistant" in roles
    texts = [m.get("content", "") for m in session.messages]
    assert any("从桌面端发起" in t for t in texts)


@pytest.mark.asyncio
async def test_require_existing_session_never_creates_a_session(tmp_path) -> None:
    """会话不存在时**绝不凭空建一个**空会话让模型对着空气回答。

    这是 ``/v1/turn`` 里 404 前置检查之外的第二道保险：拼错的 chat_id 不该在磁盘上
    留下一堆垃圾会话，也不该真的调一次模型（那是要花钱的）。

    注：``run()`` 里有一道更早的闸（``require_existing_session`` 且未缓存 → 直接
    ``continue``，连 ``_process_message`` 都不进）；这里直连 ``_process_message``
    绕过了它，于是表现为 ``_restore_turn`` 抛 ``RuntimeError``。两条路径的可观察
    结果一致：**模型没被调用、会话没被创建** —— 那才是这条要钉住的东西。
    """
    loop = _loop(tmp_path, ["不该被调用"])
    missing_key = session_key_for_channel(CHANNEL, "never-existed")

    with pytest.raises(RuntimeError):
        await loop._process_message(_desktop_turn("发给一个不存在的会话"))

    loop.provider.chat_with_retry.assert_not_awaited()
    assert loop.sessions.get_cached(missing_key) is None


@pytest.mark.asyncio
async def test_desktop_turn_counts_as_user_input(tmp_path) -> None:
    """``sender_id='desktop'`` 不能让它被当成系统消息跳过。

    ``InboundMessage.is_user_input`` 的默认判据是 ``channel != 'system'``；
    这里显式钉住，避免日后有人给桌面入口加个 'system' 渠道名导致整条链路静默失效。
    """
    loop = _loop(tmp_path, ["答复"])
    loop.sessions.get_or_create(session_key_for_channel(CHANNEL, CHAT_ID))
    msg = _desktop_turn("普通提问")

    assert msg.is_user_input is True
    await loop._process_message(msg)
    loop.provider.chat_with_retry.assert_awaited()


@pytest.mark.asyncio
async def test_two_sequential_desktop_turns_share_one_context(tmp_path) -> None:
    """连续两轮桌面发言之间上下文连续（第二轮看得见第一轮）。"""
    loop = _loop(tmp_path, ["答复一", "答复二"])
    key = session_key_for_channel(CHANNEL, CHAT_ID)
    loop.sessions.get_or_create(key)

    await loop._process_message(_desktop_turn("第一轮桌面发言"))
    await loop._process_message(_desktop_turn("第二轮桌面发言"))

    second_call = loop.provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    assert "第一轮桌面发言" in str(second_call)
    assert loop.sessions.get_cached(key) is not None


def test_chat_id_with_colon_is_preserved_in_session_key() -> None:
    """含冒号的 chatId 必须完整落在 session key 里。

    若日后有人把 ``f"{channel}:{chat_id}"`` 改成 ``split(':')[1]`` 之类按冒号切，
    这条会红。IM 侧的 chatId 由渠道决定，客户端不该假设它不含冒号。
    """
    key = session_key_for_channel("feishu.feishu", "tenant:abc:def")
    assert key == "feishu.feishu:tenant:abc:def"
    # 还原时只切掉**第一个**冒号（channel 段），其余全归 chatId
    channel, chat_id = key.split(":", 1)
    assert channel == "feishu.feishu"
    assert chat_id == "tenant:abc:def"


@pytest.mark.asyncio
async def test_run_loop_routes_desktop_turn_to_the_channel_session(tmp_path) -> None:
    """⭐ 走**真正的** ``run()`` 消费路径，钉住 session key 的解析结果。

    前面的用例直连 ``_process_message``，绕过了 ``run()`` 里的
    ``_effective_session_key`` —— 也就是说，把 session key 算错的变异
    （比如给它加上 ``api:`` 前缀、落进一个与IM 线程无关的新会话）在那些用例里
    是**测不出来的**。而那恰恰是本端点存在的全部理由：``/v1/chat`` 犯的正是
    这个错（``session_key = f"api:{session_id}"``），症状是「看起来发出去了、
    IM 那边从没收到过」，且全程零报错。

    所以这里用一条「吐一条消息后 ``sleep(3600)``」的假 ``consume_inbound``
    把 ``run()`` 真正驱动一次迭代，断言它落进的那个 key 就是渠道那一个。
    """
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.chat_with_retry = AsyncMock(
        side_effect=[LLMResponse(content="答复", usage=None)]
    )
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        cron_service=MagicMock(),
    )
    channel_key = session_key_for_channel(CHANNEL, CHAT_ID)
    loop.sessions.get_or_create(channel_key)

    # 走**真队列**而不是替换 consume_inbound：`run()` 把消费包在
    # `asyncio.wait_for(..., timeout=1.0)` 里，一个「永不返回」的假 consumer 会被
    # 每次超时取消，消息进了队列却永远送不到 run() 手上（症状：inbound 队列在涨、
    # 助手从没被调用）。预先 put 才是这条链路真实的样子。
    bus.inbound.put_nowait(_desktop_turn("经由 run() 进来"))

    task = asyncio.create_task(loop.run())
    try:
        # `run()` 把每条消息派发成独立 task，回复经 outbound 队列回来 ——
        # 等它比轮询 session 里的消息稳（不必猜 loop 内部写了哪一步）。
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=30)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert outbound.channel == CHANNEL
    assert outbound.chat_id == CHAT_ID
    assert "答复" in outbound.content

    session = loop.sessions.get_cached(channel_key)
    assert session is not None
    texts = [m.get("content", "") for m in session.messages]
    assert any("经由 run() 进来" in t for t in texts)
    # 关键断言：**没有**凭空多出一个 api: 前缀的会话
    assert loop.sessions.get_cached(f"api:{channel_key}") is None
    provider.chat_with_retry.assert_awaited()
