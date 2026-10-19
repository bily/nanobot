"""Tests for SubagentManager."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.filesystem import FileToolsConfig
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.llm_usage.context import llm_usage_source
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: LLMProvider) -> LLMRuntime:
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test", context_window_tokens=128_000)


@pytest.mark.asyncio
async def test_subagent_uses_tool_loader():
    """Verify subagent registers tools via ToolLoader, not hard-coded imports."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=Path("/tmp"),
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    tools = sm._build_tools()
    assert tools.has("read_file")
    assert tools.has("write_file")
    assert not tools.has("message")
    assert not tools.has("spawn")


@pytest.mark.asyncio
async def test_subagent_build_tools_isolates_file_read_state(tmp_path):
    """Each spawned subagent needs a fresh file-state cache."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    first_read = sm._build_tools().get("read_file")
    second_read = sm._build_tools().get("read_file")

    assert first_read is not second_read
    assert (await first_read.execute(path="note.txt")).startswith("1| hello")
    second_result = await second_read.execute(path="note.txt")
    assert second_result.startswith("1| hello")
    assert "File unchanged" not in second_result


def test_subagent_respects_file_tool_toggle(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        tools_config=ToolsConfig(file=FileToolsConfig(enable=False)),
    )

    tools = sm._build_tools()

    file_tools = {
        "apply_patch",
        "edit_file",
        "find_files",
        "grep",
        "list_dir",
        "read_file",
        "write_file",
    }
    assert file_tools.isdisjoint(tools.tool_names)


def test_subagent_prompt_keeps_agent_paths_for_selected_project(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    global_skill = agent_workspace / "skills" / "global-custom" / "SKILL.md"
    project_skill = project / "skills" / "project-custom" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    global_skill.write_text("---\ndescription: global skill\n---\nGlobal", encoding="utf-8")
    project_skill.write_text("---\ndescription: project skill\n---\nProject", encoding="utf-8")
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt(workspace=project)

    assert "one root and relative SKILL.md paths" in prompt
    assert "Join them when using `read_file`" in prompt
    assert str(project.resolve()) not in prompt
    assert f"Nanobot's agent workspace: {agent_workspace.resolve()}" in prompt
    assert f"History log: {agent_workspace.resolve() / 'memory' / 'history.jsonl'}" in prompt
    assert "global-custom" in prompt
    assert "project-custom" not in prompt


def test_subagent_prompt_uses_relative_paths_in_agent_workspace(tmp_path):
    skill = tmp_path / "skills" / "custom" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\ndescription: custom skill\n---\nCustom", encoding="utf-8")
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt()

    assert str(tmp_path.resolve()) not in prompt
    assert "History log: memory/history.jsonl" in prompt
    assert "### Workspace skills (`skills`)" in prompt


@pytest.mark.asyncio
async def test_subagent_keeps_project_runtime_scope_with_agent_owned_tools(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    agent_workspace.mkdir()
    project.mkdir()
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    manager._announce_result = AsyncMock()
    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
    )

    await manager._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "direct"},
        status,
        _runtime(provider),
        workspace_scope=build_workspace_scope(project, "restricted"),
    )

    spec = manager.runner.run.call_args.args[0]
    assert spec.workspace == project
    assert spec.tools.get("read_file")._workspace == agent_workspace.resolve()


@pytest.mark.asyncio
async def test_subagent_recovers_from_tool_error_in_same_run(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "missing.txt"},
                )
            ],
        ),
        LLMResponse(content="recovered without restarting", tool_calls=[]),
    ])
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    result = await sm.run_inline(
        task="recover after a missing file",
        session_key="test:direct",
        runtime=_runtime(provider),
    )

    assert result == "recovered without restarting"
    assert provider.chat_with_retry.await_count == 2


@pytest.mark.asyncio
async def test_spawned_subagent_inherits_llm_usage_source(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    with llm_usage_source("cron"):
        await sm.spawn(
            "automation task",
            session_key="websocket:bound-automation",
            runtime=_runtime(provider),
        )
    tasks = list(sm._running_tasks.values())
    await asyncio.gather(*tasks)

    spec = sm.runner.run.call_args.args[0]
    assert spec.llm_usage_source == "cron"


def _messages_of_call(provider: MagicMock, index: int) -> list[dict[str, Any]]:
    """Return the ``messages`` kwarg of the Nth `chat_with_retry` call."""
    return provider.chat_with_retry.await_args_list[index].kwargs["messages"]


@pytest.mark.asyncio
async def test_subagent_inherits_agent_tool_deny(tmp_path):
    """A subagent must obey the parent scope's tool blacklist.

    Without this, ``spawn`` is a **bypass around the agent tool whitelist**: a
    parent denied ``write_file`` can simply ask a child to do it, because
    ``_build_tools`` loads everything and only the parent turn carries the
    authorization gate.
    """
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="delegating the write",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "out.txt", "content": "pwned"},
                )
            ],
        ),
        LLMResponse(content="could not write", tool_calls=[]),
    ])
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    scope = build_workspace_scope(
        tmp_path, "restricted", tool_deny=["write_file"], agent_id="expert-a"
    )

    await sm.run_inline(
        task="write a file for me",
        session_key="test:direct",
        runtime=_runtime(provider),
        workspace_scope=scope,
    )

    # The refusal has to reach the child as a tool result — a denied call that
    # silently vanished would just make the subagent loop on it.
    tool_result = _messages_of_call(provider, 1)[-1]["content"]
    assert "denies the 'write_file' tool" in tool_result
    assert "expert-a" in tool_result
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_subagent_without_agent_tool_policy_can_still_write(tmp_path):
    """No declared policy must stay permissive (regression guard).

    The gate that fixes the bypass must not turn into "subagents may never use
    anything": an absent whitelist means *unrestricted*, not *deny-all*.
    """
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="writing",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "out.txt", "content": "hello"},
                )
            ],
        ),
        LLMResponse(content="wrote it", tool_calls=[]),
    ])
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    await sm.run_inline(
        task="write a file for me",
        session_key="test:direct",
        runtime=_runtime(provider),
        workspace_scope=build_workspace_scope(tmp_path, "restricted"),
    )

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_subagent_inherits_agent_tool_allow(tmp_path):
    """A declared whitelist is minimal-privilege: anything unlisted is denied."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="writing",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "out.txt", "content": "nope"},
                )
            ],
        ),
        LLMResponse(content="denied", tool_calls=[]),
    ])
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    scope = build_workspace_scope(tmp_path, "restricted", tool_allow=["read_file"])

    await sm.run_inline(
        task="write a file for me",
        session_key="test:direct",
        runtime=_runtime(provider),
        workspace_scope=scope,
    )

    tool_result = _messages_of_call(provider, 1)[-1]["content"]
    assert "does not include 'write_file'" in tool_result
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_persona_is_appended_not_replacing_the_template(tmp_path):
    """`persona` adds who the subagent is; it must not replace the template.

    The rendered template carries the **workspace boundary and skills summary**
    — a safety boundary, not a style choice. Letting a caller overwrite it would
    make "dispatch to this expert" silently drop the subagent's sandbox.
    """
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    plain = sm._build_subagent_prompt(workspace=tmp_path)
    with_persona = sm._build_subagent_prompt(
        workspace=tmp_path, persona="你是法务专家，只回答合规问题。"
    )

    assert with_persona.startswith(plain)
    assert "你是法务专家" in with_persona
    # 模板里的安全段还在 —— 被覆盖的话这两句会消失
    assert "You are a subagent" in with_persona
    assert "untrusted external data" in with_persona


@pytest.mark.asyncio
async def test_blank_persona_leaves_prompt_untouched(tmp_path):
    """A blank persona must not add stray separators to the prompt."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    plain = sm._build_subagent_prompt(workspace=tmp_path)
    assert sm._build_subagent_prompt(workspace=tmp_path, persona=None) == plain
    assert sm._build_subagent_prompt(workspace=tmp_path, persona="   \n ") == plain


@pytest.mark.asyncio
async def test_run_inline_forwards_persona_into_messages(tmp_path):
    """The dispatched persona has to reach the child's system message."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()

    await sm.run_inline(
        task="review this contract",
        session_key="test:direct",
        runtime=_runtime(provider),
        persona="你是法务专家。",
    )

    spec = sm.runner.run.call_args.args[0]
    assert spec.initial_messages[0]["role"] == "system"
    assert "你是法务专家。" in spec.initial_messages[0]["content"]
