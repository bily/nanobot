"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from loguru import logger

# [LOCAL PATCH] FR-3.12：技能 / 插件声明的工具白名单（纯策略代数，无 IO）。
from nanobot.agent.hooks.plugin_tools import (
    DeclaredToolPolicy,
    ToolDeclaration,
    combine_declarations,
    parse_declared_tools,
)
from nanobot.agent.memory import MemoryStore

# [LOCAL PATCH] FR-3.10：插件提示词片段（ChainLoader 的第 ① 级）。
# ⚠️⚠️ **`nanobot.agent.plugins` 只能函数内局部导入，不能放模块级**：
# `agent/__init__.py` 重导出本模块，而 `plugins.py` 模块级 import
# `nanobot.config.loader` —— 模块级一导就把这条链闭成环，于是
# `config/schema.py` 末尾的 `_resolve_tool_config_refs()` 静默失败，
# **任何** `Config.model_validate` 都会抛
# "`Config` is not fully defined; you should define `WebToolsConfig`"。
# 排查提示写在 schema.py 末尾；本模块正是那条提示里的典型案例。
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools import image_generation as image_generation_tools
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools import sessions as session_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_END,
    RUNTIME_CONTEXT_MESSAGE_META,
    RUNTIME_CONTEXT_TAG,
    RuntimeContextBlock,
    append_runtime_context,
)
from nanobot.security.workspace_access import WorkspaceScopeResolver
from nanobot.session.keys import last_channel_from_metadata
from nanobot.session.manager import Session
from nanobot.session.summary import SessionSummary
from nanobot.utils.helpers import detect_image_mime, load_bundled_template
from nanobot.utils.prompt_templates import PluginPromptMount, render_template

# [LOCAL PATCH] 身份文件块的标签（design §13.3）。SOUL / IDENTITY / USER 三段共用
# 一个标签块；AGENTS.md（项目约定）不在其中。客户端侧的注入用的是**同名字符串**，
# 两端必须保持一致——这个字面量在 `src/lib/constants.ts` 有一份镜像。
IDENTITY_CONTEXT_TAG_OPEN = "<identity_context>"
IDENTITY_CONTEXT_TAG_CLOSE = "</identity_context>"

# [LOCAL PATCH] 长期记忆里对「工作目录」的断言一律作废（中英双语）。
#
# 为什么需要它：长期记忆是**模型自己写的**，而它会把「当时的工作目录」当成项目事实
# 固化下来。实测事故（2026-09-19）：2026-09-16 那批会话真的跑在 `D:\test` 上，
# Dream 于是把 `- Current working directory: D:\test` 写进了
# `<agent-workspace>/memory/MEMORY.md`。此后无论会话绑到哪个工作空间，这一行都会被
# 注入——它读起来是**事实陈述**，而 `# Memory` 又排在 `# Current Project` 之后，
# 于是模型拿它当真相（原话：「根据之前的上下文，当前工作目录是 D:\test」），
# 用户设置里的 `C:\Users\xinmi\NanoWork` 形同虚设，文件写进了 `D:\test`。
#
# 工作目录是**每轮会变的运行时值**：不该、也不能存进长期记忆。这里在注入前就地
# 作废这类断言——比「删掉用户记忆里那一行」更稳，因为用户随时可能手写回去，
# Dream 也可能再次总结出来。真正的真相只有一个来源：下面 `# Current Project`。
_STALE_CWD_LINE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]*)?"
    r"(?:current[ \t]+|当前|目前的?)?"
    r"(?:working[ \t]+directory|工作目录|工作空间目录)"
    r"[ \t]*(?:(?:是|为)[ \t]*|[=:：][ \t]*)"
    r".*$",
    re.IGNORECASE | re.MULTILINE,
)

# 作废后留下的占位行。不整行删除是为了保住 Markdown 列表结构（删干净会让
# `## Project Context` 变成空小节，模型反而更容易自己补一个路径上去）。
_STALE_CWD_REPLACEMENT = (
    "- (working directory omitted on purpose: it is per-session and provided by the runtime)"
)


def neutralize_stale_working_dir(text: str) -> str:
    """清掉记忆文本里对工作目录的断言，让运行时值成为唯一来源。

    Args:
        text: 待注入的长期记忆原文（可能为空）。

    Returns:
        作废了工作目录断言后的文本；无命中时原样返回。
    """
    if not text:
        return text
    return _STALE_CWD_LINE.sub(_STALE_CWD_REPLACEMENT, text)


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted kwargs for turn-attached capabilities."""
    return (
        cli_app_utils.session_extra(metadata)
        | mcp_tools.session_extra(metadata)
        | session_tools.session_extra(metadata)
    )


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    if msg.metadata.get(INBOUND_META_RUNTIME_CONTROL) == RUNTIME_CONTROL_SESSION_DISCARD:
        await state.discard_session(msg.session_key)
        return True
    return await image_generation_tools.handle_runtime_control(state, msg, tools)


@dataclass(frozen=True, slots=True)
class PersistedPromptContextResolver:
    """Restore prompt routing context when no inbound message is available."""

    workspace_scopes: WorkspaceScopeResolver
    unified_session: bool = False

    def __call__(self, session: Session) -> tuple[str | None, Path]:
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        if self.unified_session:
            route = last_channel_from_metadata(session.metadata)
            if route is not None:
                channel = route[0]
        scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=None,
            session_metadata=session.metadata,
        )
        return channel, scope.project_path


@dataclass(frozen=True, slots=True)
class TranscriptInput:
    """Raw turn inputs from which ``ContextBuilder`` assembles a transcript."""

    history: list[dict[str, Any]]
    current_message: str | None
    media: Sequence[str] | None = None
    current_role: str = "user"
    session_summary: SessionSummary | None = None
    runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None

    @property
    def message_count(self) -> int:
        """Number of boundary-preserving messages in the assembled transcript."""
        return 1 + len(self.history) + (self.current_message is not None)


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md"]
    # 未改动的脚手架文件不注入（用户没填的模板塞进提示词只会白烧 token）。
    # IDENTITY.md 与 AGENTS.md/USER.md 同类：都是带占位符的脚手架。
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "IDENTITY.md", "USER.md"}
    _RUNTIME_CONTEXT_TAG = RUNTIME_CONTEXT_TAG
    _RUNTIME_CONTEXT_END = RUNTIME_CONTEXT_END

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        skill_overrides: dict[str, object] | None = None,
        user_skills_dir: Path | None = None,
        user_plugins_dir: Path | None = None,
    ):
        self.workspace = workspace
        self.timezone = timezone
        # FR-3.10：插件提示词片段要在这里取挂载点，故必须留在实例上
        # （只传给 SkillsLoader 的话，提示词这边就看不到插件目录了）。
        self.user_plugins_dir = user_plugins_dir
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(
            workspace,
            disabled_skills=set(disabled_skills) if disabled_skills else None,
            skill_overrides=skill_overrides,
            # 用户级技能目录由调用方注入（生产 = `default_user_skills_dir()`）。
            # 这里不兜默认值：ContextBuilder 在几十个单测里被直接构造，
            # 兜了就会把它们全都绑上开发者宿主机上的真实技能目录。
            user_skills_dir=user_skills_dir,
            # FR-3.4 用户级插件目录同理：只在生产装配路径注入。
            user_plugins_dir=user_plugins_dir,
        )

    def _plugin_prompt_mounts(self, workspace: Path) -> tuple[PluginPromptMount, ...]:
        """当前工作区已启用插件提供的提示词片段挂载点（FR-3.10）。

        **失败一律降级为空**：插件目录读不了 / 清单坏了，都不该让整轮会话起不来
        —— 那等于把「可扩展」做成了「可宕机」。取不到就是没有片段，行为回到
        引入 ChainLoader 之前。
        """
        try:
            # 局部导入：见本文件导入区的 ⚠️⚠️ 说明（模块级会把 config.loader 闭成环）
            from nanobot.agent.plugins import plugin_prompt_mounts

            return plugin_prompt_mounts(workspace, self.user_plugins_dir)
        except Exception:
            logger.warning("failed to resolve Agent Plugin prompt mounts", exc_info=True)
            return ()

    def declared_tool_policy(self, workspace: Path | None = None) -> DeclaredToolPolicy:
        """本轮生效的工具声明（FR-3.12）。语义与纪律见 `hooks/plugin_tools.py`。

        两个来源：

        - **always 技能**：它们无条件进系统提示词，所以它们的 `tools:` 对每一轮都成立；
        - **已启用插件**：清单一侧声明的 `tools:`。

        ⚠️ 口径差异（如实记下）：**按 `$skill` 显式唤起**的技能不在这里——那一步拣选
        发生在 `build_transcript` 的更下游，而钩子链必须在装配期就定下来。
        所以「临时唤起一个带 `tools:` 的技能」这一轮不会收紧工具面。

        **失败一律降级为空声明**（＝不施加约束）：声明是**作者的自律**，不是用户的
        安全设置；读不到它不会绕过任何用户设置（Agent 白名单 / 执行模式 / 审批门
        照旧由另外三道门守着）。反过来，一次读盘抖动就让整轮所有工具调用被拒，
        那是把「可扩展」做成了「可宕机」。
        """
        root = workspace or self.workspace
        declarations: list[ToolDeclaration] = []
        try:
            for name, raw in self.skills.declared_tools(self.skills.get_always_skills()):
                declarations.append(
                    ToolDeclaration(
                        source=f"skill:{name}",
                        tools=frozenset(parse_declared_tools(raw)),
                    )
                )
        except Exception:
            logger.warning("failed to read declared tools from skills", exc_info=True)
        try:
            # 局部导入：同上（模块级会把 `config.loader` 闭成环）
            from nanobot.agent.plugins import plugin_tool_declarations

            for name, tools in plugin_tool_declarations(root, self.user_plugins_dir):
                declarations.append(
                    ToolDeclaration(
                        source=f"plugin:{name}",
                        tools=frozenset(parse_declared_tools(tools)),
                    )
                )
        except Exception:
            logger.warning("failed to read declared tools from Agent Plugins", exc_info=True)
        return combine_declarations(declarations)

    def build_system_prompt(
        self,
        *,
        channel: str | None = None,
        session_summary: SessionSummary | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        root = workspace or self.workspace
        # FR-3.10：插件片段挂载点在整个装配过程里共用一份（`plugin_prompt_mounts`
        # 内部有缓存并按包指纹失效），不要在每个 render 调用点各取一次 ——
        # 那样每轮要重复扫盘，而且两次取到的可能不是同一批（改动竞态）。
        plugin_mounts = self._plugin_prompt_mounts(root)
        parts = [
            self._get_identity(channel=channel, workspace=root, plugin_mounts=plugin_mounts)
        ]

        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)

        # [LOCAL PATCH] 传 workspace：主系统提示词可被 <workspace>/prompts/ 覆盖（§13.4）
        parts.append(
            render_template(
                "agent/tool_contract.md",
                override_root=root,
                plugin_mounts=plugin_mounts,
            )
        )

        if include_memory:
            memory = self.memory.read_memory()
            if memory and not self._is_template_content(memory, "memory/MEMORY.md"):
                # [LOCAL PATCH] 注入前作废记忆里对工作目录的断言（见
                # `neutralize_stale_working_dir`）。记忆是模型写的、可能长期滞后的
                # **旧事实**，而工作目录是运行时值——两者冲突时不能让记忆赢。
                parts.append(
                    "# Memory\n\n## Long-term Memory\n"
                    f"{neutralize_stale_working_dir(memory)}"
                )

        active_skills = self.skills.get_always_skills()
        if active_skills:
            active_content = self.skills.load_skills_for_context(active_skills)
            if active_content:
                parts.append(f"# Active Skills\n\n{active_content}")

        skills_summary = self.skills.build_skills_summary(
            exclude=set(active_skills),
            workspace=root,
        )
        if skills_summary:
            parts.append(
                render_template(
                    "agent/skills_section.md",
                    skills_summary=skills_summary,
                    override_root=root,
                    plugin_mounts=plugin_mounts,
                )
            )

        if session_summary:
            parts.append(
                "[Archived Context Summary]\n\n"
                f"Previous conversation summary (last active {session_summary['last_active']}):\n"
                f"{session_summary['text']}"
            )

        # [LOCAL PATCH] 工作目录**无条件**、且**放在最后**写进提示词。
        #
        # 两条都是实测逼出来的，各自对应一种「设置里写着 A、文件落在 B」：
        #
        # 1) **无条件**。上游是 `if project_path != self.workspace: ...`——只在
        #    「本次工作空间 ≠ 引擎自身工作空间」时才追加。于是「会话没绑工作空间」
        #    和「绑的正好就是引擎默认那个」这两种情况下，提示词里**一个绝对路径
        #    都没有**（identity.md 只有显式绑定时才给绝对路径），模型只能自己猜。
        # 2) **放在最后**。此前它排在 `# Memory` **之前**，而长期记忆里恰好写着
        #    `Current working directory: D:\test`。两条消息互相矛盾、后者在下方，
        #    模型信了记忆（原话「根据之前的上下文，当前工作目录是 D:\test」）。
        #    运行时值是这一轮才成立的**唯一**真相，必须压在最后、成为最新的陈述。
        #
        # 末句是冲着这个失败模式写的：模型会为「当前工作目录是哪儿」编依据，
        # 而记忆是滞后的、可以被用户改的，工作目录只能以运行时为准。
        project_path = root.expanduser().resolve()
        parts.append(
            "# Current Project\n\n"
            f"Working directory: {project_path}\n"
            "Use it as the default root for project files and relative tool paths. "
            "This runtime value is the only source of truth for the working directory: "
            "paths mentioned in memory, skills or earlier sessions are historical — "
            "do not reuse them unless the user asks for them explicitly."
        )

        return "\n\n---\n\n".join(parts)

    def _get_identity(
        self,
        channel: str | None = None,
        workspace: Path | None = None,
        plugin_mounts: Sequence[PluginPromptMount] = (),
    ) -> str:
        """Get the core identity section."""
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        agent_workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            agent_workspace_path=agent_workspace_path,
            runtime=runtime,
            plugin_mounts=plugin_mounts,
            platform_policy=render_template(
                "agent/platform_policy.md",
                system=system,
                override_root=root,
                plugin_mounts=plugin_mounts,
            ),
            channel=channel or "",
            override_root=root,
        )

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self, workspace: Path | None = None) -> str:
        """Load project instructions plus the agent's global profile files.

        [LOCAL PATCH] 身份文件收敛进 ``<identity_context>`` 标签（design §13.3）：
        ``SOUL.md`` / ``IDENTITY.md`` / ``USER.md`` 是同**一类**东西——「我是谁、
        用户是谁」，所以它们共用一个标签块；``AGENTS.md`` 是**项目约定**，语义不同，
        留在标签外。给它们套同一个标签是为了让模型（和读提示词的人）能一眼分辨
        「这段是身份，那段是任务环境」，而不是靠四个 ``## 文件名`` 标题去猜。

        ``IDENTITY.md`` 是本次补齐的第三角：``AGENTS.md`` 的模板本就写着
        「personality → SOUL.md、durable user facts → USER.md」，但「agent 是谁」
        （名字/角色/专长）此前无处安放。
        """
        parts: list[str] = []
        project_root = workspace or self.workspace
        sources = [
            ("AGENTS.md", project_root),
            ("SOUL.md", self.workspace),
            ("IDENTITY.md", self.workspace),
            ("USER.md", self.workspace),
        ]

        identity_parts: list[str] = []
        for filename, root in sources:
            file_path = root / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                if filename == "SOUL.md" and self._is_template_content(
                    content,
                    "legacy/SOUL.md",
                ):
                    content = load_bundled_template("SOUL.md") or content
                if not content.strip():
                    continue
                if filename in self._SKIPPABLE_DEFAULTS and self._is_template_content(
                    content, filename
                ):
                    continue
                section = f"## {filename}\n\n{content}"
                if filename == "AGENTS.md":
                    parts.append(section)
                else:
                    identity_parts.append(section)

        if identity_parts:
            parts.append(
                f"{IDENTITY_CONTEXT_TAG_OPEN}\n"
                + "\n\n".join(identity_parts)
                + f"\n{IDENTITY_CONTEXT_TAG_CLOSE}"
            )

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        tpl = load_bundled_template(template_path)
        if tpl is not None:
            return content.strip() == tpl.strip()
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str | None,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        current_role: str = "user",
        session_summary: SessionSummary | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper for callers that need merged adjacent roles."""
        messages = self.build_transcript(
            TranscriptInput(
                history=history,
                current_message=current_message,
                media=media,
                current_role=current_role,
                session_summary=session_summary,
                runtime_context_blocks=runtime_context_blocks,
            ),
            channel=channel,
            workspace=workspace,
            include_memory=include_memory,
        )
        if current_message is None:
            return messages
        current = messages[-1]
        if len(messages) < 2 or messages[-2].get("role") != current.get("role"):
            return messages

        merged = dict(messages[-2])
        merged["content"] = self._merge_message_content(
            merged.get("content"),
            current.get("content"),
        )
        current_meta = current.get("_meta")
        if current.get("role") == "user" and isinstance(current_meta, dict):
            internal_meta = dict(merged.get("_meta") or {})
            internal_meta.update(cast(dict[str, Any], current_meta))
            merged["_meta"] = internal_meta
        return [*messages[:-2], merged]

    def build_transcript(
        self,
        transcript: TranscriptInput,
        *,
        channel: str | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
    ) -> list[dict[str, Any]]:
        """Build a model transcript while preserving the fresh-turn boundary."""
        root = workspace or self.workspace
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    channel=channel,
                    session_summary=transcript.session_summary,
                    workspace=root,
                    include_memory=include_memory,
                ),
            },
            *transcript.history,
        ]
        if transcript.current_message is None:
            return messages

        current = self.build_current_message(
            transcript.current_message,
            media=list(transcript.media) if transcript.media else None,
            current_role=transcript.current_role,
            runtime_context_blocks=transcript.runtime_context_blocks,
        )
        messages.append(current)
        return messages

    def build_current_message(
        self,
        current_message: str,
        *,
        media: list[str] | None = None,
        current_role: str = "user",
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
    ) -> dict[str, Any]:
        """Build only the fresh turn message without merging it into history."""
        content = self.build_user_content(current_message, image_paths=media)
        blocks: list[RuntimeContextBlock] = []
        if current_role == "user":
            blocks.extend(runtime_context_blocks or ())
            skill_context = self.skills.build_explicit_skill_runtime_context(current_message)
            if skill_context is not None and skill_context not in blocks:
                blocks.append(skill_context)
        merged, runtime_context_meta = append_runtime_context(content, blocks)
        current: dict[str, Any] = {"role": current_role, "content": merged}
        if current_role == "user" and runtime_context_meta is not None:
            current["_meta"] = {
                RUNTIME_CONTEXT_MESSAGE_META: runtime_context_meta,
            }
        return current

    def build_user_content(
        self,
        text: str,
        image_paths: list[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content from prefiltered image paths."""
        if not image_paths:
            return text

        image_blocks: list[dict[str, Any]] = []
        for path in image_paths:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            # Re-detect from the bytes used for the request: the file may have
            # changed since attachment routing, and the data URL needs its MIME.
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not image_blocks:
            return text
        return image_blocks + [{"type": "text", "text": text}]
