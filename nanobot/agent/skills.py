"""Skills loader for agent capabilities."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, cast

import yaml

from nanobot.runtime_context import RuntimeContextBlock

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)
_SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")

# —— [LOCAL PATCH] 技能三态覆盖（FR-3.1，对齐 PRD §12.2 的 skillOverrides）——
# 三态的差别只在「谁能让这个技能跑起来」：
#   on                  全可用：进 skills summary（模型能自主发现并 read_file），
#                       也能被 `$skill` 显式调用
#   user-invocable-only 只从 summary 摘除：模型看不见它存在，但 `$skill` 显式
#                       调用与客户端显式挂载照常可用 —— 这是「用户按需唤起、
#                       但不让模型到处乱用」的那一档
#   off                 完全排除：既不列出，`$skill` 也解析不到
SKILL_OVERRIDE_ON = "on"
SKILL_OVERRIDE_USER_INVOCABLE_ONLY = "user-invocable-only"
SKILL_OVERRIDE_OFF = "off"
KNOWN_SKILL_OVERRIDES = frozenset(
    {SKILL_OVERRIDE_ON, SKILL_OVERRIDE_USER_INVOCABLE_ONLY, SKILL_OVERRIDE_OFF}
)

# 数值越大越严。多来源命中同一个技能名时取**更严**的一档，
# 这样「disabled_skills 与 skill_overrides 同时写了它」不会互相抵消。
_SKILL_OVERRIDE_STRICTNESS = {
    SKILL_OVERRIDE_ON: 0,
    SKILL_OVERRIDE_USER_INVOCABLE_ONLY: 1,
    SKILL_OVERRIDE_OFF: 2,
}


def normalize_skill_override(value: object) -> str:
    """把用户写的覆盖值收敛到三个已知状态之一。

    **未知 / 畸形取值一律收敛为 ``off``（fail-closed），不是 ``on``。**
    覆盖表是一条*限制性*配置——用户写 ``"disabled"`` 或把状态名拼错时，
    意图显然是「别让它跑」；把它读成 ``on`` 等于悄悄撤销用户的限制，
    而读成 ``off`` 只是让技能从列表里消失，可见且可恢复。
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in KNOWN_SKILL_OVERRIDES:
            return normalized
    return SKILL_OVERRIDE_OFF


def is_skill_listed_to_model(state: str) -> bool:
    """该状态下技能是否进入 skills summary（模型能否自主发现它）。"""
    return state == SKILL_OVERRIDE_ON


def is_skill_invocable(state: str) -> bool:
    """该状态下技能是否还能被 ``$name`` 显式调用。"""
    return state != SKILL_OVERRIDE_OFF


def default_user_skills_dir() -> Path:
    """用户级技能目录：``~/.nanowork/skills``（``NANOWORK_HOME`` 可改基址）。

    刻意做成**函数**而不是模块常量：单测会在导入后才改环境变量，
    模块级常量会把宿主机上第一次导入时的取值冻住，结果不可复现。
    """
    home = os.environ.get("NANOWORK_HOME") or os.environ.get("CODEBUDDY_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".nanowork"
    return base / "skills"


def parse_skill_metadata(content: str) -> dict[str, object] | None:
    """Parse a skill document's YAML frontmatter."""
    if not (match := _STRIP_SKILL_FRONTMATTER.match(content)):
        return None
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in cast(dict[object, object], parsed).items()}


def valid_skill_metadata(metadata: dict[str, object], name: str) -> bool:
    """Return whether metadata satisfies the Agent Skills identity contract."""
    description = metadata.get("description")
    return (
        metadata.get("name") == name
        and len(name) <= 64
        and _SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
        skill_overrides: dict[str, object] | None = None,
        user_skills_dir: Path | None = None,
        user_plugins_dir: Path | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()
        self.skill_overrides = dict(skill_overrides or {})
        # [LOCAL PATCH] FR-3.1 用户级技能目录（跨工作区共享）。
        # 默认 ``None`` = 不扫描，由调用方显式注入产品路径。
        # 不在这里兜 `default_user_skills_dir()`：那会让每个单测都去读开发者
        # 宿主机上真实的 ~/.nanowork/skills，断言随本机装了什么而变。
        self.user_skills_dir = user_skills_dir
        # [LOCAL PATCH] FR-3.4 用户级插件目录（~/.nanowork/plugins），同上：
        # 只在生产装配路径注入，单测不注入就完全看不见用户级插件。
        self.user_plugins_dir = user_plugins_dir

    def _effective_overrides(self) -> dict[str, str]:
        """归一化后的覆盖表（含 CLI Apps 别名双向展开）。

        ``disabled_skills`` 是上一代机制（只表达「禁用」），``skill_overrides``
        是三态。两者不是二选一：同一名字同时命中时**取更严的一档**，
        否则「旧配置禁用了它、新配置写了 user-invocable-only」会把它重新放开。
        """
        effective: dict[str, str] = {}
        for name, raw in self.skill_overrides.items():
            if isinstance(name, str) and name:
                effective[name] = normalize_skill_override(raw)
        for name in self.disabled_skills:
            if isinstance(name, str) and name:
                effective[name] = SKILL_OVERRIDE_OFF
        if not effective:
            return effective

        # 别名双向展开：CLI App 的旧名与新名必须同生共死，
        # 否则「改过名」的技能会从限制表里漏出去。
        for legacy, canonical in self._skill_aliases().items():
            states = [effective[key] for key in (legacy, canonical) if key in effective]
            if not states:
                continue
            strictest = max(states, key=lambda state: _SKILL_OVERRIDE_STRICTNESS[state])
            for key in (legacy, canonical):
                effective[key] = strictest
        return effective

    def get_skill_override(self, name: str) -> str:
        """某技能当前生效的三态；未声明即 ``on``。"""
        return self._effective_overrides().get(name, SKILL_OVERRIDE_ON)

    def _skill_aliases(self) -> dict[str, str]:
        """Return compatibility aliases owned by installed CLI Apps."""
        from nanobot.apps.cli import CliAppManager

        try:
            return CliAppManager(workspace=self.workspace).installed_skill_aliases()
        except OSError:
            return {}

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        from nanobot.agent.plugins import enabled_agent_plugin_skills

        plugin_skills = enabled_agent_plugin_skills(self.workspace, self.user_plugins_dir)
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        seen_names = {entry["name"] for entry in skills}
        for name, path in plugin_skills:
            if name in seen_names:
                continue
            skills.append(
                {
                    "name": name,
                    "path": str(path),
                    "source": "plugin",
                }
            )
            seen_names.add(name)
        # 用户级排在项目级/插件之后、内置之前：PRD §12.4 的优先级链是
        # 「内置 < 用户级 < 项目级」，而这里靠 `skip_names` 实现先到先得，
        # 所以扫描顺序必须与优先级链**反向**——先扫的赢。
        if self.user_skills_dir is not None:
            user_entries = self._skill_entries_from_dir(self.user_skills_dir, "user", skip_names=seen_names)
            skills.extend(user_entries)
            seen_names.update(entry["name"] for entry in user_entries)
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=seen_names)
            )

        # 三态覆盖：只有 `off` 在此处消失（`user-invocable-only` 必须留下，
        # 它在 build_skills_summary 里才被摘掉，`$skill` 显式调用仍要能解析到）。
        overrides = self._effective_overrides()
        if overrides:
            skills = [
                skill
                for skill in skills
                if overrides.get(skill["name"], SKILL_OVERRIDE_ON) != SKILL_OVERRIDE_OFF
            ]

        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        skills = self.list_skills(filter_unavailable=False)
        available = {skill["name"] for skill in skills}
        resolved = name if name in available else self._skill_aliases().get(name, name)
        entry = next((skill for skill in skills if skill["name"] == resolved), None)
        return Path(entry["path"]).read_text(encoding="utf-8") if entry else None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Resolve ``$skill-name`` references to enabled, available skills."""
        if not text:
            return []
        available = {
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
        }
        aliases = self._skill_aliases()
        invoked: list[str] = []
        for match in _SKILL_REFERENCE.finditer(text):
            requested = match.group(1)
            name = requested if requested in available else aliases.get(requested, requested)
            if name in available and name not in invoked:
                invoked.append(name)
        return invoked

    def build_explicit_skill_runtime_context(
        self,
        text: str,
    ) -> RuntimeContextBlock | None:
        """Load non-always skills explicitly invoked by the current message."""
        skill_names = self.get_explicitly_invoked_skills(text)
        if not skill_names:
            return None
        always_active = set(self.get_always_skills())
        skill_names = [name for name in skill_names if name not in always_active]
        content = self.load_skills_for_context(skill_names)
        if not content:
            return None
        return RuntimeContextBlock(
            source="explicit_skills",
            content=(
                "[Active Skills — instructions for this user turn]\n"
                f"{content}\n"
                "[/Active Skills]"
            ),
        )

    def build_skills_summary(
        self,
        exclude: set[str] | None = None,
        *,
        workspace: Path | None = None,
    ) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Args:
            exclude: Set of skill names to omit from the summary.
            workspace: Effective project workspace used to choose safe display paths.

        Returns:
            Markdown-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        # `user-invocable-only` 的生效点就在这一行：从 summary 里摘掉 =
        # 模型不知道它存在、无法自主 read_file 加载；但它仍在 `list_skills`
        # 的返回里，所以 `$skill` 显式调用照常解析得到（见
        # get_explicitly_invoked_skills / load_skill）。这正是三态里那一档
        # 「用户能唤起、模型不能自己伸手」的语义。
        overrides = self._effective_overrides()
        all_skills = [
            entry
            for entry in all_skills
            if is_skill_listed_to_model(overrides.get(entry["name"], SKILL_OVERRIDE_ON))
        ]
        if not all_skills:
            return ""

        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = (workspace or self.workspace).expanduser().resolve()
        use_relative_roots = project_workspace == agent_workspace
        sections: list[str] = []
        groups = (
            ("Workspace skills", "workspace", self.workspace_skills),
            ("Agent Plugin skills", "plugin", self.workspace / "plugins"),
            ("User skills", "user", self.user_skills_dir),
            ("Built-in skills", "builtin", self.builtin_skills),
        )
        for label, source, root in groups:
            if root is None:
                continue
            entries = [
                entry
                for entry in all_skills
                if entry["source"] == source and (not exclude or entry["name"] not in exclude)
            ]
            if not entries:
                continue

            resolved_root = root.expanduser().resolve()
            if source == "user":
                # 用户级目录在工作区**之外**，用相对路径展示会指错地方，
                # 模型照着 `skills/foo/SKILL.md` 去 read_file 只会读到别的文件。
                display_root = resolved_root
            elif use_relative_roots:
                display_root = Path("plugins" if source == "plugin" else "skills")
            else:
                display_root = resolved_root
            lines = [f"### {label} (`{display_root}`)"]
            for entry in entries:
                skill_name = entry["name"]
                meta = self._get_skill_meta(skill_name)
                available = self._check_requirements(meta)
                desc = self.get_skill_description(skill_name)
                suffix = ""
                if not available:
                    missing = self._get_missing_requirements(meta)
                    suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
                relative_path = self._display_skill_path(entry["path"], root)
                lines.append(f"- **{skill_name}** — {desc}{suffix}  `{relative_path}`")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _display_skill_path(skill_path: str, root: Path) -> str:
        """技能文档在提示词里展示的路径（模型照着它去 read_file）。

        ``root`` 是这一组的根目录。FR-3.4 之后「插件」这一组里可能混着
        **用户级插件**（``~/.nanowork/plugins``）的技能——它们不在
        ``<workspace>/plugins`` 之下，硬算相对路径会抛 ``ValueError``，
        算出来也会指错地方。此时退回绝对路径（与 user 源同理）。
        """
        path = Path(skill_path)
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _requirement_lists(skill_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (bins, env) lists from skill metadata, tolerating null/wrong shapes."""
        requires = cast(dict[str, Any], skill_meta.get("requires") or {})
        if not isinstance(skill_meta.get("requires") or {}, dict):
            return [], []
        bins_raw: object = requires.get("bins") or []
        env_raw: object = requires.get("env") or []
        bins = [value for value in cast(list[object], bins_raw) if isinstance(value, str) and value.strip()] if isinstance(bins_raw, list) else []
        env = [value for value in cast(list[object], env_raw) if isinstance(value, str) and value.strip()] if isinstance(env_raw, list) else []
        return bins, env

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        """Get a description of missing requirements."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Return whether a skill can run and why not when it cannot."""
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """Return explicit command/env requirements and currently missing entries."""
        bins, env = self._requirement_lists(self._get_skill_meta(name))
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not shutil.which(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

    def get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        description = meta.get("description") if meta else None
        if isinstance(description, str) and description:
            return description
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict[str, Any]:
        """Extract nanobot/openclaw metadata from a frontmatter field.

        ``raw`` may be a dict (already parsed by yaml.safe_load) or a JSON str.
        """
        if isinstance(raw, dict):
            data = cast(dict[str, Any], raw)
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        data_object = cast(dict[str, Any], data)
        payload = data_object.get("nanobot", data_object.get("openclaw", {}))
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        return parse_skill_metadata(self.load_skill(name) or "")
