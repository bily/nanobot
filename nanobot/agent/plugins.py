"""Load and activate locally installed Agent Plugin packages."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import ValidationError

from nanobot.agent.skills import parse_skill_metadata, valid_skill_metadata
from nanobot.config.loader import get_config_path
from nanobot.config.schema import MCPServerConfig

# [LOCAL PATCH] FR-3.10：提示词片段的挂载形状。只取一个 dataclass，
# 不引 `utils.prompt_templates` 的其它符号（那个模块会 import jinja2）。
from nanobot.utils.prompt_templates import PluginPromptMount

AGENT_PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
AGENT_PLUGIN_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"

_PLUGIN_NAME = re.compile(r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_MCP_SERVER_FIELDS = {"type", "command", "args", "env", "cwd"}
_MAX_LOGO_BYTES = 256 * 1024

# —— [LOCAL PATCH] FR-3.8 版本化缓存布局 ——
# 目录约定（PRD §12.6）：
#   <plugins>/marketplaces/<market>/<plugin>/<version>/   源：市场拉下来的内容，可安全覆盖
#   <plugins>/cache/<market>/<plugin>/<version>/          生效副本：只增不改，按需 GC
#   <plugins>/cache/<market>/<plugin>/current             指针：当前生效版本（文件或目录）
#
# 「current」指针存在的唯一理由：**让市场刷新与正在跑的会话解耦**。
# 刷新只往 marketplaces/ 与 cache/ 里**追加**新版本、从不就地替换，也不动指针，
# 于是「长会话跨过若干次市场刷新」不会半路换掉它依赖的插件定义（环境漂移）。
# 升级 = 客户端显式改写指针，而不是扫描时「取最新」。
#
# 指针坏了（文件在、指向的版本没了）时**不回退最新版**：静默回退会让用户以为
# 升级成功了、其实跑的是另一版，这比「插件没找到」难查得多。
_CACHE_DIRNAME = "cache"
_MARKETPLACES_DIRNAME = "marketplaces"
_CURRENT_POINTER = "current"
# 目录名（市场名 / 插件目录名 / 版本号）字符集：不含路径分隔符、不以 `.` 开头——
# 这一层是**遍历防护**，别指望调用方传进来的名字是干净的。
_SAFE_NAME = re.compile(r"^(?!\.)[A-Za-z0-9._-]+$")
# 一次发现最多看多少个 (market, plugin) 组合：磁盘上堆了几千个版本时，
# 不能让一次插件发现变成一次全盘遍历。
_MAX_CACHE_ENTRIES = 2000

# —— [LOCAL PATCH] FR-3.4 组件位置候选 ——
# 索引 0 是 agent-plugins.org 规范写的位置（引擎原生就认的那一份），
# 索引 1 是 PRD §12.2 写的位置。**一个包只认一个真相**：按顺序取第一个
# 存在的文件，不让两份清单各说各话；取不到才算缺组件。
_PLUGIN_MANIFEST_RELPATHS = ("plugin.json", ".nanowork-plugin/plugin.json")
_PLUGIN_MCP_RELPATHS = ("mcp.json", "mcp/mcp.json")


def default_user_plugins_dir() -> Path:
    """用户级插件目录：``~/.nanowork/plugins``（``NANOWORK_HOME`` 可改基址）。

    与 :func:`nanobot.agent.skills.default_user_skills_dir` 同源同理——
    刻意做成**函数**而不是模块常量：单测会在导入后才改环境变量，
    模块常量会把宿主机上第一次导入时的取值冻住，结果不可复现。
    """
    home = os.environ.get("NANOWORK_HOME") or os.environ.get("CODEBUDDY_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".nanowork"
    return base / "plugins"


@dataclass(frozen=True, slots=True)
class _PackageSnapshot:
    root: Path
    fingerprint: str
    skill_dirs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _SkillCacheEntry:
    skills: tuple[tuple[str, Path], ...]
    packages: tuple[_PackageSnapshot, ...]


_SKILL_CACHE: dict[tuple[Path, Path], _SkillCacheEntry] = {}


@dataclass(frozen=True)
class AgentPlugin:
    """A validated, locally installed Agent Plugins v1 package."""

    name: str
    root: Path
    description: str
    repository: str
    display_name: str
    category: str
    accent_color: str | None
    logo: str | None
    permissions: tuple[str, ...]
    mcp_servers: tuple[str, ...] = ()
    enabled: bool = False
    # [LOCAL PATCH] FR-3.8：来源与版本。平铺安装的本地包两者都是 None；
    # 来自版本化缓存的副本带市场名与版本号，客户端据此显示「从哪来、哪一版」。
    market: str | None = None
    version: str | None = None
    # —— [LOCAL PATCH] FR-3.10 / FR-3.11：插件的文本资产 ——
    # 全部在**解析期**就用 `_contained()` 落成绝对路径：越界条目（`../`、绝对路径、
    # 指向不存在的目标）在这里被丢弃并告警，于是调用方拿到的路径**必然**在包内。
    # 若改成「存清单元数据、由消费方自己拼路径」，那层遍历防护就得在每个消费点
    # 各写一遍 —— 漏一处就是一条任意文件读取。
    prompt_dirs: tuple[Path, ...] = ()
    agent_files: tuple[Path, ...] = ()
    command_files: tuple[Path, ...] = ()
    hook_manifest: tuple[Path, ...] = ()
    # FR-3.12：声明式工具策略（**只收紧**）。空元组 = 未声明 = 不施加约束，
    # 与 `tool_allow=None`（未声明白名单）是同一条「None ≠ 空集」的纪律。
    tool_policy: tuple[str, ...] = ()


def _plugin_roots(workspace: Path, user_plugins_dir: Path | None) -> list[Path]:
    """按优先级返回要扫描的插件根目录（项目级在前）。

    **顺序即优先级**：同名去重是先到先得，所以先扫的必须是高优先级的那一侧
    ——这正是 PRD §12.4 里「用户级 < 项目级」的落地方式。
    """
    roots: list[Path] = []
    workspace = workspace.expanduser().resolve()
    project_root = _contained(workspace / "plugins", workspace, directory=True)
    if project_root is not None:
        roots.append(project_root)
    if user_plugins_dir is not None:
        candidate = user_plugins_dir.expanduser()
        # 用户级目录在工作区之外，用它的父目录做包含性检查——防的是
        # `~/.nanowork/plugins` 被换成指向别处的符号链接。
        user_root = _contained(candidate, candidate.parent.resolve(), directory=True)
        if user_root is not None:
            roots.append(user_root)
    return roots


def _version_sort_key(version: str) -> tuple[tuple[int, int, str], ...]:
    """版本号排序键：数字段按**数值**比，其余按字符串比。

    `0.1.9` 必须小于 `0.1.10`——纯字符串比会得到相反的结果，而这条错在
    「无指针时取最新版本」这条路径上会表现为「升级了却在跑旧版」。
    """
    parts = re.split(r"[._\-+]", version)
    key: list[tuple[int, int, str]] = []
    for part in parts:
        key.append((0, int(part), "") if part.isdigit() and len(part) <= 9 else (1, 0, part))
    return tuple(key)


def _cache_plugin_dirs(root: Path) -> list[tuple[str, str, Path]]:
    """列出 ``cache/<market>/<plugin>/`` 三元组（市场名、插件目录名、目录）。"""
    cache_root = _contained(root / _CACHE_DIRNAME, root, directory=True)
    if cache_root is None:
        return []
    found: list[tuple[str, str, Path]] = []
    for market in _children(cache_root, "Agent Plugins cache"):
        if not market.is_dir() or _SAFE_NAME.fullmatch(market.name) is None:
            continue
        for plugin in _children(market, f"Agent Plugins market '{market.name}'"):
            if not plugin.is_dir() or _SAFE_NAME.fullmatch(plugin.name) is None:
                continue
            found.append((market.name, plugin.name, plugin))
            if len(found) >= _MAX_CACHE_ENTRIES:
                logger.warning("Agent Plugins cache exceeded {} entries; truncating scan", _MAX_CACHE_ENTRIES)
                return found
    return found


def _resolve_cache_copy(plugin_dir: Path) -> Path | None:
    """解析某个插件**当前生效的那一版**副本。

    优先 `current` 指针（文件或目录/符号链接），无指针时取版本号最大的一版。
    返回 None = 这个插件当前没有可用副本（指针坏了也算，不回退）。
    """
    pointer = plugin_dir / _CURRENT_POINTER
    # 目录（含符号链接）形态：直接解析，比读文件更直观，也方便人工 `ln -s`
    directory = _contained(pointer, plugin_dir, directory=True)
    if directory is not None:
        return directory
    if pointer.is_file():
        try:
            target = pointer.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            target = ""
        if _SAFE_NAME.fullmatch(target) is None:
            logger.warning("Ignoring invalid version pointer for '{}'", plugin_dir)
            return None
        candidate = _contained(plugin_dir / target, plugin_dir, directory=True)
        if candidate is None:
            logger.warning(
                "Version pointer for '{}' points at missing version '{}'; treating as not installed",
                plugin_dir,
                target,
            )
            return None
        return candidate

    versions = [
        child
        for child in _children(plugin_dir, f"Agent Plugin '{plugin_dir.name}' versions")
        if child.is_dir() and _SAFE_NAME.fullmatch(child.name) is not None
    ]
    if not versions:
        return None
    return max(versions, key=lambda path: _version_sort_key(path.name))


def _installed_plugins(workspace: Path, user_plugins_dir: Path | None = None) -> list[AgentPlugin]:
    """Return installed packages found under ``<workspace>/plugins/*`` and the user dir.

    [LOCAL PATCH] FR-3.8：除平铺目录外，还解析 `<root>/cache/<market>/<plugin>/<version>/`
    下的版本化副本。**平铺优先**——那是用户自己放进去的本地包，语义上强于市场副本；
    同名冲突时缓存副本让位，而不是靠字典顺序碰运气。
    """
    plugins: dict[str, AgentPlugin] = {}
    for root in _plugin_roots(workspace, user_plugins_dir):
        # 单目录内重复身份视为脏数据，两个都丢弃（保持上游语义）；
        # 跨目录重名则是正常的优先级覆盖，先扫的赢、后者静默让位。
        seen: dict[str, AgentPlugin | None] = {}
        for candidate in _children(root, "Agent Plugins directory"):
            plugin_root = _contained(candidate, root, directory=True)
            if plugin_root is None:
                continue
            plugin = _load_manifest(plugin_root)
            if plugin is None:
                continue
            if plugin.name in seen:
                logger.warning("Ignoring duplicate Agent Plugin identity '{}'", plugin.name)
                seen[plugin.name] = None
            else:
                seen[plugin.name] = plugin
        for market, dirname, plugin_dir in _cache_plugin_dirs(root):
            copy = _resolve_cache_copy(plugin_dir)
            if copy is None:
                continue
            cached = _load_manifest(copy, market=market, version=copy.name)
            if cached is None:
                continue
            if cached.name in seen:
                logger.warning(
                    "Ignoring cached Agent Plugin copy '{}' from market '{}': a local install wins",
                    cached.name,
                    market,
                )
                continue
            seen[cached.name] = cached
        for name, plugin in seen.items():
            if plugin is None or name in plugins:
                continue
            plugins[name] = plugin
    return list(plugins.values())


def enabled_agent_plugin_skills(
    workspace: Path,
    user_plugins_dir: Path | None = None,
) -> list[tuple[str, Path]]:
    """Verify and return skills from plugins the user has explicitly enabled."""
    skills: list[tuple[str, Path]] = []
    packages: list[_PackageSnapshot] = []
    for plugin in _installed_plugins(workspace, user_plugins_dir):
        plugin_skills = _discover_plugin_skills(plugin.name, plugin.root)
        fingerprint = _enabled_package_fingerprint(workspace, plugin)
        if fingerprint is None:
            continue
        skills.extend(plugin_skills)
        if plugin_skills:
            packages.append(
                _PackageSnapshot(
                    root=plugin.root,
                    fingerprint=fingerprint,
                    skill_dirs=tuple(path.parent for _name, path in plugin_skills),
                )
            )

    key = _skill_cache_key(workspace, user_plugins_dir)
    _SKILL_CACHE[key] = _SkillCacheEntry(tuple(skills), tuple(packages))
    return skills


def enabled_agent_plugin_skill_dirs(
    workspace: Path,
    *,
    requested_path: str | Path | None = None,
    user_plugins_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Return skill roots authorized for one read, revalidating their package."""
    key = _skill_cache_key(workspace, user_plugins_dir)
    cached = _SKILL_CACHE.get(key)
    if cached is None:
        enabled_agent_plugin_skills(workspace, user_plugins_dir)
        cached = _SKILL_CACHE.get(key)
    if cached is None:
        return ()

    target = (
        Path(requested_path).expanduser().resolve(strict=False)
        if requested_path is not None
        else None
    )
    packages = tuple(
        package
        for package in cached.packages
        if target is None
        or any(target == root or target.is_relative_to(root) for root in package.skill_dirs)
    )
    if any(_package_fingerprint(package.root) != package.fingerprint for package in packages):
        # Re-run the full activation check so a changed package loses its
        # marker and cannot become readable again through this cache.
        _invalidate_skill_cache(workspace)
        enabled_agent_plugin_skills(workspace, user_plugins_dir)
        return ()

    if target is None:
        return tuple(root for package in packages for root in package.skill_dirs)
    return tuple(
        root
        for package in packages
        for root in package.skill_dirs
        if target == root or target.is_relative_to(root)
    )


@dataclass(frozen=True, slots=True)
class PluginMount:
    """一个**已启用**插件在当前工作区的挂载点（按 `name` 索引）。

    B7-05 把原先只服务 skills 的「已启用插件 → 包根」映射提升成这张表：提示词片段
    （FR-3.10）、agents / commands / hooks 清单（FR-3.11）、工具策略（FR-3.12）
    都从这里取，避免每加一类资产就把启用校验与越界校验各抄一遍。

    ⚠️ **skills 仍走它自己那条路**（`enabled_agent_plugin_skill_dirs`）：
    那个函数不只是查表，它还是「逐路径授权文件读取」的判据（`filesystem.py` 用它
    决定某个 SKILL.md 能不能被读）。把授权语义拍进通用挂载表会让两者都变模糊。
    """
    name: str
    root: Path
    fingerprint: str
    #: `trusted`（用户自己放进 `<workspace>/plugins/`）或 `user`（市场缓存副本）
    trust: str
    prompt_dirs: tuple[Path, ...] = ()
    agent_files: tuple[Path, ...] = ()
    command_files: tuple[Path, ...] = ()
    hook_manifest: tuple[Path, ...] = ()
    tool_policy: tuple[str, ...] = ()


_MOUNT_CACHE: dict[tuple[Path, Path, str], tuple[PluginMount, ...]] = {}


def plugin_trust(plugin: AgentPlugin) -> str:
    """提示词片段的信任档。

    判据取**来源**而不是「谁装的」：从市场缓存解析出来的副本内容不在用户手里
    （下一次刷新就会换一版），而 `<workspace>/plugins/` 下的包是用户自己放进去的。
    前者要包 `<untrusted_content>`，后者直注（见 FR-3.10 ④ 与 R-5）。
    """
    return "user" if plugin.market else "trusted"


def enabled_agent_plugin_mounts(
    workspace: Path,
    user_plugins_dir: Path | None = None,
) -> tuple[PluginMount, ...]:
    """当前工作区**已启用**插件的挂载表。

    每次调用都对缓存项**重算包指纹**（与 `enabled_agent_plugin_skill_dirs` 同款）：
    插件被改动后必须立刻失去挂载，否则改过的 `prompts/` 会继续进系统提示词 ——
    那正是信任分级要堵的那条路。代价是每轮提示词构建会读一遍包内文件；插件是纯
    文本资产（KB 级），这个代价换「改了就生效」是划算的。
    """
    key = _skill_cache_key(workspace, user_plugins_dir)
    cached = _MOUNT_CACHE.get(key)
    if cached is not None:
        if all(_package_fingerprint(mount.root) == mount.fingerprint for mount in cached):
            return cached
        # 有包变了：连 skills 缓存一起清掉（两者的启用判据同源，分开清必然留脏）
        _invalidate_skill_cache(workspace)

    mounts: list[PluginMount] = []
    for plugin in _installed_plugins(workspace, user_plugins_dir):
        fingerprint = _enabled_package_fingerprint(workspace, plugin)
        if fingerprint is None:
            continue
        mounts.append(
            PluginMount(
                name=plugin.name,
                root=plugin.root,
                fingerprint=fingerprint,
                trust=plugin_trust(plugin),
                prompt_dirs=plugin.prompt_dirs,
                agent_files=plugin.agent_files,
                command_files=plugin.command_files,
                hook_manifest=plugin.hook_manifest,
                tool_policy=plugin.tool_policy,
            )
        )
    result = tuple(mounts)
    _MOUNT_CACHE[key] = result
    return result


def plugin_prompt_mounts(
    workspace: Path,
    user_plugins_dir: Path | None = None,
) -> tuple[PluginPromptMount, ...]:
    """把挂载表摊平成提示词层的挂载点（**每个插件一个**，不是每个目录一个）。

    模板名基准是**包根**（``demo/fragments/domain.md``），因此一个插件只需要一个
    命名空间条目；`directories` 用来限定「能读到包内哪几个目录」。
    没声明 ``prompts`` 的插件**不产生**挂载点 —— 这样「未声明新字段」与
    「引入本特性之前」在提示词层是逐字相同的。

    转换放在这里而不是 ``prompt_templates``：那边是 ``utils`` 层，反向依赖
    ``agent`` 会成环；而这里 import 一个 dataclass 是无害的。
    """
    return tuple(
        PluginPromptMount(
            name=mount.name,
            root=mount.root,
            directories=mount.prompt_dirs,
            trust=mount.trust,
        )
        for mount in enabled_agent_plugin_mounts(workspace, user_plugins_dir)
        if mount.prompt_dirs
    )


def plugin_tool_declarations(
    workspace: Path,
    user_plugins_dir: Path | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """已启用插件声明的工具白名单（FR-3.12）。

    只返回**声明了** `tools:` 的插件：「没声明」不该出现在结果里，因为调用方
    （`combine_declarations`）把每个条目都当**收窄器** —— 掺进一个空集条目会让
    交集莫名其妙变成空，于是「某个插件没写 tools」变成「所有工具都不能用」。
    """
    return [
        (mount.name, mount.tool_policy)
        for mount in enabled_agent_plugin_mounts(workspace, user_plugins_dir)
        if mount.tool_policy
    ]


def _skill_cache_key(workspace: Path, user_plugins_dir: Path | None = None) -> tuple[Path, Path, str]:
    """Cache key：同一工作区配不同用户级目录不得互相顶掉对方的缓存。"""
    return (
        workspace.expanduser().resolve(),
        get_config_path().expanduser().resolve(),
        str(user_plugins_dir.expanduser().resolve()) if user_plugins_dir is not None else "",
    )


def _invalidate_skill_cache(workspace: Path) -> None:
    """丢弃该工作区的**全部**缓存项。

    故意不按完整 key 精确删除：失效点（``_enabled_package_fingerprint``）
    手里没有 user_plugins_dir，而漏删留下的条目会让已被停用的包继续可读。
    这里按工作区前缀清空，宁可多清也不留脏。
    """
    prefix = workspace.expanduser().resolve()
    for key in [key for key in _SKILL_CACHE if key[0] == prefix]:
        _SKILL_CACHE.pop(key, None)
    # 挂载表与 skills 缓存同源（启用判据相同），必须一起清：只清一边会出现
    # 「技能已被停用、提示词片段还在进系统提示词」这种半边生效的中间态。
    for key in [key for key in _MOUNT_CACHE if key[0] == prefix]:
        _MOUNT_CACHE.pop(key, None)


def _package_fingerprint(root: Path) -> str | None:
    """Hash package paths, link targets, and file contents."""
    digest = sha256()
    try:
        for candidate in sorted(root.rglob("*")):
            relative = candidate.relative_to(root).as_posix()
            digest.update(relative.encode())
            if candidate.is_symlink():
                digest.update(b"\0link\0")
                digest.update(candidate.readlink().as_posix().encode())
            elif candidate.is_file():
                digest.update(b"\0file\0")
                digest.update(candidate.read_bytes())
            elif candidate.is_dir():
                digest.update(b"\0dir\0")
            else:
                return None
            digest.update(b"\0")
    except OSError:
        return None
    return digest.hexdigest()


def _load_manifest(
    plugin_root: Path,
    *,
    market: str | None = None,
    version: str | None = None,
) -> AgentPlugin | None:
    # [LOCAL PATCH] FR-3.4：支持组件位置回退（PRD §12.2 的 `.nanowork-plugin/`）。
    payload = _read_first_object(plugin_root, _PLUGIN_MANIFEST_RELPATHS)
    if payload is None:
        return None
    if payload.get("$schema") != AGENT_PLUGIN_SCHEMA:
        return None
    name = payload.get("name")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or _PLUGIN_NAME.fullmatch(name) is None
    ):
        logger.warning("Ignoring Agent Plugin manifest in '{}': invalid name", plugin_root)
        return None
    extension = payload.get("extensions")
    extension_payload = cast(dict[str, object], extension) if isinstance(extension, dict) else {}
    nanobot_value = extension_payload.get("dev.nanobot")
    nanobot = cast(dict[str, object], nanobot_value) if isinstance(nanobot_value, dict) else {}
    return AgentPlugin(
        name=name,
        root=plugin_root,
        description=_string(payload.get("description")),
        repository=_string(payload.get("repository")),
        display_name=_string(nanobot.get("displayName")) or name,
        category=_string(nanobot.get("category")) or "Plugin",
        accent_color=_accent_color(nanobot.get("accentColor")),
        logo=_plugin_logo(nanobot.get("logo"), plugin_root),
        permissions=_string_tuple(nanobot.get("permissions")),
        market=market,
        version=version,
        # [LOCAL PATCH] FR-3.10 / FR-3.11：清单里的新字段。键名取 `prompts` /
        # `agents` / `commands` / `hooks` / `tools`（分析① §P1 的形状），
        # 落到 dataclass 上是带语义的 `*_dirs` / `*_files` / `tool_policy`。
        prompt_dirs=_plugin_dirs(nanobot.get("prompts"), plugin_root),
        agent_files=_plugin_files(nanobot.get("agents"), plugin_root),
        command_files=_plugin_files(nanobot.get("commands"), plugin_root),
        hook_manifest=_plugin_files(nanobot.get("hooks"), plugin_root),
        # 与 `prompts` / `agents` 同形：单值与列表都要收（`tools: "shell"` 是
        # 最容易写出来的形态，用 `_string_tuple` 会静默读成空 —— 而空 = 不约束，
        # 于是「声明了白名单却什么都没限制」，是安全方向上的静默失效）。
        tool_policy=_string_list(nanobot.get("tools")),
    )


def _string_list(value: object) -> tuple[str, ...]:
    """`string | string[]`（与 skills 同形：单值与列表都要收）。"""
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    return _string_tuple(value)


def _plugin_dirs(value: object, root: Path) -> tuple[Path, ...]:
    """清单里的包内目录 → 绝对路径；越界 / 不存在一律丢弃并告警。

    ⚠️ 必须经 `_contained()`：`Path(root) / "/etc"` 在 pathlib 里**直接变成 `/etc`**
    —— 绝对路径会覆盖左侧，所以「拼接」这一步之后再判「是否仍在包内」不能省。
    """
    resolved: list[Path] = []
    for entry in _string_list(value):
        candidate = _contained(root / entry, root, directory=True)
        if candidate is None:
            logger.warning(
                "Ignoring Agent Plugin prompt directory outside its package: {}", entry
            )
            continue
        resolved.append(candidate)
    return tuple(resolved)


def _plugin_files(value: object, root: Path) -> tuple[Path, ...]:
    """同上，但只收**文件**（agents / commands / hooks 清单）。"""
    resolved: list[Path] = []
    for entry in _string_list(value):
        candidate = _contained(root / entry, root)
        if candidate is None:
            logger.warning("Ignoring Agent Plugin file outside its package: {}", entry)
            continue
        resolved.append(candidate)
    return tuple(resolved)


def agent_plugin_mcp_servers(
    workspace: Path,
    configured: dict[str, MCPServerConfig] | None = None,
    user_plugins_dir: Path | None = None,
) -> dict[str, MCPServerConfig]:
    """Merge explicitly enabled plugin MCP servers with user configuration.

    User configuration wins on the unlikely event of a namespaced collision.
    """
    servers: dict[str, MCPServerConfig] = {}
    for plugin in _installed_plugins(workspace, user_plugins_dir):
        if not _enabled(workspace, plugin):
            continue
        plugin_servers = _plugin_mcp_servers(workspace, plugin)
        for name, server in plugin_servers.items():
            # ``--`` cannot occur in a valid plugin identity, so multi-server
            # namespaces cannot collide with a single-server plugin name.
            host_name = plugin.name if len(plugin_servers) == 1 else f"{plugin.name}--{name}"
            servers[host_name] = server
    configured = configured or {}
    if collisions := servers.keys() & configured.keys():
        logger.warning("Configured MCP servers override Agent Plugins: {}", ", ".join(sorted(collisions)))
    return servers | configured


def discover_agent_plugins(
    workspace: Path,
    user_plugins_dir: Path | None = None,
) -> list[AgentPlugin]:
    """Return component and lifecycle state for discovered plugins."""
    return [
        replace(
            plugin,
            mcp_servers=tuple(sorted(_plugin_mcp_servers(workspace, plugin))),
            enabled=_enabled(workspace, plugin),
        )
        for plugin in _installed_plugins(workspace, user_plugins_dir)
    ]


def set_agent_plugin_enabled(
    workspace: Path,
    name: str,
    enabled: bool,
    user_plugins_dir: Path | None = None,
) -> None:
    """Enable or disable one installed plugin."""
    plugin = next(
        (
            item
            for item in _installed_plugins(workspace, user_plugins_dir)
            if item.name == name
        ),
        None,
    )
    if plugin is None:
        raise ValueError(f"unknown Agent Plugin '{name}'")
    data = _plugin_data_dir(workspace, plugin.name, create=True)
    marker = data / "enabled"
    if enabled:
        entry = _activation_entry(plugin)
        if entry is None:
            raise RuntimeError(f"Agent Plugin '{name}' changed while it was being enabled")
        # 追加而不是覆盖：其他版本的签发要留着（旧会话可能还钉在旧版上）
        entries = _read_activation_entries(marker) if marker.is_file() else []
        entries.append(entry)
        _write_activation_entries(marker, entries)
    else:
        # 停用按**名字**：整份标记删掉。用户说「停用这个插件」时，
        # 留着其中某一版的签发等于没停用。
        marker.unlink(missing_ok=True)
    _invalidate_skill_cache(workspace)


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    items = cast(list[object], value) if isinstance(value, list) else []
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def _accent_color(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else None


def _plugin_logo(value: object, plugin_root: Path) -> str | None:
    """Resolve nanobot's optional packaged logo extension."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("./"):
        logger.warning("Ignoring invalid Agent Plugin logo in '{}'", plugin_root)
        return None
    logo = _contained(plugin_root / value[2:], plugin_root)
    try:
        data = logo.read_bytes() if logo is not None else b""
        suffix = logo.suffix.lower() if logo is not None else ""
        if len(data) <= _MAX_LOGO_BYTES and (
            suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n")
            or suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8\xff")
            or suffix == ".webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        ):
            mime = "jpeg" if suffix in {".jpg", ".jpeg"} else suffix[1:]
            return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except OSError:
        pass
    logger.warning("Ignoring invalid Agent Plugin logo in '{}'", plugin_root)
    return None


def _plugin_mcp_servers(workspace: Path, plugin: AgentPlugin) -> dict[str, MCPServerConfig]:
    payload = _read_first_object(plugin.root, _PLUGIN_MCP_RELPATHS)
    if payload is None:
        return {}
    raw_servers = payload.get("mcpServers")
    if (
        payload.keys() != {"$schema", "mcpServers"}
        or payload.get("$schema") != AGENT_PLUGIN_MCP_SCHEMA
        or not isinstance(raw_servers, dict)
    ):
        logger.warning("Ignoring invalid MCP component for Agent Plugin '{}'", plugin.name)
        return {}

    data = _plugin_data_dir(workspace, plugin.name, create=True)
    servers: dict[str, MCPServerConfig] = {}
    for name, raw in cast(dict[str, object], raw_servers).items():
        if not name or len(name) > 128 or any(ord(char) < 32 for char in name):
            logger.warning("Ignoring invalid MCP server name in Agent Plugin '{}'", plugin.name)
            continue
        server = _plugin_mcp_server(raw, plugin.root, data)
        if server is None:
            logger.warning("Ignoring invalid MCP server '{}' in Agent Plugin '{}'", name, plugin.name)
            continue
        servers[name] = server
    return servers


def _plugin_mcp_server(raw: object, root: Path, data: Path) -> MCPServerConfig | None:
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    if payload.keys() - _MCP_SERVER_FIELDS:
        return None
    try:
        server = MCPServerConfig.model_validate(payload)
    except ValidationError:
        return None
    command = _stdio_command(server.command, root)
    cwd = _stdio_cwd(payload.get("cwd"), root, data)
    if server.type != "stdio" or command is None or cwd is None:
        return None
    if {"PLUGIN_ROOT", "PLUGIN_DATA"} & server.env.keys():
        return None
    return server.model_copy(
        update={
            "command": command,
            "args": [_expand(item, root, data) for item in server.args],
            "env": {
                **{key: _expand(value, root, data) for key, value in server.env.items()},
                "PYTHONDONTWRITEBYTECODE": "1",
                "PLUGIN_ROOT": str(root),
                "PLUGIN_DATA": str(data),
            },
            "cwd": str(cwd),
        }
    )


def _stdio_command(value: object, root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("./"):
        executable = _contained(root / value[2:], root)
        return str(executable) if executable is not None else None
    if any(char.isspace() for char in value) or "/" in value or "\\" in value:
        return None
    return value


def _stdio_cwd(value: object, root: Path, data: Path) -> Path | None:
    if value is None:
        return root
    if not isinstance(value, str):
        return None
    if value.startswith("./"):
        return _contained(root / value[2:], root, directory=True)
    for placeholder, base in (("${PLUGIN_ROOT}", root), ("${PLUGIN_DATA}", data)):
        if value == placeholder or value.startswith(f"{placeholder}/"):
            relative = value[len(placeholder):].lstrip("/")
            candidate = (base / relative).resolve()
            if not candidate.is_relative_to(base):
                return None
            if base == data:
                candidate.mkdir(parents=True, exist_ok=True)
                candidate.chmod(0o700)
            return candidate if candidate.is_dir() else None
    return None


def _expand(value: str, root: Path, data: Path) -> str:
    return value.replace("${PLUGIN_ROOT}", str(root)).replace("${PLUGIN_DATA}", str(data))


def _plugin_data_dir(workspace: Path, name: str, *, create: bool) -> Path:
    workspace_id = sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()[:12]
    current = get_config_path().expanduser().resolve().parent
    for segment in ("plugin-data", workspace_id, name):
        path = current / segment
        if create:
            path.mkdir(parents=True, exist_ok=True)
        try:
            resolved = path.resolve(strict=create)
        except OSError as exc:
            raise RuntimeError("Agent Plugin data directory is unavailable") from exc
        if not resolved.is_relative_to(current):
            raise RuntimeError("Agent Plugin data directory escapes its parent")
        if create:
            resolved.chmod(0o700)
        current = resolved
    return current


def _activation_entry(plugin: AgentPlugin) -> tuple[str, str] | None:
    """当前副本的激活条目 ``(fingerprint, root)``；算不出指纹时返回 None。"""
    fingerprint = _package_fingerprint(plugin.root)
    if fingerprint is None:
        return None
    return (fingerprint, str(plugin.root))


def _read_activation_entries(marker: Path) -> list[tuple[str, str]]:
    """读标记文件 → 已签发的条目列表。

    三种历史形态都要认，否则一次引擎升级会把**用户已经启用的包**读成未启用：

    - v2：`{"version":2,"entries":[{"fingerprint":...,"root":...}]}`
    - v1：`{"fingerprint":...,"root":...}`（单版本时代的写法）
    - 更早：整串就是包路径
    """
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return []
    if not raw:
        return []
    if raw.startswith("{"):
        try:
            payload = cast(object, json.loads(raw))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        entries = payload.get("entries")
        if isinstance(entries, list):
            parsed: list[tuple[str, str]] = []
            for item in entries:
                if not isinstance(item, dict):
                    continue
                fingerprint = item.get("fingerprint")
                root = item.get("root")
                if isinstance(fingerprint, str) and fingerprint:
                    parsed.append((fingerprint, root if isinstance(root, str) else ""))
            return parsed
        fingerprint = payload.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            root = payload.get("root")
            return [(fingerprint, root if isinstance(root, str) else "")]
        return []
    # 早期纯路径形态：没有指纹，只能按路径匹配
    return [("", raw)]


def _write_activation_entries(marker: Path, entries: list[tuple[str, str]]) -> None:
    """写回标记文件（v2 形态），顺手剪掉副本已不在磁盘上的条目。"""
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fingerprint, root in entries:
        if fingerprint in seen:
            continue
        if root and not Path(root).is_dir():
            # 副本被 GC 掉了：留着它只会让标记文件无限长
            continue
        seen.add(fingerprint)
        kept.append((fingerprint, root))
    marker.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [{"fingerprint": fp, "root": root} for fp, root in kept],
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    marker.chmod(0o600)


def _enabled_package_fingerprint(workspace: Path, plugin: AgentPlugin) -> str | None:
    """Return the content fingerprint when this exact package copy is enabled.

    [LOCAL PATCH] FR-3.8：改成**多条目**匹配。同一插件的多版副本可以各自被签发
    （升级到 v2 并启用后，v1 的签发仍然有效），这样「市场刷新 / 升级」不会让
    还钉在旧版上的会话突然变成未启用。
    """
    marker = _plugin_data_dir(workspace, plugin.name, create=False) / "enabled"
    try:
        if not marker.is_file():
            return None
    except OSError:
        return None
    entries = _read_activation_entries(marker)
    if not entries:
        logger.warning("Ignoring unreadable activation marker for Agent Plugin '{}'", plugin.name)
        return None
    fingerprint = _package_fingerprint(plugin.root)
    if fingerprint is None:
        # 包正在被改动：算不出指纹，等于这一刻不可信
        _invalidate_skill_cache(workspace)
        return None
    for entry_fingerprint, entry_root in entries:
        if entry_fingerprint == fingerprint:
            return fingerprint
        if entry_fingerprint == "" and entry_root == str(plugin.root):
            # 早期纯路径形态命中：就地升级成带指纹的 v2 形态，下次不必再算一次
            try:
                _write_activation_entries(marker, [(fingerprint, str(plugin.root))])
            except OSError:
                return fingerprint
            return fingerprint
    return None


def _enabled(workspace: Path, plugin: AgentPlugin) -> bool:
    return _enabled_package_fingerprint(workspace, plugin) is not None


def _discover_plugin_skills(plugin_name: str, plugin_root: Path) -> list[tuple[str, Path]]:
    skills_root = _contained(plugin_root / "skills", plugin_root, directory=True)
    if skills_root is None:
        return []

    skills: list[tuple[str, Path]] = []
    for candidate in _children(skills_root, f"Agent Plugin '{plugin_name}' skills"):
        skill_root = _contained(candidate, skills_root, directory=True)
        if skill_root is None:
            continue
        skill_file = _contained(skill_root / "SKILL.md", plugin_root)
        if skill_file is None:
            continue
        try:
            metadata = parse_skill_metadata(skill_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            metadata = None
        if metadata is None or not valid_skill_metadata(metadata, candidate.name):
            logger.warning("Ignoring Agent Plugin '{}' skill '{}': invalid metadata", plugin_name, candidate.name)
            continue
        skills.append((candidate.name, skill_file))
    return skills


def _children(root: Path, label: str) -> list[Path]:
    try:
        return sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        logger.warning("Could not inspect {}: {}", label, exc)
        return []


def _contained(path: Path, root: Path, *, directory: bool = False) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    expected_kind = resolved.is_dir() if directory else resolved.is_file()
    return resolved if expected_kind and resolved.is_relative_to(root) else None


def _read_first_object(root: Path, relpaths: tuple[str, ...]) -> dict[str, object] | None:
    """读取 ``relpaths`` 里第一个**存在**的组件（顺序即优先级）。

    先判存在再解析，是为了让「高优先级位置存在但内容坏」表现为**明确的失败**，
    而不是悄悄回退到低优先级位置——回退会让一份坏清单被另一份好清单掩盖。
    """
    for relpath in relpaths:
        candidate = root / relpath
        if candidate.is_file():
            return _read_object(candidate, root)
    return None


def _read_object(path: Path, root: Path) -> dict[str, object] | None:
    contained = _contained(path, root)
    if contained is None:
        return None
    try:
        value = cast(object, json.loads(contained.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid Agent Plugin component '{}': {}", contained, exc)
        return None
    return cast(dict[str, object], value) if isinstance(value, dict) else None
