"""Load and render agent system prompt templates (Jinja2) under nanobot/templates/.

Agent prompts live in ``templates/agent/`` (pass names like ``agent/identity.md``).
Shared copy lives under ``agent/_snippets/`` and is included via
``{% include 'agent/_snippets/....md' %}``.

[LOCAL PATCH] 提示词外置（design §13.4 / ADR-007）
------------------------------------------------
调用方可传 ``override_root=``：此时先看 ``<workspace>/prompts/<name>`` 是否存在且
非空——存在则用**它**渲染，否则回退包内只读模板。目标是「改提示词不必重新发版」，
让提示词成为可 diff、可评审、可 A/B 的资产。

两个刻意的边界：

1. **只有白名单模板可被覆盖**（``OVERRIDABLE_TEMPLATES``）。覆盖面是显式的，
   否则 ``_snippets/`` 这类被 include 的片段也能被顶掉，安全审查无处落脚。
2. **覆盖文件坏掉时回退内置**（fail-open 到只读模板）。提示词语法错误不该让
   会话直接起不来——那是把「可运营」做成了「可宕机」。

⚠️ 参数名刻意叫 ``override_root`` 而**不是** ``workspace``：``agent/subagent_system.md``
等模板自身就有名为 ``workspace`` 的变量，若参数同名会被函数签名吃掉，``{{ workspace }}``
静默渲染成空串——这种失败不报错，只表现为子代理提示词缺一块。改名前先全量
``grep '{{ workspace' templates/``。

与 ``workspace_prompts.py`` 的分工：那边是**无变量**的纯文本文档（dream /
evaluator 的人写提示词），这边是**带变量**的 Jinja2 模板。两者不要混用。

[LOCAL PATCH] FR-3.10 插件提示词片段：单 loader → ChainLoader
----------------------------------------------------------
原先只有「工作区覆盖 / 内置」两级，且覆盖是 ``from_string`` —— 而
``Environment.from_string`` 产出的模板**没有 loader**，于是覆盖文件里写
``{% include %}`` 一律解析不到。第三方因此无法在不改引擎代码的前提下注入领域规程。

现在两级都做成 **loader**，按顺序串成 ChainLoader：

    ① `<plugin-name>/<relpath>`（插件包内 ``prompts/`` 目录）
    ② `<override_root>/prompts/<name>`（工作区覆盖，限白名单）
    ③ 引擎内置

命名空间就是**插件名**，因为插件身份在引擎里已经唯一（``_PLUGIN_NAME`` 正则 +
同名去重）。不加命名空间的话，两个插件都可能提供 ``domain.md``，先加载的那个
**静默**赢——那种错只能靠人比对提示词全文才发现。

信任分级（FR-3.10 ④，防 R-5 提示词注入）：平铺安装的插件算 ``trusted`` 直注；
市场缓存副本算 ``user``，其片段一律包 ``<untrusted_content>``。标签含义由
``agent/_snippets/untrusted_content.md`` 说明，而那一段**一直在**系统提示词里
（``agent/identity.md`` 与 ``agent/subagent_system.md`` 都 include 了它），
所以不必在包裹处再解释一遍。

⚠️ **自定义 loader 的 `get_source` 找不到时必须抛 `TemplateNotFound`，不能 `return None`**：
Jinja 的 ``BaseLoader.load`` 会去解包返回值，返回 None 会抛
``TypeError: cannot unpack non-iterable NoneType object``，而 ``ChoiceLoader``
**只捕 `TemplateNotFound`** → 这个 TypeError 会一路冒到业务代码，报的还是一个
跟模板名毫无关系的错。这条踩过（见 ``test_plugin_prompt_fragments.py``）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    TemplateError,
    TemplateNotFound,
)
from loguru import logger

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"

# [LOCAL PATCH] 允许被 ``<workspace>/prompts/`` 覆盖的模板名（显式白名单）。
# 新增一项前先确认：该模板的变量只来自 build_system_prompt 的调用方，
# 且改坏它不会让整轮会话起不来（有回退兜底）。
OVERRIDABLE_TEMPLATES: tuple[str, ...] = (
    "agent/identity.md",
    "agent/tool_contract.md",
    "agent/skills_section.md",
    "agent/platform_policy.md",
    "agent/subagent_system.md",
)

#: 不可信片段的包裹标签（FR-3.10 ④）。含义见 ``agent/_snippets/untrusted_content.md``。
UNTRUSTED_TAG = "untrusted_content"


@dataclass(frozen=True, slots=True)
class PluginPromptMount:
    """一个插件的提示词挂载点。

    模板名形如 ``<name>/<包内相对路径>``（例：``demo/fragments/domain.md``）——
    基准是**包根**而不是声明的那几个目录。理由是可读性与可预测性：用户看到的
    路径与插件包里的目录结构逐字对应，``fragments/`` 出现在名字里，而不是被
    声明位置「吃掉」成 ``demo/domain.md``（同一份片段在 ``prompts: "fragments"``
    与 ``prompts: "."`` 两种声明下会变成两个名字，那才是真难查）。

    ``directories`` 是**允许读取的范围**：清单声明并经 ``_contained()`` 校验过的
    那几个目录。解析出的文件必须落在其中之一 —— 声明了 ``fragments/`` 的插件
    不该能读到包里的 ``mcp/``。
    """

    name: str
    #: 包根（namespace 的基准）
    root: Path
    #: 允许读取的目录（清单声明 + 越界校验后的结果）
    directories: tuple[Path, ...]
    #: ``trusted`` 直注 / ``user`` 包 ``<untrusted_content>``
    trust: str = "trusted"


def wrap_untrusted(source: str) -> str:
    """把不可信来源的片段包起来。

    **不做转义**：这是纯文本提示词而不是 HTML，把 ``<`` 转成 ``&lt;`` 只会让模型
    读到转义符，反而更难读。边界标记的作用是「告诉模型这段从哪来」。
    """
    return f"<{UNTRUSTED_TAG}>\n{source.rstrip()}\n</{UNTRUSTED_TAG}>"


def _resolve_prompt_file(
    root: Path,
    directories: tuple[Path, ...],
    relative: str,
) -> Path | None:
    """``root/<relative>`` → 文件；越界 / 不在声明目录内 / 不存在 一律 ``None``。

    两道判据缺一不可：``is_relative_to(root)`` 拦 ``..`` 逃出包外，
    ``is_relative_to(directories[…]`` 拦「声明了 ``fragments/`` 却去读 ``mcp/``」。
    只做前者的话，插件清单里一句 ``prompts: "fragments"`` 就变成了「整个包可读」。
    """
    if not relative or relative.startswith(("/", "\\")) or ":" in relative:
        # 绝对路径（含 Windows 盘符形如 `C:\...`）直接拒
        return None
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not resolved.is_relative_to(root):
        # `..` 逃逸在这里被拦下；root 在挂载时已 resolve 过
        return None
    if not any(resolved.is_relative_to(directory) for directory in directories):
        return None
    return resolved


class PluginPromptLoader(BaseLoader):
    """把插件包内 ``prompts/`` 目录挂到 ``<plugin-name>/`` 虚拟命名空间下。"""

    def __init__(self, mounts: Sequence[PluginPromptMount]) -> None:
        self._mounts = tuple(mounts)

    def get_source(
        self,
        environment: Environment,
        template: str,
    ) -> tuple[str, str, Any]:
        for mount in self._mounts:
            prefix = f"{mount.name}/"
            if not template.startswith(prefix):
                continue
            resolved = _resolve_prompt_file(
                mount.root, mount.directories, template[len(prefix):]
            )
            if resolved is None:
                # 命名空间命中但文件不在 → 抛（不能 return None），让下一级接手。
                # 契约细节见模块 docstring 末尾的 ⚠️。
                raise TemplateNotFound(name=template)
            try:
                source = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Agent Plugin prompt fragment {} is unreadable ({}); ignoring",
                    resolved,
                    exc,
                )
                raise TemplateNotFound(name=template) from exc
            if mount.trust == "user":
                source = wrap_untrusted(source)
            return source, str(resolved), lambda: False
        raise TemplateNotFound(name=template)


class WorkspaceOverrideLoader(BaseLoader):
    """``<workspace>/prompts/<name>`` 的外置覆盖（仅白名单模板）。

    做成 loader 而不是 ``from_string``：覆盖文件里会写 ``{% include %}``
    （把插件片段拼进主提示词正是它的用法），而 ``from_string`` 生成的模板
    **没有 loader**，include 一律解析不到。白名单与「空文件视为未覆盖」的语义
    在这里与 ``load_workspace_template`` 保持一致。
    """

    def __init__(self, root: Path | str) -> None:
        self._root = root

    def get_source(
        self,
        environment: Environment,
        template: str,
    ) -> tuple[str, str, Any]:
        if template not in OVERRIDABLE_TEMPLATES:
            raise TemplateNotFound(name=template)
        text = load_workspace_template(self._root, template)
        if text is None:
            raise TemplateNotFound(name=template)
        return text, str(workspace_template_path(self._root, template)), lambda: False


@lru_cache
def _environment() -> Environment:
    """默认环境（无覆盖、无插件挂载）。保持与引入 ChainLoader 之前**完全一致**。"""
    # Plain-text prompts: do not HTML-escape variable values.
    return _build_environment(())


@lru_cache
def _environment_for(
    override_root: str | None,
    mounts: tuple[PluginPromptMount, ...],
) -> Environment:
    """按「覆盖 + 插件挂载」缓存环境。

    ``Environment`` 持有模板编译缓存，每次渲染新建一个会把编译开销重付一遍；
    而把所有变体塞进同一个 lru_cache 又会让**无挂载**（最常见）的那条路也变成
    「带参」形态。故：无参走 ``_environment()``，有参才走这里。
    """
    loaders: list[BaseLoader] = []
    if mounts:
        loaders.append(PluginPromptLoader(mounts))
    if override_root is not None:
        loaders.append(WorkspaceOverrideLoader(override_root))
    if not loaders:
        return _environment()
    # ⚠️ 内置 loader **必须始终在链尾**。少了它会退化成「只有一两级」：链上唯一那个
    # loader 抛 `TemplateNotFound` 时没有任何人接手，`agent/platform_policy.md`
    # 这类内置模板会直接解析失败 —— 而报错看起来像「模板文件不见了」，
    # 真因却是「忘了把兜底那一级接上」。
    loaders.append(FileSystemLoader(str(_TEMPLATES_ROOT)))
    return _build_environment(loaders)


def _build_environment(loaders: Sequence[BaseLoader]) -> Environment:
    loader: BaseLoader | ChoiceLoader
    if not loaders:
        loader = FileSystemLoader(str(_TEMPLATES_ROOT))
    elif len(loaders) == 1:
        loader = loaders[0]
    else:
        loader = ChoiceLoader(list(loaders))
    return Environment(
        loader=loader,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _select_environment(
    override_root: Path | str | None,
    mounts: tuple[PluginPromptMount, ...],
) -> Environment:
    if override_root is None and not mounts:
        return _environment()
    root_key = None
    if override_root is not None:
        root_key = str(Path(override_root).expanduser())
    return _environment_for(root_key, mounts)


def workspace_template_path(workspace: Path | str, name: str) -> Path:
    """[LOCAL PATCH] 外置覆盖的约定路径：``<workspace>/prompts/<name>``。

    刻意**保留子目录与扩展名**（``agent/identity.md`` → ``prompts/agent/identity.md``），
    这样用户看到的是与包内模板一一对应的树，而不是一串被压平的名字。父目录不存在
    时由写入方负责 ``mkdir(parents=True)``。
    """
    return Path(workspace).expanduser() / "prompts" / name


def load_workspace_template(workspace: Path | str, name: str) -> str | None:
    """读取外置覆盖原文；缺失 / 空文件 / 读不了 一律返回 ``None``（调用方回退内置）。"""
    path = workspace_template_path(workspace, name)
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def render_template(
    name: str,
    *,
    strip: bool = False,
    override_root: Path | str | None = None,
    plugin_mounts: Sequence[PluginPromptMount] | None = None,
    **kwargs: Any,
) -> str:
    """Render ``name`` (e.g. ``agent/identity.md``, ``agent/platform_policy.md``) under ``templates/``.

    Use ``strip=True`` for single-line user-facing strings when the file ends
    with a trailing newline you do not want preserved.

    [LOCAL PATCH] 查找顺序：**插件命名空间 → 工作区覆盖 → 内置**。

    ``plugin_mounts=`` 给出插件片段挂载点（``<plugin>/<relpath>``）。它同时进环境，
    因此**内置模板或工作区覆盖里的 ``{% include '<plugin>/...' %}`` 也能解析到**
    —— 只把插件当「另一个可覆盖的根」是不够的，那表达不了「把插件的领域规程
    include 进主提示词」这个唯一真实用法。
    """
    mounts = tuple(plugin_mounts or ())
    # 先**预检**覆盖是否存在：这次的判据不是「覆盖有没有被用上」（loader 自己会挑），
    # 而是「渲染失败时该不该把它算在覆盖头上」。少了这个预检，内置模板里的一处
    # TemplateNotFound 也会被当成「覆盖坏了」，于是真因被一条「已回退」的告警埋掉。
    has_override = (
        override_root is not None
        and name in OVERRIDABLE_TEMPLATES
        and load_workspace_template(override_root, name) is not None
    )

    text: str | None = None
    if has_override:
        try:
            text = (
                _select_environment(override_root, mounts).get_template(name).render(**kwargs)
            )
        except TemplateError as exc:
            # 覆盖整份渲染失败（语法错、include 不到、变量报错）→ fail-open 回内置。
            # `TemplateNotFound` 也在这里面：覆盖里 include 一个不存在的片段，
            # 属于「这份覆盖写坏了」，不是「引擎的内置模板不见了」。
            logger.warning(
                "workspace prompt override {} failed to render ({}); "
                "falling back to bundled template",
                name,
                exc,
            )
            text = None

    if text is None:
        try:
            text = _select_environment(None, mounts).get_template(name).render(**kwargs)
        except TemplateNotFound:
            # 只吞「插件命名空间下缺片段」这一种：片段是可选的，缺它不该把整段提示词
            # 打崩。判据必须收紧到「首段命中了某个挂载名」——写成 `"/" in name`
            # 会把 `agent/xxx.md` 这类**内置模板名写错**也一起吞掉，那种错要暴露出来。
            if any(name.startswith(f"{mount.name}/") for mount in mounts):
                logger.warning("prompt fragment {} not found; rendering empty", name)
                return ""
            raise
    return text.rstrip() if strip else text


def initialize_workspace_templates(
    workspace: Path | str,
    names: Iterable[str] | None = None,
) -> list[str]:
    """[LOCAL PATCH] 把可选覆盖模板的**默认副本**落到 ``<workspace>/prompts/``。

    语义与 ``workspace_prompts.initialize_workspace_prompt`` 一致：**不覆盖已有
    非空文件**，因此可以反复调用（幂等），用户改过的副本不会被还原。

    Returns:
        本次实际新建的相对模板名（如 ``["agent/identity.md"]``）。
    """
    created: list[str] = []
    for name in names if names is not None else OVERRIDABLE_TEMPLATES:
        if name not in OVERRIDABLE_TEMPLATES:
            continue
        source = _TEMPLATES_ROOT / name
        try:
            default_text = source.read_text(encoding="utf-8")
        except OSError:
            continue

        target = workspace_template_path(workspace, name)
        try:
            if target.is_file() and target.read_text(encoding="utf-8").strip():
                continue  # 用户已有内容，绝不覆盖
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(default_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        created.append(name)
    return created
