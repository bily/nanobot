"""Load and render agent system prompt templates (Jinja2) under nanobot/templates/.

Agent prompts live in ``templates/agent/`` (pass names like ``agent/identity.md``).
Shared copy lives under ``agent/_snippets/`` and is included via
``{% include 'agent/_snippets/....md' %}``.

[LOCAL PATCH] 提示词外置（design §13.4 / ADR-007）
------------------------------------------------
调用方可传 ``workspace=``：此时先看 ``<workspace>/prompts/<name>`` 是否存在且
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
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateError
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


@lru_cache
def _environment() -> Environment:
    # Plain-text prompts: do not HTML-escape variable values.
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


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
    **kwargs: Any,
) -> str:
    """Render ``name`` (e.g. ``agent/identity.md``, ``agent/platform_policy.md``) under ``templates/``.

    Use ``strip=True`` for single-line user-facing strings when the file ends
    with a trailing newline you do not want preserved.

    [LOCAL PATCH] 传入 ``override_root=`` 时优先渲染 ``<override_root>/prompts/<name>``
    （限 ``OVERRIDABLE_TEMPLATES``）。覆盖文件渲染失败 → 记一条 warn 并回退内置，
    绝不因此抛错（见模块 docstring 第 2 条）。参数名不叫 ``workspace`` 是为了避开
    模板变量同名冲突，见模块 docstring 的 ⚠️。
    """
    if override_root is not None and name in OVERRIDABLE_TEMPLATES:
        override = load_workspace_template(override_root, name)
        if override is not None:
            try:
                text = _environment().from_string(override).render(**kwargs)
                return text.rstrip() if strip else text
            except TemplateError as exc:
                logger.warning(
                    "workspace prompt override {} failed to render ({}); "
                    "falling back to bundled template",
                    name,
                    exc,
                )

    text = _environment().get_template(name).render(**kwargs)
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
