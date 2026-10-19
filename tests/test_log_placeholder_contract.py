"""占位符契约守卫：logger 的实现与占位符风格必须匹配。

同一个服务里**并存两套 logger**，而它们的占位符语法不同：

* `from loguru import logger` → **``{}``** 占位符；
* `logging.getLogger(__name__)` → **`%s`** 占位符（惰性 ``% args``）。

写反的后果**不对称**，而且两种都不容易被发现：

* 标准 logging 里写 ``{}`` 并传了实参 → emit 时 `TypeError: not all arguments
  converted during string formatting`。会炸，但只在日志**真的被 handler 处理时**
  才炸——没有 handler 的环境里它安安静静地从不出事。
* loguru 里写 ``%s`` 并传了实参 → **不报错**，占位符原样输出。告警照发，可消息里
  看不到具体是哪个成员 / 哪个模型 / 哪个工具——「留痕」实际上没留下可查的信息。

后者正是本测试的存在理由：它不会红、不会炸，只是让一次本可诊断的故障变得无案可查。
（真实案例：2026-10-06 专家团成员模型_pars 失败时打了 ``unresolvable model '%s'``，
日志里一个具体值都看不到。）

本测试只**报告**——给出「哪种错法 / 文件：行 / 消息模板」，以及应该写成什么样；
不做自动修复（占位符怎么补、补几个值，只有看代码的人知道）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_LEVELS = {"debug", "info", "warning", "error", "critical"}
_ROOT = Path(__file__).resolve().parents[2] / "nanobot"
#: ``{}`` / ``{name}`` / ``{name:>8}`` —— loguru 的占位符形状。刻意不含 ``{%d}``：
#: 那是标准 logging 的常见写法里的字面花括号（如 ``stash@{%d}``），误判会把第三方包
#: 的错误样本当成本仓缺陷。
_BRACE_STYLE = re.compile(r"\{\}|\{[A-Za-z_][\w.:>\[\]-]*\}")


def _violations() -> list[tuple[str, Path, int, str]]:
    """扫 `nanobot/` 包源码，返回 ``(kind, path, lineno, message_template)``。

    ⚠️ 只看**字面量**首参：f-string / 变量拼接不参与占位符契约（它们已经在调用侧
    格式化完了），把那些也算进来会产生大量噪声，反而让真问题被忽略。
    """
    found: list[tuple[str, Path, int, str]] = []
    for source in sorted(_ROOT.rglob("*.py")):
        # 第三方 / 虚拟环境不是本仓契约的一部分，也不该由本仓修
        if ".venv" in source.parts or "site-packages" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        uses_loguru = "from loguru import" in text
        defines_logger = "logging.getLogger" in text or "logger =" in text
        if not (uses_loguru or defines_logger):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - 语法坏掉别的测试会红
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "logger"
                and func.attr in _LEVELS
            ):
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            message, has_args = first.value, len(node.args) > 1
            if not has_args:
                continue
            rel = source.relative_to(_ROOT.parent)
            if not uses_loguru and _BRACE_STYLE.search(message):
                found.append(("standard-logging got {}", rel, node.lineno, message))
            elif uses_loguru and "%s" in message:
                found.append(("loguru got %s", rel, node.lineno, message))
    return found


def test_logger_placeholders_match_their_implementation() -> None:
    violations = _violations()
    if not violations:
        return
    report = "\n".join(
        f"  {kind}: {path}:{lineno}  {message!r}" for kind, path, lineno, message in violations
    )
    raise AssertionError(
        "占位符与该模块的 logger 实现不匹配"
        "（loguru 用 `{}`；`logging.getLogger` 用 `%s`）：\n"
        f"{report}\n"
        "两种错法都不会被人发现：写成 `%s` 的 loguru 日志只是不替换占位符，"
        "写成 `{}` 的标准 logging 日志要等到真有 handler 接管时才抛 TypeError。"
    )


def test_guard_still_scans_the_real_tree() -> None:
    """守卫自身不能变成空跑——包路径挪了/文件名改了会静默失去意义。

    这条断言不依赖「现在是否干净」，只确认扫描器确实看到了成规模的调用点。
    """
    scanned = [p for p in _ROOT.rglob("*.py") if ".venv" not in p.parts]
    assert len(scanned) > 100, f"扫描到的源码文件过少，路径可能失效：{_ROOT}"
    assert _LEVELS and "warning" in _LEVELS
