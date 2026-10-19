"""canonical_feishu_section 对段级策略键（tool_approval）的统一契约。

背景（真机事故 2026-10-19）：客户端把 ``tool_approval`` 写在 feishu 段级，
网关启动时 ``sync_saved_feishu_identity_boundary`` 走 upsert → canonical 回写，
段级键被丢弃、又经 inherited 泄漏进每个实例——引擎自己的读取端
（``channel_tool_approval_modes`` 只看段级）随即失效，审批静默变回 auto。
统一后的契约：**段级是唯一权威位置**——不下放实例、canonical 回写不丢弃。
"""

import pytest

from nanobot.channels.feishu.config import feishu_default_config
from nanobot.channels.feishu.instances import canonical_feishu_section


def test_canonical_preserves_section_level_tool_approval() -> None:
    section = {
        "tool_approval": "ask",
        "instances": [{"id": "feishu", "instanceId": "feishu", "appId": "cli_x"}],
    }
    out = canonical_feishu_section(section, feishu_default_config())
    assert out["tool_approval"] == "ask"
    for instance in out["instances"]:
        assert "tool_approval" not in instance


def test_canonical_promotes_instance_level_tool_approval() -> None:
    """存量污染形态（实例里有、段级没有）→ 回收到段级，实例上一份不剩。"""
    section = {
        "instances": [
            {"id": "feishu", "instanceId": "feishu", "appId": "cli_x", "tool_approval": "ask"},
            {"id": "lark", "instanceId": "lark", "domain": "lark", "appId": "cli_y", "tool_approval": "ask"},
        ],
    }
    out = canonical_feishu_section(section, feishu_default_config())
    assert out["tool_approval"] == "ask"
    for instance in out["instances"]:
        assert "tool_approval" not in instance


def test_canonical_instance_level_ignored_when_section_level_present() -> None:
    """段级已有值时实例级副本直接剥掉，不覆盖段级（权威唯一）。"""
    section = {
        "tool_approval": "auto",
        "instances": [{"id": "feishu", "tool_approval": "ask"}],
    }
    out = canonical_feishu_section(section, feishu_default_config())
    assert out["tool_approval"] == "auto"
    assert "tool_approval" not in out["instances"][0]


def test_canonical_still_rejects_duplicate_instance_ids() -> None:
    """统一不影响原有校验：重复 instance id 仍拒绝。"""
    section = {
        "instances": [
            {"id": "feishu", "instanceId": "feishu"},
            {"id": "feishu", "instanceId": "feishu"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        canonical_feishu_section(section, feishu_default_config())
