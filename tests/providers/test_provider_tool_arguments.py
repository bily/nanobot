"""Shared tool-argument parsing policy tests."""

from nanobot.providers.base import (
    ToolCallRequest,
    parse_tool_arguments,
    tool_arguments_json_for_replay,
    tool_arguments_object_for_replay,
)


def test_parse_tool_arguments_preserves_malformed_executable_arguments() -> None:
    assert parse_tool_arguments('{path:"foo.txt"}') == '{path:"foo.txt"}'


def test_parse_tool_arguments_preserves_non_object_executable_arguments() -> None:
    assert parse_tool_arguments('["foo.txt"]') == ["foo.txt"]
    assert parse_tool_arguments("false") is False
    assert parse_tool_arguments("null") == "null"


def test_tool_arguments_object_for_replay_repairs_object_like_history_arguments() -> None:
    assert tool_arguments_object_for_replay('{path:"foo.txt"}') == {"path": "foo.txt"}


def test_tool_arguments_object_for_replay_keeps_history_object_shaped() -> None:
    for arguments in ['["foo.txt"]', "false", "null", "0", ["foo.txt"], False, None, 0]:
        assert tool_arguments_object_for_replay(arguments) == {}


def test_tool_arguments_json_for_replay_returns_object_string() -> None:
    assert tool_arguments_json_for_replay('{path:"foo.txt"}') == '{"path": "foo.txt"}'


def test_has_usable_arguments_accepts_parsed_objects() -> None:
    """provider 解析后的对象（含 `parse_tool_arguments` 的产物）才是可执行参数。"""
    for arguments in [{}, {"path": "foo.txt"}, parse_tool_arguments('{"path":"foo.txt"}')]:
        call = ToolCallRequest(id="1", name="read_file", arguments=arguments)
        assert call.has_usable_arguments()


def test_has_usable_arguments_accepts_empty_payload_as_no_arg_call() -> None:
    """空 / None = 无参调用，是合法形状（不该被 malformed 守卫丢掉）。"""
    for arguments in [None, "", "   ", {}]:
        call = ToolCallRequest(id="1", name="list_exec_sessions", arguments=arguments)
        assert call.has_usable_arguments()


def test_has_usable_arguments_rejects_truncated_and_non_object_payloads() -> None:
    """与 `has_valid_name` 对称的守卫：解析不出对象的调用一律判不可用。

    最要紧的是第一条 —— 被输出长度上限截断的 JSON（截断点停在字符串里面），
    它进不了 `json.loads`，若被持久化还会被 `json_repair` 修成假参数重放。
    """
    truncated = '{"path": "a.html", "content": "<!DOCTYPE html>…function setDas(dir) { das'
    for arguments in [
        truncated,  # 被输出上限截断的对象字面量
        '{path:"foo.txt"}',  # 非法 JSON（键未加引号）
        "null",  # 合法 JSON，但不是对象
        ["foo.txt"],  # 数组
        False,  # 标量
        0,
    ]:
        call = ToolCallRequest(id="1", name="write_file", arguments=arguments)
        assert not call.has_usable_arguments(), f"{arguments!r} 应判为不可用"
