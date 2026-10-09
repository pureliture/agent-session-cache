"""S2 synthetic tests: Hermes payload find/replace. No live Hermes, no network."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def _load_payload():
    repo = Path(__file__).resolve().parents[1]
    module_path = repo / "adapters" / "hermes" / "payload.py"
    spec = importlib.util.spec_from_file_location("hermes_adapter_payload", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_adapter_payload"] = module
    spec.loader.exec_module(module)
    return module


payload = _load_payload()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_codex_find_text_output():
    request = {
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1",
             "output": "line one\nline two\n"},
        ]
    }
    results, skipped, error = payload.find_tool_results("codex_responses", request)
    assert error is None
    assert skipped == []
    assert len(results) == 1
    assert results[0].call_id == "call_1"
    assert results[0].content == "line one\nline two\n"
    assert results[0].host_handle == {
        "api_mode": "codex_responses", "index": 1, "kind": "function_call_output"}


def test_codex_parts_output_skipped():
    request = {
        "input": [
            {"type": "function_call_output", "call_id": "call_9",
             "output": [{"type": "input_text", "text": "hello"}]},
        ]
    }
    results, skipped, error = payload.find_tool_results("codex_responses", request)
    assert error is None
    assert results == []
    assert len(skipped) == 1
    assert skipped[0].reason == "parts_output"


def test_chat_find_tool_message():
    request = {
        "messages": [
            {"role": "user", "content": "please read the file"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_2", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "file body here"},
        ]
    }
    results, skipped, error = payload.find_tool_results("chat_completions", request)
    assert error is None
    assert skipped == []
    assert len(results) == 1
    assert results[0].call_id == "call_2"
    assert results[0].host_handle["index"] == 2


def test_chat_non_text_content_skipped():
    request = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_3", "content": ["part1", "part2"]},
        ]
    }
    results, skipped, error = payload.find_tool_results("chat_completions", request)
    assert error is None
    assert results == []
    assert skipped[0].reason == "non_text_content"


def test_unsupported_modes_pass_through():
    for mode in ("anthropic_messages", "bedrock_converse", "codex_app_server"):
        request = {"input": [], "messages": []}
        results, skipped, error = payload.find_tool_results(mode, request)
        assert results == [] and skipped == []
        assert error == "unsupported_api_mode"
        new_payload, applied, apply_skipped = payload.apply_replacements(
            mode, request, [])
        assert new_payload == request and applied == [] and apply_skipped == []


def test_apply_replaces_only_target_and_keeps_original():
    body = "original tool body " + "x" * 40
    request = {
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": body},
        ]
    }
    before = {"input": [dict(item) for item in request["input"]]}
    results, _, _ = payload.find_tool_results("codex_responses", request)
    plan = {
        "item_id": "item_0123456789abcdef",
        "host_handle": dict(results[0].host_handle),
        "expected_content_sha256": _sha(body),
        "replacement_text": "[hidden:item_0123456789abcdef] short summary",
    }
    new_payload, applied, skipped = payload.apply_replacements(
        "codex_responses", request, [plan])
    assert applied == ["item_0123456789abcdef"]
    assert skipped == []
    assert new_payload["input"][1]["output"] == "[hidden:item_0123456789abcdef] short summary"
    assert new_payload["input"][0] == request["input"][0]
    assert request == before


def test_apply_hash_mismatch_keeps_original():
    request = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_7", "content": "current body"},
        ]
    }
    plan = {
        "item_id": "item_aaaaaaaaaaaaaaaa",
        "host_handle": {"api_mode": "chat_completions", "index": 0, "kind": "tool_message"},
        "expected_content_sha256": _sha("different body"),
        "replacement_text": "[hidden:item_aaaaaaaaaaaaaaaa] summary",
    }
    new_payload, applied, skipped = payload.apply_replacements(
        "chat_completions", request, [plan])
    assert applied == []
    assert skipped[0]["reason"] == "hash_mismatch"
    assert new_payload["messages"][0]["content"] == "current body"


def test_apply_bad_handle_skipped():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    plan = {
        "item_id": "item_bbbbbbbbbbbbbbbb",
        "host_handle": {"api_mode": "chat_completions", "index": 5, "kind": "tool_message"},
        "expected_content_sha256": _sha("hello"),
        "replacement_text": "replacement",
    }
    new_payload, applied, skipped = payload.apply_replacements(
        "chat_completions", request, [plan])
    assert applied == []
    assert skipped[0]["reason"] == "handle_mismatch"
    assert new_payload == request
