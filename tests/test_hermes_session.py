"""S3 synthetic tests: session record + hide/list/unhide + middleware flow.

No live Hermes, no network. Synthetic provider-format requests only.
"""

from __future__ import annotations

import json

from adapters.hermes import handlers
from adapters.hermes.allowlist import AdapterRejected, filter_invocation
from adapters.hermes.session import SessionTranscriptStore
from context_hide.engine import ContextHideEngine
from context_hide.model import Scope


def _long_body() -> str:
    lines = [f"Output row {i:03d} details" for i in range(80)]
    lines.extend([
        "id: 01234567-89ab-cdef-0123-456789abcdef",
        "results/output.json generated successfully",
    ])
    return "\n".join(lines)


def _codex_request(body: str, call_id: str = "call_1") -> dict:
    return {
        "input": [
            {"type": "function_call", "call_id": call_id, "name": "read_file",
             "arguments": json.dumps({"path": "results/output.json"})},
            {"type": "function_call_output", "call_id": call_id, "output": body},
        ]
    }


def _chat_request(body: str, call_id: str = "call_2") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "please read the file"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id,
                             "function": {"name": "read_file",
                                          "arguments": json.dumps(
                                              {"path": "results/output.json"})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": body},
        ]
    }


def _scope(session: str) -> Scope:
    return Scope(adapter_id="hermes", host_profile="test", session_id=session)


def test_allowlist_filters_and_rejects():
    invocation, omitted = filter_invocation("read_file", {"path": "a.txt"})
    assert invocation == {"tool_name": "read_file", "arguments": {"path": "a.txt"}}
    assert omitted == {}
    try:
        filter_invocation("evil_tool", {"path": "a.txt"})
    except AdapterRejected as exc:
        assert exc.code == "unsupported_tool"
    else:
        raise AssertionError("expected unsupported_tool")
    try:
        filter_invocation("hide_context", {"tool_call_id": "call_1"})
    except AdapterRejected as exc:
        assert exc.code == "internal_target"
    else:
        raise AssertionError("expected internal_target")


def test_codex_hide_then_two_requests_stay_hidden_then_unhide():
    engine = ContextHideEngine()
    store = SessionTranscriptStore()
    scope = _scope("s3-codex")
    body = _long_body()
    first = _codex_request(body)

    before = json.loads(json.dumps(first))
    step1 = handlers.middleware_step(engine, store, scope, "codex_responses", first)
    assert step1["applied"] == []
    assert first == before

    hidden = handlers.hide(engine, store, scope, "call_1")
    assert hidden.ok is True
    assert hidden.plan is not None
    item_id = hidden.plan.item_id

    step2 = handlers.middleware_step(engine, store, scope, "codex_responses",
                                     _codex_request(body))
    assert step2["applied"] == [item_id]
    out2 = step2["request"]["input"][1]["output"]
    assert out2 != body
    assert item_id in out2

    step3 = handlers.middleware_step(engine, store, scope, "codex_responses",
                                     _codex_request(body))
    assert step3["applied"] == [item_id]
    assert step3["request"]["input"][1]["output"] == out2

    items = handlers.list_records(engine, scope)
    assert len(items) == 1
    assert items[0]["item_id"] == item_id

    restored = handlers.unhide(engine, scope, item_id)
    assert restored.ok is True

    step4 = handlers.middleware_step(engine, store, scope, "codex_responses",
                                     _codex_request(body))
    assert step4["applied"] == []
    assert step4["request"]["input"][1]["output"] == body


def test_chat_hide_and_unhide_exact():
    engine = ContextHideEngine()
    store = SessionTranscriptStore()
    scope = _scope("s3-chat")
    body = _long_body()

    handlers.middleware_step(engine, store, scope, "chat_completions",
                              _chat_request(body))
    hidden = handlers.hide(engine, store, scope, "call_2")
    assert hidden.ok is True
    assert hidden.plan is not None

    step = handlers.middleware_step(engine, store, scope, "chat_completions",
                                    _chat_request(body))
    assert step["applied"] == [hidden.plan.item_id]
    assert step["request"]["messages"][0] == {"role": "user", "content": "please read the file"}
    assert step["request"]["messages"][2]["content"] != body

    handlers.unhide(engine, scope, hidden.plan.item_id)
    again = handlers.middleware_step(engine, store, scope, "chat_completions",
                                     _chat_request(body))
    assert again["applied"] == []
    assert again["request"]["messages"][2]["content"] == body


def test_hide_not_found_ambiguous_and_unsupported():
    engine = ContextHideEngine()
    store = SessionTranscriptStore()
    scope = _scope("s3-edge")

    missing = handlers.hide(engine, store, scope, "call_missing")
    assert missing.ok is False and missing.error == "not_found"

    dup_body_a = _long_body()
    dup_body_b = _long_body() + "\nrow 081 extra details"
    dup = {
        "input": [
            {"type": "function_call", "call_id": "call_d", "name": "read_file",
             "arguments": json.dumps({"path": "a.txt"})},
            {"type": "function_call_output", "call_id": "call_d", "output": dup_body_a},
            {"type": "function_call_output", "call_id": "call_d", "output": dup_body_b},
        ]
    }
    handlers.middleware_step(engine, store, scope, "codex_responses", dup)
    ambiguous = handlers.hide(engine, store, scope, "call_d")
    assert ambiguous.ok is False and ambiguous.error == "ambiguous"

    evil = {
        "input": [
            {"type": "function_call", "call_id": "call_e", "name": "evil_tool",
             "arguments": json.dumps({"path": "a.txt"})},
            {"type": "function_call_output", "call_id": "call_e", "output": _long_body()},
        ]
    }
    handlers.middleware_step(engine, store, _scope("s3-evil"), "codex_responses", evil)
    rejected = handlers.hide(engine, store, _scope("s3-evil"), "call_e")
    assert rejected.ok is False and rejected.error == "unsupported_tool"


def test_middleware_unsupported_mode_leaves_request():
    engine = ContextHideEngine()
    store = SessionTranscriptStore()
    scope = _scope("s3-unsupported")
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    step = handlers.middleware_step(engine, store, scope, "anthropic_messages", payload)
    assert step["request"] == payload
    assert step["applied"] == []
    assert step["trace"]["reason"] == "unsupported_api_mode"


def test_session_bounds_ttl_and_capacity():
    now = [1000.0]
    store = SessionTranscriptStore(max_sessions=2, max_results_per_session=1,
                                   ttl_seconds=60, time_func=lambda: now[0])
    scope_a = _scope("a")
    scope_b = _scope("b")
    scope_c = _scope("c")
    two_calls = {
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": json.dumps({"path": "a.txt"})},
            {"type": "function_call_output", "call_id": "call_1", "output": "body one"},
            {"type": "function_call", "call_id": "call_2", "name": "read_file",
             "arguments": json.dumps({"path": "b.txt"})},
            {"type": "function_call_output", "call_id": "call_2", "output": "body two"},
        ]
    }
    kept, _, _ = store.record_request(scope_a.scope_key(), "codex_responses", two_calls)
    assert len(kept) == 1
    store.record_request(scope_b.scope_key(), "codex_responses", _codex_request("body b"))
    store.record_request(scope_c.scope_key(), "codex_responses", _codex_request("body c"))
    assert len(store) == 2
    assert store.entries(scope_a.scope_key()) == []
    now[0] += 61.0
    assert store.entries(scope_b.scope_key()) == []
