"""Unit tests for ContextHideEngine high-level orchestrator and contracts."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from context_hide.engine import ContextHideEngine
from context_hide.model import (
    EngineConfig,
    Scope,
    ToolResultRecord,
    compute_invocation_digest,
    compute_sha256,
)
from context_hide.transport import SummarizerError


def _make_record(
    content: str,
    call_id: str = "call_1",
    tool_name: str = "terminal",
    host_handle: Any = None,
    position: int = 0,
    invocation_args: dict[str, Any] | None = None,
) -> ToolResultRecord:
    inv = {"tool_name": tool_name, "arguments": invocation_args or {"command": "ls"}}
    return ToolResultRecord(
        call_id=call_id,
        result_position=position,
        content=content,
        content_sha256=compute_sha256(content),
        invocation=inv,
        invocation_digest=compute_invocation_digest(inv),
        status="ok",
        host_handle=host_handle,
    )


def _long_content() -> str:
    lines = [f"Output row {i:03d} details" for i in range(80)]
    lines.extend([
        "id: 01234567-89ab-cdef-0123-456789abcdef",
        "results/output.json generated successfully",
    ])
    return "\n".join(lines)


def test_engine_hide_with_rule_fallback():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_1")
    record = _make_record(_long_content())

    result = engine.hide_sync(scope, record)
    assert result.ok is True
    assert result.plan is not None
    assert result.plan.item_id.startswith("item_")
    assert result.plan.expected_content_sha256 == record.content_sha256
    assert "원문 발췌:" in result.plan.replacement_text


def test_engine_hide_with_summarizer():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_2")
    record = _make_record(_long_content())

    class MockSummarizer:
        async def summarize(self, original: str, required_evidence: tuple[str, ...], on_call=None, *, invocation=None, context=None):
            return {"summary": "Generated summary with id: 01234567-89ab-cdef-0123-456789abcdef and results/output.json."}

    result = asyncio.run(engine.hide(scope, record, MockSummarizer()))
    assert result.ok is True
    assert result.plan is not None
    assert "생성 요약" in result.plan.replacement_text
    assert result.item is not None
    assert result.item.compaction_source == "lfm"


def test_engine_hide_preserves_host_handle():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_3")
    handle = {"turn_id": 4, "subagent": "worker"}
    record = _make_record(_long_content(), host_handle=handle)

    result = engine.hide_sync(scope, record)
    assert result.ok is True
    assert result.plan is not None
    assert result.plan.host_handle == handle


def test_engine_hide_fails_open_on_protected_error():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_4")
    error_content = "Traceback (most recent call last):\n  File 'app.py', line 10\nZeroDivisionError\n" + _long_content()
    record = _make_record(error_content)

    result = engine.hide_sync(scope, record)
    assert result.ok is False
    assert result.error == "protected_error"
    assert engine.list_items(scope) == []


def test_engine_hide_fails_open_on_lfm_failure_fallback_to_rule():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_5")
    record = _make_record(_long_content())

    class FailingSummarizer:
        async def summarize(self, original: str, required_evidence: tuple[str, ...], on_call=None, *, invocation=None, context=None):
            raise SummarizerError("simulated timeout")

    result = asyncio.run(engine.hide(scope, record, FailingSummarizer()))
    assert result.ok is True
    assert result.plan is not None
    assert result.item is not None
    assert result.item.compaction_source == "rule"


def test_engine_unhide_restores_visibility_without_reexecution():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_6")
    record = _make_record(_long_content())

    hide_res = engine.hide_sync(scope, record)
    assert hide_res.ok and hide_res.plan is not None

    plans_before = engine.project(scope, [record])
    assert len(plans_before) == 1

    unhide_res = engine.unhide(scope, hide_res.plan.item_id)
    assert unhide_res.ok

    # Unhidden item produces no replacement plan (original remains in transcript)
    plans_after = engine.project(scope, [record])
    assert plans_after == []


def test_engine_project_generates_replacement_plans_for_compacted_items():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_7")

    rec1 = _make_record(_long_content() + "\n1", call_id="c1", host_handle=1)
    rec2 = _make_record(_long_content() + "\n2", call_id="c2", host_handle=2)
    rec3 = _make_record(_long_content() + "\n3", call_id="c3", host_handle=3)

    res1 = engine.hide_sync(scope, rec1)
    res2 = engine.hide_sync(scope, rec2)
    res3 = engine.hide_sync(scope, rec3)

    # Unhide rec2
    engine.unhide(scope, res2.plan.item_id)

    plans = engine.project(scope, [rec1, rec2, rec3])
    assert len(plans) == 2
    handles = [p.host_handle for p in plans]
    assert handles == [1, 3]


def test_engine_list_items_filtering_and_projection():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_8")

    rec1 = _make_record(_long_content() + "\napple finding", call_id="c1")
    rec2 = _make_record(_long_content() + "\nbanana finding", call_id="c2")

    engine.hide_sync(scope, rec1)
    engine.hide_sync(scope, rec2)

    all_items = engine.list_items(scope)
    assert len(all_items) == 2

    filtered = engine.list_items(scope, query="apple")
    assert len(filtered) == 1
    assert filtered[0].call_id == "c1"


def test_engine_budget_one_summary_call_per_turn():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_9")
    record = _make_record(_long_content())
    call_count = [0]

    class CountSummarizer:
        async def summarize(self, original: str, required_evidence: tuple[str, ...], on_call=None, *, invocation=None, context=None):
            call_count[0] += 1
            return {"summary": "summary with id: 01234567-89ab-cdef-0123-456789abcdef and results/output.json"}

    res = asyncio.run(engine.hide(scope, record, CountSummarizer()))
    assert res.ok
    assert call_count[0] == 1


def test_engine_invalid_invocation_fails_closed():
    engine = ContextHideEngine()
    scope = Scope("bridge", "test", "sess_10")

    # Negative result position
    invalid_pos = _make_record(_long_content(), position=-1)
    res_pos = engine.hide_sync(scope, invalid_pos)
    assert not res_pos.ok
    assert res_pos.error == "invalid_invocation_order"

    # Sensitive pattern in invocation arguments
    sensitive_rec = _make_record(
        _long_content(),
        invocation_args={"token": "Bearer secret_token_xyz"},
    )
    res_sens = engine.hide_sync(scope, sensitive_rec)
    assert not res_sens.ok
    assert res_sens.error == "sensitive_invocation"


def test_engine_scope_isolation():
    engine = ContextHideEngine()
    scope_a = Scope("bridge", "test", "session_A")
    scope_b = Scope("bridge", "test", "session_B")

    record = _make_record(_long_content())

    res_a = engine.hide_sync(scope_a, record)
    assert res_a.ok

    assert len(engine.list_items(scope_a)) == 1
    assert len(engine.list_items(scope_b)) == 0

    assert len(engine.project(scope_a, [record])) == 1
    assert len(engine.project(scope_b, [record])) == 0
