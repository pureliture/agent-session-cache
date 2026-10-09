"""Hermes adapter request flow and tool handlers (M2-2).

Middleware flow per outbound request:
1. Record the transmitted (call ID -> tool name/args/result body/location)
   list in the session store under this scope.
2. Ask the engine for replacement plans.
3. Apply plans to a copy and return it, never the stored history.

Tool flow (model calls): look the tool call ID up in the session record by
scope + ID. Reject when absent or duplicated, otherwise call the engine.
"""

from __future__ import annotations

from typing import Any

from .allowlist import AdapterRejected, filter_invocation
from .payload import UNSUPPORTED_API_MODE, apply_replacements
from .session import SessionTranscriptStore
from context_hide.engine import ContextHideEngine
from context_hide.model import (
    MutationResult,
    Scope,
    ToolResultRecord,
    compute_invocation_digest,
)


def hide(
    engine: ContextHideEngine,
    store: SessionTranscriptStore,
    scope: Scope,
    tool_call_id: str,
    *,
    status: str = "ok",
) -> MutationResult:
    """Hide one transmitted tool result found via the session record."""
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        return MutationResult(ok=False, error="missing_tool_call_id")
    entries = store.lookup(scope.scope_key(), tool_call_id.strip())
    if not entries:
        return MutationResult(ok=False, error="not_found")
    if len(entries) > 1:
        return MutationResult(ok=False, error="ambiguous")
    entry = entries[0]
    if entry.arguments_parse_error or not entry.tool_name:
        return MutationResult(ok=False, error="invalid_invocation")
    try:
        invocation, omitted = filter_invocation(entry.tool_name, entry.raw_arguments)
    except AdapterRejected as exc:
        return MutationResult(ok=False, error=exc.code)
    record = ToolResultRecord(
        call_id=entry.call_id,
        result_position=entry.result_position,
        content=entry.content,
        content_sha256=entry.content_sha256,
        invocation=invocation,
        invocation_digest=compute_invocation_digest(
            invocation, omitted if omitted else None),
        status=status,
        host_handle=entry.host_handle,
    )
    return engine.hide_sync(scope, record)


def list_records(
    engine: ContextHideEngine, scope: Scope, query: str = ""
) -> list[dict[str, Any]]:
    """List hidden item metadata without originals or commands."""
    return [item.to_dict() for item in engine.list_items(scope, query)]


def unhide(
    engine: ContextHideEngine, scope: Scope, item_id: str
) -> MutationResult:
    """Mark one item visible again without rerunning tools."""
    if not isinstance(item_id, str) or not item_id.strip():
        return MutationResult(ok=False, error="not_found")
    return engine.unhide(scope, item_id.strip())


def middleware_step(
    engine: ContextHideEngine,
    store: SessionTranscriptStore,
    scope: Scope,
    api_mode: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Run one middleware pass: record, plan, apply to a copy (fail-open)."""
    stored, skipped, record_error = store.record_request(
        scope.scope_key(), api_mode, payload)
    trace: dict[str, Any] = {
        "source": "context-hide-hermes",
        "api_mode": api_mode,
        "found": len(stored),
        "skipped": len(skipped),
    }
    if record_error == UNSUPPORTED_API_MODE:
        trace["reason"] = UNSUPPORTED_API_MODE
        return {
            "request": dict(payload) if isinstance(payload, dict) else payload,
            "applied": [],
            "skipped": skipped,
            "trace": trace,
        }
    records: list[ToolResultRecord] = []
    record_skipped: list[dict[str, Any]] = list(skipped)
    handle_by_call_id: dict[str, dict[str, Any]] = {}
    for entry in stored:
        handle_by_call_id.setdefault(entry.call_id, dict(entry.host_handle))
        if entry.arguments_parse_error or not entry.tool_name:
            record_skipped.append({
                "host_handle": dict(entry.host_handle),
                "reason": "invalid_invocation",
            })
            continue
        try:
            invocation, omitted = filter_invocation(
                entry.tool_name, entry.raw_arguments)
        except AdapterRejected as exc:
            record_skipped.append({
                "host_handle": dict(entry.host_handle),
                "reason": exc.code,
            })
            continue
        records.append(
            ToolResultRecord(
                call_id=entry.call_id,
                result_position=entry.result_position,
                content=entry.content,
                content_sha256=entry.content_sha256,
                invocation=invocation,
                invocation_digest=compute_invocation_digest(
                    invocation, omitted if omitted else None),
                status="ok",
                host_handle=entry.host_handle,
            )
        )
    plans = engine.project(scope, records)
    plan_dicts = [
        {
            "item_id": plan.item_id,
            "host_handle": handle_by_call_id.get(
                _call_id_for_plan(plan, records), plan.host_handle),
            "expected_content_sha256": plan.expected_content_sha256,
            "replacement_text": plan.replacement_text,
        }
        for plan in plans
    ]
    new_payload, applied, apply_skipped = apply_replacements(
        api_mode, payload, plan_dicts)
    trace["applied"] = len(applied)
    return {
        "request": new_payload,
        "applied": applied,
        "skipped": record_skipped + apply_skipped,
        "trace": trace,
    }


def _call_id_for_plan(plan: Any, records: list[ToolResultRecord]) -> str:
    for record in records:
        if record.content_sha256 == plan.expected_content_sha256:
            return record.call_id
    return ""
