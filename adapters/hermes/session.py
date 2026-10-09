"""Per-session last-transmission record (M2-2).

The plugin tool handlers only receive identifiers, never the conversation
messages. The `llm_request` middleware fills this record on every outbound
request and the hide/list/unhide handlers read it by scope + tool call ID.

Bounds: at most `max_sessions` scopes, at most `max_results_per_session`
results per scope, entries older than `ttl_seconds` expire on access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .payload import find_tool_results

TimeFunc = Callable[[], float]


@dataclass(frozen=True)
class StoredResult:
    """One tool result seen in the last transmitted request."""

    call_id: str
    content: str
    content_sha256: str
    tool_name: str
    raw_arguments: dict[str, Any]
    arguments_parse_error: bool
    host_handle: dict[str, Any]
    api_mode: str
    result_position: int
    recorded_at: float


@dataclass
class _SessionEntry:
    api_mode: str
    updated_at: float
    results: list[StoredResult] = field(default_factory=list)


class SessionTranscriptStore:
    """Keep only the last transmitted result list per scope."""

    def __init__(
        self,
        *,
        max_sessions: int = 128,
        max_results_per_session: int = 64,
        ttl_seconds: int = 86400,
        time_func: TimeFunc | None = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.max_results_per_session = max_results_per_session
        self.ttl_seconds = ttl_seconds
        self._now: TimeFunc = time_func or time.time
        self._sessions: dict[str, _SessionEntry] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def _compute_sha256(self, value: str) -> str:
        import hashlib

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _evict_expired(self) -> None:
        now = self._now()
        expired = [
            key for key, entry in self._sessions.items()
            if now - entry.updated_at >= self.ttl_seconds
        ]
        for key in expired:
            del self._sessions[key]

    def record_request(
        self, scope_key: str, api_mode: str, payload: dict[str, Any]
    ) -> tuple[list[StoredResult], list[dict[str, Any]], str | None]:
        """Replace the scope's last-transmission list from this request.

        Returns (stored, skipped, error) mirroring the payload finder. Out
        of scope keys or oversized sessions evict the oldest entries first.
        """
        self._evict_expired()
        results, skipped, error = find_tool_results(api_mode, payload)
        if error is not None:
            return [], [], error
        invocations = _extract_invocations(api_mode, payload)
        stored: list[StoredResult] = []
        positions: dict[str, int] = {}
        for found in results:
            position = positions.get(found.call_id, 0)
            positions[found.call_id] = position + 1
            tool_name, raw_args, parse_error = invocations.get(
                found.call_id, ("", {}, True))
            stored.append(
                StoredResult(
                    call_id=found.call_id,
                    content=found.content,
                    content_sha256=self._compute_sha256(found.content),
                    tool_name=tool_name,
                    raw_arguments=raw_args,
                    arguments_parse_error=parse_error,
                    host_handle=dict(found.host_handle),
                    api_mode=api_mode,
                    result_position=position,
                    recorded_at=self._now(),
                )
            )
        skipped_dicts = [
            {"host_handle": dict(item.host_handle), "reason": item.reason}
            for item in skipped
        ]
        kept = stored[: self.max_results_per_session]
        self._sessions[scope_key] = _SessionEntry(
            api_mode=api_mode, updated_at=self._now(), results=kept)
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
            del self._sessions[oldest]
        return kept, skipped_dicts, None

    def lookup(self, scope_key: str, tool_call_id: str) -> list[StoredResult]:
        """Return stored entries for one call ID, empty when absent/expired."""
        self._evict_expired()
        entry = self._sessions.get(scope_key)
        if entry is None:
            return []
        return [item for item in entry.results if item.call_id == tool_call_id]

    def entries(self, scope_key: str) -> list[StoredResult]:
        """Return all stored entries for one scope."""
        self._evict_expired()
        entry = self._sessions.get(scope_key)
        return list(entry.results) if entry is not None else []


def _extract_invocations(
    api_mode: str, payload: dict[str, Any]
) -> dict[str, tuple[str, dict[str, Any], bool]]:
    """Map call ID to (tool name, raw arguments, parse error).

    codex_responses reads `function_call` items (name + arguments).
    chat_completions reads `assistant.tool_calls` entries (function name +
    arguments). Missing or unparsable arguments are reported with
    parse_error True and decided later by the allowlist filter.
    """
    out: dict[str, tuple[str, dict[str, Any], bool]] = {}
    if not isinstance(payload, dict):
        return out
    if api_mode == "codex_responses":
        items = payload.get("input")
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            raw_name = item.get("name")
            name = raw_name if isinstance(raw_name, str) else ""
            raw = item.get("arguments", {})
            args, failed = _coerce_args(raw)
            out.setdefault(call_id.strip(), (name, args, failed))
    elif api_mode == "chat_completions":
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return out
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            calls = msg.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                raw_id = call.get("id", call.get("call_id"))
                if not isinstance(raw_id, str) or not raw_id.strip():
                    continue
                fn_raw = call.get("function")
                fn = fn_raw if isinstance(fn_raw, dict) else {}
                name_raw = fn.get("name")
                name = name_raw if isinstance(name_raw, str) else ""
                raw = fn.get("arguments", {})
                args, failed = _coerce_args(raw)
                out.setdefault(raw_id.strip(), (name, args, failed))
    return out


def _coerce_args(raw: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw, dict):
        return raw, False
    if isinstance(raw, str):
        if not raw.strip():
            return {}, False
        import json

        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}, True
        return (parsed, False) if isinstance(parsed, dict) else ({}, True)
    return {}, True
