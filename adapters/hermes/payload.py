"""Hermes adapter payload helpers (M2-1).

Provider-format request copies only. No Hermes imports, no network, no env reads.
Finds tool result bodies in `codex_responses` and `chat_completions` shapes and
applies engine replacement plans to a copy. Anything else passes through unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

SUPPORTED_API_MODES = ("codex_responses", "chat_completions")
UNSUPPORTED_API_MODES = (
    "anthropic_messages",
    "bedrock_converse",
    "codex_app_server",
)

UNSUPPORTED_API_MODE = "unsupported_api_mode"


@dataclass(frozen=True)
class FoundResult:
    """One text tool result located in the outgoing provider payload."""

    call_id: str
    content: str
    host_handle: dict[str, Any]


@dataclass(frozen=True)
class SkippedEntry:
    """A payload entry examined but not offered for hiding."""

    host_handle: dict[str, Any]
    reason: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def find_tool_results(
    api_mode: str, payload: dict[str, Any]
) -> tuple[list[FoundResult], list[SkippedEntry], str | None]:
    """Locate text tool results in a provider-format request copy.

    Returns (results, skipped, error). Unsupported modes yield
    ([], [], "unsupported_api_mode") and the caller must leave the request
    unchanged. Supported modes never error on shape gaps; unpairable entries
    are reported in `skipped`.
    """
    if api_mode in UNSUPPORTED_API_MODES or api_mode not in SUPPORTED_API_MODES:
        return [], [], UNSUPPORTED_API_MODE
    if not isinstance(payload, dict):
        return [], [], None
    if api_mode == "codex_responses":
        return _find_codex_responses(payload)
    return _find_chat_completions(payload)


def _find_codex_responses(
    payload: dict[str, Any],
) -> tuple[list[FoundResult], list[SkippedEntry], None]:
    items = payload.get("input")
    if not isinstance(items, list):
        return [], [], None
    results: list[FoundResult] = []
    skipped: list[SkippedEntry] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        handle = {"api_mode": "codex_responses", "index": index, "kind": "function_call_output"}
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            skipped.append(SkippedEntry(host_handle=handle, reason="missing_call_id"))
            continue
        output = item.get("output")
        if not isinstance(output, str):
            skipped.append(SkippedEntry(host_handle=handle, reason="parts_output"))
            continue
        results.append(
            FoundResult(call_id=call_id.strip(), content=output, host_handle=handle)
        )
    return results, skipped, None


def _find_chat_completions(
    payload: dict[str, Any],
) -> tuple[list[FoundResult], list[SkippedEntry], None]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return [], [], None
    results: list[FoundResult] = []
    skipped: list[SkippedEntry] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        handle = {"api_mode": "chat_completions", "index": index, "kind": "tool_message"}
        call_id = msg.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            skipped.append(SkippedEntry(host_handle=handle, reason="missing_call_id"))
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            skipped.append(SkippedEntry(host_handle=handle, reason="non_text_content"))
            continue
        results.append(
            FoundResult(call_id=call_id.strip(), content=content, host_handle=handle)
        )
    return results, skipped, None


def apply_replacements(
    api_mode: str,
    payload: dict[str, Any],
    plans: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Apply replacement plans to a copy of the payload (fail-open).

    Each plan needs `host_handle` (as echoed by find), `expected_content_sha256`,
    `replacement_text`, and `item_id`. Entries whose current body no longer
    matches the expected hash, whose handle is out of range, or whose shape is
    unexpected are skipped and the original body is kept. Returns
    (new_payload, applied_item_ids, skipped) where skipped entries carry
    `item_id` and `reason`.
    """
    copied = copy.deepcopy(payload)
    if api_mode in UNSUPPORTED_API_MODES or api_mode not in SUPPORTED_API_MODES:
        return copied, [], [
            {
                "item_id": str(plan.get("item_id", "")),
                "reason": UNSUPPORTED_API_MODE,
            }
            for plan in plans
            if isinstance(plan, dict)
        ]
    applied: list[str] = []
    skipped: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            skipped.append({"item_id": "", "reason": "invalid_plan"})
            continue
        item_id = str(plan.get("item_id", ""))
        handle = plan.get("host_handle")
        expected = plan.get("expected_content_sha256")
        replacement = plan.get("replacement_text")
        if (
            not isinstance(handle, dict)
            or handle.get("api_mode") != api_mode
            or not isinstance(handle.get("index"), int)
            or not isinstance(replacement, str)
            or not isinstance(expected, str)
        ):
            skipped.append({"item_id": item_id, "reason": "invalid_plan"})
            continue
        index = handle["index"]
        kind = handle.get("kind")
        if api_mode == "codex_responses":
            items = copied.get("input")
            if (
                not isinstance(items, list)
                or not 0 <= index < len(items)
                or not isinstance(items[index], dict)
                or items[index].get("type") != "function_call_output"
                or kind != "function_call_output"
            ):
                skipped.append({"item_id": item_id, "reason": "handle_mismatch"})
                continue
            current = items[index].get("output")
            if not isinstance(current, str) or _sha256(current) != expected:
                skipped.append({"item_id": item_id, "reason": "hash_mismatch"})
                continue
            items[index]["output"] = replacement
            applied.append(item_id)
        else:
            messages = copied.get("messages")
            if (
                not isinstance(messages, list)
                or not 0 <= index < len(messages)
                or not isinstance(messages[index], dict)
                or messages[index].get("role") != "tool"
                or kind != "tool_message"
            ):
                skipped.append({"item_id": item_id, "reason": "handle_mismatch"})
                continue
            current = messages[index].get("content")
            if not isinstance(current, str) or _sha256(current) != expected:
                skipped.append({"item_id": item_id, "reason": "hash_mismatch"})
                continue
            messages[index]["content"] = replacement
            applied.append(item_id)
    return copied, applied, skipped
