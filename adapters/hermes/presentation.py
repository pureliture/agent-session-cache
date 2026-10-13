"""Human-supervision display metadata for the Hermes adapter (A-plan).

The engine stores items, but it has no wall-clock "hidden at" time and no
hide order. This module keeps that *display-only* metadata in process
memory, scope-bound, and never duplicates original bodies:

- ``hidden_order``: per-scope counter of hide transitions (newest first
  when listed). Not a stable ID.
- ``hidden_at``: UTC wall-clock of the latest hide transition. A re-hide
  of an already-hidden item keeps the earlier timestamp; a hide after an
  unhide refreshes it (it is a new transition).
- ``original_bytes_size``: UTF-8 size of the original at hide time.
- ``summary_line``: derived from the stored excerpt lines, never from a
  new model call.

Everything here is best-effort: entries missing for an item simply list
with ``null`` display fields instead of invented values. Callers prune
expired entries against the engine's live item ids on each tool access.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

MAX_SUMMARY_LINE_CHARS = 160


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def summary_line_from_excerpts(excerpt_lines: Iterable[str]) -> str:
    """First non-empty excerpt line, whitespace-collapsed, length-capped."""
    for line in excerpt_lines:
        collapsed = " ".join(str(line).split())
        if collapsed:
            return collapsed[:MAX_SUMMARY_LINE_CHARS]
    return ""


@dataclass
class Presentation:
    """Display-only metadata for one hidden item."""

    item_id: str
    hidden_order: int
    hidden_at: str
    original_bytes_size: int
    last_visibility: str = "compacted"


class PresentationRegistry:
    """Scope-keyed, thread-safe display metadata (process memory only)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Presentation]] = {}
        self._next_order: dict[str, int] = {}

    def record_hidden(
        self,
        scope_key: str,
        item_id: str,
        original_bytes_size: int,
    ) -> None:
        """Record a hide transition; refresh only after a visible period."""
        with self._lock:
            entries = self._entries.setdefault(scope_key, {})
            existing = entries.get(item_id)
            if existing is not None and existing.last_visibility == "compacted":
                existing.original_bytes_size = original_bytes_size
                return
            order = self._next_order.get(scope_key, 0) + 1
            self._next_order[scope_key] = order
            entries[item_id] = Presentation(
                item_id=item_id,
                hidden_order=order,
                hidden_at=_utc_now_iso(),
                original_bytes_size=original_bytes_size,
                last_visibility="compacted",
            )

    def mark_visible(self, scope_key: str, item_id: str) -> None:
        """Record an unhide; the hide timestamp and order are retained."""
        with self._lock:
            entry = self._entries.get(scope_key, {}).get(item_id)
            if entry is not None:
                entry.last_visibility = "original"

    def prune(self, scope_key: str, live_item_ids: set[str]) -> int:
        """Drop entries whose engine item no longer exists (expired/evicted)."""
        with self._lock:
            entries = self._entries.get(scope_key)
            if not entries:
                return 0
            stale = [key for key in entries if key not in live_item_ids]
            for key in stale:
                del entries[key]
            return len(stale)

    def snapshot(self) -> dict[str, dict[str, Presentation]]:
        """Copy for assertions; never used for output on the hot path."""
        with self._lock:
            return {
                scope: dict(entries) for scope, entries in self._entries.items()
            }


def _sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    hidden = item.get("visibility") == "compacted"
    order = item.get("hidden_order")
    known = isinstance(order, int)
    return (
        0 if hidden else 1,
        0 if known else 1,
        -order if known else 0,
        str(item.get("item_id", "")),
    )


def decorate_items(
    items: list[dict[str, Any]],
    snapshot: dict[str, Presentation],
) -> list[dict[str, Any]]:
    """Add display fields, sort (newest hide first), and number the hidden.

    Restored records keep their display metadata (so the history stays
    readable) but carry ``display_number: None`` and sort after hidden
    items. Unknown-order entries sort last by item id; no value is ever
    invented.
    """
    decorated: list[dict[str, Any]] = []
    for item in items:
        entry = snapshot.get(str(item.get("item_id", "")))
        item = dict(item)
        item["original_bytes_size"] = (
            entry.original_bytes_size if entry is not None else None
        )
        item["hidden_at"] = entry.hidden_at if entry is not None else None
        item["hidden_order"] = entry.hidden_order if entry is not None else None
        item["display_number"] = None
        item["summary_line"] = summary_line_from_excerpts(
            item.get("excerpt_lines") or [])
        decorated.append(item)
    decorated.sort(key=_sort_key)
    number = 0
    for item in decorated:
        if item.get("visibility") == "compacted":
            number += 1
            item["display_number"] = number
    return decorated
