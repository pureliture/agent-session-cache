"""A-plan supervision contracts; local requests only, no agent/provider calls."""
from __future__ import annotations

import json

from _adapter_loader import hermes_adapter

plugin = hermes_adapter().plugin
SESSION = "human-supervision-session"


def _state(**overrides):
    return plugin.PluginState(
        is_enabled=lambda: True, count_middlewares=lambda: 0, **overrides,
    )


def _body(label="sample"):
    return "\n".join(f"{label} row {i:03d}: " + "a" * 60 for i in range(80))


def _request(body, call_id):
    return {"input": [
        {"type": "function_call", "call_id": call_id, "name": "read_file",
         "arguments": json.dumps({"path": "sample.txt"})},
        {"type": "function_call_output", "call_id": call_id, "output": body},
    ]}


def _hide(state, call_id="call_1", body=None, session=SESSION):
    original = _body() if body is None else body
    plugin.make_middleware(state)(
        request=_request(original, call_id), session_id=session,
        api_mode="codex_responses",
    )
    result = json.loads(plugin.make_hide_handler(state)(
        {"tool_call_id": call_id}, session_id=session,
    ))
    assert result["ok"] is True, result
    return result


def _list(state, query="", session=SESSION):
    return json.loads(plugin.make_list_handler(state)(
        {"query": query}, session_id=session,
    ))


def _unhide(state, item_id, session=SESSION):
    return json.loads(plugin.make_unhide_handler(state)(
        {"item_id": item_id}, session_id=session,
    ))


def test_list_counts_hidden_separately_from_restored_records():
    state = _state()
    empty = _list(state)
    assert empty["count"] == 0 and empty["items"] == []
    assert empty["hidden_count"] == 0
    assert empty["message"] == "현재 숨겨진 항목 없음"

    hidden = _hide(state)
    listed = _list(state)
    assert listed["count"] == len(listed["items"]) == 1
    assert listed["hidden_count"] == 1
    assert listed["message"] == "현재 숨겨진 항목 1건"

    assert _unhide(state, hidden["item_id"])["ok"] is True
    restored = _list(state)
    assert restored["count"] == len(restored["items"]) == 1
    assert restored["items"][0]["visibility"] == "original"
    assert restored["hidden_count"] == 0
    assert restored["message"] == "현재 숨겨진 항목 없음"


def test_hidden_items_carry_human_fields_and_newest_first():
    state = _state()
    body_a, body_b = _body("alpha"), _body("beta")
    first = _hide(state, "call_1", body_a)
    second = _hide(state, "call_2", body_b)
    assert first["item_id"] != second["item_id"]

    items = {item["item_id"]: item for item in _list(state)["items"]}
    top, below = items[second["item_id"]], items[first["item_id"]]
    assert [item["item_id"] for item in _list(state)["items"]] == [
        second["item_id"], first["item_id"]]
    assert [item["display_number"] for item in _list(state)["items"]] == [1, 2]
    assert top["hidden_order"] > below["hidden_order"]
    assert top["hidden_at"].endswith("Z") and len(top["hidden_at"]) == 20
    assert top["original_bytes_size"] == len(body_b.encode("utf-8"))
    line = top["summary_line"]
    assert line and "\n" not in line and len(line) <= 160
    assert line.split(" ")[0] == "beta"

    restored = _list(state)["items"][0]["item_id"]
    assert _unhide(state, restored)["ok"] is True
    after = _list(state)
    # Hidden items keep the numbers and sort first; the restored record
    # stays in the list (history) with display_number None and its
    # original hidden_at retained.
    assert [item["display_number"] for item in after["items"]] == [1, None]
    assert after["items"][0]["item_id"] == first["item_id"]
    assert after["items"][1]["hidden_at"] == top["hidden_at"]


def test_query_view_renumbers_and_reports_search_scope():
    state = _state()
    _hide(state, "call_1", _body("alpha"))
    _hide(state, "call_2", _body("beta"))
    _hide(state, "call_3", _body("gamma"))

    matched = _list(state, query="beta")
    assert matched["hidden_count"] == 1
    assert matched["message"] == "검색 조건에 맞는 숨겨진 항목 1건"
    assert [item["display_number"] for item in matched["items"]] == [1]

    none = _list(state, query="no-such-marker")
    assert none["count"] == 0 and none["hidden_count"] == 0
    assert none["message"] == "검색 조건에 맞는 숨겨진 항목 없음"

    full = _list(state)
    assert full["hidden_count"] == 3
    assert [item["display_number"] for item in full["items"]] == [1, 2, 3]


def test_expired_items_prune_their_display_metadata():
    from context_hide.engine import ContextHideEngine
    from context_hide.store import MemoryContextStore

    now = [1000.0]
    state = _state(engine=ContextHideEngine(
        store=MemoryContextStore(ttl_seconds=60, clock=lambda: now[0])))
    hidden = _hide(state)
    scope_key = next(iter(state.presentations.snapshot()))
    assert hidden["item_id"] in state.presentations.snapshot()[scope_key]

    now[0] += 61.0
    expired = _list(state)
    assert expired["count"] == 0 and expired["hidden_count"] == 0
    assert expired["message"] == "현재 숨겨진 항목 없음"
    assert state.presentations.snapshot().get(scope_key, {}) == {}


def test_rehide_after_restore_starts_a_new_hide_transition():
    state = _state()
    body = _body("cycle")
    hidden = _hide(state, "call_1", body)
    first = {item["item_id"]: item for item in _list(state)["items"]}[
        hidden["item_id"]]

    # Re-hide while already hidden: same transition, no new timestamp.
    again = _hide(state, "call_1", body)
    assert again["item_id"] == hidden["item_id"]
    same = {item["item_id"]: item for item in _list(state)["items"]}[
        hidden["item_id"]]
    assert same["hidden_order"] == first["hidden_order"]
    assert same["hidden_at"] == first["hidden_at"]

    # Restore, then hide again: a new transition gets a new order.
    assert _unhide(state, hidden["item_id"])["ok"] is True
    refreshed = _hide(state, "call_1", body)
    assert refreshed["item_id"] == hidden["item_id"]
    newest = _list(state)
    assert newest["hidden_count"] == 1
    item = newest["items"][0]
    assert item["hidden_order"] > first["hidden_order"]
    assert item["display_number"] == 1
    assert item["hidden_at"] >= first["hidden_at"]


class _SectionCtx:
    """PluginContext surface with the optional prompt-section API."""

    def __init__(self, enabled: bool = True, with_section_api: bool = True):
        self.profile_name = "default"
        self._config = {"enabled": enabled}
        self.tools: dict[str, dict] = {}
        self.middlewares: dict[str, list] = {}
        self.sections: list[dict] = []
        if with_section_api:
            self.register_system_prompt_section = self._register_section

    def _register_section(self, id, content, *, position="after_memory",
                          max_chars=1000):
        self.sections.append({
            "id": id, "content": content, "position": position,
            "max_chars": max_chars,
        })

    def get_config(self, key, default=None):
        return self._config.get(key, default)

    def register_tool(self, name, toolset, schema, handler, description="", **_extra):
        self.tools[name] = {"description": description}

    def register_middleware(self, kind, callback):
        self.middlewares.setdefault(kind, []).append(callback)


def test_enabled_plugin_registers_static_supervision_section():
    ctx = _SectionCtx()
    plugin.register(ctx)
    assert len(ctx.sections) == 1
    section = ctx.sections[0]
    assert section["id"] == "context-hide.human-supervision"
    assert section["position"] == "after_memory"
    assert section["max_chars"] == 1200
    text = section["content"]
    assert isinstance(text, str) and len(text) <= 1200
    for token in ("list_context_items", "unhide_context", "숨김"):
        assert token in text, token


def test_section_registration_degrades_without_api_or_when_disabled():
    legacy = _SectionCtx(with_section_api=False)
    plugin.register(legacy)  # must not raise on hosts without the API
    assert legacy.tools  # tools still registered

    disabled = _SectionCtx(enabled=False)
    plugin.register(disabled)
    assert disabled.sections == []


def test_tool_descriptions_route_human_list_and_restore_requests():
    sections = plugin.TOOL_SCHEMAS
    list_desc = sections["list_context_items"]["description"]
    unhide_desc = sections["unhide_context"]["description"]
    assert "사용자" in list_desc and "보여" in list_desc
    assert "복원" in unhide_desc and "item_id" in unhide_desc
    assert "목록" in unhide_desc


def test_unhide_description_routes_number_references_verbatim():
    desc = plugin.TOOL_SCHEMAS["unhide_context"]["description"]
    assert "번" in desc and "고정" in desc
    assert "복원 직전" in desc and "임의로" in desc


def test_numeric_reference_resolves_against_the_last_shown_list():
    state = _state()
    first = _hide(state, "call_1", _body("alpha"))
    second = _hide(state, "call_2", _body("beta"))
    shown = _list(state)
    assert [i["item_id"] for i in shown["items"]] == [second["item_id"], first["item_id"]]
    assert [i["display_number"] for i in shown["items"]] == [1, 2]

    restored = _unhide(state, "2번")  # digits + suffix resolve, no ID recall
    assert restored["ok"] is True
    assert restored["item_id"] == first["item_id"]

    # The shown numbers stay stable until the NEXT list call: "1" still names
    # what the user saw as #1, never a number a fresh list would issue.
    late = _unhide(state, "#1")
    assert late["ok"] is True
    assert late["item_id"] == second["item_id"]


def test_numeric_reference_fails_closed_without_a_shown_list():
    state = _state()
    hidden = _hide(state, "call_1")
    blocked = _unhide(state, "1")
    assert blocked["ok"] is False and blocked["error"] == "no_recent_list"
    # An exact item_id still works without any prior list call.
    exact = _unhide(state, hidden["item_id"])
    assert exact["ok"] is True


def test_numeric_reference_out_of_range_and_after_new_list():
    state = _state()
    _hide(state, "call_1")
    _list(state)
    bad = _unhide(state, "9")
    assert bad["ok"] is False and bad["error"] == "unknown_display_number"

    stale = _unhide(state, "1")
    assert stale["ok"] is True
    _list(state)  # fresh view: nothing hidden, no numbers
    missing = _unhide(state, "1")
    assert missing["ok"] is False and missing["error"] == "unknown_display_number"


def test_numeric_reference_keeps_working_after_a_restore():
    state = _state()
    _hide(state, "call_1", _body("alpha"))
    second = _hide(state, "call_2", _body("beta"))
    _list(state)  # 1 = second, 2 = first

    first_restore = _unhide(state, "1")
    assert first_restore["item_id"] == second["item_id"]
    second_restore = _unhide(state, " 2 ")  # surrounding whitespace tolerated
    assert second_restore["ok"] is True


def test_fixed_id_restore_after_recheck_keeps_original_target():
    state = _state()
    first = _hide(state, "call_1", _body("alpha"))
    _hide(state, "call_2", _body("beta"))
    shown = _list(state)
    fixed = next(
        item["item_id"] for item in shown["items"] if item["display_number"] == 2
    )
    assert fixed == first["item_id"]

    # New hide changes the numbering: the same number now names another item.
    _hide(state, "call_3", _body("gamma"))
    rechecked = _list(state)
    assert rechecked["items"][1]["item_id"] != fixed

    # Required path: restore the fixed exact ID, not the new number.
    restored = _unhide(state, fixed)
    assert restored["ok"] is True
    assert restored["item_id"] == fixed
    after = _list(state)
    assert next(
        item["visibility"] for item in after["items"] if item["item_id"] == fixed
    ) == "original"
