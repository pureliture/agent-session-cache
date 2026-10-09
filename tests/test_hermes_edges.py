"""S5 synthetic tests: edge conditions and the summarizer-path decision.

No live Hermes, no network. Covers compression rotation, subagent
isolation, session/engine expiry, disabled toggle, and documents the
rule-only first-version decision with a direct assertion.
"""

from __future__ import annotations

import copy
import json

from _adapter_loader import hermes_adapter as _load_hermes_adapter
from context_hide.engine import ContextHideEngine
from context_hide.model import Scope
from context_hide.store import MemoryContextStore

_adapter = _load_hermes_adapter()
plugin = _adapter.plugin
handlers = _adapter.handlers
_session_mod = _adapter.session
SessionTranscriptStore = _session_mod.SessionTranscriptStore


def _long_body(suffix: str = "") -> str:
    lines = [f"Output row {i:03d} details" for i in range(80)]
    lines.extend([
        "id: 01234567-89ab-cdef-0123-456789abcdef",
        "results/output.json generated successfully",
    ])
    if suffix:
        lines.append(suffix)
    return "\n".join(lines)


def _codex_request(body: str, call_id: str = "call_1") -> dict:
    return {
        "input": [
            {"type": "function_call", "call_id": call_id, "name": "read_file",
             "arguments": json.dumps({"path": "results/output.json"})},
            {"type": "function_call_output", "call_id": call_id, "output": body},
        ]
    }


class FakeCtx:
    def __init__(self, config: dict | None = None, profile: str = "default"):
        self.profile_name = profile
        self.config = dict(config or {})
        self.tools: dict[str, dict] = {}
        self.middlewares: dict[str, list] = {}

    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

    def register_tool(self, name, toolset, schema, handler, description="", **_extra):
        self.tools[name] = {"handler": handler, "schema": schema}

    def register_middleware(self, kind, callback):
        self.middlewares.setdefault(kind, []).append(callback)


def _enabled_pair(**overrides):
    ctx = FakeCtx(config={"enabled": True, **overrides})
    state = plugin.register(ctx)
    return ctx, state


def _mw(state, body: str, session: str, call_id: str = "call_1") -> dict:
    mw = plugin.make_middleware(state)
    return mw(request=_codex_request(body, call_id), session_id=session,
              api_mode="codex_responses")


def test_compression_rotation_starts_fresh_scope():
    ctx, state = _enabled_pair()
    hide = ctx.tools["hide_context"]["handler"]
    unhide = ctx.tools["unhide_context"]["handler"]
    body = _long_body()

    _mw(state, body, "sess-before")
    before_item = json.loads(hide({"tool_call_id": "call_1"},
                                   session_id="sess-before"))["item_id"]

    # Compression rotated the session id: the old hidden state does NOT
    # carry over, so the new scope's request passes the original body.
    rotated = _mw(state, body, "sess-after")
    assert rotated["request"]["input"][1]["output"] == body

    # The new scope manages its own independent hidden set.
    after_item = json.loads(hide({"tool_call_id": "call_1"},
                                  session_id="sess-after"))["item_id"]
    assert after_item != before_item
    assert _mw(state, body, "sess-after")["request"]["input"][1][
        "output"] != body

    # Unhiding the old scope leaves the new scope hidden (isolation).
    unhide_result = json.loads(unhide({"item_id": before_item},
                                       session_id="sess-before"))
    assert unhide_result["ok"] is True
    assert _mw(state, body, "sess-before")["request"]["input"][1][
        "output"] == body
    assert _mw(state, body, "sess-after")["request"]["input"][1][
        "output"] != body


def test_subagent_sessions_stay_isolated():
    ctx, state = _enabled_pair()
    hide = ctx.tools["hide_context"]["handler"]
    body = _long_body()

    _mw(state, body, "sess-parent")
    parent_item = json.loads(hide({"tool_call_id": "call_1"},
                                   session_id="sess-parent"))["item_id"]

    # The child scope starts unhidden even for the same body.
    child = _mw(state, body, "sess-child")
    assert child["request"]["input"][1]["output"] == body

    # Each scope hides its own copy under its own item id.
    child_item = json.loads(hide({"tool_call_id": "call_1"},
                                  session_id="sess-child"))["item_id"]
    assert child_item != parent_item

    parent = _mw(state, body, "sess-parent")
    assert parent["request"]["input"][1]["output"] != body


def test_session_expiry_forgets_quiet_scopes():
    # Every new request refreshes the "last transmission" record, so
    # expiry only bites after silence: a late hide finds nothing, while
    # a fresh request re-records and hiding works again.
    now = [1000.0]
    store = SessionTranscriptStore(ttl_seconds=60, time_func=lambda: now[0])
    state = plugin.PluginState(store=store,
                               is_enabled=lambda: True,
                               count_middlewares=lambda: 1)
    scope = plugin.scope_for(state, "sess-ttl")
    assert scope is not None
    body = _long_body()

    handlers.middleware_step(state.engine, store, scope, "codex_responses",
                             _codex_request(body))
    assert handlers.hide(state.engine, store, scope, "call_1").ok is True

    now[0] += 61.0
    # Silence longer than the TTL: the tool path finds no record.
    assert store.entries(scope.scope_key()) == []
    late = handlers.hide(state.engine, store, scope, "call_1")
    assert late.ok is False and late.error == "not_found"

    # A new request re-records, so hiding works again on the fresh entry.
    handlers.middleware_step(state.engine, store, scope, "codex_responses",
                             _codex_request(body))
    assert handlers.hide(state.engine, store, scope, "call_1").ok is True


def test_engine_store_expiry_stops_replacement():
    now = [5000.0]
    engine = ContextHideEngine(
        store=MemoryContextStore(ttl_seconds=60, clock=lambda: now[0]))
    store = SessionTranscriptStore()
    scope = Scope(adapter_id="hermes", host_profile="test",
                  session_id="sess-eng-ttl", branch_scope="unknown")
    body = _long_body()

    handlers.middleware_step(engine, store, scope, "codex_responses",
                             _codex_request(body))
    assert handlers.hide(engine, store, scope, "call_1").ok is True

    now[0] += 61.0
    later = handlers.middleware_step(engine, store, scope, "codex_responses",
                                     _codex_request(body))
    assert later["applied"] == []
    assert later["request"]["input"][1]["output"] == body


def test_disable_returns_to_original_and_reenable_applies():
    ctx = FakeCtx(config={"enabled": True})
    state = plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    hide = ctx.tools["hide_context"]["handler"]
    body = _long_body()
    kwargs = dict(session_id="sess-toggle", api_mode="codex_responses")

    mw(request=_codex_request(body), **kwargs)
    assert json.loads(hide({"tool_call_id": "call_1"},
                            session_id="sess-toggle"))["ok"] is True
    applied = mw(request=_codex_request(body), **kwargs)
    assert applied["request"]["input"][1]["output"] != body

    ctx.config["enabled"] = False
    assert mw(request=_codex_request(body), **kwargs) is None
    untouched = _codex_request(body)
    snapshot = copy.deepcopy(untouched)
    assert mw(request=untouched, **kwargs) is None
    assert untouched == snapshot

    ctx.config["enabled"] = True
    again = mw(request=_codex_request(body), **kwargs)
    assert again["request"]["input"][1]["output"] != body


def test_first_version_is_rule_only_no_external_call():
    ctx, state = _enabled_pair()
    hide = ctx.tools["hide_context"]["handler"]
    list_items = ctx.tools["list_context_items"]["handler"]
    body = _long_body()

    plugin.make_middleware(state)(request=_codex_request(body),
                                  session_id="sess-rule",
                                  api_mode="codex_responses")
    assert json.loads(hide({"tool_call_id": "call_1"},
                            session_id="sess-rule"))["ok"] is True
    listed = json.loads(list_items({}, session_id="sess-rule"))
    assert listed["count"] == 1
    assert listed["items"][0]["source"] == "rule"


def test_default_middleware_counter_degrades_to_single_owner():
    # Outside a Hermes process the lazy host import fails and must fall
    # back to 1 ("no evidence of others") instead of raising.
    assert plugin._default_count_llm_middlewares() == 1
