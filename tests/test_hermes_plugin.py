"""S4 synthetic tests: plugin registration, middleware, and tool handlers.

No live Hermes, no network. A fake ctx captures register_tool /
register_middleware calls; middleware and handlers are then driven with
synthetic provider-format kwargs shaped like the real contract.
"""

from __future__ import annotations

import copy
import json

from _adapter_loader import hermes_adapter as _load_hermes_adapter

_adapter = _load_hermes_adapter()
plugin = _adapter.plugin


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


class FakeCtx:
    """Duck-typed stand-in for Hermes PluginContext (register surface only)."""

    def __init__(self, profile: str = "default", config: dict | None = None):
        self.profile_name = profile
        self.config = dict(config or {})
        self.tools: dict[str, dict] = {}
        self.middlewares: dict[str, list] = {}

    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

    def register_tool(self, name, toolset, schema, handler, description="", **_extra):
        self.tools[name] = {
            "toolset": toolset, "schema": schema,
            "handler": handler, "description": description,
        }

    def register_middleware(self, kind, callback):
        self.middlewares.setdefault(kind, []).append(callback)


def _enabled_ctx(**overrides) -> FakeCtx:
    config = {"enabled": True}
    config.update(overrides)
    return FakeCtx(config=config)


def _mw_kwargs(request: dict, session: str, api_mode: str) -> dict:
    return {
        "request": request,
        "original_request": copy.deepcopy(request),
        "telemetry_schema_version": "hermes.observer.v1",
        "middleware_schema_version": "hermes.middleware.v1",
        "task_id": "task-1",
        "session_id": session,
        "turn_id": "turn-1",
        "api_request_id": "req-1",
        "provider": "foundry",
        "model": "test-model",
        "api_mode": api_mode,
    }


def test_register_wires_three_tools_and_one_middleware():
    ctx = FakeCtx()
    plugin.register(ctx)
    assert set(ctx.tools) == {"hide_context", "list_context_items", "unhide_context"}
    assert set(ctx.middlewares) == {"llm_request"}
    assert len(ctx.middlewares["llm_request"]) == 1
    for name, entry in ctx.tools.items():
        schema = entry["schema"]
        assert schema["name"] == name
        assert schema["parameters"]["type"] == "object"
        assert entry["toolset"] == "context_hide"
        assert callable(entry["handler"])
    hide_params = ctx.tools["hide_context"]["schema"]["parameters"]
    assert hide_params["required"] == ["tool_call_id"]
    unhide_params = ctx.tools["unhide_context"]["schema"]["parameters"]
    assert unhide_params["required"] == ["item_id"]


def test_disabled_by_default():
    ctx = FakeCtx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    request = _codex_request(_long_body())
    assert mw(**_mw_kwargs(request, "sess-off", "codex_responses")) is None
    for name, entry in ctx.tools.items():
        raw = entry["handler"]({}, session_id="sess-off")
        assert isinstance(raw, str)
        assert json.loads(raw) == {"ok": False, "error": "disabled"}


def test_enabled_end_to_end_hide_two_requests_list_unhide():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    hide = ctx.tools["hide_context"]["handler"]
    list_items = ctx.tools["list_context_items"]["handler"]
    unhide = ctx.tools["unhide_context"]["handler"]
    body = _long_body()

    before = _codex_request(body)
    snapshot = copy.deepcopy(before)
    first = mw(**_mw_kwargs(before, "sess-e2e", "codex_responses"))
    assert first is not None
    assert first["applied" if "applied" in first else "reason"] is not None
    assert first["request"]["input"][1]["output"] == body
    assert before == snapshot

    hidden = json.loads(hide({"tool_call_id": "call_1"}, session_id="sess-e2e"))
    assert hidden["ok"] is True
    item_id = hidden["item_id"]
    assert isinstance(item_id, str) and item_id.startswith("item_")

    second = mw(**_mw_kwargs(_codex_request(body), "sess-e2e", "codex_responses"))
    assert second["request"]["input"][1]["output"] != body
    assert item_id in second["request"]["input"][1]["output"]

    third = mw(**_mw_kwargs(_codex_request(body), "sess-e2e", "codex_responses"))
    assert third["request"]["input"][1]["output"] == second["request"]["input"][1]["output"]

    listed = json.loads(list_items({}, session_id="sess-e2e"))
    assert listed["ok"] is True and listed["count"] == 1
    assert listed["items"][0]["item_id"] == item_id
    # Metadata only: no full original, no command arguments.
    assert "original" not in listed["items"][0]
    assert "arguments" not in listed["items"][0]
    assert "command" not in json.dumps(listed)

    restored = json.loads(unhide({"item_id": item_id}, session_id="sess-e2e"))
    assert restored["ok"] is True

    fourth = mw(**_mw_kwargs(_codex_request(body), "sess-e2e", "codex_responses"))
    assert fourth["request"]["input"][1]["output"] == body


def test_middleware_result_shape_and_trace():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    out = mw(**_mw_kwargs(_codex_request(_long_body()), "sess-shape", "codex_responses"))
    assert set(out) == {"request", "source", "reason"}
    assert out["source"] == "context-hide-hermes"
    assert isinstance(out["reason"], str) and out["reason"]


def test_unsupported_mode_passes_through_with_reason():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    out = mw(**_mw_kwargs(payload, "sess-unsupported", "anthropic_messages"))
    assert out["request"] == payload
    assert out["reason"] == "unsupported_api_mode"


def test_missing_session_fails_open():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    payload = _codex_request(_long_body())
    out = mw(**_mw_kwargs(payload, "", "codex_responses"))
    assert out["request"] == payload
    assert out["reason"] == "missing_session"
    hide = ctx.tools["hide_context"]["handler"]
    assert json.loads(hide({"tool_call_id": "call_1"}, session_id="")) == {
        "ok": False, "error": "not_found"}


def test_hide_missing_target_and_unknown_item():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    hide = ctx.tools["hide_context"]["handler"]
    unhide = ctx.tools["unhide_context"]["handler"]
    assert json.loads(hide({}, session_id="sess-x")) == {
        "ok": False, "error": "missing_tool_call_id"}
    assert json.loads(hide({"tool_call_id": "nope"}, session_id="sess-x")) == {
        "ok": False, "error": "not_found"}
    assert json.loads(unhide({}, session_id="sess-x")) == {
        "ok": False, "error": "not_found"}
    assert json.loads(unhide({"item_id": "item_0123456789abcdef"},
                              session_id="sess-x"))["ok"] is False


def test_coexistence_records_but_does_not_apply():
    ctx = _enabled_ctx()
    state = plugin.register(ctx)
    state.count_middlewares = lambda: 2
    mw = ctx.middlewares["llm_request"][0]
    body = _long_body()
    out = mw(**_mw_kwargs(_codex_request(body), "sess-coex", "codex_responses"))
    assert out["request"]["input"][1]["output"] == body
    assert out["reason"] == "other_middleware_present"
    hide = ctx.tools["hide_context"]["handler"]
    hidden = json.loads(hide({"tool_call_id": "call_1"}, session_id="sess-coex"))
    assert hidden["ok"] is True


def test_middleware_fail_open_on_bad_shapes():
    ctx = _enabled_ctx()
    plugin.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    assert mw(request=None, session_id="s", api_mode="codex_responses") is None
    out = mw(**_mw_kwargs(_codex_request("tiny"), "s", None))  # type: ignore[arg-type]
    assert out["request"]["input"][1]["output"] == "tiny"


def test_bridge_collision_set_matches_bridge_contract():
    assert plugin.BRIDGE_COLLISION_TOOLS == {
        "hide_context", "list_context_items", "unhide_context"}
    assert plugin.bridge_should_yield(["read_file", "hide_context"]) is True
    assert plugin.bridge_should_yield(["terminal", "read_file"]) is False
    assert plugin.bridge_should_yield([]) is False
    assert plugin.bridge_should_yield(None) is False
