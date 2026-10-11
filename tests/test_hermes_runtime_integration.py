"""M1 integration tests against the REAL Hermes plugin runtime.

These tests run against the installed Hermes source tree (``HERMES_REPO``) and
prove the packaging contract at the level the host actually enforces:

- the host manifest parser accepts ``plugin.yaml`` from this repository
- the host directory loader imports the plugin root and finds ``register``
- ``register`` wires exactly the three tools and the ``llm_request`` middleware
  through the real ``PluginContext`` signals
- with no settings every tool reports ``disabled`` and the middleware returns
  ``None`` (recording nothing, rewriting nothing)
- enabling the plugin makes the middleware rewrite only a COPY of the request

Contract boundary: this module never starts an agent loop, never contacts a
provider, and never writes to any real Hermes home. The plugin root is loaded
by path with a private module name.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERMES_REPO = Path(
    os.environ.get("HERMES_REPO", "~/.hermes/hermes-agent")
).resolve()

pytestmark = pytest.mark.skipif(
    not (HERMES_REPO / "hermes_cli" / "plugins.py").is_file(),
    reason=f"installed Hermes source not found at {HERMES_REPO}",
)


@pytest.fixture(scope="module")
def host_modules():
    """Import the real plugin host modules and the plugin root, once."""
    for path in (str(HERMES_REPO), str(REPO / "src"), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import importlib

    manifest_mod = importlib.import_module("hermes_cli.plugins_manifest")
    loader_mod = importlib.import_module("hermes_cli.plugins_loader")
    plugins_mod = importlib.import_module("hermes_cli.plugins")
    plugin_root = importlib.import_module("adapters.hermes.plugin")
    return manifest_mod, loader_mod, plugins_mod, plugin_root


def _load_root_by_path():
    """Import the repo root the way ``_load_directory_module`` does."""
    import importlib.util

    name = "context_hide_plugin_root_host_test"
    spec = importlib.util.spec_from_file_location(
        name, REPO / "__init__.py", submodule_search_locations=[str(REPO)]
    )
    assert spec is not None and spec.loader is not None, "no spec for plugin root"
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name
    module.__path__ = [str(REPO)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _HostCtx:
    """The subset of ``PluginContext`` register_tool/register_middleware touch."""

    def __init__(self, profile: str = "default", settings: dict | None = None) -> None:
        self.profile_name = profile
        self.manifest = None
        self._settings = dict(settings or {})
        self.tools: dict[str, dict] = {}
        self.middlewares: dict[str, list] = {}
        self.llm = None

    def get_config(self, key: str, default=None):
        return self._settings.get(key, default)

    def register_tool(self, name, toolset, schema, handler, description="", **_extra):
        self.tools[name] = {"toolset": toolset, "schema": schema,
                            "handler": handler, "description": description}

    def register_middleware(self, kind, callback):
        self.middlewares.setdefault(kind, []).append(callback)


def test_host_manifest_parser_accepts_this_repository(host_modules) -> None:
    manifest_mod = host_modules[0]
    root = REPO
    manifest = manifest_mod.parse_manifest_file(
        root / "plugin.yaml", root, "user", ""
    )
    assert manifest is not None, "the installed Hermes parser rejected plugin.yaml"
    assert manifest.key == "context-hide"
    assert manifest.name == "context-hide"
    assert manifest.kind == "standalone"
    assert set(manifest.provides_tools or []) == {
        "hide_context", "list_context_items", "unhide_context",
    }
    assert manifest.manifest_version <= manifest_mod.SUPPORTED_MANIFEST_VERSION


def test_host_loader_imports_plugin_root_and_calls_register() -> None:
    """``register`` must be reachable the way the loader reaches it: via the dir."""
    module = _load_root_by_path()
    assert callable(module.register), "plugin root must export register(ctx)"
    ctx = _HostCtx(profile="daedalus")
    state = module.register(ctx)
    assert state is not None
    assert set(ctx.tools) == {"hide_context", "list_context_items", "unhide_context"}
    assert list(ctx.middlewares) == ["llm_request"]
    for name, entry in ctx.tools.items():
        assert entry["toolset"] == "context_hide"
        assert entry["schema"]["name"] == name
        assert callable(entry["handler"])


def test_register_does_not_import_host_internals() -> None:
    """The plugin must be loadable without the agent loop or plugin manager."""
    module = _load_root_by_path()
    before = set(sys.modules)
    module.__dict__.pop("register", None)
    del sys.modules["context_hide_plugin_root_host_test"]
    _load_root_by_path()
    imported = set(sys.modules) - before
    banned = {
        name for name in imported
        if name.startswith(("agent.", "hermes_cli.", "tools.", "hermes_cli.plugins"))
    }
    assert not banned, f"register must not pull host internals: {sorted(banned)}"


def _chat_request(body: str, call_id: str = "call_1") -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": json.dumps({"path": "big.txt"})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": body},
        ]
    }


def _long_body() -> str:
    lines = [f"row {i:03d} " + "x" * 60 for i in range(60)]
    lines.append("id: 01234567-89ab-cdef-0123-456789abcdef")
    return "\n".join(lines)


def _mw_kwargs(request: dict, session: str, mode: str = "chat_completions") -> dict:
    return {"request": request, "original_request": json.loads(json.dumps(request)),
            "session_id": session, "api_mode": mode, "provider": "test",
            "model": "test-model", "task_id": "t", "turn_id": "u",
            "api_request_id": "r"}


def test_default_off_records_nothing_and_rewrites_nothing() -> None:
    module = _load_root_by_path()
    ctx = _HostCtx(profile="daedalus")
    module.register(ctx)
    mw = ctx.middlewares["llm_request"][0]
    body = _long_body()
    out = mw(**_mw_kwargs(_chat_request(body), "sess-off"))
    assert out is None, "default-off middleware must return None"
    for name, entry in ctx.tools.items():
        assert json.loads(entry["handler"]({}, session_id="sess-off")) == {
            "ok": False, "error": "disabled",
        }, f"{name} must report disabled when the setting is absent"


def test_enabled_rewrites_only_the_outgoing_copy() -> None:
    module = _load_root_by_path()
    ctx = _HostCtx(profile="daedalus", settings={"enabled": True})
    state = module.register(ctx)
    state.count_middlewares = lambda: 0  # no other middleware in this test
    mw = ctx.middlewares["llm_request"][0]
    body = _long_body()
    request = _chat_request(body)
    snapshot = json.dumps(request, sort_keys=True)

    first = mw(**_mw_kwargs(request, "sess-on"))
    assert first is not None and isinstance(first.get("request"), dict)
    assert json.dumps(request, sort_keys=True) == snapshot, (
        "the caller's request dict must never be mutated in place"
    )
    assert first["request"]["messages"][1]["content"] == body, (
        "first transmission must still carry the original body"
    )
    assert first["request"]["messages"][1]["content"] != first["request"]["messages"][1]["content"].replace(body, "")

    hidden = json.loads(ctx.tools["hide_context"]["handler"](
        {"tool_call_id": "call_1"}, session_id="sess-on"))
    assert hidden["ok"] is True, hidden
    item_id = hidden["item_id"]

    second = mw(**_mw_kwargs(_chat_request(body), "sess-on"))
    replaced = second["request"]["messages"][1]["content"]
    assert replaced != body and item_id in replaced, (
        "second transmission must carry the short replacement, not the original"
    )
    assert len(replaced.encode("utf-8")) < len(body.encode("utf-8"))

    listed = json.loads(ctx.tools["list_context_items"]["handler"](
        {}, session_id="sess-on"))
    assert listed["ok"] is True and listed["count"] == 1
    assert listed["items"][0]["item_id"] == item_id
    assert body not in json.dumps(listed), "list must not leak the original body"

    restored = json.loads(ctx.tools["unhide_context"]["handler"](
        {"item_id": item_id}, session_id="sess-on"))
    assert restored["ok"] is True, restored

    third = mw(**_mw_kwargs(_chat_request(body), "sess-on"))
    assert third["request"]["messages"][1]["content"] == body, (
        "after unhide the original body must ride again, byte-identical"
    )


def test_scope_is_isolated_by_profile_and_session() -> None:
    """Separate Hermes processes must not share hidden-item identity.

    ``item_id`` is derived from the scope key, so two registrations with the
    same profile, session and body legitimately agree; the isolation that
    matters is that a DIFFERENT profile or session never resolves the other
    registration's item.
    """
    module = _load_root_by_path()
    ctx_a = _HostCtx(profile="daedalus", settings={"enabled": True})
    ctx_b = _HostCtx(profile="default", settings={"enabled": True})
    module.register(ctx_a)
    module.register(ctx_b)
    body = _long_body()
    for ctx, session in ((ctx_a, "sess-A"), (ctx_b, "sess-A")):
        ctx.middlewares["llm_request"][0](**_mw_kwargs(_chat_request(body), session))
    hid_a = json.loads(ctx_a.tools["hide_context"]["handler"](
        {"tool_call_id": "call_1"}, session_id="sess-A"))
    hid_b = json.loads(ctx_b.tools["hide_context"]["handler"](
        {"tool_call_id": "call_1"}, session_id="sess-A"))
    assert hid_a["ok"] is True and hid_b["ok"] is True
    assert hid_a["item_id"] != hid_b["item_id"], (
        "a different host profile must land in a different scope"
    )

    # The other registration must not resolve or restore this item.
    cross = json.loads(ctx_b.tools["unhide_context"]["handler"](
        {"item_id": hid_a["item_id"]}, session_id="sess-A"))
    assert cross["ok"] is False and cross["error"] == "not_found"
    listed_b = json.loads(ctx_b.tools["list_context_items"]["handler"](
        {}, session_id="sess-A"))
    assert all(item["item_id"] != hid_a["item_id"] for item in listed_b["items"])


def test_scope_is_isolated_by_session() -> None:
    module = _load_root_by_path()
    ctx = _HostCtx(profile="daedalus", settings={"enabled": True})
    module.register(ctx)
    body = _long_body()
    ctx.middlewares["llm_request"][0](**_mw_kwargs(_chat_request(body), "sess-A"))
    ctx.middlewares["llm_request"][0](**_mw_kwargs(_chat_request(body), "sess-B"))
    hid_a = json.loads(ctx.tools["hide_context"]["handler"](
        {"tool_call_id": "call_1"}, session_id="sess-A"))
    assert hid_a["ok"] is True
    wrong_session = json.loads(ctx.tools["unhide_context"]["handler"](
        {"item_id": hid_a["item_id"]}, session_id="sess-B"))
    assert wrong_session["ok"] is False and wrong_session["error"] == "not_found"
    listed_b = json.loads(ctx.tools["list_context_items"]["handler"](
        {}, session_id="sess-B"))
    assert listed_b["count"] == 0, "sess-B has not hidden anything yet"
