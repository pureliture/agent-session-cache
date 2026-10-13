"""M3 evidence: drive the human-supervision protocol through the REAL host.

Runs the actual ``hermes_cli.plugins.PluginManager`` discovery+load against a
throwaway ``HERMES_HOME`` that contains a full copy of this plugin directory
(root files + ``vendor/``), then:

1. asserts the three tools and the ``llm_request`` middleware registered,
2. registers nothing else; the real ``PluginContext`` validation applied to
   the static supervision section (id/position/max_chars) either accepts it or
   the probe records the host's refusal,
3. exercises one full hide -> list -> unhide cycle through the registered
   handlers and middleware callback, checking the new human-supervision
   fields (``hidden_count``, ``display_number``, ``summary_line``,
   ``original_bytes_size``, ``hidden_at``).

No provider is contacted; nothing outside the temp home is written.

Usage:
    PYTHONPATH= ./.venv/bin/python tools/probe_supervision.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLUGIN_ID = "context-hide"
SESSION = "probe-human-supervision"


def _body() -> str:
    return "\n".join(f"Row {i:03d}: " + "a" * 64 for i in range(120))


def _codex_request(body: str) -> dict:
    return {"input": [
        {"type": "function_call", "call_id": "call_probe", "name": "read_file",
         "arguments": json.dumps({"path": "probe.txt"})},
        {"type": "function_call_output", "call_id": "call_probe", "output": body},
    ]}


def main() -> int:
    hermes_repo = os.environ.get("HERMES_REPO", "~/.hermes/hermes-agent")
    if hermes_repo not in sys.path:
        sys.path.insert(0, hermes_repo)

    home = Path(tempfile.mkdtemp(prefix="hermes-supervision-probe-"))
    plugin_dir = home / "plugins" / PLUGIN_ID
    plugin_dir.mkdir(parents=True)
    for item in REPO.iterdir():
        if item.name in ("__init__.py", "plugin.yaml"):
            shutil.copy2(item, plugin_dir / item.name)
    shutil.copytree(REPO / "vendor", plugin_dir / "vendor")
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [context-hide]\n"
        "  entries:\n"
        "    context-hide:\n"
        "      settings:\n"
        "        enabled: true\n",
        encoding="utf-8",
    )
    os.environ["HERMES_HOME"] = str(home)

    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager.discover_and_load()
    loaded = getattr(manager, "_plugins", {})
    entry = loaded.get(PLUGIN_ID)
    tools = sorted(getattr(entry, "tools_registered", []) or [])
    middlewares = manager._middleware.get("llm_request", [])
    sections = manager.render_system_prompt_sections({"session_id": SESSION})

    print(f"home            : {home}")
    print(f"loaded          : {PLUGIN_ID in loaded}")
    print(f"tools_registered: {tools}")
    print(f"llm_request mws : {len(middlewares)}")
    print(f"sections        : {[(s.id, len(s.content)) for s in sections]}")

    ok = (
        entry is not None
        and callable(getattr(getattr(entry, 'module', None), 'register', None))
        and set(tools) == {"hide_context", "list_context_items", "unhide_context"}
        and len(middlewares) == 1
        and any(s.id == "context-hide.human-supervision" for s in sections)
    )
    section = next((s for s in sections if s.id == "context-hide.human-supervision"), None)
    if section is not None:
        print(f"section tokens  : {[t for t in ('list_context_items', 'unhide_context') if t in section.content]}")
        ok = ok and len(section.content) <= 1200

    if ok:
        from tools.registry import registry

        entries = {name: registry.get_entry(name) for name in (
            "hide_context", "list_context_items", "unhide_context")}
        missing = sorted(name for name, e in entries.items() if e is None)
        if missing:
            print(f"MISSING registry entries: {missing}")
            print("VERDICT         : FAIL")
            return 1
        hide_entry = registry.get_entry("hide_context")
        list_entry = registry.get_entry("list_context_items")
        unhide_entry = registry.get_entry("unhide_context")
        assert hide_entry is not None and list_entry is not None and unhide_entry is not None
        hide, list_items, unhide = (
            hide_entry.handler, list_entry.handler, unhide_entry.handler)
        mw = middlewares[0]
        body = _body()
        before = _codex_request(body)
        snapshot = copy.deepcopy(before)
        first = mw(request=before, original_request=copy.deepcopy(before),
                   session_id=SESSION, api_mode="codex_responses",
                   provider="probe", model="probe", task_id="t", turn_id="u",
                   api_request_id="r")
        ok = ok and before == snapshot and first is not None
        hidden = json.loads(hide({"tool_call_id": "call_probe"}, session_id=SESSION))
        print(f"hide            : {hidden}")
        ok = ok and hidden.get("ok") is True

        second = mw(request=_codex_request(body),
                    original_request=_codex_request(body), session_id=SESSION,
                    api_mode="codex_responses", provider="probe", model="probe",
                    task_id="t", turn_id="u", api_request_id="r")
        replaced = second["request"]["input"][1]["output"]
        ok = ok and hidden["item_id"] in replaced and replaced != body

        listed = json.loads(list_items({"query": ""}, session_id=SESSION))
        item = listed["items"][0]
        print(f"list            : hidden_count={listed['hidden_count']} "
              f"message={listed['message']!r} count={listed['count']}")
        print(f"item fields     : display_number={item['display_number']} "
              f"original_bytes_size={item['original_bytes_size']} "
              f"hidden_at={item['hidden_at']!r} "
              f"summary_line={item['summary_line'][:40]!r}...")
        ok = ok and listed["hidden_count"] == 1 and item["display_number"] == 1
        ok = ok and item["original_bytes_size"] == len(body.encode("utf-8"))
        ok = ok and isinstance(item["hidden_at"], str) and item["hidden_at"].endswith("Z")

        restored = json.loads(unhide({"item_id": "1번"}, session_id=SESSION))
        after = json.loads(list_items({"query": ""}, session_id=SESSION))
        print(f"unhide('1번')   : {restored}")
        print(f"after           : hidden_count={after['hidden_count']} "
              f"visibility={after['items'][0]['visibility']} "
              f"display_number={after['items'][0]['display_number']}")
        ok = ok and restored.get("ok") is True
        if restored.get("item_id") != hidden["item_id"]:
            print("number reference resolved to a different item")
            ok = False
        ok = ok and after["hidden_count"] == 0
        ok = ok and after["items"][0]["visibility"] == "original"
        ok = ok and after["items"][0]["display_number"] is None

    print(f"VERDICT         : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
