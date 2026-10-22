"""S3 evidence: load this repository as a Hermes directory plugin for real.

Runs the ACTUAL ``hermes_cli.plugins.PluginManager`` discovery + load against a
throwaway ``HERMES_HOME`` that contains a copy of the candidate plugin root and
manifest. Nothing outside the temp home is written; no provider is contacted.

Usage (needs ruamel.yaml AND cachetools in one interpreter):
    uv run --python <hermes-env-python> --with cachetools --isolated \
        python tools_and_tests/probe_plugin_manager.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLUGIN_FILES = ("__init__.py", "plugin.yaml")
PLUGIN_ID = "context-hide"


def main() -> int:
    hermes_repo = os.environ.get(
        "HERMES_REPO", str(Path.home() / ".hermes" / "hermes-agent")
    )
    if str(hermes_repo) not in sys.path:
        sys.path.insert(0, str(hermes_repo))

    home = Path(tempfile.mkdtemp(prefix="hermes-plugin-probe-"))
    plugin_dir = home / "plugins" / PLUGIN_ID
    plugin_dir.mkdir(parents=True)
    for name in PLUGIN_FILES:
        source = REPO / name
        if not source.is_file():
            print(f"MISSING CANDIDATE FILE: {source}")
            return 2
        (plugin_dir / name).write_bytes(source.read_bytes())
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
    tools = sorted(getattr(entry, "tools_registered", []) or []) if entry else []
    middlewares = manager._middleware.get("llm_request", [])
    module = getattr(entry, "module", None) if entry else None

    print(f"hermes_home     : {home}")
    print(f"loaded          : {'context-hide' in loaded}")
    print(f"enabled         : {getattr(entry, 'enabled', None)}")
    print(f"deferred        : {getattr(entry, 'deferred', None)}")
    print(f"module          : {module}")
    print(f"module.register : {callable(getattr(module, 'register', None))}")
    print(f"tools_registered: {tools}")
    print(f"llm_request mws : {len(middlewares)}")

    ok = (
        entry is not None
        and getattr(entry, "enabled", False)
        and callable(getattr(module, "register", None))
        and set(tools) == {"hide_context", "list_context_items", "unhide_context"}
        and len(middlewares) == 1
    )
    print(f"VERDICT         : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
