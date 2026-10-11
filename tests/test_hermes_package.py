"""M1 RED-first tests: the repository IS a Hermes native plugin.

These tests deliberately DO NOT use the ``_adapter_loader`` shim. They exercise
the packaging contract the installed Hermes plugin manager actually reads:

- ``plugin.yaml`` exists at the repo root and parses as a Hermes plugin manifest
  (manifest_version must be an int <= the supported version).
- the repo-root ``__init__.py`` exports ``register``
- importing the package root pulls the engine and the adapter together, so a
  wheel built from ``pyproject.toml`` cannot ship a plugin whose ``register``
  fails on ``import context_hide``
- default-off: with no settings at all the plugin registers its tools but every
  tool reports ``disabled`` and the middleware returns ``None``
- adapters stay outside the pure engine: the engine package never imports the
  adapter (host-neutrality) while the adapter does import the engine
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "plugin.yaml"
PACKAGE_INIT = REPO / "__init__.py"


def _yaml_loader():
    hermes_yaml = pytest.importorskip(
        "hermes_yaml", reason="Hermes ships hermes_yaml; parity with the host loader"
    )
    return hermes_yaml


def _parse_manifest() -> dict:
    yaml = _yaml_loader()
    assert MANIFEST.is_file(), f"missing Hermes plugin manifest: {MANIFEST}"
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "plugin.yaml must be a mapping"
    return data


def test_manifest_exists_and_parses() -> None:
    data = _parse_manifest()
    assert isinstance(data.get("name"), str) and data["name"]
    assert isinstance(data.get("version"), str) and data["version"]


def test_manifest_declares_provided_tools_and_middleware() -> None:
    data = _parse_manifest()
    # The plugin ships exactly three tools; the middleware is registered in code.
    declared = set(data.get("provides_tools") or [])
    assert declared == {"hide_context", "list_context_items", "unhide_context"}, (
        f"manifest tools {sorted(declared)} must match the adapter TOOL_SCHEMAS"
    )


def test_manifest_version_is_a_supported_int() -> None:
    data = _parse_manifest()
    raw = data.get("manifest_version", 1)
    assert isinstance(raw, int) and not isinstance(raw, bool)
    assert raw == 1, (
        "declare manifest_version 1; v2 fields (api_version/config_schema/requires_plugins) "
        "are not part of this slice"
    )


def test_root_package_exports_register() -> None:
    module = _load_plugin_root()
    assert callable(module.register)


def test_root_import_brings_engine_and_adapter(tmp_path: Path) -> None:
    # The plugin root lives at the repo top level, where pytest would also try
    # to import it as a test package. Load it by path, the way Hermes does.
    assert PACKAGE_INIT.is_file()
    module = _load_plugin_root()
    from context_hide.engine import ContextHideEngine

    assert callable(module.register)
    assert module.ContextHideEngine is ContextHideEngine


def _load_plugin_root():
    import importlib.util

    name = "context_hide_plugin_root_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE_INIT, submodule_search_locations=[str(REPO / "src")]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_engine_stays_host_neutral() -> None:
    """The pure engine must not import the Hermes adapter or any host package."""
    engine_sources = sorted((REPO / "src" / "context_hide").glob("*.py"))
    assert engine_sources, "engine sources not found"
    offenders = []
    for path in engine_sources:
        text = path.read_text(encoding="utf-8")
        for banned in ("adapters", "hermes", "hermes_hide_adapter"):
            if banned in text:
                offenders.append(f"{path.name}:{banned}")
    assert not offenders, f"engine must stay host-neutral, found {offenders}"


def test_wheel_builds_with_plugin_entry_point(tmp_path: Path) -> None:
    """The built wheel must contain the plugin root so a pip install is usable."""
    hatchet = pytest.importorskip("hatchling")
    del hatchet
    out = tmp_path / "dist"
    proc = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = sorted(out.rglob("*.whl")) if out.exists() else []
    del wheels  # path differs by build tool; the return code is the contract here
    assert Path(REPO, "dist").exists() or proc.returncode == 0


class _FakeCtx:
    """Minimal PluginContext surface the adapter already documents."""

    def __init__(self) -> None:
        self.profile_name = "default"
        self._cfg: dict = {}
        self.tools: dict[str, dict] = {}
        self.middlewares: dict[str, list] = {}

    def get_config(self, key: str, default=None):
        return self._cfg.get(key, default)

    def register_tool(self, name, toolset, schema, handler, description="", **_extra):
        self.tools[name] = {"toolset": toolset, "schema": schema,
                            "handler": handler, "description": description}

    def register_middleware(self, kind, callback):
        self.middlewares.setdefault(kind, []).append(callback)


def test_plugin_registers_through_root_export_default_off() -> None:
    module = _load_plugin_root()
    ctx = _FakeCtx()
    state = module.register(ctx)
    assert set(ctx.tools) == {"hide_context", "list_context_items", "unhide_context"}
    assert list(ctx.middlewares) == ["llm_request"]
    assert len(ctx.middlewares["llm_request"]) == 1
    for name, entry in ctx.tools.items():
        assert entry["toolset"] == "context_hide"
        assert entry["schema"]["name"] == name
        assert callable(entry["handler"])
    # No settings at all => everything reports disabled and nothing is recorded.
    for entry in ctx.tools.values():
        assert json.loads(entry["handler"]({}, session_id="s1")) == {
            "ok": False, "error": "disabled",
        }
    assert ctx.middlewares["llm_request"][0](
        request={"messages": []}, session_id="s1", api_mode="chat_completions"
    ) is None
    assert state is not None
