"""Context-hide: host-neutral tool result summarization and compaction engine.

This file is the HERMES PLUGIN ROOT, not the engine package ``__init__``.

Layout contract (M1):

- the pure engine installs as ``context_hide`` (from ``src/context_hide``)
- the Hermes adapter installs as the top-level package ``hermes``
  (from ``adapters/hermes``) — its ``plugin`` submodule exposes ``register``
- this root installs as ``context_hide_plugin_root`` and is what a Hermes
  directory plugin loads directly: the plugin manager calls ``register`` on it

No relative engine imports live here: this module imports the installed engine
package, exactly like the adapter does, so one install keeps one engine copy.
The adapter package name is resolved defensively so an engine-only checkout
(``pip install context-hide`` without the adapter) still imports cleanly.
"""

from __future__ import annotations

from context_hide import engine as _engine  # noqa: F401  (re-exported below)
from context_hide.engine import ContextHideEngine


def _load_register():
    """Return the adapter's ``register``, or ``None`` when it is not installed."""
    try:
        from hermes.plugin import register  # wheel layout: adapters/hermes
    except ImportError:
        try:
            from context_hide_hermes.plugin import register  # remapped layout
        except ImportError:
            try:
                import importlib.util
                from pathlib import Path

                here = Path(__file__).resolve().parent
                candidate = here.parent / "adapters" / "hermes" / "plugin.py"
                if not candidate.is_file():
                    return None
                spec = importlib.util.spec_from_file_location(
                    "context_hide_hermes_adapter_plugin", candidate
                )
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.register
            except Exception:
                return None
    return register


register = _load_register()
_ADAPTER_IMPORT_ERROR: Exception | None = (
    None if callable(register) else ImportError("hermes.plugin not importable")
)

__all__ = ["ContextHideEngine", "engine", "register"]


def __getattr__(name: str):
    if name == "register" and register is None:  # pragma: no cover
        raise ImportError(
            "context-hide is installed without its Hermes adapter: "
            f"{_ADAPTER_IMPORT_ERROR}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
