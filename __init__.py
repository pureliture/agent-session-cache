"""Context-hide: host-neutral tool result summarization and compaction engine.

This file is the HERMES PLUGIN ROOT — the Hermes plugin manager loads it as
``hermes_plugins.context_hide`` and calls ``register(ctx)`` on it.

Installed layout (a self-contained directory plugin)::

    <hermes_home>/plugins/context-hide/
        __init__.py                     <- this file (register entry point)
        plugin.yaml                     <- Hermes plugin manifest
        vendor/context_hide/            <- the pure engine (pinned copy)
        vendor/context_hide_hermes/     <- the Hermes adapter (pinned copy)

The engine and the adapter are pinned alongside the manifest so a live profile
never depends on a checkout directory, a site-packages version, or a network
install. ``vendor/`` is added to ``sys.path`` — the PARENT of both packages,
not the packages themselves, so ``import context_hide`` and
``import context_hide_hermes`` resolve — for the duration of this process only.

The plugin is disabled by default: with no ``enabled`` setting every tool reports
``disabled`` and the middleware returns ``None`` (nothing recorded, nothing
rewritten) until an operator turns it on.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor"


def _prepend_vendor() -> None:
    """Make the pinned engine and adapter importable for this plugin only.

    ``vendor/`` is the PARENT of both packages, so it is the single directory
    that has to be on ``sys.path``.
    """
    path = str(_VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_engine():
    """Import the pinned engine, returning ``(engine_module, error)``."""
    try:
        import context_hide as engine_pkg  # pinned under vendor/

        return engine_pkg, None
    except Exception as exc:  # pragma: no cover - install defect
        return None, exc


def _load_register():
    """Return the adapter's ``register(ctx)`` or ``None`` when unavailable."""
    try:
        from context_hide_hermes.plugin import register
    except Exception:
        return None
    return register


_prepend_vendor()
_engine, _engine_error = _load_engine()
ContextHideEngine = getattr(_engine, "ContextHideEngine", None)
register = _load_register()
_ADAPTER_IMPORT_ERROR: Exception | None = (
    None if callable(register) else (_engine_error or ImportError("adapter unavailable"))
)

__all__ = ["ContextHideEngine", "engine", "register"]


def __getattr__(name: str):
    if name == "register" and register is None:  # pragma: no cover
        raise ImportError(
            f"context-hide plugin is incomplete: {_ADAPTER_IMPORT_ERROR}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
