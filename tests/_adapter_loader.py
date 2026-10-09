"""Load the Hermes reference adapter by file path.

The adapter stays outside the published core wheel and is never imported
through a generic top-level package name (which can collide with the
host's own packages). Tests use `hermes_adapter()` to get the package
module loaded from this repository's `adapters/hermes/` directory under
the unique name `hermes_hide_adapter`, then reach submodules as
attributes (`adapter.handlers`, `adapter.session`, ...). Relative
imports inside the adapter package keep working under that name.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_PACKAGE_NAME = "hermes_hide_adapter"


def hermes_adapter() -> types.ModuleType:
    """Return the adapter package module, loading it from disk on first call."""
    existing = sys.modules.get(_PACKAGE_NAME)
    if existing is not None:
        return existing
    pkg_dir = Path(__file__).resolve().parents[1] / "adapters" / "hermes"
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        pkg_dir / "__init__.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    for child in ("payload", "allowlist", "session", "handlers", "plugin"):
        try:
            importlib.import_module(f"{_PACKAGE_NAME}.{child}")
        except ImportError:
            pass
    return module
