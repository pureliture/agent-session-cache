"""Hermes adapter tool allowlist (M2-2).

Adapter-owned tool rules. The engine never sees host tool names here;
it only receives the filtered invocation. Order, duplicate, secret, and
size checks stay in the engine policy. This module only decides which
tools and arguments the Hermes adapter accepts.
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED_TOOLS = ("terminal", "read_file", "search_files")

INTERNAL_TOOLS = ("hide_context", "list_context_items", "unhide_context")

FIELDS: dict[str, set[str]] = {
    "terminal": {"command", "workdir"},
    "read_file": {"path", "offset", "limit"},
    "search_files": {"pattern", "target", "path", "file_glob"},
}

DEFAULTS: dict[str, dict[str, Any]] = {
    "terminal": {
        "background": False,
        "pty": False,
        "persist_on_release": False,
        "heartbeat": 0,
        "notify": False,
    },
    "read_file": {},
    "search_files": {"limit": 50, "offset": 0, "order": "discovery",
                     "output_mode": "content", "context": 0},
}

REQUIRED: dict[str, str] = {
    "terminal": "command",
    "read_file": "path",
    "search_files": "pattern",
}


class AdapterRejected(Exception):
    """Adapter-level invocation refusal carrying a contract error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def parse_raw_arguments(raw: Any) -> dict[str, Any]:
    """Coerce raw tool arguments to a dict or raise invalid_invocation."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise AdapterRejected("invalid_invocation") from None
        if not isinstance(parsed, dict):
            raise AdapterRejected("invalid_invocation")
        return parsed
    raise AdapterRejected("invalid_invocation")


def filter_invocation(
    tool_name: Any, raw_args: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter a tool call to the engine invocation shape.

    Returns (invocation, omitted_options). Raises AdapterRejected with one
    of: invalid_invocation, internal_target, unsupported_tool,
    unsupported_arguments.
    """
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise AdapterRejected("invalid_invocation")
    name = tool_name.strip()
    if name in INTERNAL_TOOLS:
        raise AdapterRejected("internal_target")
    if name not in FIELDS:
        raise AdapterRejected("unsupported_tool")
    args = parse_raw_arguments(raw_args)
    allowed_extra = set(DEFAULTS[name]) | ({"timeout"} if name == "terminal" else set())
    if set(args) - FIELDS[name] - allowed_extra:
        raise AdapterRejected("unsupported_arguments")
    for key, default in DEFAULTS[name].items():
        if key in args and (type(args[key]) is not type(default) or args[key] != default):
            raise AdapterRejected("unsupported_arguments")
    if "timeout" in args and (type(args["timeout"]) is not int or args["timeout"] < 1):
        raise AdapterRejected("unsupported_arguments")
    required = REQUIRED[name]
    if not isinstance(args.get(required), str) or not args[required].strip():
        raise AdapterRejected("invalid_invocation")
    for key in FIELDS[name] & set(args):
        if key in {"offset", "limit"}:
            if type(args[key]) is not int or args[key] < 1:
                raise AdapterRejected("invalid_invocation")
        elif not isinstance(args[key], str):
            raise AdapterRejected("invalid_invocation")
    filtered = {key: args[key] for key in sorted(FIELDS[name] & set(args))}
    invocation = {"tool_name": name, "arguments": filtered}
    omitted = {key: value for key, value in args.items() if key not in filtered}
    return invocation, omitted
