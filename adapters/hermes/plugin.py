"""Reference Hermes plugin wiring for context-hide (S4, M2-3).

This is the host-process shim between Hermes and the pure helpers
(`payload`, `session`, `handlers`). It owns everything host-specific:

- tool registration (`hide_context`, `list_context_items`, `unhide_context`)
- the `llm_request` middleware (record -> plan -> apply on a copy)
- scope derivation from host identifiers (never model arguments)
- enabled-off default, coexistence record-only mode, fail-open errors

It owns nothing engine-specific: summarization, protection rules,
storage, and visibility stay in `context_hide`. The engine never
imports this module; the wheel never ships it.

Verified against (stale-checkout warning: the local Hermes reference
checkout lags `origin/main`, so line numbers there drift; the contract
below follows the official middleware doc plus the transport/dispatch
code actually read):

- official doc `developer-guide/middleware`: `ctx.register_middleware(
  "llm_request", ...)` callbacks receive `request`, `original_request`
  plus runtime context (`session_id`, `task_id`, `turn_id`,
  `api_request_id`, `provider`, `model`, `api_mode`, ...); returning
  `{"request": {...}}` replaces the effective provider kwargs; optional
  `source`/`reason`/`name` strings land in `middleware_trace`.
- `agent/turn_api_request.py`: `llm_request` middleware runs right after
  `_build_api_kwargs`, so it sees provider-format payloads.
- `tools/registry.py` + `model_tools.handle_function_call`: tool
  handlers run as `handler(args, **kwargs)` and must return a string;
  `kwargs` carries identifiers (`task_id`, `session_id`, ...), never
  the conversation messages.

Decisions recorded here (first-version scope):

- Summarization is rule-only by default: the plugin creates the engine
  without an external summarizer, so no model call, no credentials,
  and no extra cost happen unless an operator explicitly wires one.
  When an LFM summarizer is added later it must be a separate local
  call, never the main-model path (`ctx.llm`), to keep usage, auth,
  and trust boundaries apart and keep fail-open simple.
- `branch_scope` is `"unknown"`: Hermes exposes no exact branch API
  to the plugin, so no full fork isolation is claimed. A compression
  rotation issues a new `session_id`, which naturally starts a fresh
  scope: hidden state never carries over silently.
- `transform_tool_result` is NOT used: stored conversation history is
  never rewritten, only the outgoing request copy.
- Logs carry counts, item IDs, and error codes only, never result
  bodies or commands.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .handlers import hide as _engine_hide
from .handlers import list_records as _engine_list
from .handlers import middleware_step as _engine_step
from .handlers import unhide as _engine_unhide
from .session import SessionTranscriptStore
from context_hide.engine import ContextHideEngine
from context_hide.model import Scope

logger = logging.getLogger(__name__)

SOURCE = "context-hide-hermes"
ADAPTER_ID = "hermes"
TOOLSET = "context_hide"

#: Scope branch marker: no exact branch API exists, so fork isolation
#: beyond the session is not claimed.
BRANCH_SCOPE = "unknown"

#: Must mirror the bridge's `INTERNAL_TOOL_NAMES`
#: (`openai_compatible_bridge/context_compaction.py`): when any of these
#: names appears in the request tool list, the bridge disables its own
#: compaction (`tool_name_collision`) so exactly one hide owner remains.
BRIDGE_COLLISION_TOOLS = frozenset(
    {"hide_context", "list_context_items", "unhide_context"}
)

HIDE_TOOL = "hide_context"
LIST_TOOL = "list_context_items"
UNHIDE_TOOL = "unhide_context"

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    HIDE_TOOL: {
        "name": HIDE_TOOL,
        "description": (
            "Hide one already-transmitted tool result: later model requests "
            "carry its summary instead of the full body. "
            "Pass the tool_call_id shown by the model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The tool call ID whose result body should be hidden.",
                }
            },
            "required": ["tool_call_id"],
            "additionalProperties": False,
        },
    },
    LIST_TOOL: {
        "name": LIST_TOOL,
        "description": (
            "List hidden items (IDs and restore availability). "
            "Original bodies and commands are never shown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional filter over item IDs and summaries.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    UNHIDE_TOOL: {
        "name": UNHIDE_TOOL,
        "description": (
            "Restore one hidden item to its original body without "
            "re-running any command. Pass an item_id from list_context_items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The hidden item ID to restore.",
                }
            },
            "required": ["item_id"],
            "additionalProperties": False,
        },
    },
}


def bridge_should_yield(tool_names: Any) -> bool:
    """True when the bridge must stand down for this plugin session.

    Mirrors the bridge `tool_name_collision` rule: any overlap between
    the request tool list and `BRIDGE_COLLISION_TOOLS` disables bridge
    compaction so hiding has exactly one owner.
    """
    if isinstance(tool_names, dict):
        names = set(tool_names)
    else:
        try:
            names = set(tool_names or [])
        except TypeError:
            return False
    return bool(BRIDGE_COLLISION_TOOLS & {str(name) for name in names})


def _default_count_llm_middlewares() -> int:
    """Count registered `llm_request` middlewares, 1 when unknowable.

    Runs inside the Hermes process, so a lazy host import is fine here
    (the pure helpers stay host-free). Any import problem degrades to 1,
    i.e. "no evidence of others", keeping single-owner behavior.
    """
    try:
        from hermes_cli.plugins import _delivery_manager

        manager = _delivery_manager()
        middlewares = getattr(manager, "_middleware", {}).get("llm_request", [])
        return len(middlewares)
    except Exception:
        return 1


@dataclass
class PluginState:
    """Live per-registration plugin state (one per Hermes process load)."""

    engine: ContextHideEngine = field(default_factory=ContextHideEngine)
    store: SessionTranscriptStore = field(default_factory=SessionTranscriptStore)
    profile: str = "default"
    is_enabled: Callable[[], bool] = field(default=lambda: False)
    count_middlewares: Callable[[], int] = field(
        default=_default_count_llm_middlewares)


def scope_for(state: PluginState, session_id: Any) -> Scope | None:
    """Derive the engine scope from host identifiers, None when absent.

    Blank `session_id` would mix unrelated sessions, so the caller must
    fail open (tool: `not_found`, middleware: pass the request through).
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    profile = state.profile if isinstance(state.profile, str) and state.profile else "default"
    return Scope(
        adapter_id=ADAPTER_ID,
        host_profile=profile,
        session_id=session_id.strip(),
        branch_scope=BRANCH_SCOPE,
    )


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def make_hide_handler(state: PluginState) -> Callable[..., str]:
    """Build the `hide_context` tool handler bound to this registration."""

    def hide_context(args: Any, session_id: str = "", **_extra: Any) -> str:
        if not state.is_enabled():
            return _dump({"ok": False, "error": "disabled"})
        target = args.get("tool_call_id") if isinstance(args, dict) else None
        if not isinstance(target, str) or not target.strip():
            return _dump({"ok": False, "error": "missing_tool_call_id"})
        scope = scope_for(state, session_id)
        if scope is None:
            return _dump({"ok": False, "error": "not_found"})
        result = _engine_hide(state.engine, state.store, scope, target.strip())
        if not result.ok or result.plan is None:
            return _dump({"ok": False, "error": result.error or "not_found"})
        return _dump({
            "ok": True,
            "item_id": result.plan.item_id,
            "visibility_version": result.plan.visibility_version,
        })

    return hide_context


def make_list_handler(state: PluginState) -> Callable[..., str]:
    """Build the `list_context_items` tool handler bound to this registration."""

    def list_context_items(args: Any = None, session_id: str = "", **_extra: Any) -> str:
        if not state.is_enabled():
            return _dump({"ok": False, "error": "disabled"})
        query = ""
        if isinstance(args, dict) and isinstance(args.get("query"), str):
            query = args["query"]
        scope = scope_for(state, session_id)
        if scope is None:
            return _dump({"ok": False, "error": "not_found"})
        items = _engine_list(state.engine, scope, query)
        return _dump({"ok": True, "count": len(items), "items": items})

    return list_context_items


def make_unhide_handler(state: PluginState) -> Callable[..., str]:
    """Build the `unhide_context` tool handler bound to this registration."""

    def unhide_context(args: Any, session_id: str = "", **_extra: Any) -> str:
        if not state.is_enabled():
            return _dump({"ok": False, "error": "disabled"})
        target = args.get("item_id") if isinstance(args, dict) else None
        if not isinstance(target, str) or not target.strip():
            return _dump({"ok": False, "error": "not_found"})
        scope = scope_for(state, session_id)
        if scope is None:
            return _dump({"ok": False, "error": "not_found"})
        result = _engine_unhide(state.engine, scope, target.strip())
        if not result.ok or result.item is None:
            return _dump({"ok": False, "error": result.error or "not_found"})
        return _dump({
            "ok": True,
            "item_id": result.item.item_id,
            "visibility_version": result.item.version,
        })

    return unhide_context


def make_middleware(state: PluginState) -> Callable[..., dict[str, Any] | None]:
    """Build the `llm_request` middleware callback for this registration."""

    def on_llm_request(**kwargs: Any) -> dict[str, Any] | None:
        try:
            if not state.is_enabled():
                return None
            request = kwargs.get("request")
            if not isinstance(request, dict):
                return None
            scope = scope_for(state, kwargs.get("session_id", ""))
            if scope is None:
                return {
                    "request": copy.deepcopy(request),
                    "source": SOURCE,
                    "reason": "missing_session",
                }
            api_mode = kwargs.get("api_mode", "")
            if not isinstance(api_mode, str):
                api_mode = ""
            if state.count_middlewares() > 1:
                # Another llm_request middleware is installed: each one
                # receives the original request and only the last return
                # value survives, so applying here could silently vanish.
                # Record the transmission (tools still work) but leave
                # this request untouched.
                stored, skipped, _error = state.store.record_request(
                    scope.scope_key(), api_mode, request)
                logger.debug(
                    "context-hide coexistence: recorded %d results, skipped %d, "
                    "request unchanged", len(stored), len(skipped))
                return {
                    "request": copy.deepcopy(request),
                    "source": SOURCE,
                    "reason": "other_middleware_present",
                }
            step = _engine_step(state.engine, state.store, scope, api_mode, request)
            applied = step["applied"]
            skipped = step["skipped"]
            reason = step["trace"].get("reason", "")
            if not reason:
                reason = f"applied {len(applied)}, skipped {len(skipped)}"
            logger.debug(
                "context-hide middleware: %s (found %s)",
                reason, step["trace"].get("found", 0))
            return {
                "request": step["request"],
                "source": SOURCE,
                "reason": reason,
            }
        except Exception as exc:
            # Fail open: Hermes logs and continues with the original request.
            logger.warning(
                "context-hide middleware fail-open: %s", type(exc).__name__)
            return None

    return on_llm_request


def register(ctx: Any) -> PluginState:
    """Hermes plugin entry point: register 3 tools + 1 middleware.

    The plugin starts disabled: tools report `disabled` and the
    middleware returns `None` (no recording, no rewriting) until the
    operator sets `plugins.entries.<plugin_id>.settings.enabled: true`.
    """
    try:
        profile = getattr(ctx, "profile_name", "default") or "default"
    except Exception:
        profile = "default"

    def _is_enabled() -> bool:
        try:
            get_config = getattr(ctx, "get_config", None)
            if not callable(get_config):
                return False
            return bool(get_config("enabled", False))
        except Exception:
            return False

    state = PluginState(profile=profile, is_enabled=_is_enabled)
    ctx.register_tool(
        name=HIDE_TOOL,
        toolset=TOOLSET,
        schema=TOOL_SCHEMAS[HIDE_TOOL],
        handler=make_hide_handler(state),
        description=TOOL_SCHEMAS[HIDE_TOOL]["description"],
    )
    ctx.register_tool(
        name=LIST_TOOL,
        toolset=TOOLSET,
        schema=TOOL_SCHEMAS[LIST_TOOL],
        handler=make_list_handler(state),
        description=TOOL_SCHEMAS[LIST_TOOL]["description"],
    )
    ctx.register_tool(
        name=UNHIDE_TOOL,
        toolset=TOOLSET,
        schema=TOOL_SCHEMAS[UNHIDE_TOOL],
        handler=make_unhide_handler(state),
        description=TOOL_SCHEMAS[UNHIDE_TOOL]["description"],
    )
    ctx.register_middleware("llm_request", make_middleware(state))
    logger.debug("context-hide plugin registered (profile=%s)", profile)
    return state
