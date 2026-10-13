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
from .presentation import PresentationRegistry, decorate_items
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

#: Static system-prompt guidance for the human supervision protocol. It is
#: fixed text (never per-session state) rendered once per new session; when
#: the host lacks the section API or the plugin loads disabled, only the
#: tool descriptions carry the protocol.
SUPERVISION_SECTION_ID = "context-hide.human-supervision"
SUPERVISION_SECTION_MAX_CHARS = 1200
SUPERVISION_SECTION_TEXT = (
    "context-hide 도구는 사용자의 세션 채팅 요청에 따라 숨김 상태를 "
    "확인·복원할 때 쓴다. 사용자가 '숨김 목록 보여줘'처럼 요청하면 "
    "list_context_items를 호출해 숨김 항목만 골라 표시 번호(1부터), 도구, "
    "원문 크기, 숨김 시각, 한 줄 요약이 담긴 표로 보여준다. 'N번 복원해' "
    "요청이 오면 번호를 item_id로 바꾸려 하지 말고 사용자가 말한 숫자를 "
    "그대로 unhide_context의 item_id 자리에 넘긴다 — 도구가 마지막으로 "
    "반환한 목록의 N번으로 해석한다. 정확한 item_id를 알면 그대로 쓴다. "
    "목록을 보여준 직후에는 다시 조회하지 않는다(재조회는 번호 기준을 "
    "바꾼다). 오류 no_recent_list면 목록을 먼저 보여주고, "
    "unknown_display_number면 새 목록을 보여주고 다시 선택받는다. "
    "'전부 되돌려'는 필터 없는 목록에서 숨김 item_id를 고정한 뒤 한 건씩 "
    "복원한다. 숨김이 0건이면 '현재 숨겨진 항목 없음'이라고만 말한다. "
    "숨김 상태는 같은 프로필·세션·프로세스에서만 보이고 재시작하면 "
    "사라진다. 재시작 여부를 모를 때 원인을 단정하지 않는다."
)

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
            "이 세션에서 숨김(compacted) 처리된 도구 결과를 사람이 확인할 수 "
            "있게 반환한다. 사용자가 '숨김 목록 보여줘', '뭐 숨겨져 있어?'처럼 "
            "요청하면 이 도구를 호출하고, 결과(숨김 건수·표시 번호·도구·원문 "
            "크기·숨김 시각·한 줄 요약)를 사람이 읽기 좋은 표로 정리해 "
            "보여준다. 항목 본문과 명령 원문은 반환하지 않는다. "
            "표시 번호는 목록 호출마다 새로 부여되며, 이후 도착하는 "
            "'N번 복원해' 요청은 마지막으로 반환한 이 목록의 번호 기준으로 "
            "해석된다."
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
            "숨긴 항목 하나를 재실행 없이 원문으로 복원한다. 사용자가 "
            "'N번 복원해', '그거 되돌려'처럼 요청하면 사용자가 말한 번호를 "
            "item_id 자리에 그대로 넘긴다(예: '2', '2번', '#2') — 도구가 "
            "마지막으로 보여준 목록의 N번으로 해석하므로 번호를 기억하거나 "
            "item_id로 바꿀 필요가 없다. 정확한 item_id를 알면 그대로 쓴다. "
            "no_recent_list면 먼저 목록을 보여주고, unknown_display_number면 "
            "새 목록을 보여주고 다시 선택받는다. 존재하지 않으면 다른 ID로 "
            "대체하지 않고 오류를 사용자에게 설명한다."
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
    """Count the OTHER registered `llm_request` middleware callbacks.

    Verified on 2026-10-11 against the installed runtime
    (``hermes_cli.middleware.apply_llm_request_middleware`` plus
    ``PluginManager.invoke_middleware``): the request chain COMPOSES in
    registration order -- each callback receives the previous callback's
    payload, and the last dict returned under ``request`` is what the provider
    sees. So a later middleware can overwrite our replacement and we can
    overwrite theirs; the chain does not by itself guarantee a single hide
    owner.

    This probe therefore only reports how many OTHER callbacks exist. The
    middleware reads that number and decides what to do (see
    ``on_llm_request``). Runs inside the Hermes process, so a lazy host import
    is fine here (the pure helpers stay host-free). Any import problem degrades
    to 0 = "no evidence of others", which keeps the plugin's own behaviour
    instead of silently withholding it.
    """
    try:
        from hermes_cli.plugins import _delivery_manager

        manager = _delivery_manager()
        middlewares = getattr(manager, "_middleware", {}).get("llm_request", [])
        return max(0, len(middlewares) - 1)
    except Exception:
        return 0


@dataclass
class PluginState:
    """Live per-registration plugin state (one per Hermes process load)."""

    engine: ContextHideEngine = field(default_factory=ContextHideEngine)
    store: SessionTranscriptStore = field(default_factory=SessionTranscriptStore)
    presentations: PresentationRegistry = field(default_factory=PresentationRegistry)
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
        original_bytes = 0
        if result.item is not None:
            original_bytes = len(result.item.original.encode("utf-8"))
        state.presentations.record_hidden(
            scope.scope_key(), result.plan.item_id, original_bytes)
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
        scope_key = scope.scope_key()
        live_ids = {
            item.item_id for item in state.engine.store.items(scope)
        }
        state.presentations.prune(scope_key, live_ids)
        items = decorate_items(items, state.presentations.snapshot().get(scope_key, {}))
        state.presentations.record_view(scope_key, [
            (item["display_number"], item["item_id"])
            for item in items if item["display_number"] is not None
        ])
        hidden_count = sum(item["visibility"] == "compacted" for item in items)
        prefix = "검색 조건에 맞는" if query else "현재"
        message = f"{prefix} 숨겨진 항목 {hidden_count}건" if hidden_count else f"{prefix} 숨겨진 항목 없음"
        return _dump({
            "ok": True, "count": len(items), "items": items,
            "hidden_count": hidden_count, "message": message,
        })

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
        resolved, resolve_error = state.presentations.resolve_reference(
            scope.scope_key(), target.strip())
        if resolve_error is not None:
            # Fail closed on number references: never guess a target.
            return _dump({"ok": False, "error": resolve_error})
        result = _engine_unhide(state.engine, scope, resolved or "")
        if not result.ok or result.item is None:
            return _dump({"ok": False, "error": result.error or "not_found"})
        state.presentations.mark_visible(scope.scope_key(), result.item.item_id)
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
            if state.count_middlewares() > 0:
                # Another llm_request middleware is installed. The installed
                # chain composes rewrites in registration order, so OUR
                # replacement is not silently dropped -- but if we run first,
                # a later middleware can still overwrite it, and if we run
                # last we overwrite theirs. Report the situation instead of
                # pretending single ownership: keep recording (tools keep
                # working) but leave this request untouched.
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

    # Optional host capability: freeze the human-supervision protocol into
    # each new session prompt. Absent API, disabled plugin, or a duplicate
    # registration must never block tool/middleware registration.
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section) and state.is_enabled():
        try:
            register_section(
                SUPERVISION_SECTION_ID,
                SUPERVISION_SECTION_TEXT,
                position="after_memory",
                max_chars=SUPERVISION_SECTION_MAX_CHARS,
            )
        except Exception:
            logger.debug("context-hide: system prompt section unavailable")
    logger.debug("context-hide plugin registered (profile=%s)", profile)
    return state
