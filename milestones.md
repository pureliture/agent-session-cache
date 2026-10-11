# M1 milestones: Hermes 세션에 context-hide 연결

> **Goal (approved plan, quoted):** context-hide를 Hermes의 요청 전송 단계에 연결하고, 승인된 `daedalus` 또는 `default` 중 한 프로필에서만 숨김·조회·복원과 비대상 프로필 무영향을 검증한다.

## Status

- **Selected contract:** `agentic-execution` (사용자 명시 선택, 2026-10-11).
- **Plan:** `~/.hermes/profiles/atlas/workspace/.hermes/plans/2026-10-11_104747-context-hide-hermes-profile-pilot.md`
- **Approved scope (user, 2026-10-11):** 구현·설치·활성화 진행. 적용 프로필은 `daedalus` 또는 `default` 중 하나. Atlas·다른 프로필 자동 확대 없음. Daedalus는 활성 후보이며, `default`는 추가 대상이 아닌 대체 후보로 남긴다. 프로필 확정은 사용자의 다음 지시가 필요할 수 있다.
- **Authority boundary (unchanged):** 프로필 선택 변경, main 병합, pull request 생성, 전체 gateway 재시작, 브리지 재배포, 과금, 데이터 유실, 비가역 변경은 중단하고 승인을 요청한다. Non-goals: 양쪽 프로필 동시 적용, `atlas`/기타 프로필 적용, 유료 서비스, 외부 DB, LFM 요약기 본체 연결.

## Slices

| # | Slice | Status | Observable result |
|---|---|---|---|
| S1 | Native plugin 후보 (manifest + register export + 엔진 import + wheel) | DONE | 실제 Hermes 매니페스트 파서와 로더 계약을 만족하는 후보 |
| S2 | 어댑터 계약 정합화 (공존 판정·설명 정정) | DONE | 등록 성공이 아닌 실제 전송 요청 기준 계약 일치 |
| S3 | 분리 환경 실제 PluginContext 통합 시험 | ACTIVE | 숨김→조회→복원이 실제 전송 payload에서 관측 |
| S4 | 대상 프로필 설치·활성화 | pending | 한 프로필에서만 도구 노출, 비대상 무영향 |
| S5 | 실제 provider 요청 교체·복원 반복 | pending | 숨긴 뒤 요청이 짧아지고 복원 뒤 원문 바이트·해시 일치 |
| S6 | 롤백·최종 인계 | pending | 대상 원래 동작, 비대상 변화 없음 |

## Active slice evidence — S2 DONE

- **Verified installed chain (2026-10-11, real modules):**
  `apply_llm_request_middleware` + `PluginManager.invoke_middleware` compose
  rewrites in REGISTRATION ORDER — callback 2 receives callback 1's payload, the
  last dict under `request` wins, and `original_request` stays pristine.
  Observed: `payload={'steps': ['first','second']}`,
  `trace=[{'source':'first'},{'source':'second'}]`, `original={'steps': []}`.
- **Adapter correction:** the docstring claimed "each one receives the original
  request and only the last return value survives", and the probe counted ALL
  middlewares with a fallback of 1 — both wrong for this runtime. The probe now
  counts OTHERS (`len-1`, fallback 0) and the gate is `> 0`. Semantics unchanged
  in the single-owner case; the coexistence branch now matches the real chain.
- **Tests updated to the corrected contract:** `tests/test_hermes_edges.py`
  (counter fallback 0, `count_middlewares=lambda: 0` for the sole-middleware
  path), `tests/test_hermes_plugin.py` (coexistence at 1 other, plus a new
  `test_count_middlewares_counts_others_not_self`), and the integration test's
  sole-middleware fixture set to 0.
- **Regression after the change:** 161 passed, 4 skipped, twice (0.67s / 0.65s),
  exit 0. Wheel still builds (`Successfully built context_hide-0.1.0-py3-none-any.whl`).
- **Direct end-to-end drive through the plugin's own middleware** (project venv,
  real plugin root, real handler objects): first transmission keeps the original
  body; after `hide_context` the second transmission carries the replacement
  (4180 → 757 bytes, contains the item id); after `unhide_context` the body rides
  again byte-identical. No provider was contacted.

## Prior slice evidence — S1 DONE

- Branch/worktree: `atlas/hermes-plugin-pilot-20261011` at
  `~/.hermes/profiles/atlas/workspace/.worktrees/context-hide-hermes-pilot`,
  created from `main` `8e85a4a4d0ee28e24e4884f62191aae966c90619`.
- **Manifest (real host parser):** `hermes_cli.plugins_manifest.parse_manifest_file`
  accepted `plugin.yaml` → `key=context-hide`, `name=context-hide`,
  `kind=standalone`, `provides_tools=[hide_context, list_context_items, unhide_context]`,
  `manifest_version=1 <= 2`. `provides_tools` is informational only; tools are
  registered in code through `ctx.register_tool`.
- **Register export (real loader path):** the plugin root loads by file path
  (the way `_load_directory_module` does), exports `register(ctx)`, and `register`
  wires exactly `hide_context`, `list_context_items`, `unhide_context` plus one
  `llm_request` middleware.
- **Default off:** with no settings, every tool returns
  `{"ok": false, "error": "disabled"}` and the middleware returns `None` — nothing
  is recorded, nothing is rewritten.
- **Enabled isolation:** with `enabled=true` and one middleware, the caller's
  request dict is never mutated; the first transmission still carries the original
  body; the second carries the short replacement (smaller, contains `item_id`);
  `list` leaks no original body; after `unhide` the body rides again byte-identical.
  Scope is isolated by profile and by session (different profile/session →
  `not_found`, no cross-scope restore).
- **No host internals:** loading the plugin root imports no `agent.*`,
  `hermes_cli.*`, or `tools.*` module.
- **Wheel:** `python -m build --wheel` succeeded; the wheel contains
  `context_hide_plugin_root/__init__.py`, `context_hide_plugin_root/plugin.yaml`,
  the engine, and the adapter package.
- **Regression:** `pytest -q --ignore=tests/test_model.py --ignore=tests/test_summary_validation.py`
  → 160 passed, 4 skipped, twice (0.81s / 0.57s), exit 0.
  `tests/test_model.py` and `tests/test_summary_validation.py` fail to COLLECT on
  `main` too (`ModuleNotFoundError: No module named 'rpds.rpds'`) — pre-existing,
  not caused by this branch.

## Pre-existing condition (not a blocker, recorded honestly)

`uv sync --frozen` on `main` cannot install the project into an isolated venv:
hatchling's `sources` remap is incompatible with editable installs (editables
issue #20), and the repo has no committed `uv.lock`. Working around this by
patching `uv.lock` or hand-building the venv was **not** done — it would have
hidden the real packaging defect. The wheel build path is verified instead.

## Design amendments recorded (requirements-preserving)

1. **Adapter import name.** The wheel cannot rename `adapters/hermes` without
   breaking editable installs, so the plugin root resolves the adapter through a
   small fallback chain (`hermes.plugin`, then `context_hide_hermes.plugin`, then
   a path load). Required behaviour and non-goals unchanged.
2. **`provides_tools` is documentation, not behaviour.** Verified against the host
   parser: registration happens in `register()` via `ctx.register_tool`. The test
   asserts both stay in sync.

## Active slice evidence — S3 DONE

- **Real PluginManager load (throwaway `HERMES_HOME`):**
  `tools_and_tests/probe_plugin_manager.py` copies only `__init__.py` +
  `plugin.yaml` into a temp home and runs the installed
  `hermes_cli.plugins.PluginManager.discover_and_load()` with
  `plugins.entries.context-hide.settings.enabled: true`. Result:
  `loaded=True, enabled=True, deferred=False`, module imported as
  `hermes_plugins.context_hide`, `register` callable,
  `tools_registered=['hide_context','list_context_items','unhide_context']`,
  `llm_request` middlewares `1`, **VERDICT PASS** (twice).
- **Default-off on the real registry:** with the settings entry absent the same
  load registers the tools and middleware, and the registered middleware returns
  `None` — the body is left unchanged.
- **Enabled end-to-end through the real registry and the real middleware
  chain** (no provider contacted):
  1. first transmission keeps the original body,
  2. `hide_context` → `ok`, item `item_2c6fa15a34a3cfbb`,
  3. second transmission replaced (4180 → 757 bytes, contains the item id),
  4. `original_payload` pristine,
  5. `list_context_items` count 1, leaks no original body,
  6. `unhide_context` ok,
  7. restored byte-identical.
- **Interpreter note:** the Hermes-managed environment has ruamel.yaml but not
  cachetools; the project venv has cachetools but not ruamel. The probe ran with
  `uv run --python <hermes-env> --with cachetools --isolated`, which changes no
  user environment.
- **Regression:** 161 passed, 4 skipped, twice (0.66s / 0.47s), exit 0.
  Evidence: `artifacts/m3-plugin-manager-probe.log`, `artifacts/m3-probe-repeat.log`,
  `artifacts/m3-enabled-endtoend.log`, `artifacts/m3-full.log`, `artifacts/m3-full-2.log`.

## Next action — S4 (needs your decision)

S4 installs the candidate into ONE real profile. This is the first slice that
touches live operator state, so it stops for approval:

- **Which profile?** `daedalus` (recommended) or `default`. Only one.
- **Install path.** Native directory plugin under that profile's
  `~/.hermes/profiles/<name>/plugins/context-hide/` (or `~/.hermes/plugins/` for
  default), copied from this candidate — no pip install, no entry point, no
  shared-environment change.
- **Enable posture.** Installed but `settings.enabled: false` until S5 starts,
  so a live session of that profile cannot rewrite any request before the
  verified test session.
- **Not included:** commit, push, PR, main merge, gateway restart (tool changes
  take effect on a new session), any other profile, the bridge, or any
  credential.

Branch state: `atlas/hermes-plugin-pilot-20261011` at
`~/.hermes/profiles/atlas/workspace/.worktrees/context-hide-hermes-pilot`,
base `main` `8e85a4a4d0ee28e24e4884f62191aae966c90619`. Uncommitted, no push.

## Stop conditions

데이터 유실, 비용/유료 서비스, 사용자 접근 서비스 중단, 비가역 변경(IAM 키·시크릿 분실) 발생 시 즉시 중단하고 보고한다.

## Active slice evidence — S4 DONE

- **Target profile:** `default` (`~/.hermes`), chosen by the user.
- **Commits (local only, no push, no PR):**
  `34879cf95d4122f55028c988477de8cc2eceb7bc` — "feat(hermes): make the repo an
  installable Hermes plugin"; `5e7d68d0a560be95a9760384c05ea14a856b3cb8` —
  "fix(hermes): pin the engine and adapter inside the deployed plugin", which
  fixes the vendor `sys.path` defect found during install and makes the repo
  mirror the deployed tree byte-for-byte.
- **Install:** directory plugin at `~/.hermes/plugins/context-hide/`
  containing `__init__.py` (register entry point), `plugin.yaml`, and pinned
  `vendor/context_hide/` + `vendor/context_hide_hermes/` copies. No pip install,
  no entry point, no shared-environment change.
- **Allow-list:** `hermes -p default config set plugins.enabled` added
  `context-hide` to the existing three. Backup kept at
  `~/.hermes/config.yaml.bak-pre-context-hide-20261011`.
- **Feature posture:** `plugins.entries.context-hide.settings.enabled: false`
  (verified via `hermes -p default config get`), so no live session can rewrite a
  request yet.
- **Real default-profile load:** `discovered=True, enabled=True`, module
  `hermes_plugins.context_hide`, `register` callable,
  `tools_registered=['hide_context','list_context_items','unhide_context']`,
  `llm_request` middlewares `1`, and the three pre-existing user plugins still
  load.
- **Inert when off:** with the setting absent/false the registered middleware
  leaves the body unchanged and the tools report `disabled`.
- **Enabled end-to-end on the real registry** (no provider contacted): first
  transmission keeps the original body; `hide_context` ok; second transmission
  replaced 4180 → 757 bytes; `unhide_context` ok; restored byte-identical;
  turning the setting back off makes it inert again.
- **Non-target profiles unchanged:** `daedalus`, `atlas`, `ariadne`, `metis`,
  `plutus`, `tyche` keep their previous `plugins.enabled` values.
- **Install-time defect found and fixed:** the first installed root put the
  vendor PACKAGE directories on `sys.path`, so `import context_hide` failed and
  the host reported "has no register() function". The parent `vendor/` directory
  is the correct entry. The repository root and `vendor/` now mirror the
  installed tree byte-for-byte.

## Active slice evidence — S5 (live) — envelope fix + real provider replacement PASS

- **Live session:** `20261011_233013_544841` (default profile, model
  `step-5-preview-free` via opencode-go), resumed with the three tool steps.
  Trace: `hide=ok list=1 unhide=ok`.
- **Defect found by the first live attempt:** `hide_context` returned
  `verification_failed`. Cause: real host tool results ride as single-line JSON
  envelopes (`read_file`: `{"content": "1|..\n2|.."}` with ESCAPED newlines;
  `terminal`: `{"output": ".."}`), and the rule selector works line by line — it
  can never excerpt one long escaped line. Bridge precedent for decoding first:
  `openai_compatible_bridge/context_compaction.py` (`json.loads(raw)`).
- **Fix (engine, shared by every consumer):** new `summary.selectable_text()` +
  `strip_consecutive_numbering()`; `engine.hide()` now selects the baseline and
  rule-side required evidence from the decoded view, while identity, hashing,
  the stored original, and the payload replacement keep the raw string. LFM path
  and unknown JSON shapes are untouched (fail-open stays).
- **RED → GREEN:** `tests/test_envelope_rule_selection.py` (3 tests) failed on
  the old code (`artifacts/envelope-red.log`), passes now
  (`artifacts/envelope-green.log`).
- **Deployed-plugin drive (no provider):** hide ok → second transmission
  1395 → 1161 bytes with the item id → list count 1 → unhide ok → restored
  byte-identical.
- **Real request-dump evidence (the live session, chronological):**
  17 transmissions carry the raw 3795-byte envelope; after `hide_context` the
  next 4 transmissions carry the 1071-byte replacement
  (`[hidden:item_67076f438c7213b6]` + excerpts); after `unhide_context` the
  following transmission carries the raw 3795 bytes again, byte-identical.
  **Shrink 3795 → 1071 bytes (71.8 %).**
- **Regression:** 164 passed, 4 skipped, twice, exit 0
  (`artifacts/envelope-full-3.log`, `artifacts/envelope-full-4.log`).
- **Host-env support fix riding along:** `store.py` falls back to the pinned
  `context_hide._tlru.TLRUCache` when the host interpreter lacks `cachetools`;
  `src/` and `vendor/` are byte-identical and all three docs trees match the
  deployed plugin.
