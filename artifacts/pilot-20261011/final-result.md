# context-hide Hermes 파일럿 — 최종 결과 (S6 인계)

- 일시: 2026-10-11 ~ 2026-10-12 (KST)
- 작성: Atlas
- 상태: **완료(PASS)** — S1~S6 종료. `default` 프로필에서만 활성, push/PR 없음(별도 승인 필요).

## 결론

- 실제 요청 경로에서 도구 결과 숨김→복원이 **3사이클 반복 확인**됐다. 숨김 시 요청은 3795→1071바이트(71.8% 감소), 복원 시 원문과 바이트 단위로 완전히 같았다(SHA-256 `c25d8388…4619`).
- 기능을 끄고 플러그인을 비활성화하면 새 세션에서 도구가 아예 보이지 않고 요청은 원문 그대로 나간다. 다시 켜면 정상 동작이 재현됐다(3번째 사이클).
- 비대상 프로필 6곳(daedalus·atlas·tyche·metis·plutus·ariadne)은 설정·등록 변화가 없다. 원문 이력(state.db)도 그대로다.
- 이 작업은 브리지와 별개다. 브리지 쪽 context-hide는 2026-10-10부터 비활성 상태를 유지하고 있다(ops PR #8).

## 후보 식별자 (설치한 코드)

- worktree: `~/.hermes/profiles/atlas/workspace/.worktrees/context-hide-hermes-pilot`, 브랜치 `atlas/hermes-plugin-pilot-20261011`
- 커밋(전부 로컬, push 없음): `34879cf` → `5e7d68d` → `7166067` → `e7e2f20` → `85fa709` → `678b649`
- 배포 트리 SHA-256: `4bdf5cd3ad319321cadbe6ea4e947a5742514154be89ecd1ea50b67bd141fbe4` (S6 전후 동일)

## 설치/환경 세대

- Hermes Agent v0.21.6+373.g46d7718 (2026.9.24), 설치 HEAD `46d7718a52`
- 설치 위치: `~/.hermes/plugins/context-hide/` (directory plugin, pip/엔트리포인트 없음)
- 대상: `default` 프로필(HOME `~/.hermes`), `toolsets`에 `context_hide` 포함
- 검증용 모델 전환: `step-5-preview-free`(opencode-go). 원복 백업 `config.yaml.bak-model-before-step5-20261011`

## 수행 명령 (요약)

| 목적 | 명령 |
|---|---|
| 활성화 | `hermes -p default plugins enable context-hide --no-allow-tool-override` + `config set plugins.entries.context-hide.settings.enabled true` |
| 비활성화(롤백) | `config set plugins.entries.context-hide.settings.enabled false` + `plugins disable context-hide` |
| 실검증 | `HERMES_DUMP_REQUESTS=1 hermes -p default chat --resume <세션> --query-file <프롬프트>` |
| 상태 확인 | `hermes -p default plugins list --plain --no-bundled`, `config get plugins.entries.context-hide.settings.enabled` |

## 실제 모델 호출 수 (request dump 기준)

| 구간 | 건수 |
|---|---|
| 최종 검증 세션 4개(`233013`·`f59c4f`·`07ef5f`·`29eb72`) | 43 |
| 사전 탐색·재시도(10-11 22:25 이후) | 56 |
| 10-11 낮 사전 시도 | 5 |
| 합계(10-11~10-12) | 104 |

- 429(레이트리밋) 재시도가 포함된 수치다. 원문·명령·토큰은 덤프에 남기지 않았고, 집계는 요청 덤프 분류(`dump-accounting.txt`)로 확인했다.

## PASS/FAIL

| 슬라이스 | 판정 | 한 줄 증거 |
|---|---|---|
| S1 후보 준비 | PASS | 실제 매니페스트 파서·로더 계약, wheel 빌드 |
| S2 어댑터 계약 | PASS | 등록순 합성 관측·설명 정정 |
| S3 분리 통합 시험 | PASS | 실제 PluginManager, payload 치환 관측 |
| S4 설치·활성화 | PASS | default만 설치, 비대상 무영향 |
| S5 실요청 교체·복원 | PASS | 2사이클 바이트 일치(3795→1071→3795) |
| S6 롤백·재활성·인계 | PASS | 꺼짐 실증(도구 not_found)→재활성→3번째 사이클 일치 |

## 미검증 범위 (솔직 기록)

- LFM/외부 요약 모델 미연결 — 이번 범위는 규칙 기반 엔진만 사용.
- 프로세스 재시작·만료·압축 후 숨김 상태 유지 보장 없음(메모리 저장소). `branch_scope` unknown.
- 실제 비용/캐시 절감 미측정(바이트 감소만 확인).
- WebUI/게이트웨이 경유 세션 실사용은 미검증(CLI chat 세션으로만 검증).
- 다른 프로필 확대·push/PR 미실행(승인 필요).

## 롤백 방법

```sh
hermes -p default config set plugins.entries.context-hide.settings.enabled false
hermes -p default plugins disable context-hide
```

새 세션에서 도구 미노출·원문 유지 확인(2026-10-12 실측). 설정 백업:
- `config.yaml.bak-pre-context-hide-20261011`
- `config.yaml.bak-pre-context-hide-toolset-20261011`
- `config.yaml.bak-model-before-step5-20261011`
- `config.yaml.bak-pre-s6-rollback-20261011`

## 증거 파일 (같은 폴더)

- `dump-accounting.txt` — 세션 233013 + S6 세션의 요청 덤프 전수 분류
- `s5-cycle1-step3-3.out` / `s5-cycle2-step4-1.out` / `s6-cycle3-step5-1.out` — 3사이클 CLI 출력
- `s6-disabled-f59c4f.out`, `s6-recheck-07ef5f.out`, `s6-recheck-29eb72.out` — 꺼짐/재활성 세션
- `s6-probe-*.log`, `s6-before.txt`, `probe_toolsets.py` — 등록 상태 probe
- worktree `milestones.md` — 슬라이스별 상세 기록(커밋 `678b649`)
