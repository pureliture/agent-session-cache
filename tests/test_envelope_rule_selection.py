"""Envelope-shaped tool results must be selectable by the rule path.

Real Hermes sessions deliver tool results as single-line JSON envelopes:
``{"content": "1|..\\n2|..", "total_lines": ..}`` for read_file and
``{"output": "..", "exit_code": 0}`` for terminal runs. The rule selector works
line by line, so selection has to run on the decoded, line-number-free text the
reader sees — not on the raw envelope, which is one long escaped line that can
never be excerpted.
"""

from __future__ import annotations

import json

from context_hide.engine import ContextHideEngine
from context_hide.model import (
    Scope,
    ToolResultRecord,
    compute_invocation_digest,
    compute_sha256,
)


def _record(content: str, call_id: str = "call_env") -> ToolResultRecord:
    inv = {"tool_name": "read_file", "arguments": {"path": "/synthetic/rows.txt"}}
    return ToolResultRecord(
        call_id=call_id,
        result_position=0,
        content=content,
        content_sha256=compute_sha256(content),
        invocation=inv,
        invocation_digest=compute_invocation_digest(inv),
        status="ok",
        host_handle={"index": 1},
    )


def _read_file_envelope(rows: int = 60) -> str:
    body = "\n".join(f"{i}|Synthetic row {i:03d} detail line" for i in range(1, rows + 1))
    return json.dumps(
        {
            "content": body,
            "total_lines": rows,
            "file_size": 4096,
            "truncated": False,
            "is_binary": False,
            "is_image": False,
            "not_found": False,
        }
    )


def _terminal_envelope(lines: int = 80) -> str:
    out = "\n".join(f"Synthetic trace line {i:03d}" for i in range(1, lines + 1))
    return json.dumps({"output": out, "exit_code": 0, "error": None})


def test_rule_path_hides_read_file_envelope_and_restores():
    raw = _read_file_envelope()
    engine = ContextHideEngine()
    scope = Scope("hermes", "test", "sess-env-read")

    result = engine.hide_sync(scope, _record(raw))
    assert result.ok is True, result.error
    assert result.plan is not None
    assert result.plan.expected_content_sha256 == compute_sha256(raw)

    rendered = result.plan.replacement_text
    assert "[hidden:" in rendered
    assert "원문 발췌:" in rendered
    assert len(rendered) < len(raw)
    # The line-number noise the read_file tool adds must not ride along.
    assert "1|Synthetic row" not in rendered
    assert "Synthetic row 001" in rendered

    # Unhidden items produce no replacement plan again.
    assert engine.unhide(scope, result.plan.item_id).ok is True
    assert engine.project(scope, [_record(raw)]) == []


def test_rule_path_hides_terminal_output_envelope():
    raw = _terminal_envelope()
    engine = ContextHideEngine()
    scope = Scope("hermes", "test", "sess-env-out")

    result = engine.hide_sync(scope, _record(raw))
    assert result.ok is True, result.error
    assert result.plan is not None
    assert len(result.plan.replacement_text) < len(raw)
    assert "Synthetic trace line 001" in result.plan.replacement_text


def test_unknown_json_shape_still_refuses():
    """An envelope the decoder does not recognize must stay fail-open."""
    raw = json.dumps({"unmapped_entries": ["x" * 40 for _ in range(40)]})
    engine = ContextHideEngine()
    scope = Scope("hermes", "test", "sess-env-unknown")

    result = engine.hide_sync(scope, _record(raw))
    assert result.ok is False
    assert result.error == "verification_failed"
