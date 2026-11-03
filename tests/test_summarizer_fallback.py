"""End-to-end fail-open tests from local transport through the hide engine."""
from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from context_hide.engine import ContextHideEngine
from context_hide.model import (
    Scope,
    ToolResultRecord,
    compute_invocation_digest,
    compute_sha256,
)
from context_hide.openai_compatible import OpenAICompatibleTransport
from context_hide.summary import ModelSummarizer, SummarizerConfig


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _limit: int = -1) -> bytes:
        return self.body


class FakeOpener:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls = 0

    def open(self, _request, timeout: float):
        del timeout
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return FakeResponse(self.payload)


def _long_body() -> str:
    lines = [f"Output row {i:03d} details" for i in range(80)]
    lines.extend([
        "id: 01234567-89ab-cdef-0123-456789abcdef",
        "results/output.json generated successfully",
    ])
    return "\n".join(lines)


def _record(body: str) -> ToolResultRecord:
    invocation = {"tool_name": "terminal", "arguments": {"command": "synthetic"}}
    return ToolResultRecord(
        call_id="call-fallback",
        result_position=0,
        content=body,
        content_sha256=compute_sha256(body),
        invocation=invocation,
        invocation_digest=compute_invocation_digest(invocation),
        status="ok",
        host_handle={"message_index": 1},
    )


@pytest.mark.parametrize(
    ("failure_mode", "expected_calls"),
    [
        ("timeout", 1),
        ("connection_error", 1),
        ("invalid_model_json", 1),
        ("length", 1),
        ("input_too_large", 0),
    ],
)
def test_local_summarizer_failures_fall_back_to_rules(failure_mode, expected_calls):
    good_summary = (
        "id: 01234567-89ab-cdef-0123-456789abcdef; "
        "results/output.json generated successfully."
    )
    if failure_mode == "timeout":
        response: dict[str, Any] | Exception = TimeoutError("PRIVATE_TIMEOUT_DETAIL")
    elif failure_mode == "connection_error":
        response = urllib.error.URLError("PRIVATE_CONNECTION_DETAIL")
    else:
        content = "not-json PRIVATE_MODEL_TEXT" if failure_mode == "invalid_model_json" else json.dumps(
            {"summary": good_summary}
        )
        finish_reason = "length" if failure_mode == "length" else "stop"
        response = {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 12},
        }

    opener = FakeOpener(response)
    transport = OpenAICompatibleTransport("http://127.0.0.1/v1", opener=opener)
    config = SummarizerConfig(
        model="provider-neutral-test-model",
        max_input_bytes=1 if failure_mode == "input_too_large" else 12_288,
    )
    summarizer = ModelSummarizer(generate=transport.generate, config=config)
    body = _long_body()

    result = ContextHideEngine().hide_sync(
        Scope("hermes", "test-profile", "session-fallback"),
        _record(body),
        summarizer=summarizer,
    )

    assert result.ok is True
    assert result.item is not None
    assert result.item.compaction_source == "rule"
    assert result.plan is not None
    assert result.plan.replacement_text != body
    assert opener.calls == expected_calls
