"""Production request regression: structure-based prompts, no case-name routing."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from context_hide.summary import (
    LFMSummarizer,
    SummarizerConfig,
    _PLAIN_RESULT_PROMPT,
    _SYSTEM_PROMPT,
)
from context_hide.transport import LFMUnavailable


@pytest.mark.parametrize(
    ("original", "structured"),
    [
        (
            "arbitrary-module performs exact-name lookup.\nCommand result: 12 checks passed.",
            False,
        ),
        (
            json.dumps({"output": "2 passed units", "exit_code": 0}),
            True,
        ),
        (
            "violet-module catalog uses exact-name matching.\n" * 60
            + "Warning: optional descriptions omitted.",
            True,
        ),
    ],
)
def test_request_uses_input_structure_without_invocation_context(
    original: str, structured: bool
) -> None:
    captured: list[dict[str, Any]] = []

    async def generate(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {
            "text": json.dumps(
                {"summary": original.splitlines()[0] if not structured else "2 passed units."}
            )
        }

    async def exercise() -> None:
        try:
            await LFMSummarizer(generate=generate, config=SummarizerConfig()).summarize(
                original,
                (),
                lambda: None,
                invocation={"command": "DO_NOT_SEND"},
                context={"purpose": "DO_NOT_SEND"},
            )
        except LFMUnavailable:
            pass  # Only request assembly is under test; existing validator still rejects bad output.

    asyncio.run(exercise())
    assert len(captured) == 1
    request = captured[0]
    system = request["messages"][0]["content"]
    user = request["messages"][1]["content"]

    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"

    if structured:
        assert system == _SYSTEM_PROMPT
        assert user.startswith("Write factual findings as JSON.\n\n")
        packet = json.loads(user.split("\n\n", 1)[1])
        assert "result" in packet
        assert "source" in packet["result"]
        assert packet["required_evidence"] == []
        if original.startswith("{"):
            assert packet["result"]["source"] == {"output": "2 passed units", "exit_code": 0}
        else:
            assert isinstance(packet["result"]["source"], dict)
            assert "lossless_decode" in packet["result"]["source"]
    else:
        assert system == _PLAIN_RESULT_PROMPT
        assert "Preserve the subject name" in system
        assert not user.startswith("Write factual findings as JSON.")
        packet = json.loads(user)
        assert packet["source"] == original
        assert packet["required_evidence"] == []

    assert "DO_NOT_SEND" not in json.dumps(request)
    assert request["temperature"] == 0
    assert request["reasoning"] == {"effort": "none"}
