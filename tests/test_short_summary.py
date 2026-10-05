"""The summarizer only shortens result content; metadata is not generated."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from context_hide.model import SpanChoice
from context_hide.store import MemoryContextStore
from context_hide.summary import LFMSummarizer, SummarizerConfig, validate_summary_text


def test_one_summary_field_is_generated_applied_and_restored():
    source = "\n".join(
        ["Repeated ordinary catalog description."] * 80
        + [
            "Catalog uses exact-name matching.",
            "Warning: optional descriptions are omitted.",
        ]
    )
    invocation = {"tool_name": "read_file", "arguments": {"path": "synthetic/catalog.py"}}
    summary = {"summary": "Catalog uses exact-name matching; optional descriptions are omitted."}
    requests: list[dict[str, Any]] = []

    async def generate(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {"text": json.dumps(summary), "finish_reason": "stop"}

    summarizer = LFMSummarizer(generate=generate, config=SummarizerConfig())
    result = asyncio.run(
        summarizer.summarize(source, (), lambda: None, invocation=invocation)
    )
    assert result == summary

    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["summary"]
    assert set(schema["properties"]) == {"summary"}
    prompt = requests[0]["messages"][0]["content"]
    assert "English" in prompt
    assert "missing exit" not in prompt

    packet = json.loads(requests[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert "invocation" not in packet
    assert "verified_facts" not in packet

    store = MemoryContextStore(min_chars=100)
    saved = store.compact(
        affinity="test",
        tool_call_id="one",
        original=source,
        tool_name="read_file",
        invocation=invocation,
        choice=SpanChoice(("Warning: optional descriptions are omitted.",), "lfm", result),
    )
    assert saved.ok and saved.item is not None
    assert saved.item.compaction_source == "lfm"
    assert summary["summary"] in saved.item.compacted
    assert "実行" not in saved.item.compacted
    assert "実行:" not in saved.item.compacted
    assert "실행:" not in saved.item.compacted
    assert "미확인·한계:" not in saved.item.compacted

    assert store.unhide("test", saved.item.item_id).ok
    assert store.visible_content("test", "one", source) == source


def test_claims_are_grounded_in_result_not_invocation_or_purpose():
    invocation = {"tool_name": "read_file", "arguments": {"path": "invocation/only.py"}}
    assert validate_summary_text("Names use exact matching.", {"summary": "Names use exact matching."}, (), invocation=invocation) is not None
    assert validate_summary_text("Names use exact matching.", {"summary": "See invocation/only.py."}, (), invocation=invocation) is None
    assert validate_summary_text("10 tests passed.", {"summary": "999 tests passed."}, ()) is None
    assert validate_summary_text("10 tests passed.", {"summary": "Exit code 0."}, ()) is None
