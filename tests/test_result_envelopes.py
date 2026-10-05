"""Frozen synthetic Hermes envelopes; never derived from private session logs."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from context_hide.summary import (
    LFMSummarizer,
    SummarizerConfig,
    _SYSTEM_PROMPT,
    prepare_result_source,
    validate_summary_text,
)
from context_hide.transport import LFMUnavailable

CONFIG = '''[project]
name = "cedar-api"
requires-python = ">=3.12"
dependencies = ["starlette>=0.40", "aiohttp>=3.10", "attrs>=24", "referencing>=0.35"]
[dependency-groups]
dev = ["pytest>=8.3"]
'''

ENVELOPES = (
    (
        "R01",
        json.dumps(
            {
                "content": "\n".join(
                    f"{i}|{line}" for i, line in enumerate(CONFIG.splitlines(), 1)
                ),
                "total_lines": 6,
                "file_size": 233,
                "truncated": False,
                "is_binary": False,
                "is_image": False,
                "not_found": False,
            }
        ),
        ("3.12", "starlette", "aiohttp", "attrs", "referencing", "pytest"),
    ),
    (
        "R02",
        json.dumps(
            {
                "total_count": 3,
                "files": [
                    "/synthetic/cedar/AGENTS.md",
                    "/synthetic/cedar/components/context/AGENTS.md",
                    "/synthetic/cedar/tests/fixtures/discovery/AGENTS.md",
                ],
            }
        ),
        ("3", "AGENTS.md", "context", "discovery"),
    ),
    (
        "R04",
        json.dumps(
            {
                "output": ".. [100%]\n2 passed in 0.04s",
                "exit_code": 0,
                "error": None,
            }
        ),
        ("2", "passed", "exit", "0"),
    ),
)


def test_content_envelope_is_decoded_without_line_number_noise():
    captured: list[dict[str, Any]] = []

    async def generate(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {
            "text": json.dumps(
                {"summary": "Python 3.12; starlette, aiohttp, attrs, referencing; pytest for development."}
            )
        }

    summarizer = LFMSummarizer(generate=generate, config=SummarizerConfig())
    asyncio.run(
        summarizer.summarize(
            ENVELOPES[0][1],
            (),
            lambda: None,
            invocation={"tool_name": "read_file", "arguments": {"path": "INVOCATION_ONLY"}},
            context={"purpose": "CONTEXT_ONLY"},
        )
    )
    packet = json.loads(captured[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert packet["result"]["source"]["content"]["project"]["name"] == "cedar-api"
    assert packet["result"]["source"]["metadata"]["truncated"] is False
    assert "INVOCATION_ONLY" not in str(captured) and "CONTEXT_ONLY" not in str(captured)


def test_unrecognized_content_keeps_text_warnings_and_metadata():
    source = json.dumps(
        {
            "content": "7|Warning: signatures are unavailable.\n8|registry entry",
            "truncated": True,
            "next_offset": 9,
        }
    )
    assert prepare_result_source(source) == {
        "content": "Warning: signatures are unavailable.\nregistry entry",
        "metadata": {"truncated": True, "next_offset": 9},
    }
    ambiguous = json.dumps({"content": "1|first\n3|third", "truncated": True})
    assert prepare_result_source(ambiguous)["content"] == "1|first\n3|third"


def test_unsuccessful_repetition_encoding_retains_complete_original():
    source = "Header.\nSame line.\nOther line.\nSame line.\nSame line.\n"
    assert prepare_result_source(source) == source


def test_decoded_output_warning_cannot_be_silently_omitted():
    original = json.dumps(
        {
            "output": "The violet-module catalog uses exact-name matching.\nWarning: optional descriptions are omitted.",
            "exit_code": 0,
        }
    )
    assert (
        validate_summary_text(
            original,
            {"summary": "violet-module matches exact names; exit code 0."},
            (),
        )
        is None
    )
    assert (
        validate_summary_text(
            original,
            {"summary": "violet-module matches exact names; descriptions omitted; exit code 0."},
            (),
        )
        is None
    )
    assert (
        validate_summary_text(
            original,
            {"summary": "violet-module matches exact names; optional descriptions omitted; exit code 0."},
            (),
        )
        is not None
    )


def test_toml_non_json_values_keep_the_original_content():
    source = json.dumps({"content": "published = 2024-01-02", "truncated": False})
    assert prepare_result_source(source)["content"] == "published = 2024-01-02"


def test_decode_does_not_bypass_original_input_budget():
    config = SummarizerConfig(max_input_chars=400)
    source = json.dumps({"content": chr(34) * 250, "truncated": False})
    calls: list[dict[str, Any]] = []

    async def generate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"text": '{"summary":"anything"}'}

    with pytest.raises(LFMUnavailable, match="input_too_large"):
        asyncio.run(
            LFMSummarizer(generate=generate, config=config).summarize(
                source, (), lambda: None, invocation={}
            )
        )
    assert calls == []


def test_file_locations_are_quoted_without_changing_absolute_paths():
    content = prepare_result_source(ENVELOPES[1][1])["content"]
    assert '"/synthetic/cedar/AGENTS.md"' in content


def test_prompt_requests_explicit_observations_not_generic_attributes():
    assert "State concrete subject names, matched locations, development groups and exit code when present." in _SYSTEM_PROMPT


def test_absolute_path_sentence_punctuation_is_not_a_new_filename():
    source = ENVELOPES[1][1]
    assert (
        validate_summary_text(
            source,
            {
                "summary": "Three files: /synthetic/cedar/AGENTS.md, /synthetic/cedar/components/context/AGENTS.md, /synthetic/cedar/tests/fixtures/discovery/AGENTS.md."
            },
            (),
        )
        is not None
    )
    assert (
        validate_summary_text(
            source,
            {"summary": "See /invented/AGENTS.md."},
            (),
        )
        is None
    )


def test_generated_numbers_are_checked_against_decoded_stdout():
    original = ENVELOPES[2][1]
    assert validate_summary_text(original, {"summary": "2 tests passed; exit code 0."}, ()) is not None
    assert validate_summary_text(original, {"summary": "9 tests passed; exit code 0."}, ()) is None


def test_toml_envelope_preserves_development_group_as_structured_data():
    source = prepare_result_source(ENVELOPES[0][1])
    assert source["content"]["dependency-groups"]["dev"] == ["pytest>=8.3"]
    assert source["content"]["project"]["requires-python"] == ">=3.12"
