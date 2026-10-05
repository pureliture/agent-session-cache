"""Unit tests for context_hide core models and contracts."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from context_hide.model import (
    ContextItem,
    EngineConfig,
    ItemMetadata,
    MutationResult,
    ReplacementPlan,
    Scope,
    ToolResultRecord,
    compute_invocation_digest,
    compute_sha256,
)


def test_scope_immutability_and_key():
    scope = Scope(
        adapter_id="bridge",
        host_profile="test",
        session_id="session_42",
        branch_scope="feat_branch",
    )
    assert scope.adapter_id == "bridge"
    assert scope.host_profile == "test"
    assert scope.session_id == "session_42"
    assert scope.branch_scope == "feat_branch"
    assert scope.scope_key() == "bridge:test:session_42:feat_branch"

    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.adapter_id = "mutated"  # type: ignore

    default_scope = Scope("b", "p", "s")
    assert default_scope.branch_scope == "default"
    assert default_scope.scope_key() == "b:p:s:default"


def test_tool_result_record_immutability_and_defaults():
    record = ToolResultRecord(call_id="call-1")
    assert record.call_id == "call-1"
    assert record.result_position == 0
    assert record.content == ""
    assert record.content_sha256 == ""
    assert record.invocation == {}
    assert record.invocation_digest == ""
    assert record.status == "ok"
    assert record.host_handle is None

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = "error"  # type: ignore


def test_tool_result_record_sha256_verification():
    text = "pytest output content with logs"
    correct_sha = compute_sha256(text)
    record = ToolResultRecord(
        call_id="call-1",
        content=text,
        content_sha256=correct_sha,
    )
    assert record.verify_sha256() is True

    tampered_record = ToolResultRecord(
        call_id="call-1",
        content=text,
        content_sha256="0" * 64,
    )
    assert tampered_record.verify_sha256() is False


def test_replacement_plan_immutability_and_echo():
    handle = {"message_idx": 3, "role": "tool"}
    plan = ReplacementPlan(
        item_id="item_0123456789abcdef",
        expected_content_sha256=compute_sha256("original"),
        replacement_text="[hidden:item_0123456789abcdef]\n...",
        visibility_version=1,
        host_handle=handle,
    )
    assert plan.item_id == "item_0123456789abcdef"
    assert plan.visibility_version == 1
    assert plan.host_handle == handle

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.visibility_version = 2  # type: ignore


def test_mutation_result_variants():
    plan = ReplacementPlan(
        item_id="item_123",
        expected_content_sha256="abc",
        replacement_text="text",
        visibility_version=1,
    )
    success = MutationResult(ok=True, plan=plan)
    assert success.ok is True
    assert success.error is None
    assert success.plan == plan
    assert success.replacement == plan

    failure = MutationResult(ok=False, error="protected_error")
    assert failure.ok is False
    assert failure.error == "protected_error"
    assert failure.plan is None
    assert failure.replacement is None


def test_item_metadata_serialization():
    meta = ItemMetadata(
        item_id="item_123",
        call_id="call_456",
        content_sha256="sha_789",
        visibility="compacted",
        version=1,
        expires_at=1700000000.0,
        source="lfm",
        bytes_size=4096,
        tool_name="terminal",
        excerpt_lines=("line 1", "line 2"),
        summary="shortened summary",
    )
    assert meta.tool_call_id == "call_456"
    assert meta.compaction_source == "lfm"

    d = meta.to_dict()
    assert d["item_id"] == "item_123"
    assert d["call_id"] == "call_456"
    assert d["content_sha256"] == "sha_789"
    assert d["visibility"] == "compacted"
    assert d["version"] == 1
    assert d["source"] == "lfm"
    assert d["bytes_size"] == 4096
    assert d["tool_name"] == "terminal"
    assert d["excerpt_lines"] == ("line 1", "line 2")
    assert d["summary"] == "shortened summary"


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert cfg.ttl_seconds == 86400
    assert cfg.max_bytes == 67108864
    assert cfg.min_chars == 800
    assert cfg.max_pending == 4
    assert cfg.max_evidence_excerpts == 12
    assert cfg.max_excerpt_line_length == 240
    assert cfg.max_summary_length == 1600
    assert cfg.max_invocation_bytes == 2048
    assert cfg.lfm_max_input_bytes == 12288


def test_compute_sha256_canonical_hashing():
    raw_ascii = "sample tool output"
    expected = hashlib.sha256(raw_ascii.encode("utf-8")).hexdigest()
    assert compute_sha256(raw_ascii) == expected

    korean = "한글 도구 결과 및 오류 출력"
    expected_korean = hashlib.sha256(korean.encode("utf-8")).hexdigest()
    assert compute_sha256(korean) == expected_korean


def test_compute_invocation_digest_key_order_independence():
    inv_a = {
        "tool_name": "search_files",
        "arguments": {"path": "/app", "pattern": "*.py", "limit": 10},
    }
    inv_b = {
        "arguments": {"pattern": "*.py", "limit": 10, "path": "/app"},
        "tool_name": "search_files",
    }
    assert compute_invocation_digest(inv_a) == compute_invocation_digest(inv_b)

    digest_with_omitted = compute_invocation_digest(inv_a, omitted_options={"verbose": True})
    assert digest_with_omitted != compute_invocation_digest(inv_a)


def test_contracts_json_schema_compatibility():
    contracts_dir = Path(__file__).resolve().parent.parent / "contracts"

    with open(contracts_dir / "tool_result_record.json", encoding="utf-8") as f:
        schema_record = json.load(f)
    sample_record = {
        "call_id": "call-1",
        "result_position": 0,
        "content": "output text",
        "content_sha256": compute_sha256("output text"),
        "invocation": {"tool_name": "terminal", "arguments": {"command": "ls"}},
        "invocation_digest": compute_invocation_digest({"tool_name": "terminal", "arguments": {"command": "ls"}}),
        "status": "ok",
        "host_handle": {"idx": 1},
    }
    jsonschema.validate(sample_record, schema_record)

    with open(contracts_dir / "replacement_plan.json", encoding="utf-8") as f:
        schema_plan = json.load(f)
    sample_plan = {
        "item_id": "item_0123456789abcdef",
        "host_handle": {"idx": 1},
        "expected_content_sha256": compute_sha256("output text"),
        "replacement_text": "[hidden:item_0123456789abcdef]\n...",
        "visibility_version": 1,
    }
    jsonschema.validate(sample_plan, schema_plan)

    with open(contracts_dir / "errors.json", encoding="utf-8") as f:
        schema_errors = json.load(f)
    sample_error = {
        "error": "protected_error",
        "category": "policy_refusal",
        "description": "Output contains unresolved error or failure pattern",
        "retryable": False,
    }
    jsonschema.validate(sample_error, schema_errors)
