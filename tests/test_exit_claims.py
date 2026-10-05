"""Unit tests for exit claim verification against authoritative metadata."""
from __future__ import annotations

import pytest

from context_hide.summary import validate_summary_text, verify_exit_claims


@pytest.mark.parametrize(
    "claim",
    [
        "Exit code 0.",
        "exit_code: 1",
        "종료 코드 0",
        "Exit code 0; exit code 1.",
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        '{"exit_code": null, "stdout": "row 0 has 1 entry"}',
        '{"exit_code": "unknown", "stdout": "row 0 has 1 entry"}',
        '{"exit_code": false, "stdout": "row 0 has 1 entry"}',
        "row 0 has 1 entry",
    ],
)
def test_unknown_exit_code_cannot_borrow_unrelated_digits(source: str, claim: str):
    assert validate_summary_text(source, {"summary": claim}, ()) is None
    assert not verify_exit_claims(source, claim)


@pytest.mark.parametrize(
    "claim",
    [
        "exit code is 0",
        "exit code was 0",
        'exit code: "0"',
        "return code 0",
        "return_code=0",
        "exit status 0",
        "Exit code 0; exit code is 1",
    ],
)
def test_every_exit_claim_is_checked_even_if_digits_exist(claim: str):
    source = '{"exit_code": null, "stdout": "row 0 has 1 entry"}'
    assert validate_summary_text(source, {"summary": claim}, ()) is None
    assert not verify_exit_claims(source, claim)


@pytest.mark.parametrize(
    "claim",
    [
        "Exit code 0.",
        "exit code is 0",
        "return code 0",
    ],
)
def test_explicit_source_exit_code_can_be_summarized(claim: str):
    source = "2 tests passed; exit code 0."
    assert validate_summary_text(source, {"summary": claim}, ()) == {"summary": claim}
    assert verify_exit_claims(source, claim) is True


def test_all_exit_claims_checked_against_authoritative_metadata():
    source = '{"exit_code": 0, "stdout": "row 1; 2 passed"}'
    assert validate_summary_text(source, {"summary": "Exit code 0; exit code is 1."}, ()) is None
    assert not verify_exit_claims(source, "Exit code 0; exit code is 1.")
