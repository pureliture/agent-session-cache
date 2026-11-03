"""Provider-neutral summarizer API and deprecated-name compatibility tests."""
from __future__ import annotations

import importlib

import pytest

from context_hide.summary import ModelSummarizer
from context_hide.transport import SummarizerError, SummarizerUnavailable


def test_summarizer_unavailable_has_a_provider_neutral_public_name():
    assert SummarizerUnavailable.__name__ == "SummarizerUnavailable"
    assert issubclass(SummarizerUnavailable, SummarizerError)


def test_lfm_unavailable_name_is_only_a_deprecated_compatibility_alias():
    transport = importlib.import_module("context_hide.transport")
    with pytest.warns(DeprecationWarning, match="SummarizerUnavailable"):
        legacy = getattr(transport, "LFMUnavailable")
    assert legacy is SummarizerUnavailable


def test_lfm_summarizer_name_is_only_a_deprecated_compatibility_alias():
    summary = importlib.import_module("context_hide.summary")
    with pytest.warns(DeprecationWarning, match="ModelSummarizer"):
        legacy = getattr(summary, "LFMSummarizer")
    assert legacy is ModelSummarizer
