"""Hermetic tests for the OpenAI-compatible local summarizer transport."""
from __future__ import annotations

import asyncio
import json
import urllib.error
from email.message import Message
from typing import Any

import pytest

from context_hide.openai_compatible import OpenAICompatibleTransport
from context_hide.transport import SummarizerError


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | bytes) -> None:
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _limit: int = -1) -> bytes:
        return self.body


class FakeOpener:
    def __init__(self, payload: dict[str, Any] | bytes | Exception) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def open(self, request, timeout: float):
        self.request = request
        self.timeout = timeout
        if isinstance(self.payload, Exception):
            raise self.payload
        return FakeResponse(self.payload)


def test_openai_compatible_request_and_response_conversion():
    content = '{"summary":"Grounded local summary."}'
    opener = FakeOpener({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 93, "completion_tokens": 11},
    })
    transport = OpenAICompatibleTransport(
        "http://127.0.0.1/v1/", opener=opener,
    )
    messages = [
        {"role": "system", "content": "fixed system prompt"},
        {"role": "user", "content": "untrusted source packet"},
    ]
    response_format = {"type": "json_object"}

    result = asyncio.run(transport.generate(
        model="local-model:configured-id",
        messages=messages,
        max_tokens=57,
        temperature=0,
        response_format=response_format,
        timeout_seconds=13,
        reasoning={"effort": "none"},
    ))

    request = opener.request
    assert request is not None
    assert request.full_url == "http://127.0.0.1/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") is None
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "model": "local-model:configured-id",
        "messages": messages,
        "max_tokens": 57,
        "temperature": 0,
        "response_format": response_format,
    }
    assert opener.timeout == 13
    assert result == {
        "text": content,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 93, "completion_tokens": 11},
    }


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (TimeoutError("private source body"), "timeout"),
        (urllib.error.URLError("private prompt"), "connection_error"),
        (
            urllib.error.HTTPError(
                "http://127.0.0.1/v1/chat/completions", 307,
                "private redirect target", Message(), None,
            ),
            "redirect_rejected",
        ),
    ],
)
def test_transport_maps_connection_failures_to_safe_codes(failure, code: str):
    transport = OpenAICompatibleTransport(
        "http://localhost/v1", opener=FakeOpener(failure),
    )
    with pytest.raises(SummarizerError) as caught:
        asyncio.run(transport.generate(
            model="model-id", messages=[], max_tokens=20, temperature=0,
            timeout_seconds=2,
        ))
    assert caught.value.reason == code
    assert str(caught.value) == code
    assert "private" not in str(caught.value)
    if isinstance(failure, urllib.error.HTTPError):
        assert failure.closed


def test_transport_rejects_invalid_json_and_empty_choices_without_body_leak():
    for response in (b"not-json-private-response", {"choices": []}):
        transport = OpenAICompatibleTransport(
            "http://localhost/v1", opener=FakeOpener(response),
        )
        with pytest.raises(SummarizerError) as caught:
            asyncio.run(transport.generate(
                model="model-id", messages=[], max_tokens=20, temperature=0,
                timeout_seconds=2,
            ))
        assert caught.value.reason == "invalid_response"
        assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com/v1",
        "http://user:password@127.0.0.1/v1",
        "http://127.0.0.1/v1?token=secret",
    ],
)
def test_transport_rejects_nonlocal_or_credential_bearing_endpoints(base_url: str):
    with pytest.raises(SummarizerError) as caught:
        OpenAICompatibleTransport(base_url)
    assert caught.value.reason == "invalid_endpoint"
    assert base_url not in str(caught.value)


def test_transport_ignores_environment_proxies_for_local_requests():
    transport = OpenAICompatibleTransport("http://127.0.0.1/v1")
    import urllib.request

    proxy_handlers = [
        handler for handler in getattr(transport._opener, "handlers", [])
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert not any(
        getattr(handler, "proxies", {})
        for handler in proxy_handlers
    )
