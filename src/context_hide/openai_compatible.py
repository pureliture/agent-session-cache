"""OpenAI-compatible HTTP transport for a local summary model endpoint."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from context_hide.transport import SummarizerError

_MAX_RESPONSE_BYTES = 65_536


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Do not forward private tool output to a redirected destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _completion_url(base_url: str) -> str:
    """Build a chat-completions URL after requiring a loopback-only origin."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise SummarizerError("invalid_endpoint")
    try:
        parsed = urlsplit(base_url.strip())
        hostname = parsed.hostname
        # Force validation of malformed ports before using the URL.
        _ = parsed.port
    except ValueError:
        raise SummarizerError("invalid_endpoint") from None

    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SummarizerError("invalid_endpoint")

    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        raise SummarizerError("invalid_endpoint")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = path + "/chat/completions"
    else:
        endpoint_path = path + "/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


class OpenAICompatibleTransport:
    """Send OpenAI chat-completion requests to a loopback endpoint only.

    The transport owns HTTP request/response conversion. Prompt construction and
    summary validation remain in ``ModelSummarizer``.
    """

    def __init__(self, base_url: str, *, opener: Any | None = None) -> None:
        self.endpoint = _completion_url(base_url)
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
        )

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: float = 60,
        **_provider_options: Any,
    ) -> dict[str, Any]:
        """Send a bounded standard chat-completions request asynchronously."""
        if not isinstance(model, str) or not model.strip() or timeout_seconds <= 0:
            raise SummarizerError("invalid_request")
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            raise SummarizerError("invalid_request") from None
        return await asyncio.to_thread(self._post, body, timeout_seconds)

    def _post(self, body: bytes, timeout_seconds: float) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            code = "redirect_rejected" if 300 <= exc.code < 400 else "http_error"
            exc.close()
            raise SummarizerError(code) from None
        except (TimeoutError, socket.timeout):
            raise SummarizerError("timeout") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise SummarizerError("timeout") from None
            raise SummarizerError("connection_error") from None
        except OSError:
            raise SummarizerError("connection_error") from None

        if not isinstance(raw, bytes) or len(raw) > _MAX_RESPONSE_BYTES:
            raise SummarizerError("invalid_response")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise SummarizerError("invalid_response") from None
        if not isinstance(decoded, dict):
            raise SummarizerError("invalid_response")
        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise SummarizerError("invalid_response")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise SummarizerError("invalid_response")
        if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
            raise SummarizerError("tool_calls_rejected")
        text = message.get("content")
        finish_reason = choice.get("finish_reason")
        if not isinstance(text, str) or not isinstance(finish_reason, str):
            raise SummarizerError("invalid_response")
        usage = decoded.get("usage")
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage if isinstance(usage, dict) else {},
        }
