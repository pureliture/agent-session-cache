"""Transport protocols and pluggable summarizer interfaces."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable
import warnings

OnCall = Callable[[], None]


class SummarizerError(Exception):
    """Raised when an external summarizer encounters a failure or validation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SummarizerUnavailable(SummarizerError):
    """Raised when summarization is unavailable or its result is rejected."""


def __getattr__(name: str) -> Any:
    """Expose the former model-specific error name as a deprecated alias."""
    if name == "LFMUnavailable":
        warnings.warn(
            "LFMUnavailable is deprecated; use SummarizerUnavailable",
            DeprecationWarning,
            stacklevel=2,
        )
        return SummarizerUnavailable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for pluggable local or remote summary models."""

    async def summarize(
        self,
        original: str,
        required_evidence: tuple[str, ...],
        on_call: OnCall | None = None,
        *,
        invocation: dict[str, Any] | None = None,
        context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Request and return a validated summary dictionary containing {'summary': str}.

        Args:
            original: Full raw output of the tool result.
            required_evidence: Mandatory lines that must be retained or accurately reflected.
            on_call: Measurement hook invoked immediately before sending external request.
            invocation: Serialized tool call definition and arguments.
            context: Optional non-binding caller hints.

        Returns:
            dict containing at least {"summary": str}.

        Raises:
            SummarizerError: On transport failure.
            SummarizerUnavailable: When a summary is unavailable or rejected.
        """
        ...
