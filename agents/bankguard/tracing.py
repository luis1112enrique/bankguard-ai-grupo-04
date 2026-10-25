"""Small, sanitized execution trace for BankGuard tools and delegations."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import unquote_plus

from .schemas import AnalysisTrace, TraceEvent, TraceStatus

_SECRET_KEY_SUFFIXES = (
    "api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "authorization",
    "password",
    "passwd",
    "secret",
    "token",
    "set_cookie",
    "cookie",
)
_SECRET_NAME_PATTERN = (
    r"(?:[a-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|authorization|password|passwd|secret|token|"
    r"set[_-]?cookie|cookie)"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rf"""(?ix)
    (?P<prefix>
        (?P<quote>[\"']?)
        (?P<key>{_SECRET_NAME_PATTERN})
        (?P=quote)
        \s*[:=]\s*
    )
    (?P<value>
        bearer\s+[^\s,;}}\]&]+|
        [\"'][^\"']*[\"']|
        [^\s,;}}\]&]+
    )
    """
)
_AUTH_HEADER_PATTERN = re.compile(
    r"(?im)(?P<prefix>\b(?:authorization|proxy-authorization|x-api-key)\s*:\s*)[^\r\n]*"
)
_COOKIE_HEADER_PATTERN = re.compile(
    r"(?im)(?P<prefix>\b(?:cookie|set-cookie)\s*:\s*)[^\r\n]*"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;}}\]&]+")
_QUERY_PARAMETER_PATTERN = re.compile(
    r"(?P<separator>[?&])(?P<key>[^=&\s#]+)=(?P<value>[^&#\s]*)"
)


def _normalize_secret_key(value: str) -> str:
    """Normalize mapping/header/query keys without changing their values."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _is_secret_key(value: str) -> bool:
    """Return whether a key names a credential or authentication artifact."""
    normalized = _normalize_secret_key(value)
    return any(
        normalized == suffix or normalized.endswith(f"_{suffix}")
        for suffix in _SECRET_KEY_SUFFIXES
    )


def _redact_query_parameters(text: str) -> str:
    """Redact sensitive URL/query-string values while preserving safe params."""

    def replace(match: re.Match[str]) -> str:
        raw_key = match.group("key")
        try:
            decoded_key = unquote_plus(raw_key)
        except (UnicodeDecodeError, ValueError):
            decoded_key = raw_key
        if not _is_secret_key(decoded_key):
            return match.group(0)
        return f"{match.group('separator')}{raw_key}=<redacted>"

    return _QUERY_PARAMETER_PATTERN.sub(replace, text)


def _redact_text(text: str) -> str:
    """Redact secrets embedded in log/error text without hiding safe fields."""
    redacted = _redact_query_parameters(text)
    redacted = _AUTH_HEADER_PATTERN.sub(r"\g<prefix><redacted>", redacted)
    redacted = _COOKIE_HEADER_PATTERN.sub(r"\g<prefix><redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\g<prefix><redacted>", redacted)
    return _BEARER_PATTERN.sub("Bearer <redacted>", redacted)


def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact common secret fields while retaining JSON-safe detail."""
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            sanitized[key] = (
                "<redacted>"
                if _is_secret_key(key)
                else _sanitize_value(raw_value, depth=depth + 1)
            )
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def _safe_summary(value: Any, *, limit: int = 500) -> str | None:
    if value is None:
        return None
    sanitized = _sanitize_value(value)
    if isinstance(sanitized, str):
        text = sanitized
    else:
        text = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    text = text.replace("\n", " ").strip()
    return f"{text[: limit - 3]}..." if len(text) > limit else text


@dataclass(slots=True)
class TraceSpan:
    """Mutable details filled inside ``TraceRecorder.span``."""

    output_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """Thread-safe operational trace that never stores model reasoning."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex}"
        self.started_at = datetime.now(timezone.utc)
        self._events: list[TraceEvent] = []
        self._sequence = 0
        self._lock = threading.Lock()

    def _append(
        self,
        *,
        component: str,
        action: str,
        status: TraceStatus,
        started_at: datetime,
        elapsed_ms: float,
        input_summary: Any = None,
        output_summary: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            self._events.append(
                TraceEvent(
                    sequence=sequence,
                    component=component,
                    action=action,
                    status=status,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_ms=round(max(elapsed_ms, 0.0), 3),
                    input_summary=_safe_summary(input_summary),
                    output_summary=_safe_summary(output_summary),
                    error=_safe_summary(error),
                    metadata=_sanitize_value(metadata or {}),
                )
            )

    @contextmanager
    def span(
        self,
        component: str,
        action: str,
        *,
        input_summary: Any = None,
    ) -> Iterator[TraceSpan]:
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        state = TraceSpan()
        try:
            yield state
        except Exception as exc:
            self._append(
                component=component,
                action=action,
                status=TraceStatus.ERROR,
                started_at=started_at,
                elapsed_ms=(time.perf_counter() - started_clock) * 1_000,
                input_summary=input_summary,
                output_summary=state.output_summary,
                # Exception messages often contain local paths, endpoints, query
                # strings, or SDK diagnostics.  Operational traces only need the
                # stable exception class; detailed diagnostics belong in private
                # server logs with their own access and retention policy.
                error=type(exc).__name__,
                metadata=state.metadata,
            )
            raise
        else:
            self._append(
                component=component,
                action=action,
                status=TraceStatus.SUCCESS,
                started_at=started_at,
                elapsed_ms=(time.perf_counter() - started_clock) * 1_000,
                input_summary=input_summary,
                output_summary=state.output_summary,
                metadata=state.metadata,
            )

    def record(
        self,
        component: str,
        action: str,
        status: TraceStatus,
        *,
        input_summary: Any = None,
        output_summary: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._append(
            component=component,
            action=action,
            status=status,
            started_at=now,
            elapsed_ms=0.0,
            input_summary=input_summary,
            output_summary=output_summary,
            error=error,
            metadata=metadata,
        )

    def record_event(self, event: dict[str, Any]) -> None:
        """Accept the generic event mapping emitted by the RAG boundary."""
        raw_status = str(event.get("status", "success")).casefold()
        status = (
            TraceStatus.ERROR
            if raw_status in {"error", "unavailable", "failed"}
            else TraceStatus.SUCCESS
        )
        metadata = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "component",
                "action",
                "status",
                "timestamp",
                "duration_ms",
                "input_summary",
                "output_summary",
                "error",
                "error_type",
            }
        }
        started_at = datetime.now(timezone.utc)
        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            try:
                started_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        self._append(
            component=str(event.get("component", "external")),
            action=str(event.get("action", "event")),
            status=status,
            started_at=started_at,
            elapsed_ms=float(event.get("duration_ms", 0.0) or 0.0),
            input_summary=event.get("input_summary"),
            output_summary=event.get("output_summary"),
            error=event.get("error") or event.get("error_type"),
            metadata=metadata,
        )

    def snapshot(self, *, complete: bool = False) -> AnalysisTrace:
        with self._lock:
            events = [event.model_copy(deep=True) for event in self._events]
        return AnalysisTrace(
            run_id=self.run_id,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc) if complete else None,
            events=events,
        )


__all__ = ["TraceRecorder", "TraceSpan"]
