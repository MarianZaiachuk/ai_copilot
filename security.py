"""Secret redaction utilities, safe logging filters, and stream wrappers."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Iterable, TextIO
from urllib.parse import quote

_SECRET_LOCK = threading.Lock()
_REGISTERED_SECRETS: set[str] = set()

_IGNORED_SECRETS: frozenset[str] = frozenset({"ollama", "lm-studio", "none", "null"})

_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_PREFIX_KEY_RE = re.compile(
    r"(?<![0-9A-Za-z_\-])(?:sk-(?:proj-|svcacct-|ant-)?|gsk_|csk-|pplx-|hf_|xai-|ghp_|github_pat_)[0-9A-Za-z_\-]{12,}"
)
_TAVILY_KEY_RE = re.compile(r"\btvly-[A-Za-z0-9_-]{12,}")
_BRAVE_KEY_RE = re.compile(r"\bBSA[A-Za-z0-9_-]{16,}")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:key|api_?key|x-api-key|token|access_token|secret|password)=)([^&\s\"']+)"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{8,})")
_TOKEN_AUTH_RE = re.compile(r"(?i)(\bToken\s+)([A-Za-z0-9._~+/=-]{8,})")
_BASIC_AUTH_RE = re.compile(r"(?i)(\bBasic\s+)([A-Za-z0-9+/=]{8,})")
_HEADER_SECRET_RE = re.compile(
    r"(?i)(\b(?:X-Subscription-Token|x-goog-api-key|x-api-key|xi-api-key)\s*:\s*)([^\s,;\"']{8,})"
)
_JSON_SECRET_RE = re.compile(
    r'(?i)("(?:api_?key|secret|token|password|access_token)"\s*:\s*")([^"]{6,})(")'
)
_URL_USERINFO_RE = re.compile(r"(https?://)([^/\s:@]+:[^/\s@]+)(@)")


def register_secret(secret: str | None) -> None:
    """Register a runtime API key so it is automatically masked in logs and UI errors."""
    if not secret:
        return
    cleaned = secret.strip()
    if len(cleaned) < 8 or cleaned.lower() in _IGNORED_SECRETS:
        return
    with _SECRET_LOCK:
        _REGISTERED_SECRETS.add(cleaned)


def clear_registered_secrets() -> None:
    """Clear registered secrets (mainly for unit tests)."""
    with _SECRET_LOCK:
        _REGISTERED_SECRETS.clear()


def redact_secrets(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Mask API keys, bearer tokens, query parameters, and registered secret strings."""
    if not text:
        return text

    result = _GOOGLE_KEY_RE.sub("AIza***REDACTED***", text)
    result = _PREFIX_KEY_RE.sub("***REDACTED***", result)
    result = _TAVILY_KEY_RE.sub("***REDACTED***", result)
    result = _BRAVE_KEY_RE.sub("***REDACTED***", result)
    result = _QUERY_SECRET_RE.sub(r"\1***REDACTED***", result)
    result = _BEARER_RE.sub(r"\1***REDACTED***", result)
    result = _TOKEN_AUTH_RE.sub(r"\1***REDACTED***", result)
    result = _BASIC_AUTH_RE.sub(r"\1***REDACTED***", result)
    result = _HEADER_SECRET_RE.sub(r"\1***REDACTED***", result)
    result = _JSON_SECRET_RE.sub(r"\1***REDACTED***\3", result)
    result = _URL_USERINFO_RE.sub(r"\1***REDACTED***\3", result)

    with _SECRET_LOCK:
        candidates = set(_REGISTERED_SECRETS)

    for s in extra_secrets:
        if s and len(s.strip()) >= 8 and s.strip().lower() not in _IGNORED_SECRETS:
            candidates.add(s.strip())

    for secret in sorted(candidates, key=len, reverse=True):
        if secret in result:
            result = result.replace(secret, "***REDACTED***")
        if len(secret) >= 6:
            encoded = quote(secret, safe="")
            if encoded != secret and encoded in result:
                result = result.replace(encoded, "***REDACTED***")

    return result


class SecretRedactingFilter(logging.Filter):
    """Logging filter scrubbing secrets from log messages and cached exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            record.msg = redact_secrets(rendered)
            record.args = None
        except Exception:
            record.msg = redact_secrets(str(record.msg))
            record.args = None
        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        return True


class RedactingFormatter(logging.Formatter):
    """Logging formatter scrubbing formatted output including stack tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


class RedactingTextStream:
    """Writable text-stream wrapper scrubbing secrets before flushing to disk."""

    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped

    def write(self, data: str) -> int:
        if not data:
            return 0
        cleaned = redact_secrets(data)
        self._wrapped.write(cleaned)
        return len(data)

    def writelines(self, lines: Iterable[str]) -> None:
        self._wrapped.writelines(redact_secrets(line) for line in lines)

    @property
    def encoding(self) -> str:
        return getattr(self._wrapped, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._wrapped, "errors", None) or "replace"

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        self._wrapped.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
