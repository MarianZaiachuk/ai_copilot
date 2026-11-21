"""Web search backends (DDGS, Tavily, Brave, Wikipedia), SSRF-safe page fetching, and query extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import ipaddress
import logging
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

from ddgs import DDGS
from lxml import html as lxml_html
import requests

from config import VALID_SEARCH_PROVIDERS
from security import redact_secrets

logger = logging.getLogger(__name__)

_USER_AGENT = "ZoomAICopilot/2.1 (+desktop-assistant)"
_TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s*")
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
_PROMPT_TAG_RE = re.compile(
    r"</?\s*(?:documents_context|web_search_context|live_transcript|user_question|recent_dialogue)\s*>",
    re.IGNORECASE,
)
_BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (".local", ".internal", ".localhost")
_ALLOWED_PORTS: frozenset[int] = frozenset({80, 443, 8080, 8443})
_NONCANON_IPV4_LABEL_RE = re.compile(r"^(?:0x[0-9a-f]+|0[0-7]+|\d+)$", re.IGNORECASE)
_META_CHARSET_RE = re.compile(
    br"""<meta[^>]+charset=["']?\s*([A-Za-z0-9._-]+)""",
    re.IGNORECASE,
)
_HTML_NOISE_XPATH = (
    ".//script | .//style | .//noscript | .//svg | .//nav | .//footer | .//header | .//aside"
)
_MAX_FETCH_BYTES = 1_000_000
_MAX_REDIRECTS = 5
_SEARCH_CACHE_TTL_SEC = 180.0
_SEARCH_CACHE_MAX_ITEMS = 64
_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: dict[tuple[str, str, int, str], tuple[float, list["WebSearchResult"]]] = {}

_ORIG_REQUESTS_GET = requests.get
_ORIG_REQUESTS_POST = requests.post
_SESSION_LOCK = threading.Lock()
_API_SESSION: requests.Session | None = None
_FETCH_SESSION: requests.Session | None = None


def _get_session(*, use_env_proxies: bool) -> requests.Session:
    global _API_SESSION, _FETCH_SESSION
    with _SESSION_LOCK:
        if use_env_proxies:
            if _API_SESSION is None:
                _API_SESSION = requests.Session()
            return _API_SESSION
        if _FETCH_SESSION is None:
            sess = requests.Session()
            sess.trust_env = False
            _FETCH_SESSION = sess
        return _FETCH_SESSION


def _http_call(
    method: str,
    url: str,
    *,
    use_env_proxies: bool = True,
    **kwargs,
):
    if method == "GET" and requests.get is not _ORIG_REQUESTS_GET:
        return requests.get(url, **kwargs)
    if method == "POST" and requests.post is not _ORIG_REQUESTS_POST:
        return requests.post(url, **kwargs)
    sess = _get_session(use_env_proxies=use_env_proxies)
    return sess.request(method, url, **kwargs)


def clear_search_cache() -> None:
    """Clear in-memory web search TTL cache."""
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE.clear()


@dataclass(frozen=True)
class WebSearchResult:
    """Single normalized web search result."""

    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass(frozen=True)
class WebPageContent:
    """Extracted text content from a public web page."""

    url: str
    title: str
    text: str
    engine: str = "http"


def _sanitize_text(text: str) -> str:
    """Strip boundary XML tags and collapse excessive whitespace."""
    if not text:
        return ""
    cleaned = _PROMPT_TAG_RE.sub("", text)
    return " ".join(cleaned.split()).strip()


def _sanitize_multiline(text: str) -> str:
    """Strip boundary XML tags while preserving meaningful paragraph breaks."""
    if not text:
        return ""
    cleaned = _PROMPT_TAG_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    compact: list[str] = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty and compact:
                compact.append("")
            prev_empty = True
        else:
            compact.append(line)
            prev_empty = False
    return "\n".join(compact).strip()


def _extract_domain(url: str) -> str:
    """Extract clean display domain from an HTTP(S) URL."""
    host = (urlparse(url).hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "web"


def _is_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True strictly for globally routable unicast IPs (handles 6to4/Teredo/IPv4-mapped)."""
    if isinstance(ip_obj, ipaddress.IPv6Address):
        mapped = ip_obj.ipv4_mapped
        if mapped is not None:
            return _is_public_ip(mapped)
        sixtofour = ip_obj.sixtofour
        if sixtofour is not None and not _is_public_ip(sixtofour):
            return False
        teredo = ip_obj.teredo
        if teredo is not None:
            for embedded_ip in teredo:
                if not _is_public_ip(embedded_ip):
                    return False

    return bool(
        ip_obj.is_global
        and not (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        )
    )


def _looks_like_noncanonical_ip(hostname: str) -> bool:
    if hostname.lower().startswith("0x"):
        return True
    labels = hostname.split(".")
    return bool(labels) and all(_NONCANON_IPV4_LABEL_RE.match(lbl) for lbl in labels)


def validate_public_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate that a URL is a public HTTP(S) target (prevents SSRF against local/private nets)."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Порожній URL.")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Заборонена схема URL: '{scheme or 'none'}'. Дозволено лише http/https.")

    if not parsed.netloc or parsed.username or parsed.password or "@" in parsed.netloc:
        raise ValueError("Некоректний або небезпечний URL (userinfo не дозволено).")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Некоректний номер порту в URL.") from exc
    if port is not None and port not in _ALLOWED_PORTS:
        raise ValueError(f"Доступ до порту {port} заборонено.")

    hostname = (parsed.hostname or "").strip().rstrip(".").lower()
    if not hostname:
        raise ValueError("URL не містить імені хоста.")

    if (
        hostname == "localhost"
        or hostname.isdigit()
        or any(hostname.endswith(sfx) for sfx in _BLOCKED_HOST_SUFFIXES)
    ):
        raise ValueError(f"Доступ до локального або внутрішнього хоста заборонено: {hostname}")

    # Direct IP literal check
    try:
        ip_literal = ipaddress.ip_address(hostname)
    except ValueError:
        ip_literal = None

    if ip_literal is not None:
        if not _is_public_ip(ip_literal):
            raise ValueError(f"Доступ до приватної або локальної IP-адреси заборонено: {hostname}")
        return raw

    if _looks_like_noncanonical_ip(hostname) or "." not in hostname:
        raise ValueError(f"Некоректне або неповне ім'я хоста: {hostname}")

    # Optional DNS resolution check to catch DNS records resolving to loopback/private/link-local
    if resolve_dns:
        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Не вдалося визначити IP-адресу хоста {hostname}: {exc}") from exc

        if not addr_infos:
            raise ValueError(f"Хост {hostname} не повернув жодної IP-адреси.")

        for info in addr_infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            resolved_ip_raw = str(sockaddr[0]).split("%", 1)[0]
            try:
                resolved_ip = ipaddress.ip_address(resolved_ip_raw)
            except ValueError:
                continue
            if not _is_public_ip(resolved_ip):
                raise ValueError(
                    f"Хост {hostname} резолвиться у приватну/локальну адресу ({resolved_ip_raw})."
                )

    return raw


def extract_urls(text: str, max_urls: int = 2) -> list[str]:
    """Extract up to `max_urls` valid public HTTP(S) URLs from user text."""
    if not text or not text.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _URL_RE.findall(text):
        candidate = match.rstrip(".,;:!?")
        if candidate in seen:
            continue
        try:
            validate_public_url(candidate, resolve_dns=False)
        except ValueError:
            continue
        seen.add(candidate)
        found.append(candidate)
        if len(found) >= max_urls:
            break
    return found


def extract_search_query(transcript_text: str, manual_question: str = "") -> str:
    """Derive a concise search query from manual question or recent Ukrainian transcript."""
    manual = _sanitize_text(manual_question)
    if manual:
        return manual[:220]

    cleaned = _TIMESTAMP_RE.sub("", transcript_text or "").strip()
    if not cleaned:
        return ""

    # Prefer the most recent explicit question ending in '?'
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    if questions:
        return _sanitize_text(questions[-1])[:200]

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    tail = " ".join(lines[-2:]) if lines else cleaned
    return _sanitize_text(tail)[:200]


def _normalize_result_item(
    title: str, url: str, snippet: str, source: str
) -> WebSearchResult | None:
    clean_url = (url or "").strip()
    if not clean_url:
        return None
    try:
        validate_public_url(clean_url, resolve_dns=False)
    except ValueError:
        return None

    clean_title = _sanitize_text(html.unescape(title or "")) or _extract_domain(clean_url)
    clean_snippet = _sanitize_text(html.unescape(snippet or ""))
    return WebSearchResult(
        title=clean_title[:180],
        url=clean_url,
        snippet=clean_snippet[:300],
        source=source,
    )


def _check_cancelled_or_expired(
    cancel_event: threading.Event | None,
    deadline: float | None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Операцію веб-пошуку скасовано.")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Перевищено ліміт часу для веб-запиту.")


def _bounded_timeout(timeout: float, deadline: float | None) -> float:
    base = max(0.3, float(timeout))
    if deadline is None:
        return base
    rem = deadline - time.monotonic()
    if rem <= 0:
        raise TimeoutError("Перевищено ліміт часу для веб-запиту.")
    return min(base, max(0.25, rem))


def _search_tavily(
    query: str,
    api_key: str,
    max_results: int,
    timeout: float,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    key = api_key.strip()
    if not key:
        raise ValueError("Не задано API-ключ для Tavily (WEB_SEARCH_API_KEY).")
    _check_cancelled_or_expired(cancel_event, deadline)

    resp = _http_call(
        "POST",
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_bounded_timeout(timeout, deadline),
    )
    resp.raise_for_status()
    payload = resp.json() or {}
    out: list[WebSearchResult] = []
    for item in payload.get("results", []):
        norm = _normalize_result_item(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("content", "")),
            source="tavily",
        )
        if norm is not None:
            out.append(norm)
    return out[:max_results]


def _search_brave(
    query: str,
    api_key: str,
    max_results: int,
    timeout: float,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    key = api_key.strip()
    if not key:
        raise ValueError("Не задано API-ключ для Brave Search (WEB_SEARCH_API_KEY).")
    _check_cancelled_or_expired(cancel_event, deadline)

    resp = _http_call(
        "GET",
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={
            "X-Subscription-Token": key,
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
        timeout=_bounded_timeout(timeout, deadline),
    )
    resp.raise_for_status()
    payload = resp.json() or {}
    raw_results = (payload.get("web") or {}).get("results") or []
    out: list[WebSearchResult] = []
    for item in raw_results:
        norm = _normalize_result_item(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("description", "")),
            source="brave",
        )
        if norm is not None:
            out.append(norm)
    return out[:max_results]


def _search_ddgs(
    query: str,
    max_results: int,
    timeout: float,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    _check_cancelled_or_expired(cancel_event, deadline)
    req_timeout = _bounded_timeout(timeout, deadline)
    client = DDGS(timeout=int(max(1, min(2, round(req_timeout)))))
    raw_items = list(client.text(query, max_results=max_results) or [])
    out: list[WebSearchResult] = []
    for item in raw_items:
        norm = _normalize_result_item(
            title=str(item.get("title", "")),
            url=str(item.get("href") or item.get("url") or ""),
            snippet=str(item.get("body") or item.get("snippet") or ""),
            source="ddgs",
        )
        if norm is not None:
            out.append(norm)
    return out[:max_results]


def _unwrap_ddg_redirect(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://duckduckgo.com" + href
    parsed = urlparse(href)
    host = (parsed.hostname or "").lower()
    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg and uddg[0]:
            return unquote(uddg[0])
    return href


def _search_ddg_html(
    query: str,
    max_results: int,
    timeout: float,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    _check_cancelled_or_expired(cancel_event, deadline)
    resp = _http_call(
        "GET",
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=_bounded_timeout(timeout, deadline),
    )
    resp.raise_for_status()
    doc = lxml_html.fromstring(resp.text)
    out: list[WebSearchResult] = []
    for node in doc.xpath("//div[contains(@class, 'result')]"):
        a_nodes = node.xpath(".//a[contains(@class, 'result__a')]")
        if not a_nodes:
            continue
        link = a_nodes[0]
        href = _unwrap_ddg_redirect(str(link.get("href") or ""))
        title = "".join(link.itertext())
        snip_nodes = node.xpath(".//*[contains(@class, 'result__snippet')]")
        snippet = "".join(snip_nodes[0].itertext()) if snip_nodes else ""
        norm = _normalize_result_item(title=title, url=href, snippet=snippet, source="ddg_html")
        if norm is not None:
            out.append(norm)
            if len(out) >= max_results:
                break
    return out


def _search_wikipedia(
    query: str,
    max_results: int,
    timeout: float,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    last_exc: Exception | None = None
    for lang in ("uk", "en"):
        _check_cancelled_or_expired(cancel_event, deadline)
        try:
            resp = _http_call(
                "GET",
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": "1",
                    "srlimit": max_results,
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=_bounded_timeout(timeout, deadline),
            )
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as exc:
            last_exc = exc
            logger.warning("Wikipedia (%s) search failed: %s", lang, redact_secrets(str(exc)))
            continue

        items = (payload.get("query") or {}).get("search") or []
        if not items:
            continue
        out: list[WebSearchResult] = []
        for item in items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            page_url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            raw_snip = re.sub(r"<[^>]+>", " ", str(item.get("snippet", "")))
            norm = _normalize_result_item(
                title=f"{title} — Вікіпедія ({lang})" if lang == "uk" else f"{title} — Wikipedia",
                url=page_url,
                snippet=raw_snip,
                source=f"wikipedia:{lang}",
            )
            if norm is not None:
                out.append(norm)
        if out:
            return out[:max_results]

    if last_exc is not None:
        raise last_exc
    return []


def search_web(
    query: str,
    provider: str = "auto",
    api_key: str = "",
    max_results: int = 5,
    timeout: float = 4.0,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> list[WebSearchResult]:
    """Execute a web search using DDGS, Tavily, Brave, Wikipedia, or auto fallback chain."""
    _check_cancelled_or_expired(cancel_event, deadline)
    clean_query = _sanitize_text(query)
    if not clean_query:
        return []

    limit = max(1, min(10, int(max_results)))
    prov = (provider or "auto").strip().lower()
    if prov not in VALID_SEARCH_PROVIDERS:
        prov = "auto"

    clean_key = (api_key or "").strip()
    key_fingerprint = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()[:12] if clean_key else ""
    cache_key = (prov, clean_query.lower(), limit, key_fingerprint)
    now = time.monotonic()
    with _SEARCH_CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached is not None:
            ts, items = cached
            if (now - ts) < _SEARCH_CACHE_TTL_SEC:
                return list(items)
            _SEARCH_CACHE.pop(cache_key, None)

    def _store_cache(items: list[WebSearchResult]) -> list[WebSearchResult]:
        if not items:
            return items
        with _SEARCH_CACHE_LOCK:
            if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX_ITEMS and cache_key not in _SEARCH_CACHE:
                oldest_key = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
                _SEARCH_CACHE.pop(oldest_key, None)
            _SEARCH_CACHE[cache_key] = (time.monotonic(), list(items))
        return items

    try:
        if prov == "tavily":
            return _store_cache(
                _search_tavily(
                    clean_query, clean_key, limit, timeout,
                    cancel_event=cancel_event, deadline=deadline,
                )
            )
        if prov == "brave":
            return _store_cache(
                _search_brave(
                    clean_query, clean_key, limit, timeout,
                    cancel_event=cancel_event, deadline=deadline,
                )
            )
        if prov == "wikipedia":
            return _store_cache(
                _search_wikipedia(
                    clean_query, limit, timeout,
                    cancel_event=cancel_event, deadline=deadline,
                )
            )

        # Auto-detect explicit search API key if configured
        if prov == "auto" and clean_key:
            if clean_key.startswith("tvly-"):
                return _store_cache(
                    _search_tavily(
                        clean_query, clean_key, limit, timeout,
                        cancel_event=cancel_event, deadline=deadline,
                    )
                )
            if clean_key.startswith("BSA"):
                return _store_cache(
                    _search_brave(
                        clean_query, clean_key, limit, timeout,
                        cancel_event=cancel_event, deadline=deadline,
                    )
                )

        # Free metasearch ('ddgs' or 'auto') with deterministic fallbacks
        try:
            results = _search_ddgs(
                clean_query, limit, timeout, cancel_event=cancel_event, deadline=deadline
            )
            if results:
                return _store_cache(results)
        except (InterruptedError, TimeoutError):
            raise
        except Exception as exc:
            logger.warning("DDGS search failed, trying fallback: %s", redact_secrets(str(exc)))

        _check_cancelled_or_expired(cancel_event, deadline)
        try:
            results = _search_ddg_html(
                clean_query, limit, timeout, cancel_event=cancel_event, deadline=deadline
            )
            if results:
                return _store_cache(results)
        except (InterruptedError, TimeoutError):
            raise
        except Exception as exc:
            logger.warning("DDG HTML fallback failed: %s", redact_secrets(str(exc)))

        _check_cancelled_or_expired(cancel_event, deadline)
        try:
            return _store_cache(
                _search_wikipedia(
                    clean_query, limit, timeout, cancel_event=cancel_event, deadline=deadline
                )
            )
        except (InterruptedError, TimeoutError):
            raise
        except Exception as exc:
            logger.warning("Wikipedia fallback failed: %s", redact_secrets(str(exc)))
            return []
    except (ValueError, InterruptedError, TimeoutError):
        raise
    except Exception as exc:
        safe_msg = redact_secrets(str(exc), extra_secrets=(clean_key,))
        raise RuntimeError(f"Помилка веб-пошуку ({prov}): {safe_msg}") from exc


def _try_fetch_obscura(
    clean_url: str,
    max_chars: int,
    timeout: float,
    obscura_path: str,
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> WebPageContent | None:
    exe_str = (obscura_path or "").strip()
    if not exe_str:
        return None
    exe_path = Path(exe_str)
    if not exe_path.is_file():
        logger.warning("Configured OBSCURA_PATH does not exist: %s", exe_str)
        return None
    _check_cancelled_or_expired(cancel_event, deadline)

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc_timeout = min(10.0, _bounded_timeout(float(timeout) + 2.0, deadline))
    proc_deadline = time.monotonic() + proc_timeout
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [str(exe_path), "fetch", clean_url, "--dump", "markdown"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        while proc.poll() is None:
            if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= proc_deadline:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Завантаження сторінки скасовано.")
                logger.warning("Obscura timed out for %s; falling back to HTTP.", clean_url)
                return None
            time.sleep(0.05)

        stdout_data, _ = proc.communicate(timeout=1.0)
        if proc.returncode == 0 and stdout_data and stdout_data.strip():
            body = _sanitize_multiline(stdout_data)[:max_chars]
            return WebPageContent(
                url=clean_url,
                title=_extract_domain(clean_url),
                text=body,
                engine="obscura",
            )
        logger.warning(
            "Obscura exited with code %s for %s; falling back to HTTP.",
            proc.returncode,
            clean_url,
        )
    except InterruptedError:
        raise
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        logger.warning("Obscura execution failed (%s); falling back to HTTP.", exc)
    return None


def _decode_webpage_bytes(raw_bytes: bytes, content_type: str, resp_encoding: str | None) -> str:
    ct_charset_match = re.search(r"charset=([^\s;\"']+)", content_type, re.IGNORECASE)
    if ct_charset_match:
        enc = ct_charset_match.group(1).strip().strip("\"'")
        return raw_bytes.decode(enc, errors="replace")

    if resp_encoding and resp_encoding.strip().lower() not in {"iso-8859-1", "latin-1", "latin1", "ascii"}:
        try:
            return raw_bytes.decode(resp_encoding.strip(), errors="replace")
        except LookupError:
            pass

    meta_match = _META_CHARSET_RE.search(raw_bytes[:4096])
    if meta_match:
        meta_enc = meta_match.group(1).decode("ascii", errors="ignore").strip()
        if meta_enc:
            try:
                return raw_bytes.decode(meta_enc, errors="replace")
            except LookupError:
                pass

    for fallback_enc in ("utf-8", "cp1251"):
        try:
            return raw_bytes.decode(fallback_enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _extract_html_title_and_text(raw_html: str) -> tuple[str, str]:
    doc = lxml_html.fromstring(raw_html)
    title_nodes = doc.xpath("//title")
    title = _sanitize_text("".join(title_nodes[0].itertext())) if title_nodes else ""

    for bad in doc.xpath(_HTML_NOISE_XPATH):
        bad.drop_tree()

    main_nodes = doc.xpath("//main | //article")
    target = main_nodes[0] if main_nodes else None
    if target is not None:
        candidate_raw = "\n".join(
            line.strip() for line in target.xpath(".//text()") if line and line.strip()
        )
        candidate_text = _sanitize_multiline(candidate_raw)
        if len(candidate_text) >= 180:
            return title, candidate_text

    body_nodes = doc.xpath("//body")
    root_node = body_nodes[0] if body_nodes else doc
    raw_text = "\n".join(
        line.strip()
        for line in root_node.xpath(".//text()")
        if line and line.strip()
    )
    return title, _sanitize_multiline(raw_text)


def fetch_webpage(
    url: str,
    max_chars: int = 4500,
    timeout: float = 5.0,
    obscura_path: str = "",
    *,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> WebPageContent:
    """Fetch and extract readable text from a validated public HTTP(S) URL."""
    clean_url = validate_public_url(url)

    obscura_content = _try_fetch_obscura(
        clean_url,
        max_chars,
        timeout,
        obscura_path,
        cancel_event=cancel_event,
        deadline=deadline,
    )
    if obscura_content is not None:
        return obscura_content

    current_url = clean_url
    req_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    }
    resp = None
    for _hop in range(_MAX_REDIRECTS + 1):
        _check_cancelled_or_expired(cancel_event, deadline)
        hop_resp = _http_call(
            "GET",
            current_url,
            use_env_proxies=False,
            timeout=_bounded_timeout(timeout, deadline),
            headers=req_headers,
            allow_redirects=False,
            stream=True,
        )
        raw_status = getattr(hop_resp, "status_code", 200)
        status = raw_status if isinstance(raw_status, int) else 200
        hop_headers = getattr(hop_resp, "headers", None)
        if status in {301, 302, 303, 307, 308}:
            location = (
                (hop_headers.get("Location") or hop_headers.get("location"))
                if hop_headers is not None and hasattr(hop_headers, "get")
                else None
            )
            close_fn = getattr(hop_resp, "close", None)
            if callable(close_fn):
                close_fn()
            if not location:
                raise ValueError(f"Редирект без заголовка Location ({status}).")
            next_candidate = urljoin(current_url, str(location))
            current_url = validate_public_url(next_candidate, resolve_dns=True)
            continue
        resp = hop_resp
        break

    if resp is None:
        raise ValueError("Перевищено ліміт HTTP-редиректів (макс. 5).")

    try:
        resp.raise_for_status()
        raw_final_url = getattr(resp, "url", current_url)
        final_url = validate_public_url(
            raw_final_url if isinstance(raw_final_url, str) and raw_final_url else current_url
        )

        headers = getattr(resp, "headers", None)
        raw_ct = headers.get("Content-Type", "") if headers is not None and hasattr(headers, "get") else ""
        content_type = raw_ct.lower() if isinstance(raw_ct, str) else ""
        is_text_or_html = (
            not content_type
            or "text/" in content_type
            or "html" in content_type
            or "xml" in content_type
            or "json" in content_type
        )
        if not is_text_or_html:
            raise ValueError(f"Непідтримуваний тип контенту сторінки: {content_type}")

        raw_bytes = b""
        if hasattr(resp, "iter_content") and callable(resp.iter_content):
            chunks: list[bytes] = []
            total = 0
            for part in resp.iter_content(chunk_size=16384):
                _check_cancelled_or_expired(cancel_event, deadline)
                if isinstance(part, (bytes, bytearray)) and part:
                    chunks.append(bytes(part))
                    total += len(part)
                    if total >= _MAX_FETCH_BYTES:
                        break
            raw_bytes = b"".join(chunks)

        if not raw_bytes:
            content_attr = getattr(resp, "content", b"")
            if isinstance(content_attr, (bytes, bytearray)) and content_attr:
                raw_bytes = bytes(content_attr[:_MAX_FETCH_BYTES])
            elif isinstance(getattr(resp, "text", None), str):
                raw_bytes = resp.text[:_MAX_FETCH_BYTES].encode("utf-8", errors="replace")

        raw_enc = getattr(resp, "encoding", None)
        enc_arg = raw_enc if isinstance(raw_enc, str) else None
        decoded = _decode_webpage_bytes(raw_bytes, content_type, enc_arg)
    finally:
        close_fn = getattr(resp, "close", None)
        if callable(close_fn):
            close_fn()

    domain = _extract_domain(final_url)
    if "html" in content_type or "<html" in decoded[:512].lower() or "<body" in decoded[:1024].lower():
        try:
            title, text = _extract_html_title_and_text(decoded)
        except Exception:
            title, text = domain, _sanitize_multiline(decoded)
    else:
        title, text = domain, _sanitize_multiline(decoded)

    return WebPageContent(
        url=final_url,
        title=title or domain,
        text=text[:max_chars],
        engine="http",
    )


def format_web_results_for_prompt(
    results: list[WebSearchResult],
    fetched_pages: list[WebPageContent] | None = None,
    max_chars: int = 4200,
) -> tuple[str, list[str]]:
    """Format web search results & fetched pages for `<web_search_context>` and return UI sources."""
    sections: list[str] = []
    sources: list[str] = []
    seen_sources: set[str] = set()

    def _add_source(url: str) -> None:
        label = f"🌐 {_extract_domain(url)}"
        if label not in seen_sources:
            seen_sources.add(label)
            sources.append(label)

    for idx, res in enumerate(results, start=1):
        _add_source(res.url)
        block = f"[{idx}] {res.title} ({res.url})"
        if res.snippet:
            block += f"\n{res.snippet}"
        sections.append(block)

    for page in fetched_pages or ():
        _add_source(page.url)
        sections.append(f"### Вміст сторінки: {page.title} ({page.url})\n{page.text}")

    if not sections:
        return "", []

    joined = "\n\n".join(sections)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 3].rstrip() + "..."
    return joined, sources
