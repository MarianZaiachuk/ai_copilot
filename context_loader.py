"""Document loading, structure-aware chunking, and Ukrainian hybrid BM25/4-gram RAG."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Iterable

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".py",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".cs",
        ".csproj",
        ".sln",
        ".razor",
        ".xaml",
        ".cshtml",
        ".props",
        ".targets",
        ".config",
        ".go",
        ".rs",
        ".kt",
        ".php",
        ".rb",
        ".sql",
        ".proto",
        ".http",
        ".vue",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".css",
        ".tex",
        ".csv",
        ".ini",
        ".toml",
        ".sh",
        ".bat",
        ".ps1",
        ".ipynb",
    }
)

IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "bin",
        "obj",
        ".git",
        ".vs",
        ".idea",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "packages",
        "testresults",
        "coverage",
        "logs",
        "temp",
        "tmp",
        "artifacts",
        "publish",
        ".next",
        ".nuxt",
        "out",
        "dist",
        "build",
        "target",
        ".pytest_cache",
        ".ruff_cache",
    }
)
IGNORED_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "cargo.lock",
        "packages.lock.json",
        "composer.lock",
        "gemfile.lock",
        "thumbs.db",
        ".ds_store",
    }
)
IGNORED_FILE_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".designer.cs",
    ".g.cs",
    ".g.i.cs",
    ".assemblyinfo.cs",
    ".assemblyattributes.cs",
    ".globalusings.g.cs",
    ".deps.json",
    ".runtimeconfig.json",
    ".sourcelink.json",
    ".pdb",
    ".dll",
    ".exe",
    ".cache",
)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_FOLDER_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_EXTRACTED_CHARS = 500_000
MAX_PDF_PAGES = 250

_PROSE_MARKDOWN_EXTS: frozenset[str] = frozenset({"", ".md", ".txt", ".docx", ".pdf", ".tex", ".rst"})
_CODE_EXTS: frozenset[str] = frozenset(
    {
        ".py",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".kt",
        ".php",
        ".rb",
        ".sql",
        ".sh",
        ".bat",
        ".ps1",
        ".razor",
        ".cshtml",
        ".vue",
        ".proto",
    }
)

_TECH_TOKEN_RE = re.compile(
    r"\b(?:[A-Z]{2,}[0-9]*|[A-Za-z]+[A-Z][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9_]{3,})\b"
)
_WORD_SPLIT_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PDF_PAGE_RE = re.compile(r"^\[Сторінка\s+\d+\]$")
_MD_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$")
_CODE_DEF_RE = re.compile(
    r"^(?:"
    r"(?:async\s+)?def|class|function|struct|"
    r"export\s+(?:async\s+)?function|export\s+class|"
    r"namespace|"
    r"(?:(?:public|private|protected|internal|file)\s+)?"
    r"(?:(?:static|abstract|sealed|partial|readonly|ref)\s+)*"
    r"(?:class|interface|struct|record(?:\s+(?:class|struct))?|enum)"
    r")\s+([A-Za-z_][A-Za-z0-9_.]*)"
    r"|^(?:public|private|protected|internal)\s+(?:(?:static|virtual|override|abstract|async|sealed|new)\s+)*"
    r"(?:[A-Za-z_][A-Za-z0-9_<>,\[\]\.\?]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]+>)?\s*\("
)
_DECORATOR_ATTR_RE = re.compile(r"^(?:@[A-Za-z_]|\[[A-Z][A-Za-z0-9_.]*(?:\([^)]*\))?\]$)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HYPHEN_WRAP_RE = re.compile(r"([А-Яа-яІіЇїЄєҐґA-Za-z])-\s*\n\s*([а-яіїєґa-z])")

_STOP_LATIN_WORDS: frozenset[str] = frozenset(
    {
        "self",
        "none",
        "true",
        "false",
        "return",
        "import",
        "from",
        "class",
        "with",
        "this",
        "that",
        "have",
        "been",
        "will",
        "your",
        "about",
        "into",
        "const",
        "function",
        "export",
        "default",
        "async",
        "await",
        "string",
        "number",
        "print",
        "public",
        "private",
        "protected",
        "internal",
        "static",
        "readonly",
        "override",
        "virtual",
        "abstract",
        "sealed",
        "partial",
        "namespace",
        "using",
        "void",
        "bool",
        "object",
        "system",
        "task",
        "tasks",
        "collections",
        "generic",
        "threading",
        "microsoft",
        "null",
        "throw",
        "catch",
        "finally",
        "foreach",
        "while",
        "switch",
        "case",
        "break",
        "continue",
        "else",
        "elif",
        "interface",
        "struct",
        "enum",
        "record",
        "var",
        "let",
        "int",
        "long",
        "double",
        "float",
        "decimal",
    }
)

_RAG_STOPWORDS: frozenset[str] = frozenset(
    {
        "що",
        "як",
        "які",
        "який",
        "яка",
        "яке",
        "для",
        "про",
        "при",
        "або",
        "та",
        "і",
        "й",
        "у",
        "в",
        "на",
        "з",
        "із",
        "зі",
        "до",
        "від",
        "по",
        "за",
        "це",
        "цей",
        "ця",
        "ці",
        "тут",
        "де",
        "коли",
        "чи",
        "не",
        "так",
        "ні",
        "будь",
        "ласка",
        "розкажіть",
        "поясніть",
        "скажіть",
        "можете",
        "вашій",
        "вашої",
        "роботі",
        "через",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "how",
        "why",
    }
)

_UK_SUFFIXES: tuple[str, ...] = (
    "увати",
    "ювати",
    "ання",
    "ення",
    "іння",
    "ості",
    "ість",
    "ами",
    "ями",
    "ові",
    "еві",
    "ого",
    "ому",
    "ими",
    "іти",
    "ати",
    "яти",
    "ов",
    "ев",
    "их",
    "ій",
    "ою",
    "ею",
    "ів",
    "їв",
    "а",
    "я",
    "у",
    "ю",
    "о",
    "е",
    "и",
    "і",
)

_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "контролер": ("controller", "endpoint", "route"),
    "репозитор": ("repository", "repo"),
    "автентиф": ("auth", "authentication", "jwt", "token", "login"),
    "авториз": ("authorize", "authorization", "role", "policy"),
    "баз": ("database", "db", "dbcontext", "sql", "entity"),
    "сервіс": ("service", "handler"),
    "кеш": ("cache", "caching", "redis", "memorycache"),
    "міграц": ("migration", "migrations", "ef"),
    "транзакц": ("transaction", "transactions", "acid"),
    "інтерфейс": ("interface", "ui", "view"),
    "асинхрон": ("async", "await", "task", "thread"),
    "помилк": ("error", "exception", "try", "catch"),
    "потік": ("thread", "stream", "queue"),
    "запит": ("query", "request", "http"),
    "таблиц": ("table", "entity", "model", "schema"),
    "захист": ("security", "defense", "guard"),
    "тест": ("test", "tests", "unit", "mock"),
    "замовлен": ("order", "orders"),
    "користувач": ("user", "users", "account"),
}


def build_file_dialog_filter() -> str:
    """Build a Qt file dialog filter covering every supported context extension."""
    exts = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
    return f"Підтримувані файли ({exts});;Усі файли (*.*)"


@dataclass(frozen=True)
class LoadedDocument:
    """Represents a single parsed file loaded into the copilot context."""

    path: str
    name: str
    content: str
    char_count: int
    display_name: str = ""
    mtime_ns: int = 0
    page_count: int = 0
    truncated: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def token_estimate(self) -> int:
        return max(1, (self.char_count + 3) // 4) if self.char_count > 0 else 0


@dataclass(frozen=True)
class DocumentChunk:
    """Structure-aware document chunk used by hybrid RAG retrieval."""

    doc_path: str
    doc_name: str
    chunk_index: int
    section: str
    text: str


def _looks_like_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_count = sum(
        1 for b in sample if b < 32 and b not in (9, 10, 13, 12)
    )
    return (control_count / len(sample)) > 0.12


def _read_text_with_fallback(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(f"Файл завеликий ({mb:.1f} МБ; ліміт {MAX_FILE_SIZE_BYTES // (1024 * 1024)} МБ).")

    raw_bytes = path.read_bytes()
    if not raw_bytes:
        return ""

    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw_bytes.decode("utf-16", errors="replace")
    elif raw_bytes.startswith(b"\xef\xbb\xbf"):
        text = raw_bytes.decode("utf-8-sig", errors="replace")
    else:
        if _looks_like_binary(raw_bytes[:8192]):
            raise ValueError("Схоже на бінарний файл (пропущено).")
        text = ""
        decoded = False
        for encoding in ("utf-8", "cp1251"):
            try:
                text = raw_bytes.decode(encoding)
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            text = raw_bytes.decode("utf-8", errors="replace")

    text = text.replace("\x00", "")
    if len(text) > 50_000:
        line_count = max(1, text.count("\n") + 1)
        if (len(text) / line_count) > 500:
            raise ValueError("Мініфікований однорядковий файл — пропущено для швидкодії RAG.")
    if len(text) > MAX_EXTRACTED_CHARS:
        text = text[:MAX_EXTRACTED_CHARS] + "\n\n[... Файл скорочено через ліміт 500 тис. символів ...]"
    return text


def _normalize_pdf_text(raw: str) -> str:
    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "").replace("\u00ad", "").replace("\xa0", " ")
    cleaned = _HYPHEN_WRAP_RE.sub(r"\1\2", cleaned)
    return cleaned.strip()


def _extract_pdf_text(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(f"PDF завеликий ({mb:.1f} МБ).")
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            res = reader.decrypt("")
            if not res:
                raise ValueError("PDF захищено паролем відкриття.")
        except Exception as exc:
            raise ValueError("PDF захищено паролем і не може бути прочитаний.") from exc

    pages: list[str] = []
    total_chars = 0
    for idx, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            continue
        text = _normalize_pdf_text(raw_text)
        if text:
            block = f"[Сторінка {idx}]\n{text}"
            pages.append(block)
            total_chars += len(block)
            if total_chars >= MAX_EXTRACTED_CHARS:
                break
    return "\n\n".join(pages)


def _docx_heading_prefix(style_name: str) -> str:
    low = (style_name or "").strip().lower()
    if not low:
        return ""
    if low == "title":
        return "# "
    for level in (1, 2, 3, 4):
        if f"heading {level}" in low or f"заголовок {level}" in low:
            return "#" * level + " "
    return ""


def _extract_docx_text(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(f"DOCX завеликий ({mb:.1f} МБ).")
    doc = DocxDocument(str(path))
    blocks: list[str] = []

    iter_content = getattr(doc, "iter_inner_content", None)
    if callable(iter_content):
        from docx.text.paragraph import Paragraph

        for item in iter_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style_name = getattr(getattr(item, "style", None), "name", "") or ""
                prefix = _docx_heading_prefix(style_name)
                if prefix and not text.startswith("#"):
                    text = f"{prefix}{text}"
                blocks.append(text)
            else:
                table_rows: list[str] = []
                for row in item.rows:
                    cells: list[str] = []
                    seen_tcs: set[int] = set()
                    for cell in row.cells:
                        tc_id = id(getattr(cell, "_tc", cell))
                        if tc_id in seen_tcs:
                            continue
                        seen_tcs.add(tc_id)
                        cells.append(re.sub(r"\s+", " ", cell.text).strip())
                    if any(cells):
                        table_rows.append(" | ".join(cells))
                if table_rows:
                    blocks.append("\n".join(table_rows))
        joined = "\n\n".join(blocks)
        if len(joined) > MAX_EXTRACTED_CHARS:
            joined = joined[:MAX_EXTRACTED_CHARS] + "\n\n[... Файл скорочено через ліміт 500 тис. символів ...]"
        return joined

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)
    joined = "\n\n".join(blocks)
    if len(joined) > MAX_EXTRACTED_CHARS:
        joined = joined[:MAX_EXTRACTED_CHARS] + "\n\n[... Файл скорочено через ліміт 500 тис. символів ...]"
    return joined


def extract_text_from_file(path: str | Path) -> str:
    """Extract readable text from PDF, DOCX, plain text, or source code files."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        content = _extract_pdf_text(file_path)
    elif suffix == ".docx":
        content = _extract_docx_text(file_path)
    else:
        content = _read_text_with_fallback(file_path)

    return content.strip()


def _extract_document_payload(path: str | Path) -> tuple[str, int, bool]:
    """Extract (content, page_count, truncated) while delegating to extract_text_from_file."""
    file_path = Path(path)
    content = extract_text_from_file(file_path)
    truncated = (
        "[... Файл скорочено" in content
        or len(content) >= MAX_EXTRACTED_CHARS
    )
    page_count = 0
    if file_path.suffix.lower() == ".pdf" and content:
        page_count = sum(
            1 for line in content.splitlines() if _PDF_PAGE_RE.match(line.strip())
        )
    return content, page_count, truncated


def expand_context_paths(
    paths: Iterable[str | Path],
    *,
    max_files: int = 80,
    max_total_bytes: int = 15 * 1024 * 1024,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Expand mixed files/folders recursively into (resolved_path, display_name) pairs."""
    collected: list[tuple[Path, str]] = []
    notices: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    skipped_ignored = 0
    capped = False

    def _add_candidate(file_path: Path, display: str, strict_ext: bool) -> None:
        nonlocal total_bytes, skipped_ignored, capped
        if capped:
            return
        name_low = file_path.name.lower()
        if strict_ext:
            if name_low in IGNORED_FILE_NAMES or name_low.endswith(IGNORED_FILE_SUFFIXES):
                skipped_ignored += 1
                return
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return
        try:
            resolved = file_path.resolve()
            size = resolved.stat().st_size
        except OSError:
            return

        norm_key = os.path.normcase(os.path.normpath(str(resolved)))
        if norm_key in seen:
            return
        if strict_ext and size > MAX_FOLDER_FILE_SIZE_BYTES:
            notices.append(
                f"{file_path.name}: пропущено у папці (більше {MAX_FOLDER_FILE_SIZE_BYTES // (1024 * 1024)} МБ)"
            )
            return
        if size > MAX_FILE_SIZE_BYTES:
            notices.append(f"{file_path.name}: пропущено (більше {MAX_FILE_SIZE_BYTES // (1024 * 1024)} МБ)")
            return
        if len(collected) >= max_files or (total_bytes + size) > max_total_bytes:
            capped = True
            return

        seen.add(norm_key)
        total_bytes += size
        collected.append((resolved, display))

    for raw in paths:
        if capped:
            break
        p = Path(raw)
        if p.is_file():
            _add_candidate(p, p.name, strict_ext=False)
        elif p.is_dir():
            root_resolved = p.resolve()
            root_name = root_resolved.name or str(root_resolved)
            for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
                dirnames[:] = [
                    d
                    for d in sorted(dirnames)
                    if d.lower() not in IGNORED_DIR_NAMES and not d.startswith(".")
                ]
                for fname in sorted(filenames):
                    fpath = Path(dirpath) / fname
                    try:
                        rel = fpath.relative_to(root_resolved).as_posix()
                    except ValueError:
                        rel = fname
                    _add_candidate(fpath, f"{root_name}/{rel}", strict_ext=True)
                    if capped:
                        break
        else:
            notices.append(f"{p.name or raw}: шлях не знайдено")

    if capped:
        max_mb = max(1, max_total_bytes // (1024 * 1024))
        notices.append(f"Досягнуто ліміт пакетного імпорту (макс. {max_files} файлів або {max_mb} МБ).")
    if skipped_ignored:
        notices.append(f"Автоматично пропущено службових/мініфікованих файлів: {skipped_ignored}.")
    return collected, notices


def _format_file_count_uk(count: int) -> str:
    mod100 = count % 100
    mod10 = count % 10
    if 11 <= mod100 <= 14:
        noun = "файлів"
    elif mod10 == 1:
        noun = "файл"
    elif 2 <= mod10 <= 4:
        noun = "файли"
    else:
        noun = "файлів"
    return f"{count} {noun}"


def _split_oversized_block(block: str, max_chars: int, *, is_code: bool = False) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    pieces: list[str] = []
    if is_code:
        lines = block.splitlines()
        buf_lines: list[str] = []
        buf_len = 0
        for line in lines:
            if len(line) > max_chars:
                if buf_lines:
                    pieces.append("\n".join(buf_lines))
                    buf_lines.clear()
                    buf_len = 0
                for i in range(0, len(line), max_chars):
                    pieces.append(line[i : i + max_chars])
                continue
            extra = len(line) + (1 if buf_lines else 0)
            if buf_len + extra <= max_chars:
                buf_lines.append(line)
                buf_len += extra
            else:
                if buf_lines:
                    pieces.append("\n".join(buf_lines))
                buf_lines = [line]
                buf_len = len(line)
        if buf_lines:
            pieces.append("\n".join(buf_lines))
        return pieces or [block[:max_chars]]

    sentences = _SENTENCE_SPLIT_RE.split(block)
    if len(sentences) == 1:
        sentences = block.splitlines()
    buf = ""
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(s), max_chars):
                pieces.append(s[i : i + max_chars])
            continue
        candidate = f"{buf} {s}".strip() if buf else s
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = s
    if buf:
        pieces.append(buf)
    return pieces or [block[:max_chars]]


def chunk_document(
    doc: LoadedDocument,
    target_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[DocumentChunk]:
    """Split a document along pages, headings, code definitions, and paragraphs."""
    text = doc.content.strip()
    if not text:
        return []

    suffix = Path(doc.path or doc.name).suffix.lower()
    allow_md_headings = suffix in _PROSE_MARKDOWN_EXTS
    allow_code_defs = suffix in _CODE_EXTS

    sections_and_blocks: list[tuple[str, str]] = []
    current_section = "Початок документа"
    current_lines: list[str] = []
    pending_attrs: list[str] = []

    def _flush_lines() -> None:
        nonlocal current_lines
        joined = "\n".join(current_lines).strip()
        current_lines = []
        if joined:
            block_budget = max(300, target_chars - (overlap_chars if overlap_chars > 0 else 0))
            for part in _split_oversized_block(joined, block_budget, is_code=allow_code_defs):
                sections_and_blocks.append((current_section, part))

    for raw_line in text.splitlines():
        expanded = raw_line.replace("\t", "    ")
        stripped = expanded.strip()
        if not stripped:
            if pending_attrs:
                current_lines.extend(pending_attrs)
                pending_attrs.clear()
            _flush_lines()
            continue

        if _PDF_PAGE_RE.match(stripped):
            if pending_attrs:
                current_lines.extend(pending_attrs)
                pending_attrs.clear()
            _flush_lines()
            current_section = stripped.strip("[]")
            continue

        if allow_md_headings:
            md_match = _MD_HEADING_RE.match(stripped)
            if md_match:
                _flush_lines()
                current_section = md_match.group(1).strip()[:80]
                current_lines.append(raw_line)
                continue

        if allow_code_defs:
            indent = len(expanded) - len(expanded.lstrip(" "))
            if indent <= 8 and _DECORATOR_ATTR_RE.match(stripped):
                pending_attrs.append(raw_line)
                continue

            code_match = _CODE_DEF_RE.match(stripped) if indent <= 8 else None
            if code_match:
                sym = code_match.group(1) or code_match.group(2)
                _flush_lines()
                current_section = f"Код: {sym}"
                if pending_attrs:
                    current_lines.extend(pending_attrs)
                    pending_attrs.clear()
                current_lines.append(raw_line)
                continue

        if pending_attrs:
            current_lines.extend(pending_attrs)
            pending_attrs.clear()
        current_lines.append(raw_line)

    if pending_attrs:
        current_lines.extend(pending_attrs)
        pending_attrs.clear()
    _flush_lines()
    if not sections_and_blocks:
        return []

    chunks: list[DocumentChunk] = []
    cur_text = ""
    cur_section = sections_and_blocks[0][0]
    display_doc_name = doc.label
    max_chunk_ceiling = target_chars + overlap_chars

    for sec, block in sections_and_blocks:
        if not cur_text:
            cur_text = block
            cur_section = sec
            continue

        candidate = f"{cur_text}\n\n{block}"
        if len(candidate) <= target_chars and sec == cur_section:
            cur_text = candidate
        else:
            chunks.append(
                DocumentChunk(
                    doc_path=doc.path,
                    doc_name=display_doc_name,
                    chunk_index=len(chunks),
                    section=cur_section,
                    text=cur_text,
                )
            )
            if sec == cur_section and overlap_chars > 0 and len(cur_text) > overlap_chars:
                tail = cur_text[-overlap_chars:].lstrip()
                space_idx = tail.find(" ")
                if 0 < space_idx < len(tail) // 2:
                    tail = tail[space_idx + 1 :]
                overlapped = f"{tail}\n\n{block}".strip()
                cur_text = overlapped if len(overlapped) <= max_chunk_ceiling else block
            else:
                cur_text = block
            cur_section = sec

    if cur_text:
        chunks.append(
            DocumentChunk(
                doc_path=doc.path,
                doc_name=display_doc_name,
                chunk_index=len(chunks),
                section=cur_section,
                text=cur_text,
            )
        )

    return chunks


@lru_cache(maxsize=32768)
def _uk_stem(word: str) -> str:
    if len(word) <= 4 or word.isascii():
        return word
    for sfx in _UK_SUFFIXES:
        if len(word) - len(sfx) >= 3 and word.endswith(sfx):
            return word[: -len(sfx)]
    return word


@lru_cache(maxsize=32768)
def _split_identifier_subtokens(raw: str) -> tuple[str, ...]:
    if not raw or raw.isascii() is False:
        return (raw.lower(),) if raw else ()
    split_camel = _CAMEL_BOUNDARY_RE.sub(" ", raw)
    parts = [p.lower() for p in re.split(r"[\s_]+", split_camel) if len(p) >= 2]
    low = raw.lower()
    if len(parts) >= 2 and parts[0] == "i" and len(parts[1]) >= 3 and parts[1].isalpha():
        # Strip C# interface prefix (e.g. IOrderRepository -> i, order, repository)
        parts.append(parts[1])
    out: list[str] = [low]
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out)


def _tokenize_stems(text: str, *, is_query: bool = False) -> list[str]:
    """Extract normalized tokens, CamelCase subtokens, and Ukrainian stems."""
    cleaned = text.replace("'", "").replace("’", "").replace("ʼ", "")
    tokens: list[str] = []
    for raw in _WORD_SPLIT_RE.findall(cleaned):
        if len(raw) < 2:
            continue
        low = raw.lower()
        if low in _RAG_STOPWORDS:
            continue

        if raw.isascii():
            subtokens = _split_identifier_subtokens(raw)
            tokens.extend(subtokens)
        else:
            tokens.append(low)
            stem = _uk_stem(low)
            if stem != low:
                tokens.append(stem)

        if is_query:
            for uk_pfx, en_expansions in _QUERY_SYNONYMS.items():
                if low.startswith(uk_pfx):
                    tokens.extend(en_expansions)
    return tokens


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    cleaned = text[:4000].lower().replace("'", "").replace("’", "").replace("ʼ", "")
    grams: set[str] = set()
    for word in set(_WORD_SPLIT_RE.findall(cleaned)):
        if word in _RAG_STOPWORDS:
            continue
        if len(word) < n:
            if len(word) >= 2:
                grams.add(word)
            continue
        for i in range(len(word) - n + 1):
            grams.add(word[i : i + n])
    return grams


class ContextManager:
    """Thread-safe document store, cached hybrid RAG index, and folder/reload manager."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: dict[str, LoadedDocument] = {}
        self._revision: int = 0
        self._content_hash: str = hashlib.sha256(b"").hexdigest()
        self._chunks: list[DocumentChunk] = []
        self._chunk_tfs: list[Counter[str]] = []
        self._chunk_lens: list[int] = []
        self._chunk_ngrams: list[set[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_dl: float = 1.0
        self._doc_index_cache: dict[
            str, tuple[str, list[DocumentChunk], list[Counter[str]], list[int], list[set[str]]]
        ] = {}

    @property
    def documents(self) -> list[LoadedDocument]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: (d.label.lower(), d.path.lower()))

    @property
    def total_chars(self) -> int:
        with self._lock:
            return sum(doc.char_count for doc in self._docs.values())

    @property
    def content_revision(self) -> tuple[int, str]:
        with self._lock:
            return self._revision, self._content_hash

    def _disambiguate_display_names(self) -> None:
        name_counts = Counter(doc.name.lower() for doc in self._docs.values())
        for key, doc in list(self._docs.items()):
            if doc.display_name and "/" in doc.display_name:
                continue
            if name_counts[doc.name.lower()] > 1:
                parent = Path(doc.path).parent.name
                new_label = f"{parent}/{doc.name}" if parent else doc.name
                if new_label != doc.display_name:
                    self._docs[key] = replace(doc, display_name=new_label)

    def _rebuild_index(self) -> None:
        with self._lock:
            self._disambiguate_display_names()
            self._revision += 1
            hasher = hashlib.sha256()
            all_chunks: list[DocumentChunk] = []
            all_tfs: list[Counter[str]] = []
            all_lens: list[int] = []
            all_ngrams: list[set[str]] = []
            df: Counter[str] = Counter()
            active_keys: set[str] = set()

            for doc in self.documents:
                norm_key = os.path.normcase(os.path.normpath(doc.path))
                active_keys.add(norm_key)
                hasher.update(doc.path.encode("utf-8", errors="ignore"))
                hasher.update(b"\0")
                hasher.update(doc.content.encode("utf-8", errors="ignore"))
                hasher.update(b"\0")

                doc_digest = hashlib.sha256(
                    f"{doc.label}\0{doc.content}".encode("utf-8", errors="ignore")
                ).hexdigest()
                cached = self._doc_index_cache.get(norm_key)
                if cached is not None and cached[0] == doc_digest:
                    _, d_chunks, d_tfs, d_lens, d_ngrams = cached
                else:
                    raw_chunks = chunk_document(doc)
                    d_chunks = []
                    d_tfs = []
                    d_lens = []
                    d_ngrams = []
                    base_idx = 0
                    for ch in raw_chunks:
                        d_chunks.append(ch)
                        combined_text = f"{ch.doc_name} {ch.section} {ch.text}"
                        stems = _tokenize_stems(combined_text, is_query=False)
                        tf = Counter(stems)
                        d_tfs.append(tf)
                        d_lens.append(max(1, len(stems)))
                        d_ngrams.append(_char_ngrams(combined_text))
                        base_idx += 1
                        if base_idx % 25 == 0:
                            time.sleep(0)
                    self._doc_index_cache[norm_key] = (doc_digest, d_chunks, d_tfs, d_lens, d_ngrams)

                for ch, tf, dl, ng in zip(d_chunks, d_tfs, d_lens, d_ngrams, strict=True):
                    all_chunks.append(ch)
                    all_tfs.append(tf)
                    all_lens.append(dl)
                    all_ngrams.append(ng)
                    for term in tf:
                        df[term] += 1

            for stale_key in list(self._doc_index_cache):
                if stale_key not in active_keys:
                    self._doc_index_cache.pop(stale_key, None)

            self._content_hash = hasher.hexdigest()
            self._chunks = all_chunks
            self._chunk_tfs = all_tfs
            self._chunk_lens = all_lens
            self._chunk_ngrams = all_ngrams
            self._df = df
            self._avg_dl = sum(all_lens) / len(all_lens) if all_lens else 1.0

    def add_files(
        self,
        paths: Iterable[str | Path | tuple[str | Path, str]],
    ) -> tuple[list[LoadedDocument], list[str]]:
        """Parse and store files (or (path, display_name) pairs), updating RAG index."""
        parsed_entries: list[tuple[str, LoadedDocument]] = []
        errors: list[str] = []

        for item in paths:
            if isinstance(item, tuple) and len(item) == 2:
                raw_path, explicit_display = item[0], str(item[1] or "")
            else:
                raw_path, explicit_display = item, ""

            file_path = Path(raw_path)
            try:
                resolved_path = file_path.resolve()
                resolved = str(resolved_path)
            except OSError:
                resolved_path = file_path
                resolved = str(file_path)
            norm_key = os.path.normcase(os.path.normpath(resolved))

            try:
                content, page_count, truncated = _extract_document_payload(resolved_path)
                if not content:
                    errors.append(f"{file_path.name}: файл порожній або не містить тексту")
                    continue
                try:
                    mtime_ns = resolved_path.stat().st_mtime_ns
                except OSError:
                    mtime_ns = 0
                doc = LoadedDocument(
                    path=resolved,
                    name=file_path.name,
                    content=content,
                    char_count=len(content),
                    display_name=explicit_display or file_path.name,
                    mtime_ns=mtime_ns,
                    page_count=page_count,
                    truncated=truncated,
                )
                parsed_entries.append((norm_key, doc))
            except Exception as exc:
                errors.append(f"{file_path.name}: {exc}")
            time.sleep(0)

        if parsed_entries:
            with self._lock:
                for norm_key, doc in parsed_entries:
                    self._docs[norm_key] = doc
                self._rebuild_index()

        return [doc for _, doc in parsed_entries], errors

    def reload_modified_files(self) -> tuple[list[LoadedDocument], list[str]]:
        """Re-parse any loaded files whose modification timestamp changed on disk."""
        with self._lock:
            snapshot = list(self._docs.items())

        reloaded: list[LoadedDocument] = []
        errors: list[str] = []
        removed_keys: list[str] = []

        for norm_key, old_doc in snapshot:
            fpath = Path(old_doc.path)
            if not fpath.is_file():
                removed_keys.append(norm_key)
                errors.append(f"{old_doc.label}: файл видалено або переміщено")
                continue
            try:
                mtime_ns = fpath.stat().st_mtime_ns
            except OSError:
                continue
            if old_doc.mtime_ns and mtime_ns == old_doc.mtime_ns:
                continue
            try:
                content, page_count, truncated = _extract_document_payload(fpath)
                if not content:
                    errors.append(f"{old_doc.label}: оновлений файл порожній")
                    continue
                new_doc = LoadedDocument(
                    path=old_doc.path,
                    name=old_doc.name,
                    content=content,
                    char_count=len(content),
                    display_name=old_doc.display_name,
                    mtime_ns=mtime_ns,
                    page_count=page_count,
                    truncated=truncated,
                )
                with self._lock:
                    if norm_key in self._docs:
                        self._docs[norm_key] = new_doc
                reloaded.append(new_doc)
            except Exception as exc:
                errors.append(f"{old_doc.label}: {exc}")
            time.sleep(0)

        if removed_keys or reloaded:
            with self._lock:
                for rk in removed_keys:
                    self._docs.pop(rk, None)
                    self._doc_index_cache.pop(rk, None)
                self._rebuild_index()

        return reloaded, errors

    def remove_file(self, path: str) -> bool:
        """Remove a single document and immediately refresh the RAG index."""
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            resolved = path
        norm_key = os.path.normcase(os.path.normpath(resolved))
        raw_norm = os.path.normcase(os.path.normpath(path))

        with self._lock:
            removed = False
            for candidate in (norm_key, raw_norm):
                if candidate in self._docs:
                    del self._docs[candidate]
                    self._doc_index_cache.pop(candidate, None)
                    removed = True
                    break
            if not removed:
                for key, doc in list(self._docs.items()):
                    if doc.path in {path, resolved}:
                        del self._docs[key]
                        self._doc_index_cache.pop(key, None)
                        removed = True
                        break
            if removed:
                self._rebuild_index()
            return removed

    def clear(self) -> None:
        """Remove all loaded documents and clear the RAG index."""
        with self._lock:
            if self._docs:
                self._docs.clear()
                self._doc_index_cache.clear()
                self._rebuild_index()

    def summary(self) -> str:
        """Return a compact Ukrainian status summary of loaded context."""
        with self._lock:
            if not self._docs:
                return "Контекст не завантажено"
            count_label = _format_file_count_uk(len(self._docs))
            total = self.total_chars
        if total >= 1000:
            size_label = f"{total / 1000:.1f} тис. симв."
        else:
            size_label = f"{total} симв."
        return f"{count_label} ({size_label})"

    def extract_vocabulary_terms(self, max_terms: int = 80) -> list[str]:
        """Extract domain acronyms and CamelCase/technical terms for Whisper normalization."""
        with self._lock:
            docs = self.documents
        if not docs:
            return []

        counts: Counter[str] = Counter()
        canonical: dict[str, str] = {}

        for doc in docs:
            for token in _TECH_TOKEN_RE.findall(doc.content[:60000]):
                clean = token.strip("_")
                if len(clean) < 3 or len(clean) > 32:
                    continue
                low = clean.lower()
                if low in _STOP_LATIN_WORDS:
                    continue
                is_acronym = clean.isupper() and len(clean) >= 3
                is_mixed = any(c.isupper() for c in clean[1:]) or any(c.isdigit() for c in clean)
                weight = 3 if (is_acronym or is_mixed) else 1
                counts[low] += weight
                if low not in canonical or (is_acronym or is_mixed):
                    canonical[low] = clean

        return [canonical[k] for k, _ in counts.most_common(max_terms)]

    def get_combined_context(self, max_chars: int) -> str:
        """Combine loaded documents deterministically and enforce max_chars."""
        with self._lock:
            docs = self.documents
        if not docs or max_chars <= 0:
            return ""

        sections = [f"=== ФАЙЛ: {doc.label} ===\n{doc.content}" for doc in docs]
        combined = "\n\n".join(sections)
        if len(combined) <= max_chars:
            return combined

        suffix = "\n\n[... Контекст обрізано через ліміт CONTEXT_MAX_CHARS ...]"
        keep = max(0, max_chars - len(suffix))
        return combined[:keep].rstrip() + suffix

    def retrieve_context(
        self,
        query: str,
        max_chars: int,
        full_context_threshold: int = 24000,
    ) -> tuple[str, list[str]]:
        """Tiered CAG + Hybrid BM25/4-gram RAG context selector with diversity & manifest."""
        with self._lock:
            if not self._docs or max_chars <= 0:
                return "", []

            docs = self.documents
            header_overhead = sum(len(d.label) + 22 for d in docs)
            tier1_limit = min(max_chars, max(1000, full_context_threshold))
            if (self.total_chars + header_overhead) <= tier1_limit:
                return self.get_combined_context(max_chars), [doc.label for doc in docs]

            if not self._chunks:
                return self.get_combined_context(max_chars), [doc.label for doc in docs]

            chunks = list(self._chunks)
            chunk_tfs = list(self._chunk_tfs)
            chunk_lens = list(self._chunk_lens)
            chunk_ngrams = list(self._chunk_ngrams)
            df = self._df
            avg_dl = self._avg_dl

        query_stems = set(_tokenize_stems(query, is_query=True))
        query_grams = _char_ngrams(query)
        n_chunks = len(chunks)
        k1 = 1.5
        b = 0.75

        raw_bm25: list[float] = []
        raw_gram: list[float] = []
        raw_boost: list[float] = []

        for idx, chunk in enumerate(chunks):
            tf = chunk_tfs[idx]
            dl = chunk_lens[idx]
            bm25 = 0.0
            for term in query_stems:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                term_df = df.get(term, 1)
                idf = math.log(1.0 + (n_chunks - term_df + 0.5) / (term_df + 0.5))
                bm25 += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avg_dl))
            raw_bm25.append(bm25)

            c_grams = chunk_ngrams[idx]
            if query_grams and c_grams:
                inter = len(query_grams & c_grams)
                raw_gram.append(inter / len(query_grams))
            else:
                raw_gram.append(0.0)

            path_tokens = set(_tokenize_stems(chunk.doc_name, is_query=False))
            path_hits = len(path_tokens & query_stems)
            raw_boost.append(min(0.16, 0.08 * path_hits) if path_hits else 0.0)

        max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
        base_scores: list[float] = []
        for idx in range(n_chunks):
            norm_bm25 = (raw_bm25[idx] / max_bm25) if max_bm25 > 0.0 else 0.0
            intro_bonus = 0.015 if (n_chunks > 1 and chunks[idx].chunk_index == 0) else 0.0
            base_scores.append(0.68 * norm_bm25 + 0.32 * raw_gram[idx] + raw_boost[idx] + intro_bonus)

        manifest_block = ""
        if len(docs) > 1 and max_chars >= 2600:
            labels = ", ".join(d.label for d in docs[:15])
            if len(docs) > 15:
                labels += f", ... (+{len(docs) - 15})"
            candidate_manifest = f"[Файли проєкту ({len(docs)}): {labels}]"
            manifest_block = candidate_manifest[: min(380, max_chars // 10)]

        used_chars = len(manifest_block) + (2 if manifest_block else 0)
        chosen_indices: list[int] = []

        if not query_stems and not query_grams:
            # Round-robin across files when query carries zero lexical overlap
            doc_buckets: dict[str, list[int]] = {}
            for idx, ch in enumerate(chunks):
                doc_buckets.setdefault(ch.doc_path, []).append(idx)
            max_depth = max((len(v) for v in doc_buckets.values()), default=0)
            rr_order: list[int] = []
            for depth in range(max_depth):
                for bucket in doc_buckets.values():
                    if depth < len(bucket):
                        rr_order.append(bucket[depth])
            for idx in rr_order:
                ch = chunks[idx]
                cost = len(f"[Джерело: {ch.doc_name} | {ch.section}]\n{ch.text}") + (2 if chosen_indices else 0)
                if used_chars + cost > max_chars:
                    if not chosen_indices:
                        chosen_indices.append(idx)
                        used_chars += cost
                    break
                chosen_indices.append(idx)
                used_chars += cost
        else:
            remaining = set(range(n_chunks))
            doc_counts: Counter[str] = Counter()
            while remaining and used_chars < max_chars:
                best_idx = max(
                    remaining,
                    key=lambda i: (base_scores[i] * (0.86 ** doc_counts[chunks[i].doc_path]), -i),
                )
                remaining.remove(best_idx)
                ch = chunks[best_idx]
                formatted = f"[Джерело: {ch.doc_name} | {ch.section}]\n{ch.text}"
                cost = len(formatted) + (2 if chosen_indices else 0)
                if used_chars + cost > max_chars:
                    if not chosen_indices:
                        chosen_indices.append(best_idx)
                        used_chars += cost
                        break
                    continue
                chosen_indices.append(best_idx)
                doc_counts[ch.doc_path] += 1
                used_chars += cost
                if used_chars >= max_chars:
                    break

        chosen_indices.sort()
        sections: list[str] = [manifest_block] if manifest_block else []
        sources: list[str] = []
        seen_sources: set[str] = set()
        for idx in chosen_indices:
            ch = chunks[idx]
            sections.append(f"[Джерело: {ch.doc_name} | {ch.section}]\n{ch.text}")
            src_label = f"{ch.doc_name} ({ch.section})"
            if src_label not in seen_sources:
                seen_sources.add(src_label)
                sources.append(src_label)

        combined = "\n\n".join(sections)
        if len(combined) > max_chars:
            combined = combined[:max_chars].rstrip()
        return combined, sources
