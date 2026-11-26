"""Lightweight LaTeX-to-HTML/Unicode math renderer for QTextBrowser (no external deps)."""

from __future__ import annotations

import html
import re

_GREEK_LETTERS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

_MATH_SYMBOLS: dict[str, str] = {
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "div": "÷",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "propto": "∝",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "setminus": "∖",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "emptyset": "∅",
    "varnothing": "∅",
    "neg": "¬",
    "land": "∧",
    "lor": "∨",
    "circ": "∘",
    "degree": "°",
    "ldots": "…",
    "cdots": "⋯",
    "dots": "…",
    "vdots": "⋮",
    "ddots": "⋱",
    "angle": "∠",
    "perp": "⊥",
    "parallel": "∥",
    "triangle": "△",
    "square": "□",
    "hbar": "ℏ",
    "ell": "ℓ",
    "Re": "ℜ",
    "Im": "ℑ",
    "aleph": "ℵ",
    "prime": "′",
    "langle": "⟨",
    "rangle": "⟩",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "quad": "\u2003",
    "qquad": "\u2003\u2003",
}

_BLACKBOARD_MAP: dict[str, str] = {
    "R": "ℝ",
    "N": "ℕ",
    "Z": "ℤ",
    "Q": "ℚ",
    "C": "ℂ",
    "P": "ℙ",
    "E": "𝔼",
}

_COMBINING_ACCENTS: dict[str, str] = {
    "vec": "\u20d7",
    "bar": "\u0304",
    "overline": "\u0304",
    "hat": "\u0302",
    "tilde": "\u0303",
    "dot": "\u0307",
    "ddot": "\u0308",
}

_KNOWN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "tg",
        "ctg",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "log",
        "ln",
        "lg",
        "exp",
        "lim",
        "sup",
        "inf",
        "max",
        "min",
        "arg",
        "det",
        "dim",
        "ker",
        "deg",
        "gcd",
        "mod",
        "Pr",
    }
)

_ALL_SYMBOLS: dict[str, str] = {**_GREEK_LETTERS, **_MATH_SYMBOLS}

_CODE_SEGMENT_RE = re.compile(
    r"(^```[\s\S]*?^```|^~~~[\s\S]*?^~~~|`[^`\n]+`)",
    re.MULTILINE,
)

_DISPLAY_OR_INLINE_RE = re.compile(
    r"""
    \$\$(?P<dd>[\s\S]+?)\$\$
    | \\\[(?P<lb>[\s\S]+?)\\\]
    | \\\((?P<lp>[\s\S]+?)\\\)
    | (?<![\\$])\$(?!\s)(?P<dl>[^\n$]+?)(?<!\s)\$(?![0-9])
    """,
    re.VERBOSE,
)

_CURRENCY_ONLY_RE = re.compile(r"^[0-9][0-9\s.,]*$")
_BLOCK_PLACEHOLDER_P_RE = re.compile(
    r"<p(?:\s+[^>]*)?>\s*(ZMATHBLOCK\d+Z)\s*</p>",
    re.IGNORECASE,
)


def has_latex_math(text: str) -> bool:
    """Fast check for possible LaTeX delimiters before running parser."""
    if not text:
        return False
    return "$" in text or "\\(" in text or "\\[" in text


def _skip_spaces(src: str, idx: int) -> int:
    n = len(src)
    while idx < n and src[idx].isspace():
        idx += 1
    return idx


def _read_group(src: str, idx: int) -> tuple[str, int]:
    """Read `{...}` or `[...]` or a single LaTeX token/character starting at `idx`."""
    idx = _skip_spaces(src, idx)
    n = len(src)
    if idx >= n:
        return "", idx

    if src[idx] == "{":
        depth = 1
        pos = idx + 1
        while pos < n and depth > 0:
            ch = src[pos]
            if ch == "\\" and pos + 1 < n:
                pos += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return src[idx + 1 : pos], pos + 1
            pos += 1
        return src[idx + 1 :], n

    if src[idx] == "\\":
        pos = idx + 1
        if pos < n and src[pos].isalpha():
            while pos < n and src[pos].isalpha():
                pos += 1
            return src[idx:pos], pos
        if pos < n:
            return src[idx : pos + 1], pos + 1
        return "\\", pos

    return src[idx], idx + 1


def _read_optional_bracket(src: str, idx: int) -> tuple[str | None, int]:
    idx_ws = _skip_spaces(src, idx)
    n = len(src)
    if idx_ws >= n or src[idx_ws] != "[":
        return None, idx
    depth = 1
    pos = idx_ws + 1
    while pos < n and depth > 0:
        if src[pos] == "[":
            depth += 1
        elif src[pos] == "]":
            depth -= 1
            if depth == 0:
                return src[idx_ws + 1 : pos], pos + 1
        pos += 1
    return None, idx


def _strip_outer_tags(fragment: str) -> str:
    return re.sub(r"<[^>]+>", "", fragment)


def _format_fraction(num_raw: str, den_raw: str, *, display: bool) -> str:
    num_html = _render_expr(num_raw, display=False)
    den_html = _render_expr(den_raw, display=False)
    num_plain = _strip_outer_tags(num_html).strip()
    den_plain = _strip_outer_tags(den_html).strip()

    if display:
        return (
            '<table style="display:inline-table; vertical-align:middle; '
            'border-collapse:collapse; margin:0 3px;">'
            f'<tr><td style="border-bottom:1px solid #93C5FD; padding:0 4px 1px 4px; '
            f'text-align:center; border-left:none; border-right:none; border-top:none;">{num_html}</td></tr>'
            f'<tr><td style="padding:1px 4px 0 4px; text-align:center; '
            f'border:none;">{den_html}</td></tr>'
            "</table>"
        )

    if len(num_plain) <= 4 and len(den_plain) <= 4 and "/" not in num_plain and "/" not in den_plain:
        return f"<sup>{num_html}</sup>⁄<sub>{den_html}</sub>"
    if len(num_plain) > 1 and any(op in num_plain for op in ("+", "-", " ")):
        num_html = f"({num_html})"
    if len(den_plain) > 1 and any(op in den_plain for op in ("+", "-", " ", "·", "×")):
        den_html = f"({den_html})"
    return f"{num_html}/{den_html}"


def _render_expr(src: str, *, display: bool = False) -> str:
    out: list[str] = []
    i = 0
    n = len(src)

    while i < n:
        ch = src[i]

        if ch == "\\":
            if i + 1 >= n:
                i += 1
                continue
            nxt = src[i + 1]
            if nxt in (",", ";", ":", " "):
                out.append("\u2009" if nxt == "," else " ")
                i += 2
                continue
            if nxt == "!":
                i += 2
                continue
            if nxt == "\\":
                out.append("<br/>")
                i += 2
                continue
            if nxt in ("{", "}", "$", "%", "&", "#", "_"):
                out.append(html.escape(nxt))
                i += 2
                continue

            j = i + 1
            while j < n and src[j].isalpha():
                j += 1
            cmd = src[i + 1 : j]
            i = j

            if not cmd:
                out.append(html.escape(nxt))
                i += 1
                continue

            if cmd in ("left", "right", "big", "Big", "bigg", "Bigg"):
                continue

            if cmd in ("frac", "dfrac", "tfrac"):
                num, i = _read_group(src, i)
                den, i = _read_group(src, i)
                out.append(_format_fraction(num, den, display=display))
                continue

            if cmd == "sqrt":
                root_deg, i = _read_optional_bracket(src, i)
                radicand, i = _read_group(src, i)
                inner = _render_expr(radicand, display=False)
                deg_html = (
                    f"<sup>{_render_expr(root_deg, display=False)}</sup>"
                    if root_deg
                    else ""
                )
                plain_inner = _strip_outer_tags(inner).strip()
                wrapped = inner if len(plain_inner) == 1 else f"({inner})"
                out.append(f"{deg_html}√{wrapped}")
                continue

            if cmd in ("text", "mathrm", "textrm", "operatorname", "mbox"):
                arg, i = _read_group(src, i)
                out.append(
                    f'<span style="font-family:\'Segoe UI\',sans-serif;">{html.escape(arg)}</span>'
                )
                continue

            if cmd in ("mathbf", "textbf", "boldsymbol"):
                arg, i = _read_group(src, i)
                out.append(f"<b>{_render_expr(arg, display=False)}</b>")
                continue

            if cmd in ("mathit", "textit", "emph"):
                arg, i = _read_group(src, i)
                out.append(f"<i>{_render_expr(arg, display=False)}</i>")
                continue

            if cmd == "mathbb":
                arg, i = _read_group(src, i)
                mapped = "".join(_BLACKBOARD_MAP.get(c, html.escape(c)) for c in arg.strip())
                out.append(mapped)
                continue

            if cmd in _COMBINING_ACCENTS:
                arg, i = _read_group(src, i)
                rendered_arg = _render_expr(arg, display=False)
                comb = _COMBINING_ACCENTS[cmd]
                if len(_strip_outer_tags(rendered_arg)) == 1:
                    out.append(f"{rendered_arg}{comb}")
                else:
                    out.append(f"{rendered_arg}{comb}")
                continue

            if cmd in _ALL_SYMBOLS:
                out.append(_ALL_SYMBOLS[cmd])
                continue

            if cmd in _KNOWN_FUNCTIONS:
                out.append(
                    f'<span style="font-style:normal; font-family:\'Segoe UI\',sans-serif;">{cmd}</span>'
                )
                continue

            out.append(html.escape(cmd))
            continue

        if ch == "^":
            arg, i = _read_group(src, i + 1)
            out.append(f"<sup>{_render_expr(arg, display=False)}</sup>")
            continue

        if ch == "_":
            arg, i = _read_group(src, i + 1)
            out.append(f"<sub>{_render_expr(arg, display=False)}</sub>")
            continue

        if ch == "{":
            arg, i = _read_group(src, i)
            out.append(_render_expr(arg, display=False))
            continue

        if ch == "}":
            i += 1
            continue

        if ch == "~":
            out.append("&nbsp;")
            i += 1
            continue

        if ch in ("+", "-", "="):
            out.append(f" {ch} " if ch == "=" else html.escape(ch))
            i += 1
            continue

        if ch == "<":
            out.append("&lt;")
            i += 1
            continue
        if ch == ">":
            out.append("&gt;")
            i += 1
            continue
        if ch == "&":
            out.append("&amp;")
            i += 1
            continue

        if ch.isspace():
            if out and not out[-1].endswith((" ", "\u2009", "\u2003", "<br/>")):
                out.append(" ")
            i += 1
            continue

        out.append(html.escape(ch))
        i += 1

    result = "".join(out).strip()
    result = re.sub(r" {2,}", " ", result)
    return result


def latex_to_html(expr: str, *, display: bool = False) -> str:
    """Convert a LaTeX math expression into styled HTML/Unicode safe for QTextBrowser."""
    cleaned = (expr or "").strip()
    if not cleaned:
        return ""
    body = _render_expr(cleaned, display=display)
    if display:
        return (
            '<div style="background-color:#141C2B; border-left:3px solid #60A5FA; '
            "border-radius:4px; padding:6px 10px; margin:6px 0; "
            "font-family:'Cambria Math','Segoe UI Symbol','Times New Roman',serif; "
            f'color:#E0F2FE;">{body}</div>'
        )
    return (
        "<span style=\"font-family:'Cambria Math','Segoe UI Symbol','Times New Roman',serif; "
        f'color:#BAE6FD;">{body}</span>'
    )


def extract_latex_placeholders(markdown_text: str) -> tuple[str, dict[str, str]]:
    """Replace LaTeX math outside code blocks with safe alphanumeric placeholders."""
    if not markdown_text or not has_latex_math(markdown_text):
        return markdown_text, {}

    placeholders: dict[str, str] = {}
    counter = 0

    def _replace_in_prose(segment: str) -> str:
        nonlocal counter

        def _on_match(match: re.Match[str]) -> str:
            nonlocal counter
            dd = match.group("dd")
            lb = match.group("lb")
            lp = match.group("lp")
            dl = match.group("dl")

            if dd is not None:
                raw_expr = dd
                is_display = True
            elif lb is not None:
                raw_expr = lb
                is_display = True
            elif lp is not None:
                raw_expr = lp
                is_display = False
            else:
                assert dl is not None
                raw_expr = dl
                if _CURRENCY_ONLY_RE.match(raw_expr.strip()):
                    return match.group(0)
                is_display = False

            rendered = latex_to_html(raw_expr, display=is_display)
            if not rendered:
                return match.group(0)

            counter += 1
            if is_display:
                token = f"ZMATHBLOCK{counter}Z"
                placeholders[token] = rendered
                return f"\n\n{token}\n\n"
            token = f"ZMATHINLINE{counter}Z"
            placeholders[token] = rendered
            return token

        return _DISPLAY_OR_INLINE_RE.sub(_on_match, segment)

    out_parts: list[str] = []
    last_idx = 0
    for code_m in _CODE_SEGMENT_RE.finditer(markdown_text):
        prose = markdown_text[last_idx : code_m.start()]
        if prose:
            out_parts.append(_replace_in_prose(prose))
        out_parts.append(code_m.group(0))
        last_idx = code_m.end()

    tail = markdown_text[last_idx:]
    if tail:
        out_parts.append(_replace_in_prose(tail))

    return "".join(out_parts), placeholders


def inject_latex_html(rendered_html: str, placeholders: dict[str, str]) -> str:
    """Substitute placeholder tokens inside QTextDocument HTML with rendered formula HTML."""
    if not placeholders or not rendered_html:
        return rendered_html

    def _sub_block_p(match: re.Match[str]) -> str:
        token = match.group(1)
        return placeholders.get(token, match.group(0))

    out = _BLOCK_PLACEHOLDER_P_RE.sub(_sub_block_p, rendered_html)
    for token, formula_html in placeholders.items():
        if token in out:
            out = out.replace(token, formula_html)
    return out
