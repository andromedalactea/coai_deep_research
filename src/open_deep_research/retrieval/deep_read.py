"""Deep Read: raw content fetcher that bypasses summarization.

When Tavily's auto-summary would strip precise numerical data (error
margins, p-values, spectral coefficients …) this module fetches the
original page with httpx + BeautifulSoup and extracts structured
scientific content while preserving all quantitative detail.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Regex that catches numbers with optional uncertainty notation
# e.g.  "1.23 ± 0.04", "p < 0.001", "3.14e-5", "SNR = 42.7"
_NUMERICAL_PATTERN = re.compile(
    r"(?:"
    r"[\d]+\.[\d]+(?:[eE][+-]?\d+)?(?:\s*[±+\-]\s*[\d]+\.[\d]+(?:[eE][+-]?\d+)?)?"
    r"|p\s*[<>=≤≥]\s*[\d.]+"
    r"|SNR\s*[=:≈]\s*[\d.]+"
    r"|(?:σ|sigma)\s*[=:≈]\s*[\d.]+"
    r")"
)

# Tags typically containing main article body
_ARTICLE_SELECTORS = [
    "article",
    'div[role="main"]',
    "main",
    ".article-body",
    ".paper-content",
    "#content",
    ".entry-content",
    ".post-content",
]

# Tags to strip from output (nav, ads, scripts…)
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}

# Default user-agent for scraping
_USER_AGENT = (
    "Mozilla/5.0 (compatible; DeepResearchBot/1.0; "
    "+https://github.com/langchain-ai/open_deep_research)"
)


@dataclass
class DeepReadResult:
    """Structured result from a Deep Read fetch."""

    raw_text: str
    """Full extracted body text (stripped of boilerplate)."""

    tables: List[str] = field(default_factory=list)
    """Markdown-formatted tables found in the page."""

    numerical_excerpts: List[str] = field(default_factory=list)
    """Lines that contain significant numerical data."""

    fetch_status: str = "ok"
    """One of: 'ok', 'timeout', 'http_error', 'parse_error'."""

    status_detail: str = ""
    """Human-readable detail on failure (empty when ok)."""

    data_completeness: str = "full"
    """'full' when raw scrape succeeded, 'partial' on fallback."""


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

async def fetch_raw_content(
    url: str,
    max_chars: int = 80_000,
    timeout: float = 30.0,
) -> DeepReadResult:
    """Fetch and parse a URL into structured scientific content.

    Args:
        url: The page to fetch.
        max_chars: Hard character cap for extracted body text.
        timeout: HTTP request timeout in seconds.

    Returns:
        ``DeepReadResult`` with text, tables, numerical excerpts, and status.
    """
    try:
        html = await _fetch_html(url, timeout=timeout)
    except httpx.TimeoutException:
        return DeepReadResult(
            raw_text="",
            fetch_status="timeout",
            status_detail=f"HTTP request timed out after {timeout}s for {url}",
            data_completeness="partial",
        )
    except httpx.HTTPStatusError as exc:
        return DeepReadResult(
            raw_text="",
            fetch_status="http_error",
            status_detail=f"HTTP {exc.response.status_code} for {url}",
            data_completeness="partial",
        )
    except Exception as exc:
        return DeepReadResult(
            raw_text="",
            fetch_status="http_error",
            status_detail=f"Fetch failed for {url}: {exc}",
            data_completeness="partial",
        )

    try:
        return _parse_html(html, max_chars=max_chars)
    except Exception as exc:
        logger.warning("Deep Read parse error for %s: %s", url, exc)
        return DeepReadResult(
            raw_text=html[:max_chars],
            fetch_status="parse_error",
            status_detail=str(exc),
            data_completeness="partial",
        )


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────

async def _fetch_html(url: str, timeout: float = 30.0) -> str:
    """GET the URL and return raw HTML string."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


def _parse_html(html: str, max_chars: int = 80_000) -> DeepReadResult:
    """Parse HTML into structured DeepReadResult."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Try to locate the main article container
    body_element = None
    for selector in _ARTICLE_SELECTORS:
        body_element = soup.select_one(selector)
        if body_element:
            break
    if body_element is None:
        body_element = soup.body or soup

    # ── Extract tables as markdown ──
    tables: list[str] = []
    for table_tag in body_element.find_all("table"):
        md_table = _table_to_markdown(table_tag)
        if md_table:
            tables.append(md_table)

    # ── Extract full text ──
    raw_text = body_element.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    raw_text = raw_text[:max_chars]

    # ── Extract numerical excerpts ──
    numerical_excerpts: list[str] = []
    for line in raw_text.split("\n"):
        line_stripped = line.strip()
        if line_stripped and _NUMERICAL_PATTERN.search(line_stripped):
            numerical_excerpts.append(line_stripped)

    return DeepReadResult(
        raw_text=raw_text,
        tables=tables,
        numerical_excerpts=numerical_excerpts,
        fetch_status="ok",
        data_completeness="full",
    )


def _table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML <table> into a simple Markdown table string."""
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            (td.get_text(strip=True) or "")
            for td in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Determine column count from widest row
    n_cols = max(len(r) for r in rows)

    # Pad shorter rows
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    # Build markdown
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
