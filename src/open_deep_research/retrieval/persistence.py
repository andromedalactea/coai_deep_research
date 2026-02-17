"""Persistence layer for web retrieval results.

Saves all retrieved content (raw text, tables, scores, metadata) to the
run's ``sources_dir/web_retrieval/`` directory so that every piece of
information fetched from the internet during a research run is preserved
for traceability, reproducibility, and downstream reuse.

The directory layout mirrors the pattern used by ``scientific_tools.py``
which saves ArXiv PDFs to ``sources_dir/``.

Directory structure created per retrieval call::

    sources_dir/
    └── web_retrieval/
        └── retrieve_<timestamp>/
            ├── manifest.json          # queries, mode, accepted/rejected counts, timing
            ├── source_001_<domain>.json   # per-source metadata + scores
            ├── source_001_<domain>.txt    # full extracted content
            ├── source_001_<domain>_tables.md  # tables (if any)
            ├── source_002_<domain>.json
            ├── source_002_<domain>.txt
            └── ...
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def _get_sources_dir(config: Optional[RunnableConfig]) -> Optional[str]:
    """Extract sources_dir from runtime config.

    Checks two locations (in order):
    1. ``config["configurable"]["sources_dir"]`` — set directly by runner scripts.
    2. The ``Configuration`` object (reads from env var ``SOURCES_DIR`` or Studio UI).
    """
    if config is None:
        return None
    configurable = config.get("configurable", {})
    sources_dir = configurable.get("sources_dir", None)
    if sources_dir:
        return sources_dir
    # Fallback: check the Configuration object (picks up env vars / Studio config)
    try:
        from open_deep_research.configuration import Configuration
        cfg = Configuration.from_runnable_config(config)
        return cfg.sources_dir
    except Exception:
        return None


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    """Turn a URL domain or title into a safe filesystem name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-.]", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_len] if cleaned else "unknown"


def _extract_domain(url: str) -> str:
    """Extract domain from URL for filename."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or "unknown"
        if host.startswith("www."):
            host = host[4:]
        return host.replace(".", "_")
    except Exception:
        return "unknown"


async def save_retrieval_results(
    config: Optional[RunnableConfig],
    queries: List[str],
    mode: str,
    accepted: List[Tuple[dict, Any]],
    rejected_count: int,
    all_processed: Optional[List[dict]] = None,
) -> Optional[str]:
    """Save all retrieval results to the run's sources directory.

    Args:
        config: Runtime config containing sources_dir.
        queries: The search queries executed.
        mode: Retrieval mode used.
        accepted: List of (source_dict, SourceScore) tuples that passed scoring.
        rejected_count: Number of sources rejected by scoring.
        all_processed: Optional full list of processed sources (including rejected)
            for complete traceability.

    Returns:
        Path to the created retrieval directory, or None if sources_dir not configured.
    """
    sources_dir = _get_sources_dir(config)
    if not sources_dir:
        return None

    try:
        # Create web_retrieval subdirectory
        web_dir = Path(sources_dir) / "web_retrieval"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        retrieval_dir = web_dir / f"retrieve_{timestamp}"
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        # ── Write manifest ──
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "queries": queries,
            "mode": mode,
            "sources_accepted": len(accepted),
            "sources_rejected": rejected_count,
            "sources_total": len(accepted) + rejected_count,
        }
        manifest_path = retrieval_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        # ── Write per-source files ──
        for i, (src, score) in enumerate(accepted, 1):
            domain = _extract_domain(src.get("url", ""))
            prefix = f"source_{i:03d}_{domain}"

            # Metadata JSON (scores, URL, title, method, completeness)
            meta = {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "retrieval_method": src.get("retrieval_method", ""),
                "data_completeness": src.get("data_completeness", ""),
                "authority_label": src.get("authority", ""),
                "scores": {
                    "relevance": score.relevance,
                    "authority": score.authority,
                    "recency": score.recency,
                    "methodology_quality": score.methodology_quality,
                    "overall": score.overall,
                    "has_quantitative_data": score.has_quantitative_data,
                },
                "query_matched": src.get("query", ""),
            }
            meta_path = retrieval_dir / f"{prefix}.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

            # Full content text
            content = src.get("content", "")
            if content:
                content_path = retrieval_dir / f"{prefix}.txt"
                content_path.write_text(content, encoding="utf-8")

            # Tables (if Deep Read extracted them and they're in the content)
            # Check if there's a tables section in the content
            if "### Tables" in content:
                tables_start = content.index("### Tables")
                tables_section = content[tables_start:]
                # Find end of tables section (next ### or end)
                next_section = tables_section.find("\n### ", 1)
                if next_section > 0:
                    tables_section = tables_section[:next_section]
                tables_path = retrieval_dir / f"{prefix}_tables.md"
                tables_path.write_text(tables_section, encoding="utf-8")

        # ── Also save rejected sources for full traceability ──
        if all_processed:
            accepted_urls = {src.get("url") for src, _ in accepted}
            rejected_sources = [
                s for s in all_processed
                if s.get("url") not in accepted_urls
            ]
            if rejected_sources:
                rejected_path = retrieval_dir / "rejected_sources.json"
                rejected_data = []
                for src in rejected_sources:
                    rejected_data.append({
                        "url": src.get("url", ""),
                        "title": src.get("title", ""),
                        "authority": src.get("authority", ""),
                        "retrieval_method": src.get("retrieval_method", ""),
                    })
                rejected_path.write_text(
                    json.dumps(rejected_data, indent=2, ensure_ascii=False)
                )

        logger.info("Saved %d retrieval results to %s", len(accepted), retrieval_dir)
        return str(retrieval_dir)

    except Exception as exc:
        logger.warning("Failed to save retrieval results: %s", exc)
        return None


async def save_null_result(
    config: Optional[RunnableConfig],
    queries: List[str],
    mode: str,
    rejected_count: int = 0,
) -> Optional[str]:
    """Save a null-result record for traceability.

    Even when no evidence is found, the fact that these queries were
    executed and returned nothing is itself a scientific finding worth
    preserving.
    """
    sources_dir = _get_sources_dir(config)
    if not sources_dir:
        return None

    try:
        web_dir = Path(sources_dir) / "web_retrieval"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        retrieval_dir = web_dir / f"retrieve_{timestamp}_null"
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "queries": queries,
            "mode": mode,
            "status": "no_evidence",
            "sources_accepted": 0,
            "sources_rejected": rejected_count,
            "message": (
                "No significant data found for the executed queries. "
                "This is a valid scientific finding."
            ),
        }
        manifest_path = retrieval_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        logger.info("Saved null-result retrieval record to %s", retrieval_dir)
        return str(retrieval_dir)

    except Exception as exc:
        logger.warning("Failed to save null-result record: %s", exc)
        return None
