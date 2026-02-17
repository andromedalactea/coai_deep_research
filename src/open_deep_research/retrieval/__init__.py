"""Enhanced scientific retrieval module.

Provides ``contextual_retrieve`` — a mode-aware, always-on search tool
with Deep Read, relevance/authority scoring, and null-result handling.

Quick start::

    from open_deep_research.retrieval import contextual_retrieve, RetrievalMode

Modules
-------
contextual_retrieve
    Main LangChain tool callable by any agent.
deep_read
    Raw content fetcher (httpx + BeautifulSoup) that preserves numerical data.
scoring
    Source relevance, authority, recency, and methodology scoring.
domains
    High-authority domain registry and deterministic trust scoring.
persistence
    Saves retrieval results to disk (sources_dir/web_retrieval/) for traceability.
"""

from open_deep_research.retrieval.contextual_retrieve import (
    CONTEXTUAL_RETRIEVE_DESCRIPTION,
    RetrievalMode,
    contextual_retrieve,
)
from open_deep_research.retrieval.deep_read import DeepReadResult, fetch_raw_content
from open_deep_research.retrieval.domains import (
    get_authority_label,
    get_domain_trust_score,
    is_high_authority,
)
from open_deep_research.retrieval.persistence import (
    save_null_result,
    save_retrieval_results,
)
from open_deep_research.retrieval.scoring import SourceScore, score_source

__all__ = [
    "contextual_retrieve",
    "CONTEXTUAL_RETRIEVE_DESCRIPTION",
    "RetrievalMode",
    "DeepReadResult",
    "fetch_raw_content",
    "SourceScore",
    "score_source",
    "is_high_authority",
    "get_domain_trust_score",
    "get_authority_label",
    "save_retrieval_results",
    "save_null_result",
]
