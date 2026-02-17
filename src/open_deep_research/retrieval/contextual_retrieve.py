"""Contextual Retrieve — the always-on scientific retrieval tool.

This tool replaces / augments the original ``tavily_search`` with:
* **Mode-aware retrieval** (exploratory, verification, methods, novelty).
* **Deep Read** on high-authority domains for verification / methods modes.
* **Relevance + authority scoring** with configurable rejection threshold.
* **Null-result handling** that distinguishes *absence of evidence* from
  *search errors*, so downstream agents never hallucinate a loose match.

It is designed to be bound to *every* agent role (supervisor, researcher,
computational discovery nodes) via ``get_all_tools()`` so that retrieval
is a continuous cognitive capability, not a one-shot pipeline step.
"""

import asyncio
import json
import logging
from typing import Annotated, List, Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

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

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Tool description (shown to the LLM when choosing tools)
# ─────────────────────────────────────────────────────────────────

CONTEXTUAL_RETRIEVE_DESCRIPTION = (
    "Search the internet for information with mode-aware scientific retrieval. "
    "Modes: 'exploratory' (broad search with summaries), "
    "'verification' (deep-reads high-authority sources for precise data), "
    "'methods' (fetches full methodology/reproducibility details), "
    "'novelty' (flags unusual or underexplored claims). "
    "Returns scored sources with authority labels and data-completeness flags. "
    "Handles null results explicitly — absence of evidence is reported as a "
    "valid scientific finding, never hallucinated."
)

# Type alias for retrieval modes
RetrievalMode = Literal["exploratory", "verification", "methods", "novelty"]


# ─────────────────────────────────────────────────────────────────
# Main tool
# ─────────────────────────────────────────────────────────────────

@tool(description=CONTEXTUAL_RETRIEVE_DESCRIPTION)
async def contextual_retrieve(
    queries: List[str],
    mode: RetrievalMode = "exploratory",
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None,
) -> str:
    """Search the web with scientific-grade retrieval and scoring.

    Args:
        queries: Search queries to execute.
        mode: Retrieval mode — determines summarization vs deep-read behaviour.
        max_results: Max results per query (injected from config).
        config: Runtime config (API keys, model settings, thresholds).

    Returns:
        Formatted string with scored sources, or a structured null/error report.
    """
    from open_deep_research.configuration import Configuration
    from open_deep_research.utils import (
        get_api_key_for_model,
        get_base_url_for_model,
        get_tavily_api_key,
        summarize_webpage,
        tavily_search_async,
    )
    from langchain.chat_models import init_chat_model
    from open_deep_research.utils import get_effective_model_name
    from open_deep_research.state import Summary

    configurable = Configuration.from_runnable_config(config)

    # Import trace helper (safe to fail — tracing is optional)
    try:
        from open_deep_research.computational.traceability import trace_web_retrieval
    except ImportError:
        trace_web_retrieval = None  # type: ignore[assignment]

    # ── Step 1: execute Tavily search ──
    try:
        search_results = await tavily_search_async(
            queries,
            max_results=max_results,
            topic="general",
            include_raw_content=True,
            config=config,
        )
    except Exception as exc:
        if trace_web_retrieval:
            trace_web_retrieval(queries=queries, mode=mode, sources_accepted=0, sources_rejected=0, status="error")
        return _format_error("search_api_error", str(exc), queries)

    # ── Step 2: deduplicate by URL ──
    unique_results: dict[str, dict] = {}
    for response in search_results:
        for result in response.get("results", []):
            url = result.get("url", "")
            if url and url not in unique_results:
                unique_results[url] = {**result, "query": response.get("query", "")}

    # ── Handle genuine absence of evidence ──
    if not unique_results:
        saved_path = await save_null_result(config, queries, mode)
        if trace_web_retrieval:
            trace_web_retrieval(queries=queries, mode=mode, sources_accepted=0, sources_rejected=0, saved_path=saved_path, status="null")
        return _format_no_evidence(queries)

    # ── Step 3: per-mode content processing ──
    deep_read_timeout = float(getattr(configurable, "deep_read_timeout", 30))
    max_char = configurable.max_content_length
    enable_deep_read = getattr(configurable, "enable_deep_read", True)

    processed_sources: list[dict] = []

    if mode in ("verification", "methods") and enable_deep_read:
        processed_sources = await _process_deep_read(
            unique_results, mode, max_char, deep_read_timeout
        )
    elif mode == "novelty":
        processed_sources = await _process_exploratory(
            unique_results, configurable, config
        )
    else:
        # exploratory — standard Tavily summarization path
        processed_sources = await _process_exploratory(
            unique_results, configurable, config
        )

    # ── Step 4: score all sources ──
    combined_query = " | ".join(queries)
    min_score = float(getattr(configurable, "min_source_score", 0.3))
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    model_base_url = get_base_url_for_model(configurable.summarization_model, config)

    scores: list[SourceScore] = []
    scoring_tasks = [
        score_source(
            url=src["url"],
            content=src["content"][:6000],
            query=combined_query,
            model_name=configurable.summarization_model,
            model_api_key=model_api_key,
            model_base_url=model_base_url,
            max_structured_output_retries=configurable.max_structured_output_retries,
            min_score=min_score,
        )
        for src in processed_sources
    ]
    scores = list(await asyncio.gather(*scoring_tasks))

    # ── Step 5: filter and format ──
    accepted: list[tuple[dict, SourceScore]] = []
    rejected_count = 0
    for src, sc in zip(processed_sources, scores):
        if sc.rejection_reason:
            rejected_count += 1
            continue
        accepted.append((src, sc))

    if not accepted:
        saved_path = await save_null_result(config, queries, mode, rejected_count=rejected_count)
        if trace_web_retrieval:
            trace_web_retrieval(queries=queries, mode=mode, sources_accepted=0, sources_rejected=rejected_count, saved_path=saved_path, status="null")
        return _format_no_evidence(queries, rejected_count=rejected_count)

    # ── Step 6: persist results to sources_dir for traceability ──
    saved_path = await save_retrieval_results(
        config=config,
        queries=queries,
        mode=mode,
        accepted=accepted,
        rejected_count=rejected_count,
        all_processed=processed_sources,
    )

    if trace_web_retrieval:
        trace_web_retrieval(
            queries=queries,
            mode=mode,
            sources_accepted=len(accepted),
            sources_rejected=rejected_count,
            saved_path=saved_path,
            source_urls=[src.get("url", "") for src, _ in accepted],
            status="complete",
        )

    return _format_output(accepted, mode, queries, rejected_count)


# ─────────────────────────────────────────────────────────────────
# Processing strategies
# ─────────────────────────────────────────────────────────────────

async def _process_deep_read(
    unique_results: dict[str, dict],
    mode: str,
    max_chars: int,
    timeout: float,
) -> list[dict]:
    """For verification/methods: deep-read high-authority; summarize the rest."""
    tasks = []
    urls = list(unique_results.keys())
    results = list(unique_results.values())

    async def _deep_or_fallback(url: str, result: dict) -> dict:
        if is_high_authority(url):
            dr = await fetch_raw_content(url, max_chars=max_chars, timeout=timeout)
            if dr.fetch_status == "ok":
                content_parts = [dr.raw_text]
                if dr.tables:
                    content_parts.append("\n\n### Tables\n" + "\n\n".join(dr.tables))
                if dr.numerical_excerpts and mode == "verification":
                    content_parts.append(
                        "\n\n### Key Numerical Data\n"
                        + "\n".join(f"- {ex}" for ex in dr.numerical_excerpts[:30])
                    )
                return {
                    "url": url,
                    "title": result.get("title", ""),
                    "content": "\n".join(content_parts),
                    "data_completeness": dr.data_completeness,
                    "retrieval_method": "deep_read",
                    "authority": get_authority_label(url),
                }
            # Deep read failed — fall back to Tavily raw content
            raw = result.get("raw_content") or result.get("content", "")
            return {
                "url": url,
                "title": result.get("title", ""),
                "content": raw[:max_chars],
                "data_completeness": "partial",
                "retrieval_method": "deep_read_fallback",
                "authority": get_authority_label(url),
            }
        else:
            # Non-high-authority: use Tavily content directly
            raw = result.get("raw_content") or result.get("content", "")
            return {
                "url": url,
                "title": result.get("title", ""),
                "content": raw[:max_chars],
                "data_completeness": "full" if result.get("raw_content") else "partial",
                "retrieval_method": "tavily_raw",
                "authority": get_authority_label(url),
            }

    tasks = [_deep_or_fallback(url, res) for url, res in zip(urls, results)]
    return list(await asyncio.gather(*tasks))


async def _process_exploratory(
    unique_results: dict[str, dict],
    configurable,
    config: RunnableConfig,
) -> list[dict]:
    """Standard summarization path (mirrors original tavily_search behaviour)."""
    from open_deep_research.utils import (
        get_api_key_for_model,
        get_base_url_for_model,
        get_effective_model_name,
        summarize_webpage,
    )
    from langchain.chat_models import init_chat_model
    from open_deep_research.state import Summary

    max_chars = configurable.max_content_length

    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    model_base_url = get_base_url_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=get_effective_model_name(configurable.summarization_model),
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        base_url=model_base_url,
        tags=["langsmith:nostream"],
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    async def _noop():
        return None

    urls = list(unique_results.keys())
    results = list(unique_results.values())

    summarization_tasks = [
        _noop() if not res.get("raw_content")
        else summarize_webpage(summarization_model, res["raw_content"][:max_chars])
        for res in results
    ]
    summaries = await asyncio.gather(*summarization_tasks)

    processed: list[dict] = []
    for url, res, summary in zip(urls, results, summaries):
        content = res["content"] if summary is None else str(summary)
        processed.append({
            "url": url,
            "title": res.get("title", ""),
            "content": content,
            "data_completeness": "full" if res.get("raw_content") else "partial",
            "retrieval_method": "tavily_summary",
            "authority": get_authority_label(url),
        })
    return processed


# ─────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────

def _format_output(
    accepted: list[tuple[dict, SourceScore]],
    mode: str,
    queries: list[str],
    rejected_count: int,
) -> str:
    """Format accepted sources with scores and metadata."""
    parts = [
        f"Contextual Retrieve Results  [mode={mode}]  "
        f"[queries={len(queries)}]  [accepted={len(accepted)}, rejected={rejected_count}]\n"
    ]

    for i, (src, sc) in enumerate(accepted, 1):
        parts.append(f"\n--- SOURCE {i}: {src['title']} ---")
        parts.append(f"URL: {src['url']}")
        parts.append(
            f"Scores: relevance={sc.relevance:.2f}  authority={sc.authority:.2f}  "
            f"recency={sc.recency:.2f}  methodology={sc.methodology_quality:.2f}  "
            f"overall={sc.overall:.2f}"
        )
        parts.append(f"Authority: {src['authority']}  |  Data completeness: {src['data_completeness']}  |  Method: {src['retrieval_method']}")
        has_quant = "yes" if sc.has_quantitative_data else "no"
        parts.append(f"Has quantitative data: {has_quant}")
        parts.append(f"\nCONTENT:\n{src['content']}\n")
        parts.append("-" * 80)

    return "\n".join(parts)


def _format_no_evidence(queries: list[str], rejected_count: int = 0) -> str:
    """Return a structured 'absence of evidence' report — never hallucinate."""
    report = {
        "status": "no_evidence",
        "queries_executed": queries,
        "results_found": 0,
        "results_rejected_by_scoring": rejected_count,
        "message": (
            "No significant data found for the executed queries. "
            "This is a valid scientific finding — absence of evidence "
            "should be reported as such. Do NOT retry with looser queries "
            "or hallucinate results."
        ),
    }
    return (
        "CONTEXTUAL RETRIEVE: NO EVIDENCE\n"
        + json.dumps(report, indent=2)
    )


def _format_error(error_type: str, detail: str, queries: list[str]) -> str:
    """Return a structured error report (tool failure, not null result)."""
    report = {
        "status": "error",
        "error_type": error_type,
        "queries_executed": queries,
        "message": f"Search tool error: {detail}. This is a tool failure, not an absence of evidence.",
    }
    return (
        "CONTEXTUAL RETRIEVE: ERROR\n"
        + json.dumps(report, indent=2)
    )
