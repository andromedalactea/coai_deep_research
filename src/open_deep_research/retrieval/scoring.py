"""Source relevance and authority scoring for scientific retrieval.

Each retrieved source is scored across four dimensions before it enters
the shared evidence context.  Authority scoring is deterministic (domain
based); relevance and methodology quality use a lightweight LLM call via
the configured summarization model.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.retrieval.domains import get_domain_trust_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Structured output for the LLM relevance assessment
# ─────────────────────────────────────────────────────────────────

class RelevanceAssessment(BaseModel):
    """LLM-produced relevance and methodology assessment for a source."""

    relevance: float = Field(
        description="How relevant is this source to the query? 0.0 = unrelated, 1.0 = directly answers it."
    )
    methodology_quality: float = Field(
        description="Quality of the scientific methodology described. 0.0 = no methodology, 1.0 = rigorous peer-reviewed."
    )
    has_quantitative_data: bool = Field(
        description="Does the source contain concrete numerical data (measurements, statistics, p-values)?"
    )
    brief_justification: str = Field(
        description="One-sentence justification for the scores."
    )


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

@dataclass
class SourceScore:
    """Final composite score for a single retrieved source."""

    relevance: float       # 0-1
    authority: float       # 0-1 (domain-based, no LLM)
    recency: float         # 0-1
    methodology_quality: float  # 0-1
    has_quantitative_data: bool
    overall: float         # weighted composite
    rejection_reason: Optional[str] = None  # set when filtered out


async def score_source(
    url: str,
    content: str,
    query: str,
    *,
    model_name: str = "openai:gpt-4.1-mini",
    model_api_key: Optional[str] = None,
    model_base_url: Optional[str] = None,
    max_structured_output_retries: int = 3,
    min_score: float = 0.3,
) -> SourceScore:
    """Score a single source on four dimensions.

    Args:
        url: Source URL (used for authority scoring).
        content: Extracted text or summary of the source.
        query: The original search query for relevance comparison.
        model_name: LLM for relevance assessment (cheap model recommended).
        model_api_key: Optional API key override.
        model_base_url: Optional base-URL override.
        max_structured_output_retries: Retry count for structured output.
        min_score: Overall threshold below which ``rejection_reason`` is set.

    Returns:
        ``SourceScore`` with individual and composite scores.
    """
    # ── Authority (deterministic) ──
    authority = get_domain_trust_score(url)

    # ── Recency (heuristic: extract year from content) ──
    recency = _estimate_recency(content)

    # ── Relevance + methodology (LLM) ──
    try:
        assessment = await _llm_assess(
            content[:6000],  # keep prompt small for speed
            query,
            model_name=model_name,
            model_api_key=model_api_key,
            model_base_url=model_base_url,
            max_retries=max_structured_output_retries,
        )
        relevance = max(0.0, min(1.0, assessment.relevance))
        methodology = max(0.0, min(1.0, assessment.methodology_quality))
        has_quant = assessment.has_quantitative_data
    except Exception as exc:
        logger.warning("Scoring LLM call failed for %s: %s – using defaults", url, exc)
        relevance = 0.5
        methodology = 0.3
        has_quant = False

    # ── Composite (weighted average) ──
    overall = (
        0.40 * relevance
        + 0.25 * authority
        + 0.15 * recency
        + 0.20 * methodology
    )

    rejection_reason = None
    if overall < min_score:
        rejection_reason = f"Overall score {overall:.2f} below threshold {min_score}"

    return SourceScore(
        relevance=relevance,
        authority=authority,
        recency=recency,
        methodology_quality=methodology,
        has_quantitative_data=has_quant,
        overall=overall,
        rejection_reason=rejection_reason,
    )


async def score_sources_parallel(
    sources: list[dict],
    query: str,
    **kwargs,
) -> list[SourceScore]:
    """Score multiple sources concurrently.

    Each element of *sources* must have keys ``url`` and ``content``.
    """
    tasks = [
        score_source(s["url"], s["content"], query, **kwargs)
        for s in sources
    ]
    return list(await asyncio.gather(*tasks))


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

_YEAR_PATTERN = re.compile(r"\b(20[12]\d)\b")


def _estimate_recency(text: str) -> float:
    """Return a 0-1 recency score based on the most recent year found."""
    current_year = datetime.now().year
    years = [int(y) for y in _YEAR_PATTERN.findall(text)]
    if not years:
        return 0.3  # unknown age – conservative default

    most_recent = max(years)
    age = current_year - most_recent
    if age <= 0:
        return 1.0
    if age <= 1:
        return 0.9
    if age <= 3:
        return 0.7
    if age <= 5:
        return 0.5
    return 0.3


_ASSESSMENT_PROMPT = """You are a scientific source evaluator.  Given a QUERY and a SOURCE excerpt, produce a JSON assessment.

<query>{query}</query>

<source_excerpt>
{content}
</source_excerpt>

Score the source on:
1. **relevance** (0.0-1.0): How directly does this source address the query?
2. **methodology_quality** (0.0-1.0): How rigorous is the scientific methodology described?
   - 0.0 = blog post / opinion / no methodology
   - 0.5 = describes methods but lacks rigour
   - 1.0 = peer-reviewed with detailed, reproducible methodology
3. **has_quantitative_data** (boolean): Does it contain concrete measurements, statistics, or numerical results?
4. **brief_justification**: One sentence explaining your scores.

Return ONLY the JSON object."""


async def _llm_assess(
    content: str,
    query: str,
    *,
    model_name: str,
    model_api_key: Optional[str],
    model_base_url: Optional[str],
    max_retries: int,
) -> RelevanceAssessment:
    """Run a lightweight LLM call to assess relevance and methodology."""
    from open_deep_research.utils import get_effective_model_name

    model = init_chat_model(
        model=get_effective_model_name(model_name),
        max_tokens=512,
        api_key=model_api_key,
        base_url=model_base_url,
        tags=["langsmith:nostream"],
    ).with_structured_output(RelevanceAssessment).with_retry(
        stop_after_attempt=max_retries
    )

    prompt = _ASSESSMENT_PROMPT.format(query=query, content=content)
    return await asyncio.wait_for(
        model.ainvoke([HumanMessage(content=prompt)]),
        timeout=30.0,
    )
