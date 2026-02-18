"""Multi-round literature novelty engine inspired by AI-Scientist loops."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.computational.configuration import ComputationalConfiguration
from open_deep_research.utils import get_model_runtime_config

S2_API_KEY = os.getenv("S2_API_KEY")


class NoveltyAssessmentResult(BaseModel):
    """Structured result returned by novelty assessment."""

    verdict: str = "inconclusive"  # novel | not_novel | inconclusive
    confidence: float = 0.0
    reasoning: str = ""
    rounds_used: int = 0
    searched_queries: List[str] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)


def _search_semantic_scholar(query: str, result_limit: int = 8) -> List[Dict[str, Any]]:
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationCount",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", []) or []


def _search_openalex(query: str, result_limit: int = 8) -> List[Dict[str, Any]]:
    url = "https://api.openalex.org/works"
    response = requests.get(
        url,
        params={"search": query, "per_page": result_limit},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results", []) or []
    papers: List[Dict[str, Any]] = []
    for item in results:
        authors = [
            (auth.get("author") or {}).get("display_name", "Unknown")
            for auth in item.get("authorships", [])[:8]
        ]
        papers.append(
            {
                "title": item.get("title", ""),
                "authors": ", ".join(authors),
                "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                "year": item.get("publication_year"),
                "abstract": item.get("abstract_inverted_index", {}),
                "citationCount": item.get("cited_by_count", 0),
            }
        )
    return papers


def _extract_json_block(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output.")


async def assess_novelty(
    hypothesis_text: str,
    research_brief: str,
    config: RunnableConfig,
    configurable: ComputationalConfiguration,
) -> NoveltyAssessmentResult:
    """Run a bounded novelty loop with model-guided literature queries."""

    model_name = configurable.novelty_model or configurable.worker_model or configurable.research_model
    max_tokens = configurable.worker_model_max_tokens or configurable.research_model_max_tokens
    model_factory = init_chat_model(configurable_fields=("model", "max_tokens", "api_key", "base_url"))
    primary_model = model_factory.with_config(
        get_model_runtime_config(model_name, config, max_tokens=max_tokens, tags=["langsmith:nostream", "phase:novelty"])
    )
    fallback_model = None
    if configurable.fallback_fast_model:
        fallback_model = model_factory.with_config(
            get_model_runtime_config(
                configurable.fallback_fast_model,
                config,
                max_tokens=max_tokens,
                tags=["langsmith:nostream", "phase:novelty", "fallback"],
            )
        )

    searched_queries: List[str] = []
    collected_papers: List[Dict[str, Any]] = []
    evidence_blob = "No papers retrieved yet."
    verdict = "inconclusive"
    confidence = 0.0
    reasoning = ""

    for round_idx in range(configurable.novelty_max_rounds):
        prompt = f"""
You are evaluating novelty for a scientific hypothesis.

Research brief:
{research_brief[:2500]}

Hypothesis:
{hypothesis_text}

Current literature evidence:
{evidence_blob[:5000]}

Respond ONLY with JSON:
{{
  "query": "optional search query",
  "decision": "continue|novel|not_novel",
  "confidence": 0.0,
  "reasoning": "brief justification"
}}
"""
        try:
            response = await primary_model.ainvoke([HumanMessage(content=prompt)])
        except Exception:
            if fallback_model is None:
                raise
            response = await fallback_model.ainvoke([HumanMessage(content=prompt)])
        parsed = _extract_json_block(str(response.content))
        decision = str(parsed.get("decision", "continue")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reasoning = str(parsed.get("reasoning", "")).strip()
        query = str(parsed.get("query", "")).strip()

        if decision in {"novel", "not_novel"}:
            verdict = decision
            return NoveltyAssessmentResult(
                verdict=verdict,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
                rounds_used=round_idx + 1,
                searched_queries=searched_queries,
                papers=collected_papers[:20],
            )

        if not query:
            query = hypothesis_text[:180]
        searched_queries.append(query)

        try:
            if configurable.novelty_engine == "openalex":
                papers = _search_openalex(query)
            else:
                papers = _search_semantic_scholar(query)
        except Exception as exc:  # noqa: BLE001
            papers = [{"title": "search_error", "abstract": str(exc), "citationCount": 0}]

        collected_papers.extend(papers)
        compact = []
        for idx, paper in enumerate(papers[:8]):
            title = paper.get("title", "untitled")
            venue = paper.get("venue", "")
            year = paper.get("year", "")
            cites = paper.get("citationCount", 0)
            abstract = str(paper.get("abstract", ""))[:400]
            compact.append(f"{idx+1}. {title} ({venue}, {year}, cites={cites}) :: {abstract}")
        evidence_blob = "\n".join(compact) if compact else "No papers found."

    return NoveltyAssessmentResult(
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning or "Novelty loop reached max rounds without a definitive decision.",
        rounds_used=configurable.novelty_max_rounds,
        searched_queries=searched_queries,
        papers=collected_papers[:20],
    )

