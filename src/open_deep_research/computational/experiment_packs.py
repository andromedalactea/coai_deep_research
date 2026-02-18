"""Experiment pack abstractions for reusable discovery profiles.

This module introduces a lightweight contract similar to AI-Scientist templates,
but adapted to the graph-native architecture of computational discovery.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ExperimentPack(BaseModel):
    """Declarative configuration for domain-specific experiment behavior."""

    id: str
    display_name: str
    domain: str = "general"
    task_context: str = ""
    required_outputs: List[str] = Field(default_factory=list)
    preferred_metrics: List[str] = Field(default_factory=list)
    baseline_expectations: List[str] = Field(default_factory=list)
    planning_hints: List[str] = Field(default_factory=list)
    writeup_hints: List[str] = Field(default_factory=list)


_PACKS: Dict[str, ExperimentPack] = {
    "astronomy_default": ExperimentPack(
        id="astronomy_default",
        display_name="Astronomy Discovery Default",
        domain="astronomy",
        task_context="Favor real observational archives before simulation.",
        required_outputs=[
            "statistical_result",
            "figure",
        ],
        preferred_metrics=[
            "p_value",
            "confidence_interval",
            "effect_size",
        ],
        baseline_expectations=[
            "At least one external data source must be queried if available.",
            "Claims should include uncertainty quantification.",
        ],
        planning_hints=[
            "Search VizieR/MAST/NASA archives before synthesizing data.",
            "Design one run for retrieval sanity checks and one for hypothesis tests.",
        ],
        writeup_hints=[
            "Include data provenance and archive query details.",
            "Report detectability limits and uncertainty bars.",
        ],
    ),
    "general_default": ExperimentPack(
        id="general_default",
        display_name="General Science Default",
        domain="general",
        task_context="Prioritize reproducible quantitative experiments with explicit assumptions.",
        required_outputs=["statistical_result"],
        preferred_metrics=["p_value", "confidence_interval"],
        baseline_expectations=["At least one falsifiable hypothesis per cycle."],
        planning_hints=["Plan incremental runs: baseline -> perturbation -> robustness."],
        writeup_hints=["Explicitly separate exploratory findings from validated claims."],
    ),
}


def get_experiment_pack(pack_id: str) -> ExperimentPack:
    """Return a registered experiment pack, falling back to astronomy default."""

    if pack_id in _PACKS:
        return _PACKS[pack_id]
    return _PACKS["astronomy_default"]


def list_experiment_packs() -> List[ExperimentPack]:
    """List all available experiment packs."""

    return list(_PACKS.values())

