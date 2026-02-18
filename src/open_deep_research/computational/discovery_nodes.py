"""Discovery Nodes for the Computational Scientific Discovery System.

This module implements the node functions for the LangGraph-based
scientific discovery workflow, handling:
- Hypothesis generation and refinement
- Experiment design and execution
- Result analysis and interpretation
- Iteration decisions
- Final report synthesis
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from open_deep_research.computational.code_interpreter import (
    execute_code,
    execute_experiment,
    format_outputs_for_ai,
    format_outputs_for_report,
    SandboxManager,
    analyze_images_with_vision,
    validate_experiment_quality,
    get_quality_improvement_suggestions,
)
from open_deep_research.computational.traceability import (
    TraceEventType,
    TraceManager,
    trace_node_enter,
    trace_node_exit,
    trace_phase_start,
    trace_phase_end,
    trace_performance_metric,
    trace_code_execution,
    trace_supervisor_decision,
    trace_hypothesis_created,
    trace_experiment_started,
    trace_finding_recorded,
    trace_output_generated,
    trace_claim_created,
    trace_claim_validated,
    trace_claim_rejected,
    trace_replication_check,
)
from open_deep_research.computational.prompts import (
    discovery_clarification_prompt,
    discovery_supervisor_prompt,
    experiment_design_prompt,
    final_report_synthesis_prompt,
    hypothesis_generation_prompt,
    iteration_decision_prompt,
    knowledge_gathering_prompt,
    research_brief_generation_prompt,
    result_analysis_prompt,
)
from open_deep_research.computational.scientific_tools import (
    get_scientific_tools,
    search_arxiv_papers,
)
from open_deep_research.computational.state import (
    ComputationalDiscoveryState,
    ComputationalOutput,
    ComputationResult,
    ExperimentAnalysis,
    ExperimentDesign,
    ExperimentPlan,
    ExperimentRecord,
    ExperimentStatus,
    ClaimValidationRecord,
    ClaimStatus,
    ReplicationStatus,
    Finding,
    GeneratedHypothesis,
    HypothesisRecord,
    HypothesisStatus,
    IterationDecision,
    OutputType,
    PaperData,
    ScientificDataSource,
    DataSourceType,
)
from open_deep_research.computational.configuration import (
    ComputationalConfiguration,
    ScientificDomain,
    get_domain_prompt_context,
)
from open_deep_research.utils import (
    get_model_runtime_config,
    get_today_str,
    think_tool,
)

logger = logging.getLogger(__name__)

# Initialize configurable model
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_supervisor_model_config(configurable: ComputationalConfiguration, config: RunnableConfig) -> Dict[str, Any]:
    """Get supervisor model configuration with fallback to research model."""
    model_name = configurable.supervisor_model or configurable.research_model
    max_tokens = configurable.supervisor_model_max_tokens or configurable.research_model_max_tokens
    return get_model_runtime_config(
        model_name,
        config,
        max_tokens=max_tokens,
        tags=["langsmith:nostream"],
    )


def get_worker_model_config(configurable: ComputationalConfiguration, config: RunnableConfig) -> Dict[str, Any]:
    """Get worker model configuration with fallback to research model."""
    model_name = configurable.worker_model or configurable.research_model
    max_tokens = configurable.worker_model_max_tokens or configurable.research_model_max_tokens
    return get_model_runtime_config(
        model_name,
        config,
        max_tokens=max_tokens,
        tags=["langsmith:nostream"],
    )


def escape_format_braces(text: str) -> str:
    """Escape curly braces in text to prevent KeyErrors during string formatting.
    
    This is critical because dynamic content (e.g., experiment objectives, findings)
    may contain patterns like {T} or {value} which would cause KeyError when
    the content is formatted into prompts.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


# Default timeout for model invocations (seconds)
MODEL_INVOKE_TIMEOUT = int(os.getenv("MODEL_INVOKE_TIMEOUT", "180"))


async def invoke_model_with_timeout(model, messages, timeout: int = None, label: str = "model"):
    """Invoke a model with a timeout to prevent indefinite hangs.
    
    Args:
        model: The LangChain model to invoke
        messages: Messages to send
        timeout: Timeout in seconds (defaults to MODEL_INVOKE_TIMEOUT)
        label: Human-readable label for logging
    
    Returns:
        The model response
    
    Raises:
        asyncio.TimeoutError: If the model doesn't respond in time
    """
    timeout = timeout or MODEL_INVOKE_TIMEOUT
    try:
        return await asyncio.wait_for(
            model.ainvoke(messages),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Model invocation timed out after {timeout}s for: {label}")
        raise TimeoutError(
            f"Model '{label}' did not respond within {timeout} seconds. "
            f"This may indicate the model is too slow for this role. "
            f"Consider using a faster model (e.g., deepseek-chat instead of deepseek-reasoner) "
            f"or increasing MODEL_INVOKE_TIMEOUT."
        )


def summarize_papers(papers: List[Any]) -> str:
    """Create a summary of gathered papers."""
    if not papers:
        return "No papers gathered yet."
    
    summaries = []
    for paper in papers[:10]:  # Limit to 10
        title = escape_format_braces(str(paper.title)) if paper.title else "Unknown"
        arxiv_id = escape_format_braces(str(paper.arxiv_id)) if paper.arxiv_id else "N/A"
        summaries.append(f"- {title} (ArXiv: {arxiv_id})")
        if paper.abstract:
            abstract = escape_format_braces(paper.abstract[:200])
            summaries.append(f"  Abstract: {abstract}...")
    
    return "\n".join(summaries)


def summarize_data_sources(data_sources: List[Any]) -> str:
    """Create a summary of data sources."""
    if not data_sources:
        return "No external data sources accessed yet."
    
    summaries = []
    for source in data_sources:
        name = escape_format_braces(str(source.name)) if source.name else "Unknown"
        source_type = escape_format_braces(str(source.source_type.value)) if source.source_type else "N/A"
        summaries.append(f"- {name} ({source_type})")
        if source.metadata:
            count = source.metadata.get('count', 'N/A')
            summaries.append(f"  Records: {count}")
    
    return "\n".join(summaries)


def summarize_hypotheses(hypotheses: List[HypothesisRecord]) -> str:
    """Create a summary of tested hypotheses."""
    if not hypotheses:
        return "No hypotheses tested yet."
    
    summaries = []
    for h in hypotheses:
        status_emoji = {
            HypothesisStatus.PROPOSED: "📋",
            HypothesisStatus.TESTING: "🔬",
            HypothesisStatus.SUPPORTED: "✅",
            HypothesisStatus.REFUTED: "❌",
            HypothesisStatus.INCONCLUSIVE: "❓",
            HypothesisStatus.REFINED: "🔄",
        }
        statement = escape_format_braces(str(h.statement)) if h.statement else "Unknown"
        status_val = escape_format_braces(str(h.status.value)) if h.status else "N/A"
        summaries.append(f"{status_emoji.get(h.status, '•')} {statement} [{status_val}]")
    
    return "\n".join(summaries)


def summarize_findings(findings: List[Finding]) -> str:
    """Create a summary of findings."""
    if not findings:
        return "No findings recorded yet."
    
    summaries = []
    for f in findings:
        novelty = "🆕" if f.is_novel else ""
        summaries.append(f"{novelty} {f.statement}")
        if f.statistical_evidence:
            summaries.append(f"   Evidence: {f.statistical_evidence}")
    
    return "\n".join(summaries)


def _elapsed_seconds(start: float) -> float:
    """Return elapsed seconds from a perf_counter start mark."""
    return max(0.0, perf_counter() - start)


def summarize_claim_ledger(claim_ledger: List[ClaimValidationRecord]) -> str:
    """Create a concise summary of validated/provisional/rejected claims."""
    if not claim_ledger:
        return "No claim validation records available."

    sections = []
    status_groups = {
        ClaimStatus.VALIDATED: [],
        ClaimStatus.PROVISIONAL: [],
        ClaimStatus.INCONCLUSIVE: [],
        ClaimStatus.REJECTED: [],
    }
    for claim in claim_ledger:
        status_groups.get(claim.status, status_groups[ClaimStatus.PROVISIONAL]).append(claim)

    for status in (
        ClaimStatus.VALIDATED,
        ClaimStatus.PROVISIONAL,
        ClaimStatus.INCONCLUSIVE,
        ClaimStatus.REJECTED,
    ):
        claims = status_groups.get(status, [])
        if not claims:
            continue
        sections.append(f"### Claims [{status.value}] ({len(claims)})")
        for claim in claims:
            sections.append(
                f"- {claim.claim_text[:180]} "
                f"(novelty_confidence={claim.novelty_confidence:.2f}, "
                f"replication={claim.replication_status.value})"
            )
            if claim.verdict_reason:
                sections.append(f"  Reason: {claim.verdict_reason[:220]}")
        sections.append("")
    return "\n".join(sections)


def _extract_primary_source_urls(state: ComputationalDiscoveryState, limit: int = 6) -> List[str]:
    """Extract representative URLs from knowledge/data source state for claim provenance."""
    urls: List[str] = []
    for paper in state.get("papers", []):
        if getattr(paper, "url", None):
            urls.append(str(paper.url))
    for source in state.get("data_sources", []):
        candidate = getattr(source, "url", None)
        if candidate:
            urls.append(str(candidate))
    deduped = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped[:limit]


def _estimate_novelty_confidence(
    confidence_level: str,
    has_stats: bool,
    has_contradictions: bool,
    simulation_penalty: bool,
) -> float:
    """Deterministic confidence estimator for claim novelty confidence."""
    base = {"low": 0.35, "medium": 0.6, "high": 0.8}.get((confidence_level or "").lower(), 0.4)
    if has_stats:
        base += 0.1
    if has_contradictions:
        base -= 0.25
    if simulation_penalty:
        base -= 0.2
    return max(0.0, min(1.0, base))


def _build_replication_code(original_code: str) -> str:
    """Inject a reproducibility preamble and perturb deterministic seeds."""
    patched = re.sub(r"np\.random\.seed\(\s*\d+\s*\)", "np.random.seed(1337)", original_code)
    patched = re.sub(r"random\.seed\(\s*\d+\s*\)", "random.seed(1337)", patched)
    preamble = (
        "# --- Replication preamble (auto-generated) ---\n"
        "import random\n"
        "random.seed(1337)\n"
        "try:\n"
        "    import numpy as np\n"
        "    np.random.seed(1337)\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    return preamble + patched


def _extract_boolean_hypothesis_signal(text: str) -> Optional[bool]:
    """Infer support/refute signal from experiment output text if available."""
    if not text:
        return None
    upper = text.upper()
    if "HYPOTHESIS NOT SUPPORTED" in upper or "NOT SUPPORTED" in upper:
        return False
    if "HYPOTHESIS SUPPORTED" in upper or "SUPPORTED" in upper:
        return True
    return None


def summarize_experiments(experiments: List[ExperimentRecord]) -> str:
    """Create a summary of experiments."""
    if not experiments:
        return "No experiments conducted yet."
    
    summaries = []
    for exp in experiments:
        exp_id = escape_format_braces(str(exp.id)) if exp.id else "Unknown"
        objective = escape_format_braces(str(exp.design.objective)) if exp.design and exp.design.objective else "Unknown"
        summaries.append(f"- Experiment {exp_id}: {objective}")
        status = escape_format_braces(str(exp.status.value)) if exp.status else "N/A"
        summaries.append(f"  Status: {status}")
        if exp.findings_summary:
            findings = escape_format_braces(exp.findings_summary[:100])
            summaries.append(f"  Finding: {findings}...")
    
    return "\n".join(summaries)


def summarize_data_explorations(data_explorations: List[Dict[str, Any]]) -> str:
    """Create a summary of data exploration findings for experiment design context."""
    if not data_explorations:
        return "No data explorations performed yet. Consider using ExploreData first to discover database schemas."
    
    summaries = []
    for exp in data_explorations:
        data_source = escape_format_braces(str(exp.get("data_source", "Unknown")))
        goal = escape_format_braces(str(exp.get("goal", "")))
        success = exp.get("success", False)
        findings = escape_format_braces(str(exp.get("findings", "")[:2000]))
        
        status = "SUCCESS" if success else "FAILED"
        summaries.append(f"### Exploration: {data_source} [{status}]")
        summaries.append(f"Goal: {goal}")
        if findings:
            summaries.append(f"Findings:\n{findings}")
        summaries.append("")
    
    return "\n".join(summaries)


def build_progress_summary(state: ComputationalDiscoveryState) -> str:
    """Build a concise progress summary for the supervisor system prompt.
    
    This replaces the noisy accumulation of all previous messages with a
    clean, structured summary of what has been accomplished.
    """
    parts = []
    
    # Knowledge gathered
    knowledge_summary = state.get("knowledge_summary", "")
    if knowledge_summary:
        parts.append(f"**Knowledge Gathered:**\n{knowledge_summary[:1500]}")
    
    # Data explorations / discovery results
    data_explorations = state.get("data_explorations", [])
    if data_explorations:
        successful = [e for e in data_explorations if e.get("success")]
        failed = [e for e in data_explorations if not e.get("success")]
        parts.append(f"**Data Discovery Results:** {len(successful)} successful, {len(failed)} failed")
        for exp in successful[:3]:
            parts.append(f"  - [{exp.get('data_source', '?')}] {exp.get('goal', '')[:100]}")
            # Include key findings preview so supervisor knows what data is available
            stdout_preview = exp.get("stdout_preview", "")
            if stdout_preview:
                # Extract lines mentioning FOUND or AVAILABLE
                found_lines = [
                    line.strip() for line in stdout_preview.split("\n")
                    if any(kw in line.upper() for kw in ["FOUND", "AVAILABLE", "CATALOG", "SPECTRA"])
                ]
                if found_lines:
                    for line in found_lines[:3]:
                        parts.append(f"    → {line[:120]}")
    else:
        parts.append("**Data Discovery:** Not yet performed. Use ExploreData to search for available data!")
    
    # Hypotheses
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        parts.append(f"\n**Hypotheses ({len(hypotheses)}):**")
        parts.append(summarize_hypotheses(hypotheses))
    
    # Experiments
    experiments = state.get("experiments", [])
    if experiments:
        parts.append(f"\n**Experiments ({len(experiments)}):**")
        parts.append(summarize_experiments(experiments))
    
    # Key findings
    findings = state.get("findings", [])
    if findings:
        parts.append(f"\n**Key Findings ({len(findings)}):**")
        parts.append(summarize_findings(findings))
    
    # New questions
    new_questions = state.get("new_questions", [])
    if new_questions:
        parts.append(f"\n**Open Questions:**")
        for q in new_questions[-3:]:
            parts.append(f"  - {q}")
    
    return "\n".join(parts) if parts else "No progress yet. Start by gathering knowledge."


def create_outputs_catalogue(state: ComputationalDiscoveryState) -> str:
    """Create a catalogue of all computational outputs for report generation.
    
    Provides explicit embedding instructions for each figure to ensure
    they are included in the final report.
    """
    all_outputs = state.get("all_outputs", {})
    
    if not all_outputs:
        return "No computational outputs generated."
    
    catalogue_parts = [
        "=" * 70,
        "COMPUTATIONAL OUTPUTS - MUST BE INCLUDED IN REPORT",
        "=" * 70
    ]
    
    # Group by type
    images = []
    tables = []
    stats = []
    text_outputs = []
    
    for output_id, output in all_outputs.items():
        if output.output_type == OutputType.IMAGE:
            images.append(output)
        elif output.output_type == OutputType.TABLE:
            tables.append(output)
        elif output.output_type == OutputType.STATISTICAL_RESULT:
            stats.append(output)
        elif output.output_type == OutputType.TEXT:
            text_outputs.append(output)
    
    if images:
        catalogue_parts.append(f"\n### FIGURES TO EMBED ({len(images)} total) ###")
        catalogue_parts.append("Copy these EXACT markdown lines into your report:\n")
        
        for i, img in enumerate(images):
            label = escape_format_braces(img.citation_label or f"Figure {i+1}")
            img_format = img.image_format or 'png'
            desc = escape_format_braces(str(img.description)) if img.description else f"Computational result {i+1}"
            caption = escape_format_braces(str(img.caption)) if img.caption else desc
            
            # Provide the EXACT markdown to copy
            catalogue_parts.append(f"**{label}** (ID: {img.id})")
            catalogue_parts.append(f"COPY THIS LINE INTO REPORT:")
            catalogue_parts.append(f"![{caption}](output_{img.id}.{img_format})")
            catalogue_parts.append(f"Description: {desc}")
            
            # Include experiment source if available
            if img.source_experiment_id:
                catalogue_parts.append(f"From Experiment: {img.source_experiment_id}")
            
            if img.interpretation:
                # Include key points from vision analysis
                catalogue_parts.append(f"Key Observations:")
                for line in img.interpretation.split('\n')[:8]:  # First 8 lines
                    if line.strip():
                        escaped_line = escape_format_braces(line.strip())
                        catalogue_parts.append(f"  - {escaped_line}")
            catalogue_parts.append("")  # Blank line between figures
    
    if stats:
        catalogue_parts.append(f"\n### STATISTICAL RESULTS ({len(stats)} total) ###")
        catalogue_parts.append("Include these values in your Results section:\n")
        for i, stat in enumerate(stats):
            text = escape_format_braces(str(stat.text_content)) if stat.text_content else "N/A"
            # Truncate very long results
            if len(text) > 500:
                text = text[:500] + "..."
            catalogue_parts.append(f"Result {i+1}: {text}")
            catalogue_parts.append("")
    
    if text_outputs:
        catalogue_parts.append(f"\n### TEXT OUTPUTS ({len(text_outputs)} total) ###")
        catalogue_parts.append("Key computational outputs:\n")
        for i, txt in enumerate(text_outputs):
            content = escape_format_braces(str(txt.text_content)) if txt.text_content else "N/A"
            # Truncate very long content
            if len(content) > 1000:
                content = content[:1000] + "..."
            desc = escape_format_braces(str(txt.description)) if txt.description else f"Output {i+1}"
            catalogue_parts.append(f"**{desc}:**")
            catalogue_parts.append(f"{content}")
            catalogue_parts.append("")
    
    if tables:
        catalogue_parts.append(f"\n### DATA TABLES ({len(tables)} total) ###")
        for i, tbl in enumerate(tables):
            label = escape_format_braces(tbl.citation_label or f"Table {i+1}")
            catalogue_parts.append(f"**{label}** (ID: {tbl.id})")
            desc = escape_format_braces(str(tbl.description)) if tbl.description else "No description"
            catalogue_parts.append(f"Description: {desc}")
            # Include table data if available
            if tbl.table_data:
                try:
                    # Try to render as markdown table
                    if isinstance(tbl.table_data, list) and len(tbl.table_data) > 0:
                        headers = list(tbl.table_data[0].keys())
                        catalogue_parts.append("| " + " | ".join(str(h) for h in headers) + " |")
                        catalogue_parts.append("| " + " | ".join("---" for _ in headers) + " |")
                        for row in tbl.table_data[:10]:  # First 10 rows
                            catalogue_parts.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
                except Exception:
                    pass
            catalogue_parts.append("")
    
    catalogue_parts.append("=" * 70)
    catalogue_parts.append("END OF OUTPUTS - ALL FIGURES ABOVE MUST APPEAR IN YOUR REPORT")
    catalogue_parts.append("=" * 70)
    
    return "\n".join(catalogue_parts)


# =============================================================================
# Node Implementations
# =============================================================================

async def clarify_discovery_query(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["generate_research_brief", "__end__"]]:
    """Clarify the user's discovery query if needed.
    
    This node analyzes the user's research query and determines if
    clarification is needed before proceeding with the discovery process.
    """
    node_start = perf_counter()
    configurable = ComputationalConfiguration.from_runnable_config(config)
    trace_node_enter("clarify_discovery_query")
    trace_phase_start("clarify_discovery_query.total", node_name="clarify_discovery_query")
    
    # Check if clarification is allowed
    # Also check environment variable directly for robustness
    env_allow = os.getenv("ALLOW_CLARIFICATION", "").lower()
    if not configurable.allow_clarification or env_allow in ("false", "0", "no"):
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "clarify_discovery_query.total",
            total_duration,
            node_name="clarify_discovery_query",
            data={"skipped": True, "reason": "clarification_disabled"},
        )
        trace_node_exit(
            "clarify_discovery_query",
            success=True,
            data={"duration_seconds": total_duration, "skipped": True},
        )
        logger.info("Timing clarify_discovery_query total=%.2fs (skipped)", total_duration)
        return Command(goto="generate_research_brief")
    
    messages = state.get("messages", [])
    model_config = get_supervisor_model_config(configurable, config)
    
    prompt = discovery_clarification_prompt.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    
    # Try structured output first, fall back to manual parsing
    try:
        from open_deep_research.state import ClarifyWithUser
        
        clarification_model = (
            configurable_model
            .with_structured_output(ClarifyWithUser)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        
        model_start = perf_counter()
        response = await invoke_model_with_timeout(
            clarification_model, [HumanMessage(content=prompt)],
            timeout=MODEL_INVOKE_TIMEOUT, label="clarify_discovery_query"
        )
        model_duration = _elapsed_seconds(model_start)
        trace_performance_metric(
            "clarify_discovery_query.model_invoke",
            model_duration,
            node_name="clarify_discovery_query",
            data={"model": configurable.research_model},
        )
        
        if response.need_clarification:
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "clarify_discovery_query.total",
                total_duration,
                node_name="clarify_discovery_query",
                data={"need_clarification": True, "model_duration_seconds": model_duration},
            )
            trace_node_exit(
                "clarify_discovery_query",
                success=True,
                data={"duration_seconds": total_duration, "need_clarification": True},
            )
            return Command(
                goto=END,
                update={"messages": [AIMessage(content=response.question)]}
            )
        else:
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "clarify_discovery_query.total",
                total_duration,
                node_name="clarify_discovery_query",
                data={"need_clarification": False, "model_duration_seconds": model_duration},
            )
            trace_node_exit(
                "clarify_discovery_query",
                success=True,
                data={"duration_seconds": total_duration, "need_clarification": False},
            )
            logger.info(
                "Timing clarify_discovery_query total=%.2fs model=%.2fs",
                total_duration,
                model_duration,
            )
            return Command(
                goto="generate_research_brief",
                update={"messages": [AIMessage(content=response.verification)]}
            )
            
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a structured output compatibility issue
        if "response_format" in error_str or "unavailable" in error_str or "json" in error_str:
            logger.warning(f"Structured output not supported for clarification, using fallback: {e}")
            
            # Fallback: just proceed without clarification
            # Since the query is likely detailed enough if clarification is enabled
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "clarify_discovery_query.total",
                total_duration,
                node_name="clarify_discovery_query",
                data={"fallback_used": True, "fallback_reason": "structured_output_unsupported"},
            )
            trace_node_exit(
                "clarify_discovery_query",
                success=True,
                data={"duration_seconds": total_duration, "fallback_used": True},
            )
            return Command(goto="generate_research_brief")
        else:
            # Re-raise other errors
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "clarify_discovery_query.total",
                total_duration,
                success=False,
                node_name="clarify_discovery_query",
                data={"error": str(e)[:500]},
            )
            trace_node_exit(
                "clarify_discovery_query",
                success=False,
                data={"duration_seconds": total_duration, "error": str(e)[:500]},
            )
            raise


async def generate_research_brief(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor"]]:
    """Generate a comprehensive research brief from the user's query.
    
    This transforms the user's messages into a structured research brief
    that will guide the entire discovery process.
    """
    node_start = perf_counter()
    configurable = ComputationalConfiguration.from_runnable_config(config)
    messages = state.get("messages", [])
    model_config = get_supervisor_model_config(configurable, config)
    
    # Initialize trace if not already done
    trace_node_enter("generate_research_brief")
    trace_phase_start("generate_research_brief.total", node_name="generate_research_brief")
    
    research_query = get_buffer_string(messages)

    if TraceManager.get_trace() is None:
        TraceManager.start_trace(
            research_query=research_query,
            config={"configurable": dict(config.get("configurable", {}))} if config else {},
        )
    
    research_model = configurable_model.with_config(model_config)
    
    # Get domain context for the prompt
    domain = configurable.scientific_domain
    domain_context = get_domain_prompt_context(domain)
    
    prompt = research_brief_generation_prompt.format(
        messages=research_query,
        date=get_today_str(),
        domain_context=domain_context
    )
    
    model_start = perf_counter()
    response = await invoke_model_with_timeout(
        research_model, [HumanMessage(content=prompt)],
        timeout=MODEL_INVOKE_TIMEOUT, label="generate_research_brief"
    )
    model_duration = _elapsed_seconds(model_start)
    research_brief = response.content
    
    # Record the research brief generation
    TraceManager.add_event(
        event_type=TraceEventType.NODE_EXIT,
        title="Research Brief Generated",
        node_name="generate_research_brief",
        description=f"Generated research brief from user query",
        data={
            "research_query_length": len(research_query),
            "research_brief_length": len(research_brief),
            "model": configurable.research_model,
            "scientific_domain": domain.value if hasattr(domain, 'value') else str(domain),
            "model_duration_seconds": model_duration,
        },
        success=True
    )
    
    # Update trace with research context
    trace = TraceManager.get_trace()
    if trace:
        trace.research_query = research_query
        trace.research_brief = research_brief

    total_duration = _elapsed_seconds(node_start)
    trace_performance_metric(
        "generate_research_brief.model_invoke",
        model_duration,
        node_name="generate_research_brief",
        data={"model": configurable.research_model},
    )
    trace_phase_end(
        "generate_research_brief.total",
        total_duration,
        node_name="generate_research_brief",
        data={
            "model_duration_seconds": model_duration,
            "research_query_length": len(research_query),
            "research_brief_length": len(research_brief),
        },
    )
    trace_node_exit(
        "generate_research_brief",
        success=True,
        data={
            "duration_seconds": total_duration,
            "model_duration_seconds": model_duration,
            "research_brief_length": len(research_brief),
        },
    )
    logger.info(
        "Timing generate_research_brief total=%.2fs model=%.2fs",
        total_duration,
        model_duration,
    )
    
    # Generate trace ID for this run
    trace_id = str(uuid.uuid4())[:12]
    
    # NOTE: We do NOT pre-build the supervisor system prompt here.
    # The supervisor node rebuilds it fresh each iteration with current state.
    # We only pass the research brief and initial context.
    
    return Command(
        goto="discovery_supervisor",
        update={
            "research_brief": research_brief,
            "research_query": research_query,
            "scientific_domain": domain.value if hasattr(domain, 'value') else str(domain),
            "trace_id": trace_id,
            "trace_started_at": datetime.now().isoformat(),
            "trace_events": [{
                "event_type": "research_brief_generated",
                "timestamp": datetime.now().isoformat(),
                "title": "Research Brief Generated",
                "data": {
                    "research_query": research_query[:1000],
                    "research_brief": research_brief[:2000],
                    "model": configurable.research_model,
                    "node_duration_seconds": total_duration,
                    "model_duration_seconds": model_duration,
                }
            }],
            # Initialize supervisor_messages with just the research brief
            "supervisor_messages": {
                "type": "override",
                "value": [
                    HumanMessage(content=f"Research Brief:\n{research_brief}")
                ]
            }
        }
    )


async def discovery_supervisor(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """Main discovery supervisor that orchestrates the research process.
    
    KEY DESIGN: The system prompt is rebuilt fresh each iteration with:
    - The research brief (constant)
    - A compressed progress summary (replaces noisy message history)
    - Domain-specific context
    - Only recent supervisor messages (last 6) to maintain conversation flow
    
    This prevents context pollution from accumulated tool messages and
    ensures the supervisor always has clean, relevant context.
    """
    node_start = perf_counter()
    iteration = state.get("discovery_iterations", 0)
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_supervisor_model_config(configurable, config)
    trace_node_enter("discovery_supervisor", iteration=iteration)
    trace_phase_start(
        "discovery_supervisor.total",
        node_name="discovery_supervisor",
        iteration=iteration,
    )
    
    # Define supervisor tools
    from pydantic import BaseModel, Field
    
    class GatherKnowledge(BaseModel):
        """Gather scientific knowledge from papers and databases."""
        knowledge_request: str = Field(
            description="What specific knowledge to gather"
        )
    
    class RunExperiment(BaseModel):
        """Design and run a computational experiment. BOTH arguments are REQUIRED."""
        hypothesis: str = Field(
            description=(
                "REQUIRED: A testable scientific statement that can be true or false. "
                "Example: 'Sulfur-based metabolism via H2S → S8 is thermodynamically favorable (ΔG < 0) "
                "in super-Earth atmospheres at 300-600K and 1-100 bar pressure.'"
            )
        )
        experiment_description: str = Field(
            description=(
                "REQUIRED: Detailed description of the Python code to execute. "
                "Example: 'Calculate Gibbs free energy for H2S → S8 reaction across T=200-500K, "
                "P=0.01-100 bar. Use scipy for integration, matplotlib for phase diagram.'"
            )
        )
    
    class SynthesizeFindings(BaseModel):
        """Signal that discovery is complete and synthesize findings."""
        reason: str = Field(
            description="Why discovery is being concluded"
        )
    
    class ExploreData(BaseModel):
        """Explore database schemas and APIs BEFORE running experiments. 
        Use this to discover correct column names, data types, and query syntax."""
        data_source: str = Field(
            description=(
                "The data source to explore. Examples: 'NASA Exoplanet Archive', "
                "'SIMBAD', 'VizieR/Gaia DR3', 'MAST'"
            )
        )
        exploration_goal: str = Field(
            description=(
                "What you want to discover. Example: 'Discover column names for "
                "exoplanet radius, mass, equilibrium temperature, and discovery method'"
            )
        )
    
    # Include contextual_retrieve so the discovery supervisor can search the
    # web directly when experiments yield unexpected results or when it needs
    # to verify a hypothesis against existing literature mid-discovery.
    try:
        from open_deep_research.retrieval import contextual_retrieve as _cr_tool
        supervisor_tools = [GatherKnowledge, ExploreData, RunExperiment, SynthesizeFindings, think_tool, _cr_tool]
    except ImportError:
        supervisor_tools = [GatherKnowledge, ExploreData, RunExperiment, SynthesizeFindings, think_tool]
    
    supervisor_model = (
        configurable_model
        .bind_tools(supervisor_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # =========================================================================
    # PRIORITY 1: Rebuild system prompt fresh with compressed state
    # =========================================================================
    research_brief = state.get("research_brief", "")
    domain_str = state.get("scientific_domain", "general")
    
    # Map string domain to enum for context lookup
    try:
        domain_enum = ScientificDomain(domain_str)
    except (ValueError, KeyError):
        domain_enum = ScientificDomain.GENERAL
    
    domain_context = get_domain_prompt_context(domain_enum)
    
    # Inject data discovery guidance for astronomy domain
    if domain_str.lower() in ["astronomy", "astro", "exoplanet", "stellar", "general"]:
        from open_deep_research.computational.context import get_data_discovery_guidance
        domain_context += "\n" + get_data_discovery_guidance()
    
    prompt_build_start = perf_counter()
    progress_summary = build_progress_summary(state)
    
    # Build fresh system prompt with current state
    # Escape braces in dynamic content to prevent KeyError during format()
    system_prompt = discovery_supervisor_prompt.format(
        date=get_today_str(),
        research_brief=escape_format_braces(research_brief),
        domain_context=escape_format_braces(domain_context),
        iteration=state.get('discovery_iterations', 0),
        max_iterations=configurable.max_discovery_iterations,
        hypotheses_count=len(state.get('hypotheses', [])),
        experiments_count=len(state.get('experiments', [])),
        findings_count=len(state.get('findings', [])),
        progress_summary=escape_format_braces(progress_summary)
    )
    
    # Keep only the last N supervisor messages for conversation continuity
    # This prevents context window pollution from accumulated tool messages
    supervisor_messages = state.get("supervisor_messages", [])
    MAX_RECENT_MESSAGES = 6
    recent_messages = supervisor_messages[-MAX_RECENT_MESSAGES:] if len(supervisor_messages) > MAX_RECENT_MESSAGES else supervisor_messages
    
    # =========================================================================
    # SANITIZE MESSAGE HISTORY: Ensure valid tool_calls / ToolMessage pairing
    # =========================================================================
    # Some APIs (DeepSeek, etc.) require that every ToolMessage is preceded by
    # an AIMessage with tool_calls. When we truncate the history, we can break
    # this pairing. Fix by:
    # 1. Finding the first valid start position (not a ToolMessage without parent)
    # 2. Removing orphaned ToolMessages
    sanitized = []
    has_pending_tool_calls = False
    for msg in recent_messages:
        if isinstance(msg, ToolMessage):
            if has_pending_tool_calls:
                sanitized.append(msg)
            else:
                # Orphaned ToolMessage -- skip it
                logger.debug("Skipping orphaned ToolMessage in supervisor history")
                continue
        elif isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            sanitized.append(msg)
            has_pending_tool_calls = True
        else:
            sanitized.append(msg)
            has_pending_tool_calls = False
    
    recent_messages = sanitized
    
    # Build the message list: fresh system prompt + recent conversation
    messages_for_model = [SystemMessage(content=system_prompt)] + recent_messages
    prompt_build_duration = _elapsed_seconds(prompt_build_start)
    
    # Invoke supervisor with timeout protection
    model_start = perf_counter()
    response = await invoke_model_with_timeout(
        supervisor_model, messages_for_model,
        timeout=MODEL_INVOKE_TIMEOUT, label="discovery_supervisor"
    )
    model_duration = _elapsed_seconds(model_start)
    total_duration = _elapsed_seconds(node_start)
    trace_performance_metric(
        "discovery_supervisor.prompt_build",
        prompt_build_duration,
        node_name="discovery_supervisor",
        iteration=iteration,
        data={"recent_messages_count": len(recent_messages)},
    )
    trace_performance_metric(
        "discovery_supervisor.model_invoke",
        model_duration,
        node_name="discovery_supervisor",
        iteration=iteration,
        data={"model": configurable.supervisor_model or configurable.research_model},
    )
    trace_phase_end(
        "discovery_supervisor.total",
        total_duration,
        node_name="discovery_supervisor",
        iteration=iteration,
        data={
            "prompt_build_duration_seconds": prompt_build_duration,
            "model_duration_seconds": model_duration,
            "message_count": len(messages_for_model),
        },
    )
    trace_node_exit(
        "discovery_supervisor",
        iteration=iteration,
        success=True,
        data={
            "duration_seconds": total_duration,
            "model_duration_seconds": model_duration,
            "tool_calls_count": len(getattr(response, "tool_calls", []) or []),
        },
    )
    logger.info(
        "Timing discovery_supervisor iteration=%s total=%.2fs prompt=%.2fs model=%.2fs",
        iteration,
        total_duration,
        prompt_build_duration,
        model_duration,
    )
    
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "discovery_iterations": state.get("discovery_iterations", 0) + 1
        }
    )


async def supervisor_tools(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor", "gather_knowledge", "explore_data", "run_experiment", "synthesize_findings", "__end__"]]:
    """Execute tools called by the discovery supervisor."""
    node_start = perf_counter()
    configurable = ComputationalConfiguration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    discovery_iterations = state.get("discovery_iterations", 0)
    trace_node_enter("supervisor_tools", iteration=discovery_iterations)
    trace_phase_start(
        "supervisor_tools.total",
        node_name="supervisor_tools",
        iteration=discovery_iterations,
    )
    
    most_recent_message = supervisor_messages[-1]
    
    # Check exit conditions
    exceeded_iterations = discovery_iterations > configurable.max_discovery_iterations
    no_tool_calls = not hasattr(most_recent_message, 'tool_calls') or not most_recent_message.tool_calls
    
    if exceeded_iterations or no_tool_calls:
        # Record exit decision
        exit_reason = "max_iterations_exceeded" if exceeded_iterations else "no_tool_calls"
        TraceManager.add_event(
            event_type=TraceEventType.SUPERVISOR_DECISION,
            title=f"Supervisor Exit: {exit_reason}",
            node_name="supervisor_tools",
            iteration=discovery_iterations,
            data={"reason": exit_reason, "exceeded_iterations": exceeded_iterations},
            success=True
        )
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "supervisor_tools.total",
            total_duration,
            node_name="supervisor_tools",
            iteration=discovery_iterations,
            data={"exit_reason": exit_reason},
        )
        trace_node_exit(
            "supervisor_tools",
            iteration=discovery_iterations,
            success=True,
            data={"duration_seconds": total_duration, "exit_reason": exit_reason},
        )
        return Command(goto="synthesize_findings")
    
    # Generate ToolMessage responses for ALL tool calls to satisfy API requirements
    all_tool_messages = []
    primary_action = None
    primary_update = {}
    
    tool_timing_details: List[Dict[str, Any]] = []
    for tool_call in most_recent_message.tool_calls:
        tool_start = perf_counter()
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if tool_name == "GatherKnowledge":
            all_tool_messages.append(ToolMessage(
                content="Proceeding to gather knowledge...",
                name=tool_name,
                tool_call_id=tool_id
            ))
            if primary_action is None:
                primary_action = "gather_knowledge"
                primary_update["_knowledge_request"] = tool_args.get("knowledge_request", "")
        
        elif tool_name == "ExploreData":
            data_source = tool_args.get("data_source", "").strip()
            exploration_goal = tool_args.get("exploration_goal", "").strip()
            
            all_tool_messages.append(ToolMessage(
                content=f"Proceeding to explore data source: {data_source}...",
                name=tool_name,
                tool_call_id=tool_id
            ))
            if primary_action is None:
                primary_action = "explore_data"
                primary_update["_exploration_data_source"] = data_source
                primary_update["_exploration_goal"] = exploration_goal
        
        elif tool_name == "RunExperiment":
            hypothesis = tool_args.get("hypothesis", "").strip()
            experiment_desc = tool_args.get("experiment_description", "").strip()
            
            # Validate hypothesis is provided and meaningful
            if not hypothesis or len(hypothesis) < 20:
                all_tool_messages.append(ToolMessage(
                    content=(
                        "ERROR: Invalid hypothesis provided. The 'hypothesis' argument must be a "
                        "testable scientific statement (at least 20 characters). "
                        "Example: 'Sulfur-based metabolism via H2S → S8 is thermodynamically "
                        "favorable (ΔG < 0) at 300-600K and 1-100 bar pressure.' "
                        "Please call RunExperiment again with a proper hypothesis."
                    ),
                    name=tool_name,
                    tool_call_id=tool_id
                ))
                # Don't set primary_action so supervisor continues
                continue
            
            if not experiment_desc or len(experiment_desc) < 30:
                all_tool_messages.append(ToolMessage(
                    content=(
                        "ERROR: Invalid experiment_description provided. Must describe the code "
                        "to execute (at least 30 characters). "
                        "Example: 'Calculate Gibbs free energy for H2S → S8 reaction across "
                        "T=200-500K, P=0.01-100 bar. Generate phase diagram with matplotlib.' "
                        "Please call RunExperiment again with a proper description."
                    ),
                    name=tool_name,
                    tool_call_id=tool_id
                ))
                continue
            
            all_tool_messages.append(ToolMessage(
                content=f"Proceeding to run experiment: {hypothesis[:100]}...",
                name=tool_name,
                tool_call_id=tool_id
            ))
            if primary_action is None:
                primary_action = "run_experiment"
                primary_update["_experiment_hypothesis"] = hypothesis
                primary_update["_experiment_description"] = experiment_desc
        
        elif tool_name == "SynthesizeFindings":
            all_tool_messages.append(ToolMessage(
                content=f"Proceeding to synthesize findings: {tool_args.get('reason', '')}",
                name=tool_name,
                tool_call_id=tool_id
            ))
            if primary_action is None:
                primary_action = "synthesize_findings"
        
        elif tool_name == "think_tool":
            reflection = tool_args.get("reflection", "")
            all_tool_messages.append(ToolMessage(
                content=f"Reflection recorded: {reflection}",
                name="think_tool",
                tool_call_id=tool_id
            ))
        
        elif tool_name == "contextual_retrieve":
            # Execute contextual_retrieve directly so the supervisor can
            # search the web mid-discovery when unexpected results appear.
            try:
                from open_deep_research.retrieval import contextual_retrieve as _cr
                result = await _cr.ainvoke(tool_args, config)
                all_tool_messages.append(ToolMessage(
                    content=str(result),
                    name="contextual_retrieve",
                    tool_call_id=tool_id
                ))
            except Exception as e:
                all_tool_messages.append(ToolMessage(
                    content=f"contextual_retrieve error: {e}",
                    name="contextual_retrieve",
                    tool_call_id=tool_id
                ))
        
        else:
            # Unknown tool - still respond to avoid API errors
            all_tool_messages.append(ToolMessage(
                content=f"Tool {tool_name} acknowledged.",
                name=tool_name,
                tool_call_id=tool_id
            ))

        tool_duration = _elapsed_seconds(tool_start)
        tool_timing_details.append(
            {
                "tool": tool_name,
                "duration_seconds": tool_duration,
                "has_error": any("error" in str(msg.content).lower() for msg in all_tool_messages[-1:]),
            }
        )
        trace_performance_metric(
            f"supervisor_tools.tool.{tool_name}",
            tool_duration,
            node_name="supervisor_tools",
            iteration=discovery_iterations,
            data={"tool_name": tool_name, "tool_call_id": tool_id},
        )
    
    # ==========================================================================
    # TRACEABILITY: Record the supervisor decision
    # ==========================================================================
    # Extract reasoning from AI message if available
    reasoning = ""
    if hasattr(most_recent_message, 'content') and most_recent_message.content:
        reasoning = str(most_recent_message.content)[:500]
    
    # Build tool call summary for trace
    tool_call_summary = []
    for tc in most_recent_message.tool_calls:
        tool_call_summary.append({
            "name": tc.get("name"),
            "args": {k: str(v)[:200] for k, v in tc.get("args", {}).items()}
        })
    
    # Create supervisor decision trace
    decision_trace = {
        "timestamp": datetime.now().isoformat(),
        "iteration": discovery_iterations,
        "action_chosen": primary_action or "return_to_supervisor",
        "action_parameters": primary_update,
        "tool_calls": tool_call_summary,
        "reasoning": reasoning,
        "state_summary": {
            "hypotheses_count": len(state.get("hypotheses", [])),
            "experiments_count": len(state.get("experiments", [])),
            "findings_count": len(state.get("findings", [])),
        },
        "timing": {
            "tools": tool_timing_details,
        },
    }
    
    # Also emit to TraceManager
    trace_supervisor_decision(
        iteration=discovery_iterations,
        action=primary_action or "return_to_supervisor",
        action_params=primary_update,
        state_summary=f"Hypotheses: {len(state.get('hypotheses', []))}, Experiments: {len(state.get('experiments', []))}, Findings: {len(state.get('findings', []))}",
        reasoning=reasoning,
        tool_calls=tool_call_summary,
        hypotheses_count=len(state.get("hypotheses", [])),
        experiments_count=len(state.get("experiments", [])),
        findings_count=len(state.get("findings", [])),
    )

    total_duration = _elapsed_seconds(node_start)
    trace_phase_end(
        "supervisor_tools.total",
        total_duration,
        node_name="supervisor_tools",
        iteration=discovery_iterations,
        data={
            "tool_calls_count": len(most_recent_message.tool_calls),
            "selected_action": primary_action or "return_to_supervisor",
            "tool_timings": tool_timing_details,
        },
    )
    trace_node_exit(
        "supervisor_tools",
        iteration=discovery_iterations,
        success=True,
        data={
            "duration_seconds": total_duration,
            "selected_action": primary_action or "return_to_supervisor",
        },
    )
    logger.info(
        "Timing supervisor_tools iteration=%s total=%.2fs tools=%s",
        discovery_iterations,
        total_duration,
        ", ".join([f"{t['tool']}:{t['duration_seconds']:.2f}s" for t in tool_timing_details]) or "none",
    )
    
    # Determine where to route
    if primary_action == "gather_knowledge":
        return Command(
            goto="gather_knowledge",
            update={
                "supervisor_messages": all_tool_messages,
                "supervisor_decision_traces": [decision_trace],
                **primary_update
            }
        )
    
    elif primary_action == "explore_data":
        return Command(
            goto="explore_data",
            update={
                "supervisor_messages": all_tool_messages,
                "supervisor_decision_traces": [decision_trace],
                **primary_update
            }
        )
    
    elif primary_action == "run_experiment":
        return Command(
            goto="run_experiment",
            update={
                "supervisor_messages": all_tool_messages,
                "supervisor_decision_traces": [decision_trace],
                **primary_update
            }
        )
    
    elif primary_action == "synthesize_findings":
        return Command(
            goto="synthesize_findings",
            update={
                "supervisor_messages": all_tool_messages,
                "supervisor_decision_traces": [decision_trace],
            }
        )
    
    # Default: return to supervisor (e.g., if only think_tool was called)
    return Command(
        goto="discovery_supervisor",
        update={
            "supervisor_messages": all_tool_messages,
            "supervisor_decision_traces": [decision_trace],
        }
    )


async def gather_knowledge(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor"]]:
    """Gather scientific knowledge from papers and databases.
    
    This node uses scientific tools to gather information needed
    for hypothesis generation and experiment design.
    
    KEY IMPROVEMENTS:
    - Populates papers and data_sources in state (was missing before)
    - Builds a cumulative knowledge_summary for context management
    - Extracts structured data from tool results
    """
    node_start = perf_counter()
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    iteration = state.get("discovery_iterations", 0)
    trace_node_enter("gather_knowledge", iteration=iteration)
    trace_phase_start("gather_knowledge.total", node_name="gather_knowledge", iteration=iteration)
    
    knowledge_request = state.get("_knowledge_request", state.get("research_brief", ""))
    
    # Get available tools
    scientific_tools = get_scientific_tools()
    tools = scientific_tools + [think_tool]
    
    knowledge_model = (
        configurable_model
        .bind_tools(tools)
        .with_config(model_config)
    )
    
    # Build prompt
    prompt = knowledge_gathering_prompt.format(
        date=get_today_str(),
        research_brief=state.get("research_brief", ""),
        knowledge_request=knowledge_request,
        papers_count=len(state.get("papers", [])),
        data_sources_count=len(state.get("data_sources", []))
    )
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Gather knowledge about: {knowledge_request}")
    ]
    
    # Run knowledge gathering loop (limited iterations)
    gathered_info = []
    new_papers = []
    new_data_sources = []
    max_iterations = 5
    loop_timing: List[Dict[str, Any]] = []
    
    for iter_num in range(max_iterations):
        iteration_start = perf_counter()
        model_iter_start = perf_counter()
        response = await invoke_model_with_timeout(
            knowledge_model, messages,
            timeout=MODEL_INVOKE_TIMEOUT, label=f"gather_knowledge_iter_{iter_num}"
        )
        model_iter_duration = _elapsed_seconds(model_iter_start)
        trace_performance_metric(
            "gather_knowledge.model_iteration",
            model_iter_duration,
            node_name="gather_knowledge",
            iteration=iteration,
            data={"loop_iteration": iter_num},
        )
        messages.append(response)
        
        if not response.tool_calls:
            loop_timing.append(
                {
                    "iteration": iter_num,
                    "model_duration_seconds": model_iter_duration,
                    "tool_calls_count": 0,
                    "iteration_duration_seconds": _elapsed_seconds(iteration_start),
                }
            )
            break
        
        # Execute tool calls
        iter_tools: List[Dict[str, Any]] = []
        for tool_call in response.tool_calls:
            tool = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool:
                try:
                    tool_start = perf_counter()
                    result = await tool.ainvoke(tool_call["args"], config)
                    tool_duration = _elapsed_seconds(tool_start)
                    result_str = str(result)
                    gathered_info.append({
                        "tool": tool_call["name"],
                        "args": tool_call["args"],
                        "result": result_str
                    })
                    messages.append(ToolMessage(
                        content=result_str[:5000],
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))
                    iter_tools.append(
                        {
                            "tool": tool_call["name"],
                            "duration_seconds": tool_duration,
                            "success": True,
                            "result_length": len(result_str),
                        }
                    )
                    trace_performance_metric(
                        f"gather_knowledge.tool.{tool_call['name']}",
                        tool_duration,
                        node_name="gather_knowledge",
                        iteration=iteration,
                        data={"loop_iteration": iter_num},
                    )
                    
                    # Extract structured data from tool results
                    tool_name = tool_call["name"]
                    
                    if tool_name == "search_arxiv_papers":
                        # Try to extract paper data from ArXiv results
                        try:
                            if isinstance(result, str) and "Title:" in result:
                                # Parse simple text results
                                for paper_block in result.split("\n\n"):
                                    title_match = re.search(r'Title:\s*(.+)', paper_block)
                                    arxiv_match = re.search(r'ArXiv ID:\s*(.+)', paper_block)
                                    abstract_match = re.search(r'Abstract:\s*(.+)', paper_block, re.DOTALL)
                                    if title_match:
                                        new_papers.append(PaperData(
                                            title=title_match.group(1).strip(),
                                            arxiv_id=arxiv_match.group(1).strip() if arxiv_match else None,
                                            abstract=abstract_match.group(1).strip()[:500] if abstract_match else None,
                                        ))
                        except Exception as parse_err:
                            logger.debug(f"Could not parse ArXiv results: {parse_err}")
                    
                    elif tool_name in ("query_nasa_exoplanet_archive", "query_mast_archive", "query_sdss_database"):
                        # Record data source access
                        source_map = {
                            "query_nasa_exoplanet_archive": ("NASA Exoplanet Archive", DataSourceType.DATABASE),
                            "query_mast_archive": ("NASA MAST Archive", DataSourceType.TELESCOPE_ARCHIVE),
                            "query_sdss_database": ("SDSS", DataSourceType.DATABASE),
                        }
                        name, src_type = source_map[tool_name]
                        new_data_sources.append(ScientificDataSource(
                            name=name,
                            source_type=src_type,
                            query_used=str(tool_call["args"])[:500],
                            metadata={"result_length": len(result_str)}
                        ))
                    
                except Exception as e:
                    tool_duration = _elapsed_seconds(tool_start)
                    messages.append(ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))
                    iter_tools.append(
                        {
                            "tool": tool_call["name"],
                            "duration_seconds": tool_duration,
                            "success": False,
                            "error": str(e)[:300],
                        }
                    )
                    trace_performance_metric(
                        f"gather_knowledge.tool.{tool_call['name']}",
                        tool_duration,
                        node_name="gather_knowledge",
                        iteration=iteration,
                        data={"loop_iteration": iter_num, "error": str(e)[:300]},
                    )
        loop_timing.append(
            {
                "iteration": iter_num,
                "model_duration_seconds": model_iter_duration,
                "tool_calls_count": len(response.tool_calls),
                "tool_timings": iter_tools,
                "iteration_duration_seconds": _elapsed_seconds(iteration_start),
            }
        )
    
    # Build knowledge summary
    raw_summary = "\n\n".join([
        f"[{info['tool']}]\n{info['result'][:2000]}"
        for info in gathered_info
    ])
    
    # Accumulate with previous knowledge_summary
    previous_summary = state.get("knowledge_summary", "")
    if previous_summary:
        updated_summary = f"{previous_summary}\n\n--- New Knowledge (Request: {knowledge_request[:100]}) ---\n{raw_summary[:3000]}"
    else:
        updated_summary = f"--- Knowledge (Request: {knowledge_request[:100]}) ---\n{raw_summary[:5000]}"
    
    # Limit total knowledge summary size
    if len(updated_summary) > 8000:
        updated_summary = updated_summary[:8000] + "\n...(truncated)"

    total_duration = _elapsed_seconds(node_start)
    trace_phase_end(
        "gather_knowledge.total",
        total_duration,
        node_name="gather_knowledge",
        iteration=iteration,
        data={
            "request": knowledge_request[:300],
            "iterations_run": len(loop_timing),
            "papers_added": len(new_papers),
            "data_sources_added": len(new_data_sources),
            "loop_timing": loop_timing,
        },
    )
    trace_node_exit(
        "gather_knowledge",
        iteration=iteration,
        success=True,
        data={
            "duration_seconds": total_duration,
            "papers_added": len(new_papers),
            "data_sources_added": len(new_data_sources),
        },
    )
    logger.info(
        "Timing gather_knowledge total=%.2fs iterations=%d papers=%d sources=%d",
        total_duration,
        len(loop_timing),
        len(new_papers),
        len(new_data_sources),
    )
    
    return Command(
        goto="discovery_supervisor",
        update={
            "supervisor_messages": [
                HumanMessage(content=f"Knowledge Gathered:\n{raw_summary[:10000]}")
            ],
            "papers": new_papers,
            "data_sources": new_data_sources,
            "knowledge_summary": updated_summary,
            # Clear the temp field
            "_knowledge_request": "",
        }
    )


async def explore_data(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor"]]:
    """Explore database schemas and APIs, and DISCOVER available data sources.
    
    This node is the core of the data discovery self-reflection system.
    It runs exploratory code to:
    - Search VizieR for relevant catalogs (millions of catalogs available!)
    - Discover database schemas and column names
    - Find MAST observations for specific targets
    - List available astroquery modules and specialized databases
    - Test data availability before the main experiment
    
    The findings are passed back to the supervisor to inform experiment design
    and prevent unnecessary simulation when real data is available.
    """
    from open_deep_research.computational.prompts import data_exploration_prompt
    
    node_start = perf_counter()
    configurable = ComputationalConfiguration.from_runnable_config(config)
    iteration = state.get("discovery_iterations", 0)
    trace_node_enter("explore_data", iteration=iteration)
    trace_phase_start("explore_data.total", node_name="explore_data", iteration=iteration)
    
    data_source = state.get("_exploration_data_source", "")
    exploration_goal = state.get("_exploration_goal", "")
    
    if not data_source:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "explore_data.total",
            total_duration,
            success=False,
            node_name="explore_data",
            iteration=iteration,
            data={"error": "missing_data_source"},
        )
        trace_node_exit(
            "explore_data",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "missing_data_source"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="ExploreData called without data_source. Please specify which database to explore.")
                ]
            }
        )
    
    logger.info(f"Exploring data source: {data_source}")
    logger.info(f"Exploration goal: {exploration_goal}")
    
    # Use CODE_FIXER_MODEL for generating exploration code (it's better at coding)
    code_fixer_model = os.getenv("CODE_FIXER_MODEL", configurable.research_model)
    
    # Build rich context including previous explorations and research brief
    previous_explorations_text = ""
    data_explorations = state.get("data_explorations", [])
    if data_explorations:
        previous_explorations_text = "\nPrevious exploration results:\n"
        for exp in data_explorations[-3:]:
            status = "SUCCESS" if exp.get("success") else "FAILED"
            previous_explorations_text += f"  - [{status}] {exp.get('data_source', '')}: {exp.get('goal', '')[:100]}\n"
            if exp.get("success") and exp.get("findings"):
                # Include key findings from previous explorations
                findings_preview = exp["findings"][:500]
                previous_explorations_text += f"    Key output: {findings_preview}\n"
    
    context = f"""
Research Brief: {state.get('research_brief', 'Not available')}

Scientific Domain: {state.get('scientific_domain', 'general')}

Previous exploration attempts: {len(data_explorations)}
{previous_explorations_text}

Data source to explore: {data_source}

IMPORTANT: Your goal is to DISCOVER data that might be useful for the research.
Search broadly across databases, especially VizieR (which hosts millions of catalogs).
Don't just check one source - cast a wide net to find all relevant data.
"""
    
    exploration_prompt = data_exploration_prompt.format(
        exploration_goal=exploration_goal,
        context=context
    )
    
    # Generate exploration code using the better coding model
    exploration_model = configurable_model.with_config(
        get_model_runtime_config(
            code_fixer_model,
            config,
            max_tokens=6000,
            tags=["langsmith:nostream"],
        )
    )
    
    code_gen_duration = 0.0
    try:
        code_gen_start = perf_counter()
        response = await invoke_model_with_timeout(
            exploration_model,
            [HumanMessage(content=exploration_prompt)],
            timeout=MODEL_INVOKE_TIMEOUT,
            label="explore_data_code_gen"
        )
        code_gen_duration = _elapsed_seconds(code_gen_start)
        trace_performance_metric(
            "explore_data.code_generation",
            code_gen_duration,
            node_name="explore_data",
            iteration=iteration,
            data={"model": code_fixer_model},
        )
        
        # Extract code from response
        code_content = response.content
        
        # Try to extract code from markdown blocks
        if "```python" in code_content:
            code_blocks = re.findall(r'```python\s*(.*?)```', code_content, re.DOTALL)
            if code_blocks:
                exploration_code = max(code_blocks, key=len).strip()
            else:
                exploration_code = code_content
        elif "```" in code_content:
            code_blocks = re.findall(r'```\s*(.*?)```', code_content, re.DOTALL)
            if code_blocks:
                exploration_code = max(code_blocks, key=len).strip()
            else:
                exploration_code = code_content
        else:
            exploration_code = code_content
        
        logger.info(f"Generated exploration code: {len(exploration_code)} chars")
        
    except Exception as e:
        logger.error(f"Failed to generate exploration code: {e}")
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "explore_data.total",
            total_duration,
            success=False,
            node_name="explore_data",
            iteration=iteration,
            data={"error": str(e)[:500]},
        )
        trace_node_exit(
            "explore_data",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": str(e)[:500]},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content=f"Failed to generate exploration code: {e}\n\nPlease try ExploreData again or proceed with RunExperiment.")
                ]
            }
        )
    
    # Execute exploration code in E2B with generous timeout for network queries
    from open_deep_research.computational.code_interpreter import execute_code
    
    execution_start = perf_counter()
    exploration_result = await execute_code(
        code=exploration_code,
        purpose=f"Data Discovery - {data_source}: {exploration_goal}",
        timeout=180,  # 3 minutes for broad discovery (VizieR searches can take time)
        config=config
    )
    execution_duration = _elapsed_seconds(execution_start)
    trace_performance_metric(
        "explore_data.code_execution",
        execution_duration,
        node_name="explore_data",
        iteration=iteration,
        data={"code_length": len(exploration_code), "success": exploration_result.success},
    )
    
    # Format the exploration findings with rich discovery-oriented summary
    if exploration_result.success:
        stdout_content = exploration_result.stdout[:10000] if exploration_result.stdout else "No output"
        
        # Detect if catalogs/data were actually found
        data_indicators = [
            "FOUND:", "AVAILABLE", "data_found", "catalogs found",
            "spectra found", "observations found", "columns"
        ]
        has_data = any(ind.lower() in stdout_content.lower() for ind in data_indicators)
        
        findings = f"""
## Data Discovery {'Successful - Data Found!' if has_data else 'Complete'}

**Data Source:** {data_source}
**Goal:** {exploration_goal}

### Discovery Output:
```
{stdout_content}
```

### Discovery Assessment:
{"REAL DATA was found! Use the catalogs and data sources identified above in your experiments." if has_data else "No directly matching data was found in this search. Consider: (1) broader VizieR keywords, (2) different database sources, (3) theoretical calculations with known physics as a fallback."}

**IMPORTANT: If catalogs were discovered above, USE THEM in your next experiment instead of simulating data!**
"""
        logger.info(f"Data exploration successful (data found: {has_data})")
    else:
        # Even failed explorations often contain useful partial output
        partial_stdout = exploration_result.stdout[:5000] if exploration_result.stdout else "No output"
        
        findings = f"""
## Data Exploration Encountered Errors

**Data Source:** {data_source}
**Goal:** {exploration_goal}

### Error:
```
{exploration_result.error_message or "Unknown error"}
```

### Partial Output (may still contain useful information):
```
{partial_stdout}
```

### Standard Error:
```
{exploration_result.stderr[:2000] if exploration_result.stderr else "No errors"}
```

### Recovery Suggestions:
1. Try ExploreData again with different keywords or a different data source
2. Try a simpler exploration (e.g., just search VizieR for one keyword)
3. If network issues, the data may still exist - try again
4. As last resort, proceed with RunExperiment using theoretical calculations (NOT simulated data)
"""
        logger.warning(f"Data exploration had errors: {exploration_result.error_message}")
    
    # Store exploration results
    data_explorations.append({
        "data_source": data_source,
        "goal": exploration_goal,
        "success": exploration_result.success,
        "findings": findings,
        "code": exploration_code[:3000],  # Store more code for reference
        "stdout_preview": (exploration_result.stdout or "")[:2000],
    })

    total_duration = _elapsed_seconds(node_start)
    trace_phase_end(
        "explore_data.total",
        total_duration,
        success=exploration_result.success,
        node_name="explore_data",
        iteration=iteration,
        data={
            "data_source": data_source,
            "goal": exploration_goal[:300],
            "code_generation_duration_seconds": code_gen_duration,
            "execution_duration_seconds": execution_duration,
            "code_length": len(exploration_code),
        },
    )
    trace_node_exit(
        "explore_data",
        iteration=iteration,
        success=exploration_result.success,
        data={
            "duration_seconds": total_duration,
            "code_generation_duration_seconds": code_gen_duration,
            "execution_duration_seconds": execution_duration,
        },
    )
    logger.info(
        "Timing explore_data total=%.2fs code_gen=%.2fs execution=%.2fs success=%s",
        total_duration,
        code_gen_duration,
        execution_duration,
        exploration_result.success,
    )
    
    return Command(
        goto="discovery_supervisor",
        update={
            "supervisor_messages": [
                HumanMessage(content=findings)
            ],
            "data_explorations": data_explorations,
            # Clear consumed temp fields
            "_exploration_data_source": "",
            "_exploration_goal": "",
        }
    )


async def run_experiment(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["analyze_results", "discovery_supervisor"]]:
    """Design and run a computational experiment.
    
    This node:
    1. Creates a hypothesis record
    2. Designs the experiment
    3. Executes the code in E2B
    4. Captures all outputs
    """
    node_start = perf_counter()
    iteration = state.get("discovery_iterations", 0)
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    trace_node_enter("run_experiment", iteration=iteration)
    trace_phase_start("run_experiment.total", node_name="run_experiment", iteration=iteration)
    
    hypothesis_text = state.get("_experiment_hypothesis", "").strip()
    experiment_description = state.get("_experiment_description", "").strip()
    
    # Validate hypothesis is provided
    if not hypothesis_text or len(hypothesis_text) < 10:
        logger.warning(f"Run experiment called without valid hypothesis: '{hypothesis_text}'")
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "run_experiment.total",
            total_duration,
            success=False,
            node_name="run_experiment",
            iteration=iteration,
            data={"error": "invalid_hypothesis", "hypothesis_length": len(hypothesis_text)},
        )
        trace_node_exit(
            "run_experiment",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "invalid_hypothesis"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content=(
                        "⚠️ EXPERIMENT FAILED: No valid hypothesis provided.\n\n"
                        "When calling RunExperiment, you MUST provide:\n"
                        "1. hypothesis: A testable scientific statement (e.g., 'Reaction X is "
                        "thermodynamically favorable at T=300-500K')\n"
                        "2. experiment_description: What code to run (e.g., 'Calculate ΔG for "
                        "reaction X across temperature range, generate phase diagram')\n\n"
                        "Please call RunExperiment again with BOTH arguments properly filled."
                    ))
                ],
                # Clear temp fields
                "_experiment_hypothesis": "",
                "_experiment_description": "",
            }
        )
    
    # Create hypothesis record
    hypothesis = HypothesisRecord(
        statement=hypothesis_text,
        rationale=experiment_description,
        status=HypothesisStatus.TESTING
    )
    
    # Build context for experiment design
    papers_summary = summarize_papers(state.get("papers", []))
    data_summary = summarize_data_sources(state.get("data_sources", []))
    prev_experiments = summarize_experiments(state.get("experiments", []))
    
    # Priority 3: Include data exploration findings in experiment design context
    data_explorations_summary = summarize_data_explorations(state.get("data_explorations", []))
    
    design_prompt = experiment_design_prompt.format(
        date=get_today_str(),
        hypothesis=hypothesis_text,
        available_data=f"Papers:\n{papers_summary}\n\nData Sources:\n{data_summary}",
        previous_experiments=prev_experiments,
        data_explorations=data_explorations_summary
    )
    
    # Try structured output first, fall back to manual parsing for models that don't support it
    experiment_plan = None
    design_duration = 0.0
    design_fallback_used = False
    try:
        design_start = perf_counter()
        experiment_model = (
            configurable_model
            .with_structured_output(ExperimentPlan)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        experiment_plan = await invoke_model_with_timeout(
            experiment_model,
            [HumanMessage(content=design_prompt)],
            timeout=MODEL_INVOKE_TIMEOUT,
            label="experiment_design"
        )
        design_duration = _elapsed_seconds(design_start)
    except Exception as e:
        design_duration = _elapsed_seconds(design_start)
        error_str = str(e).lower()
        # Check if it's a structured output compatibility issue
        if "response_format" in error_str or "unavailable" in error_str or "json" in error_str:
            logger.warning(f"Structured output not supported, using fallback parser: {e}")
            print(f"Structured output not supported, using fallback parser")
            design_fallback_used = True
            
            # Use fallback: ask model to output code in a very specific format
            fallback_prompt = f"""{design_prompt}

## RESPONSE FORMAT (FOLLOW EXACTLY!)

Since you cannot use structured JSON output, respond in this EXACT text format.
Put ONLY the Python code between the code markers. No other text inside the code block.

---BEGIN_RESPONSE---

OBJECTIVE: [One sentence describing what this experiment will determine]

---BEGIN_PYTHON_CODE---
# Your complete Python code here
import numpy as np
import pandas as pd
# ... rest of your code ...
# The code must be complete and syntactically valid
# DO NOT include any explanatory text inside this block
# ONLY Python code
---END_PYTHON_CODE---

EXPECTED_OUTPUTS: [figures, numerical results, statistical tests]

INTERPRETATION: [How to interpret the results]

---END_RESPONSE---

Remember: The code between ---BEGIN_PYTHON_CODE--- and ---END_PYTHON_CODE--- must be
pure Python with NO markdown, NO explanations, and MUST be syntactically complete.
"""
            try:
                fallback_start = perf_counter()
                fallback_response = await invoke_model_with_timeout(
                    configurable_model.with_config(model_config),
                    [HumanMessage(content=fallback_prompt)],
                    timeout=MODEL_INVOKE_TIMEOUT,
                    label="experiment_design_fallback"
                )
                fallback_duration = _elapsed_seconds(fallback_start)
                design_duration += fallback_duration
                trace_performance_metric(
                    "run_experiment.design_fallback_model_invoke",
                    fallback_duration,
                    node_name="run_experiment",
                    iteration=iteration,
                    data={"model": configurable.research_model},
                )
                # Parse the response manually
                content = fallback_response.content
                
                print(f"Fallback response received: {len(content)} chars")
                
                # =================================================================
                # IMPROVED CODE EXTRACTION with multiple fallback strategies
                # =================================================================
                python_code = ""
                
                # Strategy 1: Look for our custom markers
                if "---BEGIN_PYTHON_CODE---" in content and "---END_PYTHON_CODE---" in content:
                    code_start = content.find("---BEGIN_PYTHON_CODE---") + len("---BEGIN_PYTHON_CODE---")
                    code_end = content.find("---END_PYTHON_CODE---")
                    python_code = content[code_start:code_end].strip()
                    print("Extracted code using custom markers")
                
                # Strategy 2: Look for ```python ... ``` blocks
                elif "```python" in content:
                    # Find ALL python code blocks and take the longest one
                    code_blocks = re.findall(r'```python\s*(.*?)```', content, re.DOTALL)
                    if code_blocks:
                        python_code = max(code_blocks, key=len).strip()
                        print(f"Extracted code from markdown block ({len(code_blocks)} blocks found)")
                
                # Strategy 3: Look for any ``` ... ``` block that looks like Python
                elif "```" in content:
                    code_blocks = re.findall(r'```\s*(.*?)```', content, re.DOTALL)
                    for block in code_blocks:
                        block = block.strip()
                        # Check if it looks like Python (starts with import or has def/class)
                        if block.startswith(('import ', 'from ', '#', 'def ', 'class ', '"""')):
                            if len(block) > len(python_code):
                                python_code = block
                    if python_code:
                        print("Extracted code from generic markdown block")
                
                # Strategy 4: Look for PYTHON_CODE: marker
                elif "PYTHON_CODE:" in content:
                    code_start = content.find("PYTHON_CODE:") + len("PYTHON_CODE:")
                    code_end = len(content)
                    for marker in ["EXPECTED_OUTPUTS:", "INTERPRETATION:", "---END"]:
                        pos = content.find(marker, code_start)
                        if pos > 0 and pos < code_end:
                            code_end = pos
                    python_code = content[code_start:code_end].strip()
                    print("Extracted code using PYTHON_CODE marker")
                
                # Strategy 5: Just try to find Python-like content
                if not python_code or len(python_code) < 100:
                    # Look for content that starts with common Python patterns
                    matches = re.findall(
                        r'((?:import|from|#\s*=+|#!/usr/bin).*?)(?=\n\n[A-Z_]+:|$)',
                        content,
                        re.DOTALL
                    )
                    if matches:
                        longest = max(matches, key=len)
                        if len(longest) > len(python_code):
                            python_code = longest.strip()
                            print("Extracted code using pattern matching")
                
                # Clean up the code
                if python_code:
                    # Remove any leading language identifier
                    if python_code.startswith("python\n"):
                        python_code = python_code[7:]
                    elif python_code.startswith("python"):
                        python_code = python_code[6:]
                    
                    # Remove any trailing markers
                    for marker in ["---END", "EXPECTED_OUTPUTS:", "INTERPRETATION:"]:
                        if marker in python_code:
                            python_code = python_code[:python_code.find(marker)].strip()
                
                print(f"Final extracted code: {len(python_code)} chars, {python_code.count(chr(10))} lines")
                
                # Extract objective
                objective = "Test hypothesis computationally"
                if "OBJECTIVE:" in content:
                    obj_start = content.find("OBJECTIVE:") + len("OBJECTIVE:")
                    obj_end = content.find("\n", obj_start)
                    if obj_end > obj_start:
                        objective = content[obj_start:obj_end].strip()
                
                # Extract expected outputs
                expected_outputs = ["numerical results", "visualization"]
                if "EXPECTED_OUTPUTS:" in content:
                    out_start = content.find("EXPECTED_OUTPUTS:") + len("EXPECTED_OUTPUTS:")
                    out_end = content.find("\n\n", out_start)
                    if out_end < 0:
                        out_end = content.find("INTERPRETATION:", out_start)
                    if out_end > out_start:
                        outputs_text = content[out_start:out_end].strip()
                        expected_outputs = [o.strip() for o in outputs_text.split(",") if o.strip()]
                
                # Extract interpretation guide
                interpretation = "Compare results against hypothesis predictions"
                if "INTERPRETATION:" in content:
                    interp_start = content.find("INTERPRETATION:") + len("INTERPRETATION:")
                    interp_end = content.find("---END", interp_start)
                    if interp_end < 0:
                        interp_end = len(content)
                    interpretation = content[interp_start:interp_end].strip()[:500]
                
                if python_code and len(python_code) > 50:
                    experiment_plan = ExperimentPlan(
                        objective=objective,
                        python_code=python_code,
                        expected_outputs=expected_outputs,
                        interpretation_guide=interpretation
                    )
                    logger.info(f"Successfully parsed experiment plan from fallback (code: {len(python_code)} chars)")
                else:
                    raise ValueError(f"Failed to extract valid Python code from response (got {len(python_code)} chars)")
                    
            except Exception as parse_error:
                logger.error(f"Fallback parsing also failed: {parse_error}")
                total_duration = _elapsed_seconds(node_start)
                trace_phase_end(
                    "run_experiment.total",
                    total_duration,
                    success=False,
                    node_name="run_experiment",
                    iteration=iteration,
                    data={"error": str(parse_error)[:500], "phase": "design_fallback"},
                )
                trace_node_exit(
                    "run_experiment",
                    iteration=iteration,
                    success=False,
                    data={"duration_seconds": total_duration, "error": str(parse_error)[:500]},
                )
                return Command(
                    goto="discovery_supervisor",
                    update={
                        "supervisor_messages": [
                            HumanMessage(content=f"Experiment design failed (model may not support code generation): {str(parse_error)}")
                        ]
                    }
                )
        else:
            logger.error(f"Experiment design failed: {e}")
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "run_experiment.total",
                total_duration,
                success=False,
                node_name="run_experiment",
                iteration=iteration,
                data={"error": str(e)[:500], "phase": "design"},
            )
            trace_node_exit(
                "run_experiment",
                iteration=iteration,
                success=False,
                data={"duration_seconds": total_duration, "error": str(e)[:500]},
            )
            return Command(
                goto="discovery_supervisor",
                update={
                    "supervisor_messages": [
                        HumanMessage(content=f"Experiment design failed: {str(e)}")
                    ]
                }
            )
    
    if not experiment_plan:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "run_experiment.total",
            total_duration,
            success=False,
            node_name="run_experiment",
            iteration=iteration,
            data={"error": "empty_experiment_plan"},
        )
        trace_node_exit(
            "run_experiment",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "empty_experiment_plan"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="Experiment design failed: Could not generate experiment plan")
                ]
            }
        )

    trace_performance_metric(
        "run_experiment.design_generation_total",
        design_duration,
        node_name="run_experiment",
        iteration=iteration,
        data={"fallback_used": design_fallback_used},
    )
    
    # ==========================================================================
    # SIMULATION DETECTION: Check if code is generating synthetic data
    # ==========================================================================
    from open_deep_research.computational.context.astroquery_discovery import (
        check_code_for_simulation_patterns,
    )
    
    sim_check = check_code_for_simulation_patterns(experiment_plan.python_code)
    
    if sim_check["has_simulation"] and not sim_check["is_legitimate"]:
        # Code appears to be simulating data -- check if we already explored for real data
        data_explorations = state.get("data_explorations", [])
        successful_explorations = [e for e in data_explorations if e.get("success")]
        
        if not successful_explorations:
            # No successful data explorations yet -- warn the supervisor
            logger.warning(
                f"Simulation detected in experiment code without prior data discovery. "
                f"Patterns: {sim_check['simulation_patterns']}"
            )
            print(f"\n{'='*70}")
            print("SIMULATION DETECTION WARNING")
            print(f"{'='*70}")
            print(f"Detected simulation patterns: {sim_check['simulation_patterns']}")
            print(f"Recommendation: {sim_check['recommendation']}")
            print(f"{'='*70}\n")
            
            # Add warning to experiment description for the code fixer to see
            experiment_description += (
                f"\n\n⚠️ SIMULATION WARNING: The generated code appears to use synthetic data "
                f"(patterns: {', '.join(sim_check['simulation_patterns'][:3])}). "
                f"If possible, replace simulated data with real data from astroquery databases. "
                f"Search VizieR catalogs for relevant data before simulating."
            )
        else:
            logger.info(
                "Simulation detected but data explorations were done -- may be intentional."
            )
    
    # Create experiment record
    experiment = ExperimentRecord(
        hypothesis_id=hypothesis.id,
        design=ExperimentDesign(
            objective=experiment_plan.objective,
            methodology=experiment_plan.interpretation_guide,
            code_outline=experiment_plan.python_code[:500],
            expected_outputs=experiment_plan.expected_outputs
        ),
        status=ExperimentStatus.RUNNING,
        code_executed=experiment_plan.python_code
    )
    
    # ==========================================================================
    # TRACEABILITY: Record hypothesis and experiment start
    # ==========================================================================
    trace_hypothesis_created(
        hypothesis_id=hypothesis.id,
        statement=hypothesis_text,
        rationale=experiment_description
    )
    
    trace_experiment_started(
        experiment_id=experiment.id,
        hypothesis_id=hypothesis.id,
        objective=experiment_plan.objective
    )
    
    # Store experiment design for traceability
    experiment_trace_events = [{
        "event_type": "experiment_designed",
        "timestamp": datetime.now().isoformat(),
        "title": f"Experiment Designed: {experiment_plan.objective[:80]}",
        "experiment_id": experiment.id,
        "hypothesis_id": hypothesis.id,
        "data": {
            "objective": experiment_plan.objective,
            "expected_outputs": experiment_plan.expected_outputs,
            "interpretation_guide": experiment_plan.interpretation_guide[:500],
            "code_length": len(experiment_plan.python_code),
        }
    }]
    
    # Execute with retry loop for code fixing
    current_code = experiment_plan.python_code
    max_fix_attempts = int(os.getenv("MAX_CODE_FIX_ATTEMPTS", "3"))
    computation_result = None
    code_execution_traces = []  # Track all execution attempts
    execution_attempt_timings: List[Dict[str, Any]] = []
    fix_model_timings: List[Dict[str, Any]] = []
    
    # Per-experiment wall-clock timeout (prevents single experiments from running forever)
    experiment_wall_timeout = int(os.getenv("EXPERIMENT_WALL_TIMEOUT", "600"))  # 10 min default
    experiment_start_wall = datetime.now()
    
    # Priority 7: Use persistent sandbox if configured
    sandbox_id = state.get("sandbox_id") if configurable.use_persistent_sandbox else None
    
    for attempt in range(max_fix_attempts + 1):
        # Check wall-clock timeout
        elapsed_wall = (datetime.now() - experiment_start_wall).total_seconds()
        if elapsed_wall > experiment_wall_timeout:
            logger.warning(
                f"Experiment wall-clock timeout reached ({elapsed_wall:.0f}s > {experiment_wall_timeout}s). "
                f"Stopping after {attempt} attempts."
            )
            print(f"\n⏰ EXPERIMENT TIMEOUT: {elapsed_wall:.0f}s elapsed (limit: {experiment_wall_timeout}s)")
            break
        
        execution_start_time = datetime.now()
        execution_start_perf = perf_counter()
        
        # Execute the experiment in E2B with persistent sandbox
        computation_result = await execute_experiment(
            experiment_code=current_code,
            experiment_id=experiment.id,
            hypothesis=hypothesis_text,
            sandbox_id=sandbox_id,
            config=config
        )
        
        execution_time = (datetime.now() - execution_start_time).total_seconds()
        execution_time_perf = _elapsed_seconds(execution_start_perf)
        execution_attempt_timings.append(
            {
                "attempt": attempt + 1,
                "execution_time_seconds": execution_time,
                "execution_time_perf_seconds": execution_time_perf,
                "success": computation_result.success,
                "outputs_count": len(computation_result.outputs),
            }
        )
        
        # ==========================================================================
        # TRACEABILITY: Record code execution attempt
        # ==========================================================================
        code_exec_trace = {
            "timestamp": datetime.now().isoformat(),
            "attempt_number": attempt + 1,
            "code": current_code,
            "purpose": f"Test hypothesis: {hypothesis_text[:200]}",
            "success": computation_result.success,
            "execution_time_seconds": execution_time,
            "stdout": computation_result.stdout[:5000] if computation_result.stdout else "",
            "stderr": computation_result.stderr[:2000] if computation_result.stderr else "",
            "error_message": computation_result.error_message,
            "experiment_id": experiment.id,
            "hypothesis_id": hypothesis.id,
            "output_ids": [o.id for o in computation_result.outputs],
        }
        code_execution_traces.append(code_exec_trace)
        
        # Also emit to TraceManager
        trace_code_execution(
            code=current_code,
            purpose=f"Test hypothesis: {hypothesis_text[:200]}",
            success=computation_result.success,
            stdout=computation_result.stdout or "",
            stderr=computation_result.stderr or "",
            error_message=computation_result.error_message,
            execution_time=execution_time,
            attempt_number=attempt + 1,
            experiment_id=experiment.id,
            hypothesis_id=hypothesis.id,
            output_ids=[o.id for o in computation_result.outputs],
        )
        
        # If successful, break out of retry loop
        if computation_result.success:
            logger.info(f"Code execution succeeded on attempt {attempt + 1}")
            break
        
        # If failed and we have retries left, try to fix the code
        if attempt < max_fix_attempts:
            # ==========================================================================
            # VERBOSE ERROR REPORTING
            # ==========================================================================
            print(f"\n{'='*70}")
            print(f"CODE EXECUTION FAILED (attempt {attempt + 1}/{max_fix_attempts + 1})")
            print(f"{'='*70}")
            print(f"ERROR TYPE: {type(computation_result.error_message).__name__ if computation_result.error_message else 'Unknown'}")
            print(f"ERROR MESSAGE: {computation_result.error_message}")
            
            logger.warning(f"Code execution failed (attempt {attempt + 1}/{max_fix_attempts + 1})")
            logger.warning(f"Error: {computation_result.error_message}")
            
            # Get stdout/stderr for debugging context
            stdout_text = ""
            stderr_text = ""
            for output in computation_result.outputs:
                if output.output_type == OutputType.LOG and output.text_content:
                    if "stdout" in output.description.lower():
                        stdout_text = output.text_content
                    elif "stderr" in output.description.lower():
                        stderr_text = output.text_content
            
            if stdout_text:
                print(f"\nSTDOUT (last 500 chars):\n{stdout_text[-500:]}")
            if stderr_text:
                print(f"\nSTDERR:\n{stderr_text}")
            
            # Check code length - often truncation is the issue
            code_lines = current_code.count('\n')
            print(f"\nCODE INFO: {len(current_code)} chars, {code_lines} lines")
            if code_lines > 250:
                print("WARNING: Code is very long, may have been truncated!")
            print(f"{'='*70}\n")
            
            # Try to fix the code using AI
            try:
                from .prompts import code_fix_prompt
                
                # Include data exploration findings in the fix context
                exploration_context = ""
                data_explorations = state.get("data_explorations", [])
                if data_explorations:
                    successful_explorations = [e for e in data_explorations if e.get("success")]
                    if successful_explorations:
                        exploration_context = "\n\n## DATA EXPLORATION FINDINGS (use these correct column names!):\n"
                        for exp in successful_explorations[-2:]:  # Last 2 successful explorations
                            exploration_context += f"Source: {exp.get('data_source', '')}\n"
                            exploration_context += f"{exp.get('findings', '')[:1500]}\n\n"
                
                fix_prompt = code_fix_prompt.format(
                    hypothesis=hypothesis_text,
                    purpose=experiment_description + exploration_context,
                    original_code=current_code,
                    error_message=computation_result.error_message or "Unknown error",
                    stdout=stdout_text or "(empty)",
                    stderr=stderr_text or "(empty)"
                )
                
                # ==========================================================================
                # USE DEDICATED CODE FIXER MODEL IF CONFIGURED
                # ==========================================================================
                code_fixer_model_name = os.getenv("CODE_FIXER_MODEL")
                fix_start = perf_counter()
                
                if code_fixer_model_name:
                    # Use the dedicated code fixer model (e.g., Claude Opus)
                    print(f"Using CODE_FIXER_MODEL: {code_fixer_model_name}")
                    logger.info(f"Using dedicated code fixer model: {code_fixer_model_name}")
                    
                    fixer_config = get_model_runtime_config(
                        code_fixer_model_name,
                        config,
                        max_tokens=16000,  # Allow long code
                        tags=["langsmith:nostream"],
                    )
                    fix_response = await invoke_model_with_timeout(
                        configurable_model.with_config(fixer_config),
                        [HumanMessage(content=fix_prompt)],
                        timeout=MODEL_INVOKE_TIMEOUT,
                        label=f"code_fix_attempt_{attempt+2}"
                    )
                else:
                    # Use the same model as research
                    fix_response = await invoke_model_with_timeout(
                        configurable_model.with_config(model_config),
                        [HumanMessage(content=fix_prompt)],
                        timeout=MODEL_INVOKE_TIMEOUT,
                        label=f"code_fix_attempt_{attempt+2}"
                    )
                fix_duration = _elapsed_seconds(fix_start)
                fix_model_name = code_fixer_model_name or configurable.research_model
                fix_model_timings.append(
                    {
                        "attempt": attempt + 2,
                        "model": fix_model_name,
                        "duration_seconds": fix_duration,
                    }
                )
                trace_performance_metric(
                    "run_experiment.code_fix_model_invoke",
                    fix_duration,
                    node_name="run_experiment",
                    iteration=iteration,
                    data={"attempt": attempt + 2, "model": fix_model_name},
                )
                
                # Extract the fixed code from the response
                fixed_code = fix_response.content
                
                # Clean up the response - extract code from markdown if present
                if "```python" in fixed_code:
                    # Get the LAST python code block (in case there are multiple)
                    code_blocks = fixed_code.split("```python")
                    if len(code_blocks) > 1:
                        last_block = code_blocks[-1]
                        if "```" in last_block:
                            fixed_code = last_block.split("```")[0].strip()
                        else:
                            fixed_code = last_block.strip()
                elif "```" in fixed_code:
                    fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
                
                if fixed_code and len(fixed_code) > 50:
                    print(f"AI generated fix: {len(fixed_code)} chars, {fixed_code.count(chr(10))} lines")
                    logger.info(f"AI generated fix (attempt {attempt + 2}): {len(fixed_code)} chars")
                    
                    # ==========================================================================
                    # TRACEABILITY: Record code fix attempt
                    # ==========================================================================
                    fix_trace = {
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "code_fix_attempt",
                        "attempt_number": attempt + 2,
                        "original_error": computation_result.error_message,
                        "fix_model": code_fixer_model_name or configurable.research_model,
                        "original_code_length": len(current_code),
                        "fixed_code_length": len(fixed_code),
                    }
                    experiment_trace_events.append(fix_trace)
                    
                    TraceManager.add_event(
                        event_type=TraceEventType.CODE_FIX_ATTEMPT,
                        title=f"Code Fix Generated (Attempt #{attempt + 2})",
                        node_name="run_experiment",
                        experiment_id=experiment.id,
                        data={
                            "original_error": computation_result.error_message[:500] if computation_result.error_message else None,
                            "fix_model": code_fixer_model_name or configurable.research_model,
                            "code_length_change": len(fixed_code) - len(current_code),
                        }
                    )
                    
                    current_code = fixed_code
                    experiment.code_executed = current_code  # Update with fixed code
                else:
                    print("WARNING: AI fix was too short or empty, stopping retries")
                    logger.warning("AI fix was too short or empty, stopping retries")
                    break
                    
            except Exception as fix_error:
                print(f"ERROR: Failed to generate code fix: {fix_error}")
                logger.error(f"Failed to generate code fix: {fix_error}")
                break
        else:
            print(f"\n{'='*70}")
            print(f"CODE EXECUTION FAILED AFTER {max_fix_attempts + 1} ATTEMPTS")
            print(f"{'='*70}\n")
            logger.error(f"Code execution failed after {max_fix_attempts + 1} attempts")
    
    # Update experiment with results
    experiment.computation_results.append(computation_result)
    experiment.status = ExperimentStatus.COMPLETED if computation_result.success else ExperimentStatus.FAILED
    
    # ==========================================================================
    # VISION ANALYSIS: Analyze generated images with AI vision
    # ==========================================================================
    vision_duration = 0.0
    if computation_result.success:
        # Check if vision analysis is enabled
        if os.getenv("VISION_MODEL") or os.getenv("GOOGLE_API_KEY"):
            try:
                vision_start = perf_counter()
                # Build context for vision analysis
                experiment_context = f"""
Experiment: {experiment_plan.objective}
Hypothesis: {hypothesis_text}
Description: {experiment_description}
"""
                
                # Analyze images with vision model
                computation_result = await analyze_images_with_vision(
                    computation_result=computation_result,
                    hypothesis=hypothesis_text,
                    experiment_context=experiment_context
                )
                vision_duration = _elapsed_seconds(vision_start)
                trace_performance_metric(
                    "run_experiment.vision_analysis",
                    vision_duration,
                    node_name="run_experiment",
                    iteration=iteration,
                    data={"images_count": len(computation_result.get_images())},
                )
                logger.info("Vision analysis completed for generated images")
                
            except Exception as vision_error:
                logger.warning(f"Vision analysis failed (continuing without): {vision_error}")
    
    # ==========================================================================
    # QUALITY VALIDATION: Check experiment output quality
    # ==========================================================================
    quality_start = perf_counter()
    quality = validate_experiment_quality(
        computation_result=computation_result,
        require_statistical_results=True,
        require_visualizations=True,
        min_stdout_length=100
    )
    quality_duration = _elapsed_seconds(quality_start)
    trace_performance_metric(
        "run_experiment.quality_validation",
        quality_duration,
        node_name="run_experiment",
        iteration=iteration,
        data={"quality_score": quality.get("quality_score", 0)},
    )
    
    # Log quality assessment
    logger.info(
        f"Experiment quality: score={quality['quality_score']:.2f}, "
        f"valid={quality['is_valid']}, "
        f"numerical={quality['has_numerical_results']}, "
        f"stats={quality['has_statistical_tests']}, "
        f"viz={quality['has_visualizations']}"
    )
    
    if quality['warnings']:
        logger.warning(f"Quality warnings: {quality['warnings']}")
    
    # Generate quality improvement message for supervisor
    quality_message = ""
    if not quality['is_valid'] or quality['quality_score'] < 0.6:
        suggestions = get_quality_improvement_suggestions(quality)
        quality_message = (
            f"\n\n⚠️ EXPERIMENT QUALITY ISSUES (score: {quality['quality_score']*100:.0f}%):\n"
            + "\n".join(f"- {w}" for w in quality['warnings'])
            + "\n\nSuggestions to improve:\n"
            + "\n".join(f"- {s}" for s in suggestions)
        )
    
    # Register all outputs - EXPLICITLY MERGE with existing outputs
    # (The merge_dict_reducer may not work correctly with Pydantic objects due to serialization)
    existing_outputs = state.get("all_outputs", {})
    
    # Convert existing outputs to proper format if they were serialized
    # This handles cases where Pydantic objects became dicts during checkpoint serialization
    merged_outputs = {}
    for output_id, output_data in existing_outputs.items():
        if isinstance(output_data, dict) and 'output_type' in output_data:
            # Was serialized to dict, convert back to ComputationalOutput
            try:
                merged_outputs[output_id] = ComputationalOutput(**output_data)
            except Exception:
                # Keep as-is if conversion fails
                merged_outputs[output_id] = output_data
        else:
            merged_outputs[output_id] = output_data
    
    logger.info(f"Merging {len(computation_result.outputs)} new outputs with {len(merged_outputs)} existing outputs")
    
    figure_counter = state.get("figure_counter", 0)
    table_counter = state.get("table_counter", 0)
    
    for output in computation_result.outputs:
        output.source_experiment_id = experiment.id
        
        if output.output_type == OutputType.IMAGE:
            figure_counter += 1
            output.citation_label = f"Figure {figure_counter}"
        elif output.output_type == OutputType.TABLE:
            table_counter += 1
            output.citation_label = f"Table {table_counter}"
        
        merged_outputs[output.id] = output
    
    logger.info(f"Total outputs after merge: {len(merged_outputs)}")
    
    # ==========================================================================
    # TRACEABILITY: Record outputs generated
    # ==========================================================================
    for output in computation_result.outputs:
        trace_output_generated(
            output_id=output.id,
            output_type=output.output_type.value,
            description=output.description
        )
    
    # Add experiment completion event
    experiment_trace_events.append({
        "event_type": "experiment_completed",
        "timestamp": datetime.now().isoformat(),
        "title": f"Experiment {'Succeeded' if computation_result.success else 'Failed'}: {experiment_plan.objective[:60]}",
        "experiment_id": experiment.id,
        "hypothesis_id": hypothesis.id,
        "success": computation_result.success,
        "data": {
            "total_attempts": len(code_execution_traces),
            "outputs_generated": len(computation_result.outputs),
            "quality_score": quality.get("quality_score", 0),
            "design_duration_seconds": design_duration,
            "vision_duration_seconds": vision_duration,
            "quality_duration_seconds": quality_duration,
            "execution_attempts": execution_attempt_timings,
            "code_fix_timings": fix_model_timings,
        }
    })

    total_duration = _elapsed_seconds(node_start)
    trace_phase_end(
        "run_experiment.total",
        total_duration,
        success=computation_result.success,
        node_name="run_experiment",
        iteration=iteration,
        data={
            "hypothesis": hypothesis_text[:300],
            "design_duration_seconds": design_duration,
            "vision_duration_seconds": vision_duration,
            "quality_duration_seconds": quality_duration,
            "execution_attempts": execution_attempt_timings,
            "code_fix_timings": fix_model_timings,
            "outputs_generated": len(computation_result.outputs),
            "quality_score": quality.get("quality_score", 0),
        },
    )
    trace_node_exit(
        "run_experiment",
        iteration=iteration,
        success=computation_result.success,
        data={
            "duration_seconds": total_duration,
            "attempts": len(code_execution_traces),
            "outputs_generated": len(computation_result.outputs),
        },
    )
    logger.info(
        "Timing run_experiment total=%.2fs design=%.2fs quality=%.2fs attempts=%d success=%s",
        total_duration,
        design_duration,
        quality_duration,
        len(code_execution_traces),
        computation_result.success,
    )
    
    return Command(
        goto="analyze_results",
        update={
            "hypotheses": [hypothesis],
            "experiments": [experiment],
            "computation_results": [computation_result],
            "all_outputs": merged_outputs,
            "figure_counter": figure_counter,
            "table_counter": table_counter,
            "_current_experiment": experiment,
            "_current_hypothesis": hypothesis,
            "_experiment_quality": quality,
            "_quality_message": quality_message,
            # Clear consumed temp fields
            "_experiment_hypothesis": "",
            "_experiment_description": "",
            # Store sandbox_id for persistent sandbox reuse
            "sandbox_id": sandbox_id,
            # Traceability data
            "trace_events": experiment_trace_events,
            "code_execution_traces": code_execution_traces,
        }
    )


async def analyze_results(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["validate_claims", "discovery_supervisor"]]:
    """Analyze experiment results and determine implications.
    
    This node:
    1. Interprets computational outputs
    2. Determines if hypothesis is supported
    3. Creates findings
    4. Suggests next steps
    """
    node_start = perf_counter()
    iteration = state.get("discovery_iterations", 0)
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    trace_node_enter("analyze_results", iteration=iteration)
    trace_phase_start("analyze_results.total", node_name="analyze_results", iteration=iteration)
    
    experiment = state.get("_current_experiment")
    hypothesis = state.get("_current_hypothesis")
    
    if not experiment or not hypothesis:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "analyze_results.total",
            total_duration,
            success=False,
            node_name="analyze_results",
            iteration=iteration,
            data={"error": "missing_experiment_or_hypothesis"},
        )
        trace_node_exit(
            "analyze_results",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "missing_experiment_or_hypothesis"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="No experiment results to analyze.")
                ]
            }
        )
    
    # Get computation results
    computation_results = experiment.computation_results
    if not computation_results:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "analyze_results.total",
            total_duration,
            success=False,
            node_name="analyze_results",
            iteration=iteration,
            data={"error": "missing_computation_results"},
        )
        trace_node_exit(
            "analyze_results",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "missing_computation_results"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="Experiment produced no results.")
                ]
            }
        )
    
    # Format outputs for analysis
    all_outputs = []
    for result in computation_results:
        all_outputs.extend(result.outputs)
    
    outputs_summary = format_outputs_for_ai(all_outputs)
    
    # ==========================================================================
    # INCLUDE VISION ANALYSIS IN CONTEXT
    # ==========================================================================
    # Add vision analysis interpretations to give the AI richer context about what's
    # actually visible in the generated plots, enabling deeper reasoning about results
    vision_analysis_section = ""
    images_with_interpretations = [
        o for o in all_outputs 
        if o.output_type == OutputType.IMAGE and o.interpretation
    ]
    
    if images_with_interpretations:
        vision_analysis_section = "\n\n## VISION ANALYSIS OF GENERATED FIGURES\n"
        vision_analysis_section += "(AI vision model has analyzed these images and extracted the following observations)\n\n"
        
        for i, img in enumerate(images_with_interpretations, 1):
            vision_analysis_section += f"### {img.citation_label or f'Figure {i}'}\n"
            if img.caption:
                vision_analysis_section += f"Caption: {img.caption}\n"
            vision_analysis_section += f"AI Interpretation:\n{img.interpretation}\n\n"
        
        outputs_summary += vision_analysis_section
    
    # Get quality feedback from run_experiment
    quality = state.get("_experiment_quality", {})
    quality_message = state.get("_quality_message", "")
    
    # Add quality context if available
    quality_context = ""
    if quality:
        quality_context = f"""

## EXPERIMENT QUALITY METRICS
- Quality Score: {quality.get('quality_score', 0)*100:.0f}%
- Has Numerical Results: {quality.get('has_numerical_results', False)}
- Has Statistical Tests: {quality.get('has_statistical_tests', False)}
- Has Visualizations: {quality.get('has_visualizations', False)}
- Has Clear Conclusion: {quality.get('has_conclusion', False)}
"""
        if quality.get('warnings'):
            quality_context += "- Warnings: " + "; ".join(quality['warnings']) + "\n"
    
    # ==========================================================================
    # SIMULATION DETECTION in results: flag if experiment used synthetic data
    # ==========================================================================
    simulation_warning = ""
    if experiment.code_executed:
        from open_deep_research.computational.context.astroquery_discovery import (
            check_code_for_simulation_patterns,
        )
        sim_check = check_code_for_simulation_patterns(experiment.code_executed)
        if sim_check["has_simulation"] and not sim_check["is_legitimate"]:
            simulation_warning = f"""

## ⚠️ SIMULATION DATA WARNING
The experiment code contains patterns suggesting SYNTHETIC/SIMULATED data was used:
- Patterns detected: {', '.join(sim_check['simulation_patterns'][:5])}

**This is a CRITICAL limitation.** Results based on simulated data cannot support real scientific conclusions.

**RECOMMENDATION for next steps:**
1. Use ExploreData to search VizieR catalogs for real data (e.g., atmospheric retrievals, spectral surveys)
2. Search MAST for JWST/HST spectroscopic observations of the target planets
3. Only after confirming real data doesn't exist, use theoretical calculations (NOT random data)
"""
    
    # Build analysis prompt with research context and previous findings
    previous_findings_text = summarize_findings(state.get("findings", []))
    
    analysis_prompt = result_analysis_prompt.format(
        date=get_today_str(),
        research_brief=state.get("research_brief", "Not available")[:2000],
        previous_findings=previous_findings_text,
        hypothesis=hypothesis.statement,
        experiment_design=f"Objective: {experiment.design.objective}\nMethodology: {experiment.design.methodology}",
        computation_results=computation_results[0].to_ai_summary() if computation_results else "No results",
        outputs_summary=outputs_summary + quality_context + simulation_warning
    )
    
    # Try structured output, fall back to manual parsing
    analysis = None
    analysis_model_duration = 0.0
    used_fallback = False
    try:
        analysis_model = (
            configurable_model
            .with_structured_output(ExperimentAnalysis)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        analysis_start = perf_counter()
        analysis = await invoke_model_with_timeout(
            analysis_model,
            [HumanMessage(content=analysis_prompt)],
            timeout=MODEL_INVOKE_TIMEOUT,
            label="analyze_results"
        )
        analysis_model_duration = _elapsed_seconds(analysis_start)
    except Exception as e:
        error_str = str(e).lower()
        if "response_format" in error_str or "unavailable" in error_str or "json" in error_str or "timeout" in error_str:
            logger.warning(f"Structured output not supported for analysis, using fallback: {e}")
            used_fallback = True
            
            # Fallback: ask for structured text response
            fallback_prompt = f"""{analysis_prompt}

Please analyze these results and respond in this EXACT format:

FINDINGS_SUMMARY: [One paragraph summarizing what was found]

SUPPORTS_HYPOTHESIS: [YES or NO]

CONFIDENCE_LEVEL: [LOW, MEDIUM, or HIGH]

KEY_INSIGHTS:
- [First key insight]
- [Second key insight]
- [Third key insight]

RECOMMENDED_NEXT_STEPS:
- [First recommended step]
- [Second recommended step]

SHOULD_REFINE_HYPOTHESIS: [YES or NO]

REFINED_HYPOTHESIS: [If yes above, provide refined hypothesis. If no, write "N/A"]
"""
            try:
                fallback_start = perf_counter()
                fallback_response = await invoke_model_with_timeout(
                    configurable_model.with_config(model_config),
                    [HumanMessage(content=fallback_prompt)],
                    timeout=MODEL_INVOKE_TIMEOUT,
                    label="analyze_results_fallback"
                )
                fallback_duration = _elapsed_seconds(fallback_start)
                analysis_model_duration += fallback_duration
                trace_performance_metric(
                    "analyze_results.fallback_model_invoke",
                    fallback_duration,
                    node_name="analyze_results",
                    iteration=iteration,
                    data={"model": configurable.analysis_model or configurable.research_model},
                )
                content = fallback_response.content
                
                # Parse findings summary
                findings_summary = "Analysis completed"
                if "FINDINGS_SUMMARY:" in content:
                    start = content.find("FINDINGS_SUMMARY:") + len("FINDINGS_SUMMARY:")
                    end = content.find("\n\n", start)
                    if end < 0:
                        end = content.find("SUPPORTS_HYPOTHESIS:", start)
                    if end > 0:
                        findings_summary = content[start:end].strip()
                
                # Parse supports hypothesis
                supports_hypothesis = True
                if "SUPPORTS_HYPOTHESIS:" in content:
                    start = content.find("SUPPORTS_HYPOTHESIS:") + len("SUPPORTS_HYPOTHESIS:")
                    end = content.find("\n", start)
                    answer = content[start:end].strip().upper()
                    supports_hypothesis = "YES" in answer or "TRUE" in answer
                
                # Parse confidence level
                confidence_level = "medium"
                if "CONFIDENCE_LEVEL:" in content:
                    start = content.find("CONFIDENCE_LEVEL:") + len("CONFIDENCE_LEVEL:")
                    end = content.find("\n", start)
                    level = content[start:end].strip().lower()
                    if "high" in level:
                        confidence_level = "high"
                    elif "low" in level:
                        confidence_level = "low"
                
                # Parse key insights
                key_insights = ["Results analyzed successfully"]
                if "KEY_INSIGHTS:" in content:
                    start = content.find("KEY_INSIGHTS:") + len("KEY_INSIGHTS:")
                    end = content.find("RECOMMENDED_NEXT_STEPS:", start)
                    if end > 0:
                        insights_text = content[start:end].strip()
                        key_insights = [line.strip().lstrip("-•").strip() for line in insights_text.split("\n") if line.strip() and line.strip() != "-"]
                
                # Parse recommended next steps
                recommended_next_steps = ["Continue investigation"]
                if "RECOMMENDED_NEXT_STEPS:" in content:
                    start = content.find("RECOMMENDED_NEXT_STEPS:") + len("RECOMMENDED_NEXT_STEPS:")
                    end = content.find("SHOULD_REFINE_HYPOTHESIS:", start)
                    if end > 0:
                        steps_text = content[start:end].strip()
                        recommended_next_steps = [line.strip().lstrip("-•").strip() for line in steps_text.split("\n") if line.strip() and line.strip() != "-"]
                
                # Parse should refine
                should_refine_hypothesis = False
                if "SHOULD_REFINE_HYPOTHESIS:" in content:
                    start = content.find("SHOULD_REFINE_HYPOTHESIS:") + len("SHOULD_REFINE_HYPOTHESIS:")
                    end = content.find("\n", start)
                    answer = content[start:end].strip().upper()
                    should_refine_hypothesis = "YES" in answer or "TRUE" in answer
                
                # Parse refined hypothesis
                refined_hypothesis = None
                if should_refine_hypothesis and "REFINED_HYPOTHESIS:" in content:
                    start = content.find("REFINED_HYPOTHESIS:") + len("REFINED_HYPOTHESIS:")
                    refined_hypothesis = content[start:].strip().split("\n")[0].strip()
                    if "N/A" in refined_hypothesis.upper():
                        refined_hypothesis = None
                
                analysis = ExperimentAnalysis(
                    findings_summary=findings_summary,
                    supports_hypothesis=supports_hypothesis,
                    confidence_level=confidence_level,
                    key_insights=key_insights[:5],  # Limit to 5
                    recommended_next_steps=recommended_next_steps[:3],  # Limit to 3
                    should_refine_hypothesis=should_refine_hypothesis,
                    refined_hypothesis=refined_hypothesis
                )
                logger.info("Successfully parsed analysis from fallback response")
                
            except Exception as parse_error:
                logger.error(f"Fallback analysis parsing failed: {parse_error}")
                # Create a minimal analysis so we can continue
                analysis = ExperimentAnalysis(
                    findings_summary="Analysis could not be completed due to parsing error",
                    supports_hypothesis=False,
                    confidence_level="low",
                    key_insights=["Parsing error occurred"],
                    recommended_next_steps=["Retry with different approach"],
                    should_refine_hypothesis=False,
                    refined_hypothesis=None
                )
        else:
            logger.error(f"Analysis failed: {e}")
            total_duration = _elapsed_seconds(node_start)
            trace_phase_end(
                "analyze_results.total",
                total_duration,
                success=False,
                node_name="analyze_results",
                iteration=iteration,
                data={"error": str(e)[:500]},
            )
            trace_node_exit(
                "analyze_results",
                iteration=iteration,
                success=False,
                data={"duration_seconds": total_duration, "error": str(e)[:500]},
            )
            return Command(
                goto="discovery_supervisor",
                update={
                    "supervisor_messages": [
                        HumanMessage(content=f"Result analysis failed: {str(e)}")
                    ]
                }
            )
    
    if not analysis:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "analyze_results.total",
            total_duration,
            success=False,
            node_name="analyze_results",
            iteration=iteration,
            data={"error": "analysis_none"},
        )
        trace_node_exit(
            "analyze_results",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "analysis_none"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="Analysis failed: Could not generate analysis")
                ]
            }
        )
    
    # Update hypothesis status
    if analysis.supports_hypothesis:
        hypothesis.status = HypothesisStatus.SUPPORTED
    else:
        hypothesis.status = HypothesisStatus.REFUTED
    
    hypothesis.supporting_evidence = analysis.key_insights
    
    # Update experiment
    experiment.findings_summary = analysis.findings_summary
    experiment.supports_hypothesis = analysis.supports_hypothesis
    experiment.status = ExperimentStatus.ANALYZING
    
    # Determine novelty based on actual criteria:
    # A finding is novel if it reveals something unexpected or previously unknown,
    # NOT simply because the hypothesis was refuted.
    # Novel findings include: unexpected patterns, statistically significant results,
    # new correlations, or results that contradict established literature.
    is_novel = (
        analysis.confidence_level == "high"  # Strong evidence for any conclusion
        or analysis.should_refine_hypothesis  # Led to hypothesis refinement (new insight)
        or (not analysis.supports_hypothesis and analysis.confidence_level == "medium")  # Confident refutation
    )
    
    # Capture statistical evidence from key insights
    statistical_evidence = ""
    for insight in analysis.key_insights:
        if any(term in insight.lower() for term in ["p-value", "p =", "correlation", "significant", "confidence"]):
            statistical_evidence += insight + "; "
    
    # Create finding
    finding = Finding(
        statement=analysis.findings_summary,
        significance=", ".join(analysis.key_insights),
        statistical_evidence=statistical_evidence.strip("; ") if statistical_evidence else None,
        supporting_experiments=[experiment.id],
        supporting_outputs=[o.id for o in all_outputs if o.output_type in [OutputType.IMAGE, OutputType.STATISTICAL_RESULT]],
        is_novel=is_novel,
        visualization_ids=[o.id for o in all_outputs if o.output_type == OutputType.IMAGE]
    )
    
    # ==========================================================================
    # TRACEABILITY: Record finding
    # ==========================================================================
    trace_finding_recorded(
        finding_id=finding.id,
        statement=analysis.findings_summary,
        is_novel=finding.is_novel
    )
    
    # Build summary for supervisor
    summary_parts = [
        f"=== Experiment Analysis Complete ===",
        f"Hypothesis: {hypothesis.statement}",
        f"Result: {'SUPPORTED' if analysis.supports_hypothesis else 'NOT SUPPORTED'}",
        f"Confidence: {analysis.confidence_level}",
    ]
    
    # Add quality score if available
    if quality:
        quality_score = quality.get('quality_score', 0)
        summary_parts.append(f"Quality Score: {quality_score*100:.0f}%")
    
    summary_parts.append(f"\nKey Insights:")
    for insight in analysis.key_insights:
        summary_parts.append(f"  • {insight}")
    
    summary_parts.append(f"\nRecommended Next Steps:")
    for step in analysis.recommended_next_steps:
        summary_parts.append(f"  • {step}")
    
    if analysis.should_refine_hypothesis and analysis.refined_hypothesis:
        summary_parts.append(f"\nSuggested Refined Hypothesis: {analysis.refined_hypothesis}")
    
    # Include figure references with vision analysis summaries
    images = [o for o in all_outputs if o.output_type == OutputType.IMAGE]
    if images:
        summary_parts.append(f"\nGenerated {len(images)} visualization(s):")
        for img in images:
            if img.citation_label:
                summary_parts.append(f"  • {img.citation_label}")
                # Include brief vision interpretation if available
                if img.interpretation:
                    # Extract first 2 lines of interpretation for summary
                    interp_lines = img.interpretation.strip().split('\n')[:2]
                    interp_summary = ' '.join(line.strip() for line in interp_lines if line.strip())
                    if len(interp_summary) > 200:
                        interp_summary = interp_summary[:200] + "..."
                    summary_parts.append(f"    Vision analysis: {interp_summary}")
    
    # Add quality warnings/suggestions if quality is low
    if quality_message:
        summary_parts.append(quality_message)
    
    # ==========================================================================
    # TRACEABILITY: Record analysis completion
    # ==========================================================================
    analysis_trace = {
        "event_type": "analysis_completed",
        "timestamp": datetime.now().isoformat(),
        "title": f"Analysis Complete: {'Supported' if analysis.supports_hypothesis else 'Not Supported'}",
        "experiment_id": experiment.id,
        "hypothesis_id": hypothesis.id,
        "data": {
            "supports_hypothesis": analysis.supports_hypothesis,
            "confidence_level": analysis.confidence_level,
            "key_insights": analysis.key_insights,
            "recommended_next_steps": analysis.recommended_next_steps,
            "finding_statement": analysis.findings_summary[:500],
            "quality_score": quality.get("quality_score", 0) if quality else 0,
            "analysis_model_duration_seconds": analysis_model_duration,
            "analysis_fallback_used": used_fallback,
        }
    }

    total_duration = _elapsed_seconds(node_start)
    trace_performance_metric(
        "analyze_results.model_invoke_total",
        analysis_model_duration,
        node_name="analyze_results",
        iteration=iteration,
        data={"fallback_used": used_fallback},
    )
    trace_phase_end(
        "analyze_results.total",
        total_duration,
        node_name="analyze_results",
        iteration=iteration,
        data={
            "analysis_model_duration_seconds": analysis_model_duration,
            "fallback_used": used_fallback,
            "supports_hypothesis": analysis.supports_hypothesis,
            "finding_id": finding.id,
        },
    )
    trace_node_exit(
        "analyze_results",
        iteration=iteration,
        success=True,
        data={
            "duration_seconds": total_duration,
            "analysis_model_duration_seconds": analysis_model_duration,
            "finding_id": finding.id,
        },
    )
    logger.info(
        "Timing analyze_results total=%.2fs model=%.2fs fallback=%s",
        total_duration,
        analysis_model_duration,
        used_fallback,
    )
    
    # Extract recommended next steps as new questions for discovery tracking
    new_questions = [step for step in analysis.recommended_next_steps if step]
    
    return Command(
        goto="validate_claims",
        update={
            "findings": [finding],
            "new_questions": new_questions,
            "supervisor_messages": [
                HumanMessage(content="\n".join(summary_parts))
            ],
            # Keep current entities for claim validation gate
            "_current_finding": finding,
            # Traceability data
            "trace_events": [analysis_trace],
        }
    )


async def validate_claims(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor"]]:
    """Validate analyzed findings before they can be treated as robust claims."""
    node_start = perf_counter()
    iteration = state.get("discovery_iterations", 0)
    configurable = ComputationalConfiguration.from_runnable_config(config)
    trace_node_enter("validate_claims", iteration=iteration)
    trace_phase_start("validate_claims.total", node_name="validate_claims", iteration=iteration)
    finding = state.get("_current_finding")
    experiment = state.get("_current_experiment")
    hypothesis = state.get("_current_hypothesis")

    if not finding or not experiment:
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "validate_claims.total",
            total_duration,
            success=False,
            node_name="validate_claims",
            iteration=iteration,
            data={"error": "missing_finding_or_experiment"},
        )
        trace_node_exit(
            "validate_claims",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "error": "missing_finding_or_experiment"},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="Claim validation skipped: no current finding/experiment context.")
                ]
            }
        )

    quality_message = state.get("_quality_message", "")
    contradiction_notes: List[str] = []
    if "SIMULATION DATA WARNING" in quality_message.upper():
        contradiction_notes.append("Simulation-based evidence warning detected in experiment quality checks.")
    if "NOT SUPPORTED" in finding.statement.upper() and finding.is_novel:
        contradiction_notes.append("Claim flagged as novel while primary conclusion states not supported.")

    has_stats = bool(finding.statistical_evidence)
    simulation_penalty = bool(contradiction_notes)
    novelty_confidence = _estimate_novelty_confidence(
        confidence_level="high" if finding.is_novel else "medium",
        has_stats=has_stats,
        has_contradictions=bool(contradiction_notes),
        simulation_penalty=simulation_penalty,
    )

    requires_replication = (
        configurable.require_replication_for_novel_claims
        and finding.is_novel
        and novelty_confidence >= configurable.claim_validation_min_novelty_confidence
    )
    replication_status = ReplicationStatus.REQUIRED if requires_replication else ReplicationStatus.NOT_REQUIRED
    replication_reason = "Replication required for high-impact novel claim." if requires_replication else "Replication not required."
    replication_attempts = 0

    # Bounded replication gate
    replication_passed = False
    replication_timings: List[Dict[str, Any]] = []
    if requires_replication and experiment.code_executed:
        trace_replication_check(
            claim_id=finding.id,
            status="started",
            details={"experiment_id": experiment.id, "max_attempts": configurable.max_replication_attempts},
        )
        primary_signal = experiment.supports_hypothesis
        for attempt in range(configurable.max_replication_attempts):
            rep_start = perf_counter()
            replication_attempts += 1
            replication_code = _build_replication_code(experiment.code_executed)
            replication_result = await execute_experiment(
                experiment_code=replication_code,
                experiment_id=f"{experiment.id}_rep{attempt+1}",
                hypothesis=hypothesis.statement if hypothesis else finding.statement,
                sandbox_id=state.get("sandbox_id"),
                config=config,
            )
            rep_duration = _elapsed_seconds(rep_start)
            text_blob = f"{replication_result.stdout}\n{replication_result.stderr}"
            inferred_signal = _extract_boolean_hypothesis_signal(text_blob)
            consistent_signal = (
                inferred_signal is None or primary_signal is None or inferred_signal == primary_signal
            )
            has_rep_stats = any(
                token in text_blob.lower() for token in ("p-value", "p =", "confidence", "correlation")
            )
            if replication_result.success and consistent_signal and (has_stats or has_rep_stats):
                replication_passed = True
                replication_timings.append(
                    {
                        "attempt": attempt + 1,
                        "duration_seconds": rep_duration,
                        "success": replication_result.success,
                        "consistent_signal": consistent_signal,
                        "has_rep_stats": has_rep_stats,
                    }
                )
                trace_performance_metric(
                    "validate_claims.replication_attempt",
                    rep_duration,
                    node_name="validate_claims",
                    iteration=iteration,
                    data={"attempt": attempt + 1, "success": replication_result.success},
                )
                break
            replication_timings.append(
                {
                    "attempt": attempt + 1,
                    "duration_seconds": rep_duration,
                    "success": replication_result.success,
                    "consistent_signal": consistent_signal,
                    "has_rep_stats": has_rep_stats,
                }
            )
            trace_performance_metric(
                "validate_claims.replication_attempt",
                rep_duration,
                node_name="validate_claims",
                iteration=iteration,
                data={"attempt": attempt + 1, "success": replication_result.success},
            )

        replication_status = ReplicationStatus.PASSED if replication_passed else ReplicationStatus.FAILED
        replication_reason = (
            "Replication run produced consistent support signal and sufficient statistical output."
            if replication_passed
            else "Replication run failed or did not reproduce support signal/statistical evidence."
        )
        trace_replication_check(
            claim_id=finding.id,
            status="completed",
            details={
                "attempts": replication_attempts,
                "passed": replication_passed,
                "reason": replication_reason,
            },
        )

    # Final claim verdict
    if contradiction_notes:
        claim_status = ClaimStatus.REJECTED
        verdict_reason = "; ".join(contradiction_notes)
    elif requires_replication and not replication_passed:
        claim_status = ClaimStatus.INCONCLUSIVE
        verdict_reason = replication_reason
    elif (
        novelty_confidence >= configurable.claim_validation_min_novelty_confidence
        and (not configurable.require_statistical_evidence or has_stats)
        and (not requires_replication or replication_passed)
    ):
        claim_status = ClaimStatus.VALIDATED
        verdict_reason = "Evidence quality and confidence thresholds satisfied."
    else:
        claim_status = ClaimStatus.PROVISIONAL
        verdict_reason = "Claim is plausible but does not yet meet strict validation thresholds."

    # Keep novelty reserved for validated claims in synthesis layer
    finding.is_novel = claim_status == ClaimStatus.VALIDATED

    claim_record = ClaimValidationRecord(
        finding_id=finding.id,
        claim_text=finding.statement,
        hypothesis_id=hypothesis.id if hypothesis else None,
        experiment_ids=[experiment.id],
        source_urls=_extract_primary_source_urls(state),
        confidence_level="high" if finding.is_novel else "medium",
        statistical_evidence=finding.statistical_evidence,
        contradiction_notes=contradiction_notes,
        simulation_penalty_applied=simulation_penalty,
        evidence_summary=finding.significance[:1000] if finding.significance else "",
        status=claim_status,
        verdict_reason=verdict_reason,
        novelty_confidence=novelty_confidence,
        replication_status=replication_status,
        replication_reason=replication_reason,
        replication_attempts=replication_attempts,
        updated_at=datetime.now(),
    )

    trace_claim_created(claim_record.id, finding.id, finding.statement)
    if claim_status == ClaimStatus.VALIDATED:
        trace_claim_validated(claim_record.id, verdict_reason, novelty_confidence)
    else:
        trace_claim_rejected(claim_record.id, verdict_reason, novelty_confidence)

    existing_findings = state.get("findings", [])
    updated_findings = [
        finding if getattr(f, "id", None) == finding.id else f
        for f in existing_findings
    ]

    validation_summary = (
        "=== Claim Validation Gate ===\n"
        f"Claim status: {claim_status.value.upper()}\n"
        f"Novelty confidence: {novelty_confidence:.2f}\n"
        f"Replication: {replication_status.value}\n"
        f"Reason: {verdict_reason}"
    )

    claim_trace_event = {
        "event_type": "claim_validation",
        "timestamp": datetime.now().isoformat(),
        "title": f"Claim {claim_status.value.title()}",
        "finding_id": finding.id,
        "experiment_id": experiment.id,
        "data": {
            "claim_id": claim_record.id,
            "status": claim_status.value,
            "novelty_confidence": novelty_confidence,
            "replication_status": replication_status.value,
            "replication_attempts": replication_attempts,
            "replication_timings": replication_timings,
            "reason": verdict_reason,
        },
    }

    total_duration = _elapsed_seconds(node_start)
    trace_phase_end(
        "validate_claims.total",
        total_duration,
        node_name="validate_claims",
        iteration=iteration,
        data={
            "claim_id": claim_record.id,
            "status": claim_status.value,
            "replication_attempts": replication_attempts,
            "replication_timings": replication_timings,
        },
        success=claim_status in (ClaimStatus.VALIDATED, ClaimStatus.PROVISIONAL, ClaimStatus.INCONCLUSIVE),
    )
    trace_node_exit(
        "validate_claims",
        iteration=iteration,
        success=True,
        data={
            "duration_seconds": total_duration,
            "claim_id": claim_record.id,
            "status": claim_status.value,
        },
    )
    logger.info(
        "Timing validate_claims total=%.2fs status=%s replication_attempts=%d",
        total_duration,
        claim_status.value,
        replication_attempts,
    )

    return Command(
        goto="discovery_supervisor",
        update={
            "findings": {"type": "override", "value": updated_findings},
            "claim_ledger": [claim_record],
            "supervisor_messages": [HumanMessage(content=validation_summary)],
            "trace_events": [claim_trace_event],
            "_current_finding": None,
            "_current_experiment": None,
            "_current_hypothesis": None,
            "_experiment_quality": {},
            "_quality_message": "",
        }
    )


async def synthesize_findings(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["__end__", "discovery_supervisor"]]:
    """Synthesize all findings into a comprehensive research report.
    
    This is the final node that creates the complete research report,
    incorporating all computational outputs, findings, and evidence.
    
    IMPORTANT: Will redirect back to supervisor if insufficient experiments
    have been run (requires at least 1 experiment with outputs).
    """
    node_start = perf_counter()
    iteration = state.get("discovery_iterations", 0)
    configurable = ComputationalConfiguration.from_runnable_config(config)
    trace_node_enter("synthesize_findings", iteration=iteration)
    trace_phase_start("synthesize_findings.total", node_name="synthesize_findings", iteration=iteration)
    # Get all components
    research_brief = state.get("research_brief", "")
    hypotheses = state.get("hypotheses", [])
    experiments = state.get("experiments", [])
    findings = state.get("findings", [])
    claim_ledger = state.get("claim_ledger", [])
    all_outputs = state.get("all_outputs", {})
    computation_results = state.get("computation_results", [])
    discovery_iterations = state.get("discovery_iterations", 0)
    
    # Check if we have sufficient computational evidence
    # If not, redirect back to supervisor to run more experiments
    # (unless we've hit max iterations)
    max_iterations = configurable.max_discovery_iterations
    has_experiments = len(experiments) > 0
    has_outputs = len(all_outputs) > 0
    has_computation_results = len(computation_results) > 0
    
    insufficient_computation = (
        not has_experiments or 
        not has_outputs or 
        not has_computation_results
    )
    
    # Only force more experiments if we haven't hit max iterations
    if insufficient_computation and discovery_iterations < max_iterations:
        logger.warning(
            f"Synthesis requested but insufficient computational evidence. "
            f"Experiments: {len(experiments)}, Outputs: {len(all_outputs)}, "
            f"Computation results: {len(computation_results)}. "
            f"Redirecting to supervisor to run experiments."
        )
        total_duration = _elapsed_seconds(node_start)
        trace_phase_end(
            "synthesize_findings.total",
            total_duration,
            success=False,
            node_name="synthesize_findings",
            iteration=iteration,
            data={
                "insufficient_computation": True,
                "experiments": len(experiments),
                "outputs": len(all_outputs),
                "computation_results": len(computation_results),
            },
        )
        trace_node_exit(
            "synthesize_findings",
            iteration=iteration,
            success=False,
            data={"duration_seconds": total_duration, "redirected_to_supervisor": True},
        )
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="""
⚠️ INSUFFICIENT COMPUTATIONAL EVIDENCE FOR SYNTHESIS ⚠️

You attempted to synthesize findings, but NO computational experiments have been executed.

Current state:
- Experiments run: {experiments}
- Computational outputs: {outputs}
- Computation results: {results}

REQUIRED ACTIONS:
1. You MUST call RunExperiment at least 2-3 times
2. Execute actual Python code (thermodynamics, statistics, spectra)
3. Generate visualizations and numerical results
4. ONLY THEN can you call SynthesizeFindings

Do NOT call SynthesizeFindings again until you have run computational experiments.
Use RunExperiment NOW to execute code for your hypothesis.
""".format(
                        experiments=len(experiments),
                        outputs=len(all_outputs),
                        results=len(computation_results)
                    ))
                ]
            }
        )
    
    # Build comprehensive context
    hypotheses_text = "\n".join([
        f"- {h.statement} [{h.status.value}]\n  Rationale: {h.rationale}"
        for h in hypotheses
    ]) if hypotheses else "No hypotheses were formally tested."
    
    experiments_text = "\n".join([
        f"Experiment {exp.id}:\n  Objective: {exp.design.objective}\n  Finding: {exp.findings_summary or 'No summary'}"
        for exp in experiments
    ]) if experiments else "No experiments were conducted."
    
    if claim_ledger:
        findings_text = summarize_claim_ledger(claim_ledger)
    else:
        findings_text = "\n".join([
            f"- {f.statement}\n  Significance: {f.significance}"
            for f in findings
        ]) if findings else "No formal findings recorded."
    
    outputs_catalogue = create_outputs_catalogue(state)
    
    # Get raw notes from computation results
    raw_notes_parts = []
    for result in state.get("computation_results", []):
        raw_notes_parts.append(result.to_ai_summary())
    raw_notes = "\n\n".join(raw_notes_parts)
    messages_text = get_buffer_string(state.get("messages", []))
    
    # Generate final report
    report_model = configurable_model.with_config(
        get_model_runtime_config(
            configurable.final_report_model,
            config,
            max_tokens=configurable.final_report_model_max_tokens,
            tags=["langsmith:nostream"],
        )
    )
    
    synthesis_prompt = final_report_synthesis_prompt.format(
        date=get_today_str(),
        messages=messages_text[:12000],  # Limit size
        report_language=configurable.report_language,
        research_brief=research_brief,
        hypotheses=hypotheses_text,
        experiments=experiments_text,
        findings=findings_text,
        outputs_catalogue=outputs_catalogue,
        raw_notes=raw_notes[:30000]  # Limit size
    )
    
    report_generation_duration = 0.0
    try:
        report_start = perf_counter()
        response = await report_model.ainvoke([
            HumanMessage(content=synthesis_prompt)
        ])
        report_generation_duration = _elapsed_seconds(report_start)
        final_report = response.content
    except Exception as e:
        report_generation_duration = _elapsed_seconds(report_start)
        logger.error(f"Report synthesis failed: {e}")
        final_report = f"""
# Research Report

## Error

An error occurred during report synthesis: {str(e)}

## Summary of Findings

{findings_text}

## Hypotheses Tested

{hypotheses_text}

## Experiments Conducted

{experiments_text}
"""
    
    # Close any persistent sandboxes
    sandbox_id = state.get("sandbox_id")
    if sandbox_id:
        SandboxManager.close_sandbox(sandbox_id)
    
    # ==========================================================================
    # TRACEABILITY: Finalize the trace
    # ==========================================================================
    # Calculate summary statistics
    total_duration = 0
    trace_started = state.get("trace_started_at")
    if trace_started:
        try:
            from datetime import datetime as dt
            start_time = dt.fromisoformat(trace_started)
            total_duration = (datetime.now() - start_time).total_seconds()
        except:
            pass
    
    # Access current timing summary before creating the final state event
    trace = TraceManager.get_trace()
    timing_breakdown = trace.get_timing_breakdown() if trace else {}

    # Create final trace event
    final_trace_event = {
        "event_type": "report_generated",
        "timestamp": datetime.now().isoformat(),
        "title": "Final Report Generated",
        "data": {
            "report_length": len(final_report),
            "total_hypotheses": len(hypotheses),
            "total_experiments": len(experiments),
            "total_findings": len(findings),
            "total_outputs": len(all_outputs),
            "total_duration_seconds": total_duration,
            "total_iterations": discovery_iterations,
            "report_generation_duration_seconds": report_generation_duration,
            "timing_breakdown_top": list(timing_breakdown.items())[:15],
        },
        "success": True
    }
    
    # Finalize TraceManager
    TraceManager.add_event(
        event_type=TraceEventType.REPORT_GENERATED,
        title="Final Report Generated",
        node_name="synthesize_findings",
        data={
            "report_length": len(final_report),
            "total_hypotheses": len(hypotheses),
            "total_experiments": len(experiments),
            "total_findings": len(findings),
            "report_generation_duration_seconds": report_generation_duration,
            "timing_breakdown_top": list(timing_breakdown.items())[:15],
        },
        success=True
    )
    trace_performance_metric(
        "synthesize_findings.report_generation",
        report_generation_duration,
        node_name="synthesize_findings",
        iteration=iteration,
        data={"model": configurable.final_report_model},
    )
    total_duration_node = _elapsed_seconds(node_start)
    trace_phase_end(
        "synthesize_findings.total",
        total_duration_node,
        node_name="synthesize_findings",
        iteration=iteration,
        data={
            "report_generation_duration_seconds": report_generation_duration,
            "report_length": len(final_report),
            "timing_breakdown_top": list(timing_breakdown.items())[:15],
        },
    )
    trace_node_exit(
        "synthesize_findings",
        iteration=iteration,
        success=True,
        data={
            "duration_seconds": total_duration_node,
            "report_generation_duration_seconds": report_generation_duration,
            "report_length": len(final_report),
        },
    )
    logger.info(
        "Timing synthesize_findings total=%.2fs report_generation=%.2fs",
        total_duration_node,
        report_generation_duration,
    )
    TraceManager.finalize(final_report)
    
    return Command(
        goto=END,
        update={
            "final_report": final_report,
            "messages": [AIMessage(content=final_report)],
            # Final traceability data
            "trace_events": [final_trace_event],
        }
    )
