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
    trace_code_execution,
    trace_supervisor_decision,
    trace_hypothesis_created,
    trace_experiment_started,
    trace_finding_recorded,
    trace_output_generated,
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
    get_api_key_for_model,
    get_today_str,
    think_tool,
)

logger = logging.getLogger(__name__)

# Initialize configurable model
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_supervisor_model_config(configurable: ComputationalConfiguration, config: RunnableConfig) -> Dict[str, Any]:
    """Get supervisor model configuration with fallback to research model."""
    model_name = configurable.supervisor_model or configurable.research_model
    max_tokens = configurable.supervisor_model_max_tokens or configurable.research_model_max_tokens
    return {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"]
    }


def get_worker_model_config(configurable: ComputationalConfiguration, config: RunnableConfig) -> Dict[str, Any]:
    """Get worker model configuration with fallback to research model."""
    model_name = configurable.worker_model or configurable.research_model
    max_tokens = configurable.worker_model_max_tokens or configurable.research_model_max_tokens
    return {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"]
    }


def escape_format_braces(text: str) -> str:
    """Escape curly braces in text to prevent KeyErrors during string formatting.
    
    This is critical because dynamic content (e.g., experiment objectives, findings)
    may contain patterns like {T} or {value} which would cause KeyError when
    the content is formatted into prompts.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


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
    
    # Data explorations
    data_explorations = state.get("data_explorations", [])
    if data_explorations:
        successful = [e for e in data_explorations if e.get("success")]
        failed = [e for e in data_explorations if not e.get("success")]
        parts.append(f"**Data Explorations:** {len(successful)} successful, {len(failed)} failed")
        for exp in successful[:3]:
            parts.append(f"  - {exp.get('data_source', '?')}: {exp.get('goal', '')[:100]}")
    
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    
    # Check if clarification is allowed
    # Also check environment variable directly for robustness
    env_allow = os.getenv("ALLOW_CLARIFICATION", "").lower()
    if not configurable.allow_clarification or env_allow in ("false", "0", "no"):
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
        
        response = await clarification_model.ainvoke([HumanMessage(content=prompt)])
        
        if response.need_clarification:
            return Command(
                goto=END,
                update={"messages": [AIMessage(content=response.question)]}
            )
        else:
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
            return Command(goto="generate_research_brief")
        else:
            # Re-raise other errors
            raise


async def generate_research_brief(
    state: ComputationalDiscoveryState,
    config: RunnableConfig
) -> Command[Literal["discovery_supervisor"]]:
    """Generate a comprehensive research brief from the user's query.
    
    This transforms the user's messages into a structured research brief
    that will guide the entire discovery process.
    """
    configurable = ComputationalConfiguration.from_runnable_config(config)
    messages = state.get("messages", [])
    model_config = get_supervisor_model_config(configurable, config)
    
    # Initialize trace if not already done
    trace_node_enter("generate_research_brief")
    
    research_query = get_buffer_string(messages)
    
    research_model = configurable_model.with_config(model_config)
    
    # Get domain context for the prompt
    domain = configurable.scientific_domain
    domain_context = get_domain_prompt_context(domain)
    
    prompt = research_brief_generation_prompt.format(
        messages=research_query,
        date=get_today_str(),
        domain_context=domain_context
    )
    
    response = await research_model.ainvoke([HumanMessage(content=prompt)])
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
        },
        success=True
    )
    
    # Update trace with research context
    trace = TraceManager.get_trace()
    if trace:
        trace.research_query = research_query
        trace.research_brief = research_brief
    
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_supervisor_model_config(configurable, config)
    
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
    
    # Build the message list: fresh system prompt + recent conversation
    messages_for_model = [SystemMessage(content=system_prompt)] + recent_messages
    
    # Invoke supervisor
    response = await supervisor_model.ainvoke(messages_for_model)
    
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    discovery_iterations = state.get("discovery_iterations", 0)
    
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
        return Command(goto="synthesize_findings")
    
    # Generate ToolMessage responses for ALL tool calls to satisfy API requirements
    all_tool_messages = []
    primary_action = None
    primary_update = {}
    
    for tool_call in most_recent_message.tool_calls:
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
        
        else:
            # Unknown tool - still respond to avoid API errors
            all_tool_messages.append(ToolMessage(
                content=f"Tool {tool_name} acknowledged.",
                name=tool_name,
                tool_call_id=tool_id
            ))
    
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
        }
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    
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
    
    for _ in range(max_iterations):
        response = await knowledge_model.ainvoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            break
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool:
                try:
                    result = await tool.ainvoke(tool_call["args"], config)
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
                    messages.append(ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))
    
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
    """Explore database schemas and APIs before running experiments.
    
    This node runs exploratory code to discover:
    - Available column names in databases
    - Data types and value ranges
    - API-specific syntax requirements
    - Sample data to understand the structure
    
    The findings are passed back to the supervisor to inform experiment design.
    """
    from open_deep_research.computational.prompts import data_exploration_prompt
    
    configurable = ComputationalConfiguration.from_runnable_config(config)
    
    data_source = state.get("_exploration_data_source", "")
    exploration_goal = state.get("_exploration_goal", "")
    
    if not data_source:
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
    
    # Build exploration prompt
    context = f"""
Research Brief: {state.get('research_brief', 'Not available')}

Previous exploration attempts: {len(state.get('data_explorations', []))}

Data source to explore: {data_source}
"""
    
    exploration_prompt = data_exploration_prompt.format(
        exploration_goal=exploration_goal,
        context=context
    )
    
    # Get API key for code fixer model
    code_fixer_api_key = get_api_key_for_model(code_fixer_model, config)
    
    # Generate exploration code using the better coding model
    exploration_model = configurable_model.with_config({
        "model": code_fixer_model,
        "max_tokens": 4096,
        "api_key": code_fixer_api_key,
        "tags": ["langsmith:nostream"]
    })
    
    try:
        response = await exploration_model.ainvoke([
            HumanMessage(content=exploration_prompt)
        ])
        
        # Extract code from response
        code_content = response.content
        
        # Try to extract code from markdown blocks
        if "```python" in code_content:
            import re
            code_blocks = re.findall(r'```python\s*(.*?)```', code_content, re.DOTALL)
            if code_blocks:
                exploration_code = max(code_blocks, key=len).strip()
            else:
                exploration_code = code_content
        elif "```" in code_content:
            import re
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
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content=f"Failed to generate exploration code: {e}\n\nPlease try ExploreData again or proceed with RunExperiment.")
                ]
            }
        )
    
    # Execute exploration code in E2B
    from open_deep_research.computational.code_interpreter import execute_code
    
    exploration_result = await execute_code(
        code=exploration_code,
        purpose=f"Explore {data_source}: {exploration_goal}",
        timeout=120,  # 2 minutes should be enough for exploration
        config=config
    )
    
    # Format the exploration findings
    if exploration_result.success:
        findings = f"""
## Data Exploration Successful!

**Data Source:** {data_source}
**Goal:** {exploration_goal}

### Exploration Output:
```
{exploration_result.stdout[:8000] if exploration_result.stdout else "No output"}
```

### Key Findings:
The exploration code ran successfully. The output above shows:
- Available column names
- Sample data values
- Correct query syntax

**Use these findings when designing your experiment!**
"""
        logger.info("Data exploration successful")
    else:
        findings = f"""
## Data Exploration Failed

**Data Source:** {data_source}
**Goal:** {exploration_goal}

### Error:
```
{exploration_result.error_message or "Unknown error"}
```

### Standard Output (partial):
```
{exploration_result.stdout[:3000] if exploration_result.stdout else "No output"}
```

### Standard Error:
```
{exploration_result.stderr[:2000] if exploration_result.stderr else "No errors"}
```

### Next Steps:
The exploration failed, but the error message may reveal useful information about:
- Correct column names (if "invalid identifier" error)
- Required packages (if import error)
- API syntax requirements

You can:
1. Try ExploreData again with a different approach
2. Proceed with RunExperiment - the code fixer will use this error info to fix queries
"""
        logger.warning(f"Data exploration failed: {exploration_result.error_message}")
    
    # Store exploration results
    data_explorations = state.get("data_explorations", [])
    data_explorations.append({
        "data_source": data_source,
        "goal": exploration_goal,
        "success": exploration_result.success,
        "findings": findings,
        "code": exploration_code[:2000]  # Store partial code for reference
    })
    
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    
    hypothesis_text = state.get("_experiment_hypothesis", "").strip()
    experiment_description = state.get("_experiment_description", "").strip()
    
    # Validate hypothesis is provided
    if not hypothesis_text or len(hypothesis_text) < 10:
        logger.warning(f"Run experiment called without valid hypothesis: '{hypothesis_text}'")
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
    try:
        experiment_model = (
            configurable_model
            .with_structured_output(ExperimentPlan)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        experiment_plan = await experiment_model.ainvoke([
            HumanMessage(content=design_prompt)
        ])
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a structured output compatibility issue
        if "response_format" in error_str or "unavailable" in error_str or "json" in error_str:
            logger.warning(f"Structured output not supported, using fallback parser: {e}")
            print(f"Structured output not supported, using fallback parser")
            
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
                fallback_response = await configurable_model.with_config(model_config).ainvoke([
                    HumanMessage(content=fallback_prompt)
                ])
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
            return Command(
                goto="discovery_supervisor",
                update={
                    "supervisor_messages": [
                        HumanMessage(content=f"Experiment design failed: {str(e)}")
                    ]
                }
            )
    
    if not experiment_plan:
        return Command(
            goto="discovery_supervisor",
            update={
                "supervisor_messages": [
                    HumanMessage(content="Experiment design failed: Could not generate experiment plan")
                ]
            }
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
    
    # Priority 7: Use persistent sandbox if configured
    sandbox_id = state.get("sandbox_id") if configurable.use_persistent_sandbox else None
    
    for attempt in range(max_fix_attempts + 1):
        execution_start_time = datetime.now()
        
        # Execute the experiment in E2B with persistent sandbox
        computation_result = await execute_experiment(
            experiment_code=current_code,
            experiment_id=experiment.id,
            hypothesis=hypothesis_text,
            sandbox_id=sandbox_id,
            config=config
        )
        
        execution_time = (datetime.now() - execution_start_time).total_seconds()
        
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
                
                if code_fixer_model_name:
                    # Use the dedicated code fixer model (e.g., Claude Opus)
                    print(f"Using CODE_FIXER_MODEL: {code_fixer_model_name}")
                    logger.info(f"Using dedicated code fixer model: {code_fixer_model_name}")
                    
                    fixer_api_key = get_api_key_for_model(code_fixer_model_name, config)
                    fixer_config = {
                        "model": code_fixer_model_name,
                        "max_tokens": 16000,  # Allow long code
                        "api_key": fixer_api_key,
                        "tags": ["langsmith:nostream"]
                    }
                    fix_response = await configurable_model.with_config(fixer_config).ainvoke([
                        HumanMessage(content=fix_prompt)
                    ])
                else:
                    # Use the same model as research
                    fix_response = await configurable_model.with_config(model_config).ainvoke([
                        HumanMessage(content=fix_prompt)
                    ])
                
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
    if computation_result.success:
        # Check if vision analysis is enabled
        if os.getenv("VISION_MODEL") or os.getenv("GOOGLE_API_KEY"):
            try:
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
                logger.info("Vision analysis completed for generated images")
                
            except Exception as vision_error:
                logger.warning(f"Vision analysis failed (continuing without): {vision_error}")
    
    # ==========================================================================
    # QUALITY VALIDATION: Check experiment output quality
    # ==========================================================================
    quality = validate_experiment_quality(
        computation_result=computation_result,
        require_statistical_results=True,
        require_visualizations=True,
        min_stdout_length=100
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
        }
    })
    
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
) -> Command[Literal["discovery_supervisor"]]:
    """Analyze experiment results and determine implications.
    
    This node:
    1. Interprets computational outputs
    2. Determines if hypothesis is supported
    3. Creates findings
    4. Suggests next steps
    """
    configurable = ComputationalConfiguration.from_runnable_config(config)
    model_config = get_worker_model_config(configurable, config)
    
    experiment = state.get("_current_experiment")
    hypothesis = state.get("_current_hypothesis")
    
    if not experiment or not hypothesis:
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
    
    # Build analysis prompt with research context and previous findings
    previous_findings_text = summarize_findings(state.get("findings", []))
    
    analysis_prompt = result_analysis_prompt.format(
        date=get_today_str(),
        research_brief=state.get("research_brief", "Not available")[:2000],
        previous_findings=previous_findings_text,
        hypothesis=hypothesis.statement,
        experiment_design=f"Objective: {experiment.design.objective}\nMethodology: {experiment.design.methodology}",
        computation_results=computation_results[0].to_ai_summary() if computation_results else "No results",
        outputs_summary=outputs_summary + quality_context
    )
    
    # Try structured output, fall back to manual parsing
    analysis = None
    try:
        analysis_model = (
            configurable_model
            .with_structured_output(ExperimentAnalysis)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        analysis = await analysis_model.ainvoke([
            HumanMessage(content=analysis_prompt)
        ])
    except Exception as e:
        error_str = str(e).lower()
        if "response_format" in error_str or "unavailable" in error_str or "json" in error_str:
            logger.warning(f"Structured output not supported for analysis, using fallback: {e}")
            
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
                fallback_response = await configurable_model.with_config(model_config).ainvoke([
                    HumanMessage(content=fallback_prompt)
                ])
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
            return Command(
                goto="discovery_supervisor",
                update={
                    "supervisor_messages": [
                        HumanMessage(content=f"Result analysis failed: {str(e)}")
                    ]
                }
            )
    
    if not analysis:
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
        }
    }
    
    # Extract recommended next steps as new questions for discovery tracking
    new_questions = [step for step in analysis.recommended_next_steps if step]
    
    return Command(
        goto="discovery_supervisor",
        update={
            "findings": [finding],
            "new_questions": new_questions,
            "supervisor_messages": [
                HumanMessage(content="\n".join(summary_parts))
            ],
            # Clear consumed temp fields
            "_current_experiment": None,
            "_current_hypothesis": None,
            "_experiment_quality": {},
            "_quality_message": "",
            # Traceability data
            "trace_events": [analysis_trace],
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
    configurable = ComputationalConfiguration.from_runnable_config(config)
    # Get all components
    research_brief = state.get("research_brief", "")
    hypotheses = state.get("hypotheses", [])
    experiments = state.get("experiments", [])
    findings = state.get("findings", [])
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
    
    # Generate final report
    report_model = configurable_model.with_config({
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    synthesis_prompt = final_report_synthesis_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        hypotheses=hypotheses_text,
        experiments=experiments_text,
        findings=findings_text,
        outputs_catalogue=outputs_catalogue,
        raw_notes=raw_notes[:30000]  # Limit size
    )
    
    try:
        response = await report_model.ainvoke([
            HumanMessage(content=synthesis_prompt)
        ])
        final_report = response.content
    except Exception as e:
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
        },
        success=True
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
