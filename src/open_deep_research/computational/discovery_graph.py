"""Main LangGraph Implementation for Computational Scientific Discovery.

This module defines the complete graph for the self-driving scientific
research system that:
- Generates and tests hypotheses
- Runs computational experiments
- Analyzes results iteratively
- Produces comprehensive research reports

The graph follows the discovery loop pattern:
    clarify → research_brief → supervisor ↔ [gather, experiment, analyze] → synthesize
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from open_deep_research.computational.discovery_nodes import (
    analyze_results,
    check_novelty,
    clarify_discovery_query,
    discovery_supervisor,
    explore_data,
    gather_knowledge,
    generate_research_brief,
    reflect_hypothesis,
    review_paper_draft,
    run_experiment,
    supervisor_tools,
    synthesize_findings,
    write_paper_draft,
    validate_claims,
)
from open_deep_research.computational.state import (
    ComputationalDiscoveryInputState,
    ComputationalDiscoveryState,
)
from open_deep_research.computational.configuration import ComputationalConfiguration


# =============================================================================
# Main Computational Discovery Graph
# =============================================================================

def build_computational_discovery_graph():
    """Build the complete computational discovery graph.
    
    Graph Structure:
    ================
    
    START
      │
      ▼
    clarify_discovery_query ──────────────────────► END (if clarification needed)
      │
      │ (no clarification needed)
      ▼
    generate_research_brief
      │
      ▼
    ┌─────────────────────────────────────────────┐
    │         DISCOVERY SUPERVISOR LOOP            │
    │                                              │
    │   discovery_supervisor ◄─────────┐           │
    │      │                           │           │
    │      ▼                           │           │
    │   supervisor_tools               │           │
    │      │                           │           │
    │      ├── GatherKnowledge ────────┤           │
    │      │      │                    │           │
    │      │      ▼                    │           │
    │      │   gather_knowledge ───────┘           │
    │      │                                       │
    │      ├── RunExperiment ──────────┐           │
    │      │      │                    │           │
    │      │      ▼                    │           │
    │      │   run_experiment          │           │
    │      │      │                    │           │
    │      │      ▼                    │           │
    │      │   analyze_results ────────┘           │
    │      │                                       │
    │      └── SynthesizeFindings                  │
    │             │                                │
    │             ▼                                │
    │       synthesize_findings                    │
    │             │                                │
    └─────────────┼────────────────────────────────┘
                  │
                  ▼
                 END
                 
    Note: This graph uses LangGraph's Command-based routing.
    Each node returns a Command that specifies which node to go to next.
    This enables dynamic routing based on tool calls and state.
    """
    
    # Build main graph - use ComputationalConfiguration to expose all discovery settings
    discovery_builder = StateGraph(
        ComputationalDiscoveryState,
        input=ComputationalDiscoveryInputState,
        config_schema=ComputationalConfiguration
    )
    
    # Add all nodes
    discovery_builder.add_node("clarify_discovery_query", clarify_discovery_query)
    discovery_builder.add_node("generate_research_brief", generate_research_brief)
    discovery_builder.add_node("discovery_supervisor", discovery_supervisor)
    discovery_builder.add_node("supervisor_tools", supervisor_tools)
    discovery_builder.add_node("gather_knowledge", gather_knowledge)
    discovery_builder.add_node("explore_data", explore_data)  # Schema discovery before experiments
    discovery_builder.add_node("check_novelty", check_novelty)
    discovery_builder.add_node("reflect_hypothesis", reflect_hypothesis)
    discovery_builder.add_node("run_experiment", run_experiment)
    discovery_builder.add_node("analyze_results", analyze_results)
    discovery_builder.add_node("validate_claims", validate_claims)
    discovery_builder.add_node("synthesize_findings", synthesize_findings)
    discovery_builder.add_node("write_paper_draft", write_paper_draft)
    discovery_builder.add_node("review_paper_draft", review_paper_draft)
    
    # Define edges
    # Entry point - start with clarification
    discovery_builder.add_edge(START, "clarify_discovery_query")
    
    # The routing is handled by Command returns from each node:
    # - clarify_discovery_query → generate_research_brief OR END
    # - generate_research_brief → discovery_supervisor
    # - discovery_supervisor → supervisor_tools
    # - supervisor_tools → discovery_supervisor OR gather_knowledge OR explore_data OR check_novelty OR reflect_hypothesis OR run_experiment OR synthesize_findings
    # - check_novelty → discovery_supervisor
    # - reflect_hypothesis → run_experiment OR discovery_supervisor
    # - gather_knowledge → discovery_supervisor
    # - explore_data → discovery_supervisor (returns schema discovery findings)
    # - run_experiment → analyze_results
    # - analyze_results → validate_claims
    # - validate_claims → discovery_supervisor
    # - synthesize_findings → END OR discovery_supervisor OR write_paper_draft
    # - write_paper_draft → review_paper_draft
    # - review_paper_draft → END
    
    # Note: No static edges needed - all routing done via Command returns
    # synthesize_findings can now redirect back to supervisor if not enough
    # computational experiments were run
    
    return discovery_builder.compile()


# =============================================================================
# Compiled Graphs
# =============================================================================

# Main computational discovery graph
computational_discovery = build_computational_discovery_graph()

# Alias for consistency with the main deep_researcher
computational_researcher = computational_discovery


# =============================================================================
# Factory Functions
# =============================================================================

def create_discovery_graph(
    enable_clarification: bool = True,
    max_iterations: int = 5
):
    """Create a configured discovery graph.
    
    Args:
        enable_clarification: Whether to enable user clarification
        max_iterations: Maximum discovery iterations
        
    Returns:
        Compiled StateGraph for computational discovery
    """
    return build_computational_discovery_graph()


# =============================================================================
# Utility for Running the Graph
# =============================================================================

async def run_computational_discovery(
    query: str,
    config: dict = None
) -> dict:
    """Run the computational discovery process on a research query.
    
    Args:
        query: The research question or topic to investigate
        config: Optional configuration overrides
        
    Returns:
        Dictionary containing the final state with report and findings
    """
    from langchain_core.messages import HumanMessage
    
    default_config = {
        "configurable": {
            "research_model": "openai:gpt-4o",
            "final_report_model": "openai:gpt-4o",
            "max_researcher_iterations": 5,
            "allow_clarification": True
        }
    }
    
    if config:
        default_config["configurable"].update(config.get("configurable", {}))
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    result = await computational_discovery.ainvoke(initial_state, default_config)
    
    return result


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "computational_discovery",
    "computational_researcher",
    "create_discovery_graph",
    "run_computational_discovery",
    "build_computational_discovery_graph",
]
