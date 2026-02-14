"""Computational Scientific Discovery Module.

This module provides the infrastructure for self-driving scientific research
through computational experimentation, hypothesis testing, and discovery.

Key Components:
- state: State definitions for computational discovery workflow
- code_interpreter: E2B-based code execution with output management
- scientific_tools: Tools for accessing scientific data sources
- prompts: Agent prompts for the discovery workflow
- discovery_nodes: Node implementations for the discovery graph
- discovery_graph: The main LangGraph implementation

Usage:
    from open_deep_research.computational import computational_discovery
    
    result = await computational_discovery.ainvoke({
        "messages": [HumanMessage(content="Your research query")]
    })
"""

from open_deep_research.computational.state import (
    ComputationalDiscoveryState,
    ComputationalDiscoveryInputState,
    HypothesisRecord,
    HypothesisStatus,
    ExperimentRecord,
    ExperimentStatus,
    ExperimentDesign,
    ComputationResult,
    ComputationalOutput,
    OutputType,
    Finding,
    PaperData,
    ScientificDataSource,
    DataSourceType,
    ExtractedEquation,
    ExtractedDataTable,
    GeneratedHypothesis,
    ExperimentPlan,
    ExperimentAnalysis,
    IterationDecision,
)

from open_deep_research.computational.code_interpreter import (
    CodeInterpreterManager,
    SandboxManager,
    OutputProcessor,
    execute_code,
    execute_experiment,
    execute_python_code,
    run_simulation,
    statistical_test,
    format_outputs_for_ai,
    format_outputs_for_report,
)

from open_deep_research.computational.scientific_tools import (
    search_arxiv_papers,
    query_nasa_exoplanet_archive,
    query_mast_archive,
    query_sdss_database,
    fetch_scientific_api,
    extract_paper_data,
    get_scientific_tools,
)

from open_deep_research.computational.discovery_graph import (
    computational_discovery,
    computational_researcher,
    run_computational_discovery,
    create_discovery_graph,
)

from open_deep_research.computational.configuration import (
    ComputationalConfiguration,
    ScientificDomain,
    get_domain_specific_packages,
    get_domain_data_sources,
    get_domain_prompt_context,
)

from open_deep_research.computational.traceability import (
    TraceEventType,
    TraceEvent,
    CodeExecutionTrace,
    SupervisorDecisionTrace,
    ResearchTrace,
    TraceManager,
)

__all__ = [
    # Main Graph
    "computational_discovery",
    "computational_researcher",
    "run_computational_discovery",
    "create_discovery_graph",
    
    # State Models
    "ComputationalDiscoveryState",
    "ComputationalDiscoveryInputState",
    "HypothesisRecord",
    "HypothesisStatus",
    "ExperimentRecord",
    "ExperimentStatus", 
    "ExperimentDesign",
    "ComputationResult",
    "ComputationalOutput",
    "OutputType",
    "Finding",
    "PaperData",
    "ScientificDataSource",
    "DataSourceType",
    "ExtractedEquation",
    "ExtractedDataTable",
    "GeneratedHypothesis",
    "ExperimentPlan",
    "ExperimentAnalysis",
    "IterationDecision",
    
    # Code Interpreter
    "CodeInterpreterManager",
    "SandboxManager",
    "OutputProcessor",
    "execute_code",
    "execute_experiment",
    "execute_python_code",
    "run_simulation",
    "statistical_test",
    "format_outputs_for_ai",
    "format_outputs_for_report",
    
    # Scientific Tools
    "search_arxiv_papers",
    "query_nasa_exoplanet_archive",
    "query_mast_archive",
    "query_sdss_database",
    "fetch_scientific_api",
    "extract_paper_data",
    "get_scientific_tools",
    
    # Configuration
    "ComputationalConfiguration",
    "ScientificDomain",
    "get_domain_specific_packages",
    "get_domain_data_sources",
    "get_domain_prompt_context",
    
    # Traceability
    "TraceEventType",
    "TraceEvent",
    "CodeExecutionTrace",
    "SupervisorDecisionTrace",
    "ResearchTrace",
    "TraceManager",
]
