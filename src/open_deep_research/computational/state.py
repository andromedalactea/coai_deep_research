"""State definitions for the Computational Scientific Discovery System.

This module defines the state models that track the entire scientific discovery
workflow, from hypothesis generation through computational experimentation
to final synthesis of findings.

TRACEABILITY NOTE:
The state includes trace_events for capturing the research process in a way
that enables human (or AI) review and replication of the research.
"""

import operator
import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# =============================================================================
# Enums for State Management
# =============================================================================

class HypothesisStatus(str, Enum):
    """Status of a hypothesis in the discovery workflow."""
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    REFINED = "refined"


class ExperimentStatus(str, Enum):
    """Status of a computational experiment."""
    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"


class OutputType(str, Enum):
    """Types of outputs from code execution."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DATAFRAME = "dataframe"
    FIGURE = "figure"
    STATISTICAL_RESULT = "statistical_result"
    ERROR = "error"
    LOG = "log"


class DataSourceType(str, Enum):
    """Types of scientific data sources."""
    ARXIV_PAPER = "arxiv_paper"
    PUBMED_PAPER = "pubmed_paper"
    TELESCOPE_ARCHIVE = "telescope_archive"
    EXOPLANET_CATALOG = "exoplanet_catalog"
    ASTRONOMICAL_DATABASE = "astronomical_database"
    COMPUTED_RESULT = "computed_result"
    USER_PROVIDED = "user_provided"


# =============================================================================
# Output Management Models
# =============================================================================

class ComputationalOutput(BaseModel):
    """A single output from code execution - properly managed for AI consumption.
    
    This model handles all types of outputs including binary data (images),
    making them accessible throughout the system and for the final report.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    output_type: OutputType
    
    # Content - one of these will be populated based on output_type
    text_content: Optional[str] = None
    image_base64: Optional[str] = None  # Base64 encoded image
    image_format: Optional[str] = None  # "png", "jpg", "svg"
    table_data: Optional[List[Dict[str, Any]]] = None  # For tabular outputs
    
    # Metadata for AI interpretation
    description: str = Field(
        default="",
        description="Human-readable description of what this output represents"
    )
    interpretation: Optional[str] = Field(
        default=None,
        description="AI-generated interpretation of this output"
    )
    
    # Reference information for citations
    source_experiment_id: Optional[str] = None
    source_code_snippet: Optional[str] = None
    
    # For report generation
    caption: Optional[str] = None
    citation_label: Optional[str] = None  # e.g., "Figure 1", "Table 2"
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_ai_description(self) -> str:
        """Generate a description suitable for AI model consumption."""
        parts = [f"[Output {self.id}] Type: {self.output_type.value}"]
        
        if self.description:
            parts.append(f"Description: {self.description}")
        
        if self.output_type == OutputType.TEXT and self.text_content:
            parts.append(f"Content:\n{self.text_content}")
        
        elif self.output_type == OutputType.IMAGE:
            parts.append(f"Image Format: {self.image_format or 'unknown'}")
            if self.caption:
                parts.append(f"Caption: {self.caption}")
            if self.interpretation:
                parts.append(f"Visual Interpretation: {self.interpretation}")
            parts.append(f"Reference: [Image output_{self.id}]")
        
        elif self.output_type == OutputType.TABLE and self.table_data:
            parts.append(f"Table with {len(self.table_data)} rows")
            if self.table_data:
                columns = list(self.table_data[0].keys()) if self.table_data else []
                parts.append(f"Columns: {', '.join(columns)}")
        
        elif self.output_type == OutputType.STATISTICAL_RESULT:
            parts.append(f"Statistical Result: {self.text_content}")
        
        if self.interpretation:
            parts.append(f"Interpretation: {self.interpretation}")
        
        return "\n".join(parts)
    
    def to_report_reference(self) -> str:
        """Generate a reference string for use in final reports."""
        if self.output_type == OutputType.IMAGE:
            label = self.citation_label or f"Figure (figure_{self.id})"
            caption = self.caption or self.description
            # Use relative path that matches save_outputs() in run_discovery.py
            return f"![{caption}](figure_{self.id}.{self.image_format or 'png'})"
        elif self.output_type == OutputType.TABLE:
            return f"[{self.citation_label or f'Table figure_{self.id}'}]"
        else:
            return f"[Computational Result {self.id}]"


class ComputationResult(BaseModel):
    """Complete result from a code execution, including all outputs."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Execution details
    code_executed: str
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Outputs - multiple outputs possible per execution
    outputs: List[ComputationalOutput] = Field(default_factory=list)
    
    # Raw outputs for reference
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[str] = None
    
    # Metadata
    purpose: str = ""  # Why was this code executed
    experiment_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_images(self) -> List[ComputationalOutput]:
        """Get all image outputs from this result."""
        return [o for o in self.outputs if o.output_type == OutputType.IMAGE]
    
    def get_text_outputs(self) -> List[ComputationalOutput]:
        """Get all text outputs."""
        return [o for o in self.outputs if o.output_type == OutputType.TEXT]
    
    def to_ai_summary(self) -> str:
        """Generate a comprehensive summary for AI consumption."""
        parts = [
            f"=== Computation Result [{self.id}] ===",
            f"Purpose: {self.purpose}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
        ]
        
        if not self.success:
            parts.append(f"Error: {self.error_message}")
        
        if self.stdout:
            parts.append(f"\nStandard Output:\n{self.stdout}")
        
        if self.return_value:
            parts.append(f"\nReturn Value: {self.return_value}")
        
        if self.outputs:
            parts.append(f"\n--- Generated Outputs ({len(self.outputs)}) ---")
            for output in self.outputs:
                parts.append(output.to_ai_description())
                parts.append("")
        
        return "\n".join(parts)


# =============================================================================
# Scientific Data Models
# =============================================================================

class ExtractedEquation(BaseModel):
    """An equation extracted from a paper, ready for computational use."""
    latex: str
    python_equivalent: Optional[str] = None
    variables: Dict[str, str] = Field(default_factory=dict)  # var_name -> description
    source_paper_id: Optional[str] = None
    context: str = ""  # Surrounding text explaining the equation


class ExtractedDataTable(BaseModel):
    """Data table extracted from a paper or database."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    data: List[Dict[str, Any]]  # List of row dictionaries
    columns: List[str]
    units: Dict[str, str] = Field(default_factory=dict)  # column -> unit
    source: str = ""
    source_type: DataSourceType = DataSourceType.ARXIV_PAPER


class PaperData(BaseModel):
    """Structured data extracted from an academic paper."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    authors: List[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: str = ""
    
    # Extracted content
    abstract: str = ""
    full_text: Optional[str] = None
    equations: List[ExtractedEquation] = Field(default_factory=list)
    data_tables: List[ExtractedDataTable] = Field(default_factory=list)
    methodology_summary: Optional[str] = None
    key_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # For citation
    citation_key: Optional[str] = None
    
    def to_citation(self) -> str:
        """Generate citation string."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{authors_str}. {self.title}. {self.url}"


class ScientificDataSource(BaseModel):
    """A scientific data source (telescope archive, catalog, etc.)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    source_type: DataSourceType
    url: Optional[str] = None
    query_used: Optional[str] = None
    
    # Data retrieved
    data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # File references (for FITS files, etc.)
    file_paths: List[str] = Field(default_factory=list)
    
    retrieved_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Hypothesis and Experiment Models
# =============================================================================

class HypothesisRecord(BaseModel):
    """A scientific hypothesis to be tested computationally."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # The hypothesis itself
    statement: str = Field(
        description="Clear statement of the hypothesis to test"
    )
    rationale: str = Field(
        default="",
        description="Scientific rationale for this hypothesis"
    )
    
    # Status tracking
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    
    # Related elements
    derived_from_papers: List[str] = Field(default_factory=list)  # Paper IDs
    derived_from_hypotheses: List[str] = Field(default_factory=list)  # Parent hypothesis IDs
    
    # Testing results
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    statistical_significance: Optional[float] = None  # p-value if applicable
    confidence_level: Optional[float] = None
    
    # Evolution
    refined_to: Optional[str] = None  # ID of refined hypothesis
    refinement_reason: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ExperimentDesign(BaseModel):
    """Design specification for a computational experiment."""
    objective: str = Field(
        description="What this experiment aims to determine"
    )
    methodology: str = Field(
        description="How the experiment will be conducted"
    )
    
    # Code specification
    code_outline: str = Field(
        description="High-level description of code to execute"
    )
    required_data: List[str] = Field(
        default_factory=list,
        description="Data sources needed for this experiment"
    )
    required_packages: List[str] = Field(
        default_factory=list,
        description="Python packages needed"
    )
    
    # Expected outputs
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="What outputs this experiment should produce"
    )
    success_criteria: str = Field(
        default="",
        description="How to determine if the experiment succeeded"
    )


class ExperimentRecord(BaseModel):
    """Record of a computational experiment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Link to hypothesis
    hypothesis_id: str
    
    # Design
    design: ExperimentDesign
    
    # Execution
    status: ExperimentStatus = ExperimentStatus.DESIGNED
    code_executed: Optional[str] = None
    
    # Results
    computation_results: List[ComputationResult] = Field(default_factory=list)
    
    # Analysis
    findings_summary: Optional[str] = None
    supports_hypothesis: Optional[bool] = None
    confidence_notes: Optional[str] = None
    
    # Iteration
    follow_up_experiments: List[str] = Field(default_factory=list)  # IDs
    
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def get_all_outputs(self) -> List[ComputationalOutput]:
        """Get all outputs from all computation results."""
        outputs = []
        for result in self.computation_results:
            outputs.extend(result.outputs)
        return outputs


class Finding(BaseModel):
    """A scientific finding from the discovery process."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # The finding
    statement: str = Field(
        description="Clear statement of what was discovered"
    )
    significance: str = Field(
        default="",
        description="Why this finding matters"
    )
    
    # Evidence
    supporting_experiments: List[str] = Field(default_factory=list)
    supporting_outputs: List[str] = Field(default_factory=list)  # Output IDs
    statistical_evidence: Optional[str] = None
    
    # Classification
    is_novel: bool = False
    confirms_existing: bool = False
    contradicts_existing: Optional[str] = None  # What it contradicts
    
    # For report
    visualization_ids: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Reducers for State Management
# =============================================================================

def override_or_append_reducer(current_value, new_value):
    """Reducer that allows overriding or appending to lists."""
    if isinstance(new_value, dict):
        if new_value.get("type") == "override":
            return new_value.get("value", [])
        elif new_value.get("type") == "append":
            return current_value + new_value.get("value", [])
    return current_value + (new_value if isinstance(new_value, list) else [new_value])


def merge_dict_reducer(current_value, new_value):
    """Reducer that merges dictionaries."""
    if current_value is None:
        current_value = {}
    if new_value is None:
        return current_value
    if isinstance(new_value, dict):
        if new_value.get("type") == "override":
            return new_value.get("value", {})
        return {**current_value, **new_value}
    return current_value


# =============================================================================
# Main State Definitions
# =============================================================================

class ComputationalDiscoveryInputState(MessagesState):
    """Input state for the computational discovery workflow."""
    pass


class ComputationalDiscoveryState(MessagesState):
    """Main state for the Computational Scientific Discovery System.
    
    This state tracks the entire discovery workflow from hypothesis
    generation through computational experimentation to synthesis.
    """
    
    # Research context
    research_query: str = ""
    research_brief: str = ""
    scientific_domain: str = "general"  # Domain for adaptive prompts
    
    # Hypothesis tracking
    hypotheses: Annotated[List[HypothesisRecord], override_or_append_reducer] = []
    current_hypothesis_id: Optional[str] = None
    
    # Data layer - gathered knowledge
    papers: Annotated[List[PaperData], override_or_append_reducer] = []
    data_sources: Annotated[List[ScientificDataSource], override_or_append_reducer] = []
    extracted_equations: Annotated[List[ExtractedEquation], override_or_append_reducer] = []
    data_explorations: Annotated[List[Dict[str, Any]], override_or_append_reducer] = []  # Schema exploration results
    
    # Accumulated knowledge summary - compressed, structured knowledge base
    # Updated after each gather_knowledge call with a refined summary
    knowledge_summary: str = ""
    
    # Computational layer - E2B managed
    sandbox_id: Optional[str] = None  # For persistent sandbox
    experiments: Annotated[List[ExperimentRecord], override_or_append_reducer] = []
    computation_results: Annotated[List[ComputationResult], override_or_append_reducer] = []
    
    # Output management - critical for report generation
    all_outputs: Annotated[Dict[str, ComputationalOutput], merge_dict_reducer] = {}
    figure_counter: int = 0
    table_counter: int = 0
    
    # Discovery tracking
    findings: Annotated[List[Finding], override_or_append_reducer] = []
    new_questions: Annotated[List[str], override_or_append_reducer] = []
    
    # Iteration control
    discovery_iterations: int = 0
    max_iterations_reached: bool = False
    
    # Final output
    final_report: str = ""
    
    # Internal tracking
    supervisor_messages: Annotated[List[MessageLikeRepresentation], override_or_append_reducer] = []
    
    # Temporary routing fields (used to pass data between nodes)
    # These are cleared at the start of each consuming node
    _experiment_hypothesis: str = ""
    _experiment_description: str = ""
    _knowledge_request: str = ""
    _exploration_data_source: str = ""  # Data source to explore
    _exploration_goal: str = ""  # What to discover about the data source
    
    # Current experiment/hypothesis being processed (passed between nodes)
    _current_experiment: Optional["ExperimentRecord"] = None
    _current_hypothesis: Optional["HypothesisRecord"] = None
    
    # Experiment quality tracking (populated by run_experiment)
    _experiment_quality: Dict[str, Any] = {}  # Quality validation results
    _quality_message: str = ""  # Quality warnings and suggestions
    
    # ==========================================================================
    # TRACEABILITY - Captures the entire research process for review/replication
    # ==========================================================================
    
    # Trace events - serializable list of all events that occurred during research
    # Each event is a dict with: event_type, timestamp, title, description, data, etc.
    trace_events: Annotated[List[Dict[str, Any]], override_or_append_reducer] = []
    
    # Code execution traces - detailed records of all code executions including:
    # - The code that was executed
    # - Attempt number (for retries)
    # - Success/failure status
    # - stdout/stderr
    # - Error messages and fix reasoning
    code_execution_traces: Annotated[List[Dict[str, Any]], override_or_append_reducer] = []
    
    # Supervisor decision traces - records of all supervisor decisions including:
    # - What action was chosen and why
    # - State at decision time
    # - Tool calls made
    supervisor_decision_traces: Annotated[List[Dict[str, Any]], override_or_append_reducer] = []
    
    # Trace metadata
    trace_id: str = ""  # Unique identifier for this research trace
    trace_started_at: Optional[str] = None  # ISO timestamp


class ExperimentDesignerState(TypedDict):
    """State for the experiment designer agent."""
    hypothesis: HypothesisRecord
    available_data: List[str]  # Descriptions of available data
    available_tools: List[str]  # Available computational tools
    messages: Annotated[List[MessageLikeRepresentation], operator.add]
    designed_experiment: Optional[ExperimentDesign]


class ComputationExecutorState(TypedDict):
    """State for the computation executor."""
    experiment: ExperimentRecord
    sandbox_id: Optional[str]
    code_to_execute: str
    messages: Annotated[List[MessageLikeRepresentation], operator.add]
    results: List[ComputationResult]


class ResultAnalyzerState(TypedDict):
    """State for the result analyzer agent."""
    experiment: ExperimentRecord
    computation_results: List[ComputationResult]
    hypothesis: HypothesisRecord
    messages: Annotated[List[MessageLikeRepresentation], operator.add]
    analysis_complete: bool
    should_iterate: bool
    new_hypothesis: Optional[HypothesisRecord]


# =============================================================================
# Structured Output Models for Agents
# =============================================================================

class GeneratedHypothesis(BaseModel):
    """Structured output for hypothesis generation."""
    hypothesis_statement: str = Field(
        description="Clear, testable hypothesis statement"
    )
    rationale: str = Field(
        description="Scientific rationale based on gathered information"
    )
    testable_prediction: str = Field(
        description="Specific prediction that can be tested computationally"
    )
    required_data: List[str] = Field(
        description="Data needed to test this hypothesis"
    )
    suggested_methodology: str = Field(
        description="How to computationally test this hypothesis"
    )


class ExperimentPlan(BaseModel):
    """Structured output for experiment planning."""
    objective: str = Field(
        description="What this experiment will determine"
    )
    python_code: str = Field(
        description="Complete Python code to execute the experiment"
    )
    expected_outputs: List[str] = Field(
        description="What outputs the code will produce"
    )
    interpretation_guide: str = Field(
        description="How to interpret the results"
    )


class ExperimentAnalysis(BaseModel):
    """Structured output for experiment analysis."""
    findings_summary: str = Field(
        description="Summary of what was found"
    )
    supports_hypothesis: bool = Field(
        description="Whether results support the hypothesis"
    )
    confidence_level: str = Field(
        description="How confident we are in this conclusion (low/medium/high)"
    )
    key_insights: List[str] = Field(
        description="Key insights from the results"
    )
    recommended_next_steps: List[str] = Field(
        description="What to do next based on these results"
    )
    should_refine_hypothesis: bool = Field(
        description="Whether the hypothesis should be refined"
    )
    refined_hypothesis: Optional[str] = Field(
        default=None,
        description="Refined hypothesis if applicable"
    )


class IterationDecision(BaseModel):
    """Structured output for deciding whether to continue iterating."""
    should_continue: bool = Field(
        description="Whether to continue with more experiments"
    )
    reason: str = Field(
        description="Reason for this decision"
    )
    next_hypothesis: Optional[str] = Field(
        default=None,
        description="Next hypothesis to test if continuing"
    )
    sufficient_findings: bool = Field(
        description="Whether we have sufficient findings for a report"
    )
