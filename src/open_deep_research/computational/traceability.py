"""Traceability System for Computational Scientific Discovery.

This module provides comprehensive traceability for the entire research process,
enabling humans (or AI reviewers) to:
- Understand exactly what the AI did at each step
- Verify the reasoning and decision-making process
- Replicate the research by following the trace
- Audit the code execution and results
- Review all intermediate outputs and how they were used

The trace captures:
1. Timeline of all events with timestamps
2. All supervisor decisions and reasoning
3. All code executed (including failed attempts and fixes)
4. All outputs generated (with provenance)
5. All knowledge gathered and how it was used
6. Hypothesis evolution and status changes
7. Full conversation history for context
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Event Type Enumeration
# =============================================================================

class TraceEventType(str, Enum):
    """Types of events that can occur during the research process."""
    
    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    
    # Node execution events
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    PERFORMANCE_METRIC = "performance_metric"
    
    # Supervisor decision events
    SUPERVISOR_DECISION = "supervisor_decision"
    SUPERVISOR_REASONING = "supervisor_reasoning"
    TOOL_CALL = "tool_call"
    
    # Knowledge gathering events
    KNOWLEDGE_SEARCH = "knowledge_search"
    KNOWLEDGE_RESULT = "knowledge_result"
    DATA_EXPLORATION = "data_exploration"
    
    # Web retrieval events (contextual_retrieve)
    WEB_RETRIEVAL_START = "web_retrieval_start"
    WEB_RETRIEVAL_COMPLETE = "web_retrieval_complete"
    WEB_RETRIEVAL_NULL = "web_retrieval_null"
    WEB_RETRIEVAL_ERROR = "web_retrieval_error"
    
    # Hypothesis events
    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_STATUS_CHANGE = "hypothesis_status_change"
    
    # Experiment events
    EXPERIMENT_DESIGNED = "experiment_designed"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    
    # Code execution events
    CODE_GENERATED = "code_generated"
    CODE_EXECUTION_START = "code_execution_start"
    CODE_EXECUTION_SUCCESS = "code_execution_success"
    CODE_EXECUTION_FAILURE = "code_execution_failure"
    CODE_FIX_ATTEMPT = "code_fix_attempt"
    
    # Output events
    OUTPUT_GENERATED = "output_generated"
    IMAGE_ANALYZED = "image_analyzed"
    
    # Analysis events
    ANALYSIS_START = "analysis_start"
    ANALYSIS_RESULT = "analysis_result"
    FINDING_RECORDED = "finding_recorded"

    # Claim validation / replication events
    CLAIM_CREATED = "claim_created"
    CLAIM_VALIDATED = "claim_validated"
    CLAIM_REJECTED = "claim_rejected"
    REPLICATION_STARTED = "replication_started"
    REPLICATION_COMPLETED = "replication_completed"
    
    # Report events
    REPORT_GENERATION_START = "report_generation_start"
    REPORT_GENERATED = "report_generated"
    
    # Error events
    ERROR = "error"
    WARNING = "warning"


# =============================================================================
# Trace Event Models
# =============================================================================

class CodeExecutionTrace(BaseModel):
    """Detailed trace of a code execution attempt."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Code details
    code: str = Field(description="The Python code that was executed")
    purpose: str = Field(description="Why this code was executed")
    
    # Execution context
    attempt_number: int = Field(default=1, description="Which attempt this is (1=first try)")
    experiment_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    
    # Results
    success: bool = False
    execution_time_seconds: float = 0.0
    
    # Outputs
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    
    # Generated outputs (output IDs)
    output_ids: List[str] = Field(default_factory=list)
    
    # If this was a fix attempt, reference the previous trace
    previous_attempt_id: Optional[str] = None
    fix_reasoning: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert to markdown for human review."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"### Code Execution [{self.id}] - {status}",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Purpose:** {self.purpose}",
            f"**Attempt:** #{self.attempt_number}",
            f"**Execution Time:** {self.execution_time_seconds:.2f}s",
            "",
            "#### Python Code",
            "```python",
            self.code,
            "```",
            ""
        ]
        
        if self.stdout:
            lines.extend([
                "#### Standard Output",
                "```",
                self.stdout[:5000],
                "```" if len(self.stdout) <= 5000 else f"```\n... (truncated, {len(self.stdout)} total chars)",
                ""
            ])
        
        if self.stderr:
            lines.extend([
                "#### Standard Error",
                "```",
                self.stderr[:2000],
                "```",
                ""
            ])
        
        if not self.success and self.error_message:
            lines.extend([
                "#### Error Message",
                "```",
                self.error_message,
                "```",
                ""
            ])
        
        if self.fix_reasoning:
            lines.extend([
                "#### Fix Reasoning",
                self.fix_reasoning,
                ""
            ])
        
        if self.output_ids:
            lines.extend([
                "#### Generated Outputs",
                ", ".join([f"`{oid}`" for oid in self.output_ids]),
                ""
            ])
        
        return "\n".join(lines)


class SupervisorDecisionTrace(BaseModel):
    """Trace of a supervisor decision."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int = Field(description="Which iteration of the discovery loop")
    
    # State at decision time
    state_summary: str = Field(description="Summary of state when decision was made")
    hypotheses_count: int = 0
    experiments_count: int = 0
    findings_count: int = 0
    
    # Decision details
    action_chosen: str = Field(description="What action was chosen")
    action_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Reasoning (from AI)
    reasoning: Optional[str] = None
    
    # Tool calls made
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert to markdown for human review."""
        lines = [
            f"### Supervisor Decision [{self.id}]",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Iteration:** {self.iteration}",
            f"**Action Chosen:** `{self.action_chosen}`",
            "",
            "#### State at Decision Time",
            f"- Hypotheses: {self.hypotheses_count}",
            f"- Experiments: {self.experiments_count}",
            f"- Findings: {self.findings_count}",
            "",
        ]
        
        if self.reasoning:
            lines.extend([
                "#### Reasoning",
                self.reasoning,
                ""
            ])
        
        if self.action_parameters:
            lines.extend([
                "#### Action Parameters",
                "```json",
                json.dumps(self.action_parameters, indent=2, default=str),
                "```",
                ""
            ])
        
        if self.tool_calls:
            lines.extend([
                "#### Tool Calls",
                "```json",
                json.dumps(self.tool_calls, indent=2, default=str),
                "```",
                ""
            ])
        
        return "\n".join(lines)


class TraceEvent(BaseModel):
    """A single event in the research trace."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: TraceEventType
    
    # Context
    node_name: Optional[str] = None
    iteration: Optional[int] = None
    
    # Event data
    title: str = Field(description="Human-readable title for this event")
    description: str = Field(default="", description="Detailed description")
    
    # Associated data (flexible - can hold various types)
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # References to related entities
    hypothesis_id: Optional[str] = None
    experiment_id: Optional[str] = None
    code_execution_id: Optional[str] = None
    output_ids: List[str] = Field(default_factory=list)
    
    # Metadata
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "node_name": self.node_name,
            "iteration": self.iteration,
            "title": self.title,
            "description": self.description,
            "data": self.data,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "code_execution_id": self.code_execution_id,
            "output_ids": self.output_ids,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown for human review."""
        status_emoji = ""
        if self.success is not None:
            status_emoji = " ✅" if self.success else " ❌"
        
        lines = [
            f"#### [{self.event_type.value}]{status_emoji} {self.title}",
            f"*{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*",
        ]
        
        if self.node_name:
            lines.append(f"**Node:** `{self.node_name}`")
        
        if self.iteration is not None:
            lines.append(f"**Iteration:** {self.iteration}")
        
        if self.duration_seconds:
            lines.append(f"**Duration:** {self.duration_seconds:.2f}s")
        
        if self.description:
            lines.extend(["", self.description])
        
        if self.data:
            # Only show certain keys for readability
            safe_data = {}
            for k, v in self.data.items():
                if isinstance(v, str) and len(v) > 500:
                    safe_data[k] = v[:500] + f"... ({len(v)} chars total)"
                elif isinstance(v, (list, dict)) and len(str(v)) > 500:
                    safe_data[k] = f"[{type(v).__name__} with {len(v)} items]"
                else:
                    safe_data[k] = v
            
            lines.extend([
                "",
                "**Data:**",
                "```json",
                json.dumps(safe_data, indent=2, default=str),
                "```"
            ])
        
        lines.append("")
        return "\n".join(lines)


# =============================================================================
# Research Trace Container
# =============================================================================

class ResearchTrace(BaseModel):
    """Complete trace of a research session.
    
    This is the main container that holds all traceability information
    for a complete research run.
    """
    
    # Identification
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Research context
    research_query: str = Field(default="", description="Original research query")
    research_brief: str = Field(default="", description="Generated research brief")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Run configuration")
    
    # Timeline of events
    events: List[TraceEvent] = Field(default_factory=list)
    
    # Detailed traces
    code_executions: List[CodeExecutionTrace] = Field(
        default_factory=list,
        description="All code execution attempts with full details"
    )
    supervisor_decisions: List[SupervisorDecisionTrace] = Field(
        default_factory=list,
        description="All supervisor decision points"
    )
    
    # Conversation history (for context)
    supervisor_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full supervisor conversation history"
    )
    
    # Summary statistics
    total_iterations: int = 0
    total_code_executions: int = 0
    successful_code_executions: int = 0
    failed_code_executions: int = 0
    total_outputs_generated: int = 0
    total_hypotheses: int = 0
    total_experiments: int = 0
    total_findings: int = 0
    
    # Final results
    final_report: str = ""

    def get_timing_breakdown(self) -> Dict[str, float]:
        """Aggregate elapsed durations by phase/node from event timeline."""
        phase_totals: Dict[str, float] = {}
        for event in self.events:
            if event.duration_seconds is None:
                continue
            phase_name = (
                event.data.get("phase_name")
                or event.data.get("metric_name")
                or event.node_name
                or event.title
            )
            if not phase_name:
                continue
            phase_totals[phase_name] = phase_totals.get(phase_name, 0.0) + float(event.duration_seconds)
        return dict(sorted(phase_totals.items(), key=lambda item: item[1], reverse=True))
    
    def add_event(
        self,
        event_type: TraceEventType,
        title: str,
        description: str = "",
        node_name: str = None,
        iteration: int = None,
        data: Dict[str, Any] = None,
        hypothesis_id: str = None,
        experiment_id: str = None,
        code_execution_id: str = None,
        output_ids: List[str] = None,
        duration_seconds: float = None,
        success: bool = None,
    ) -> TraceEvent:
        """Add a new event to the trace."""
        event = TraceEvent(
            event_type=event_type,
            title=title,
            description=description,
            node_name=node_name,
            iteration=iteration,
            data=data or {},
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            code_execution_id=code_execution_id,
            output_ids=output_ids or [],
            duration_seconds=duration_seconds,
            success=success,
        )
        self.events.append(event)
        return event
    
    def add_code_execution(self, trace: CodeExecutionTrace):
        """Add a code execution trace."""
        self.code_executions.append(trace)
        self.total_code_executions += 1
        if trace.success:
            self.successful_code_executions += 1
        else:
            self.failed_code_executions += 1
    
    def add_supervisor_decision(self, decision: SupervisorDecisionTrace):
        """Add a supervisor decision trace."""
        self.supervisor_decisions.append(decision)
    
    def finalize(self, final_report: str = ""):
        """Mark the trace as complete."""
        self.completed_at = datetime.now()
        self.final_report = final_report
        
        # Add completion event
        duration = (self.completed_at - self.created_at).total_seconds()
        self.add_event(
            event_type=TraceEventType.WORKFLOW_END,
            title="Research Workflow Complete",
            description=f"Completed after {duration:.1f} seconds",
            data={
                "duration_seconds": duration,
                "total_iterations": self.total_iterations,
                "total_code_executions": self.total_code_executions,
                "successful_code_executions": self.successful_code_executions,
                "total_outputs_generated": self.total_outputs_generated,
                "total_hypotheses": self.total_hypotheses,
                "total_experiments": self.total_experiments,
                "total_findings": self.total_findings,
            },
            success=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "research_query": self.research_query,
            "research_brief": self.research_brief,
            "config": self.config,
            "events": [e.to_dict() for e in self.events],
            "code_executions": [ce.model_dump() for ce in self.code_executions],
            "supervisor_decisions": [sd.model_dump() for sd in self.supervisor_decisions],
            "supervisor_messages": self.supervisor_messages,
            "summary": {
                "total_iterations": self.total_iterations,
                "total_code_executions": self.total_code_executions,
                "successful_code_executions": self.successful_code_executions,
                "failed_code_executions": self.failed_code_executions,
                "total_outputs_generated": self.total_outputs_generated,
                "total_hypotheses": self.total_hypotheses,
                "total_experiments": self.total_experiments,
                "total_findings": self.total_findings,
                "timing_breakdown_seconds": self.get_timing_breakdown(),
            },
            "final_report": self.final_report,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_markdown(self) -> str:
        """Convert to comprehensive markdown document for human review."""
        duration = 0
        if self.completed_at:
            duration = (self.completed_at - self.created_at).total_seconds()
        
        lines = [
            "# Research Traceability Report",
            "",
            f"**Trace ID:** `{self.trace_id}`",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if self.completed_at:
            lines.extend([
                f"**Completed:** {self.completed_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Duration:** {duration/60:.1f} minutes ({duration:.1f} seconds)",
            ])
        
        lines.extend([
            "",
            "---",
            "",
            "## Summary Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Iterations | {self.total_iterations} |",
            f"| Total Code Executions | {self.total_code_executions} |",
            f"| Successful Code Executions | {self.successful_code_executions} |",
            f"| Failed Code Executions | {self.failed_code_executions} |",
            f"| Total Outputs Generated | {self.total_outputs_generated} |",
            f"| Total Hypotheses | {self.total_hypotheses} |",
            f"| Total Experiments | {self.total_experiments} |",
            f"| Total Findings | {self.total_findings} |",
            "",
        ])

        timing_breakdown = self.get_timing_breakdown()
        if timing_breakdown:
            lines.extend([
                "### Timing Breakdown (Top Phases)",
                "",
                "| Phase | Duration (s) |",
                "|-------|--------------|",
            ])
            for phase_name, elapsed in list(timing_breakdown.items())[:25]:
                lines.append(f"| {phase_name} | {elapsed:.2f} |")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Research Query",
            "",
            self.research_query or "*No query recorded*",
            "",
        ])
        
        if self.research_brief:
            lines.extend([
                "## Research Brief",
                "",
                self.research_brief,
                "",
            ])
        
        if self.config:
            lines.extend([
                "## Configuration",
                "",
                "```json",
                json.dumps(self.config, indent=2, default=str),
                "```",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "## Event Timeline",
            "",
        ])
        
        for event in self.events:
            lines.append(event.to_markdown())
        
        if self.supervisor_decisions:
            lines.extend([
                "---",
                "",
                "## Supervisor Decisions",
                "",
            ])
            for decision in self.supervisor_decisions:
                lines.append(decision.to_markdown())
        
        if self.code_executions:
            lines.extend([
                "---",
                "",
                "## Code Execution Details",
                "",
            ])
            for code_exec in self.code_executions:
                lines.append(code_exec.to_markdown())
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert to HTML document for human review with better formatting."""
        import html
        
        duration = 0
        if self.completed_at:
            duration = (self.completed_at - self.created_at).total_seconds()
        
        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Research Trace - {self.trace_id}</title>",
            "<style>",
            """
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
                   max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }
            h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }
            h2 { color: #16213e; margin-top: 30px; }
            h3 { color: #0f3460; }
            .event { background: #f8f9fa; border-left: 4px solid #4361ee; padding: 15px; margin: 10px 0; }
            .event.success { border-left-color: #2ecc71; }
            .event.failure { border-left-color: #e74c3c; }
            .event-header { font-weight: bold; color: #1a1a2e; }
            .event-time { color: #666; font-size: 0.9em; }
            .code-block { background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 5px; 
                          overflow-x: auto; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; }
            .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            .stats-table th { background: #16213e; color: white; }
            .stats-table tr:nth-child(even) { background: #f2f2f2; }
            .decision-box { background: #e8f4f8; border: 1px solid #3498db; border-radius: 5px; padding: 15px; margin: 15px 0; }
            .code-exec-box { background: #f5f5f5; border: 1px solid #95a5a6; border-radius: 5px; padding: 15px; margin: 15px 0; }
            .code-exec-box.success { border-color: #27ae60; background: #e8f8f5; }
            .code-exec-box.failure { border-color: #e74c3c; background: #fdedec; }
            .section { margin: 30px 0; padding: 20px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-right: 5px; }
            .tag.node { background: #3498db; color: white; }
            .tag.iteration { background: #9b59b6; color: white; }
            .collapsible { cursor: pointer; padding: 10px; background: #eee; border: none; width: 100%; text-align: left; }
            .collapsible:after { content: '+'; float: right; }
            .collapsible.active:after { content: '-'; }
            .content { max-height: 0; overflow: hidden; transition: max-height 0.3s; }
            .content.show { max-height: none; }
            """,
            "</style>",
            "</head>",
            "<body>",
            f"<h1>🔬 Research Traceability Report</h1>",
            f"<p><strong>Trace ID:</strong> <code>{self.trace_id}</code></p>",
            f"<p><strong>Created:</strong> {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]
        
        if self.completed_at:
            html_parts.extend([
                f"<p><strong>Completed:</strong> {self.completed_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
                f"<p><strong>Duration:</strong> {duration/60:.1f} minutes ({duration:.1f} seconds)</p>",
            ])
        
        # Summary Statistics
        html_parts.extend([
            "<div class='section'>",
            "<h2>📊 Summary Statistics</h2>",
            "<table class='stats-table'>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Total Iterations</td><td>{self.total_iterations}</td></tr>",
            f"<tr><td>Total Code Executions</td><td>{self.total_code_executions}</td></tr>",
            f"<tr><td>Successful Code Executions</td><td>{self.successful_code_executions}</td></tr>",
            f"<tr><td>Failed Code Executions</td><td>{self.failed_code_executions}</td></tr>",
            f"<tr><td>Total Outputs Generated</td><td>{self.total_outputs_generated}</td></tr>",
            f"<tr><td>Total Hypotheses</td><td>{self.total_hypotheses}</td></tr>",
            f"<tr><td>Total Experiments</td><td>{self.total_experiments}</td></tr>",
            f"<tr><td>Total Findings</td><td>{self.total_findings}</td></tr>",
            "</table>",
            "</div>",
        ])
        
        # Research Query
        if self.research_query:
            html_parts.extend([
                "<div class='section'>",
                "<h2>🔍 Research Query</h2>",
                f"<p>{html.escape(self.research_query)}</p>",
                "</div>",
            ])
        
        # Research Brief
        if self.research_brief:
            html_parts.extend([
                "<div class='section'>",
                "<h2>📋 Research Brief</h2>",
                f"<pre style='white-space: pre-wrap;'>{html.escape(self.research_brief)}</pre>",
                "</div>",
            ])
        
        # Event Timeline
        html_parts.extend([
            "<div class='section'>",
            "<h2>📅 Event Timeline</h2>",
        ])
        
        for event in self.events:
            success_class = ""
            if event.success is True:
                success_class = "success"
            elif event.success is False:
                success_class = "failure"
            
            status_icon = ""
            if event.success is True:
                status_icon = "✅ "
            elif event.success is False:
                status_icon = "❌ "
            
            html_parts.extend([
                f"<div class='event {success_class}'>",
                f"<div class='event-header'>{status_icon}[{event.event_type.value}] {html.escape(event.title)}</div>",
                f"<div class='event-time'>{event.timestamp.strftime('%H:%M:%S')}</div>",
            ])
            
            if event.node_name:
                html_parts.append(f"<span class='tag node'>{event.node_name}</span>")
            if event.iteration is not None:
                html_parts.append(f"<span class='tag iteration'>Iteration {event.iteration}</span>")
            
            if event.description:
                html_parts.append(f"<p>{html.escape(event.description)}</p>")
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        
        # Code Executions
        if self.code_executions:
            html_parts.extend([
                "<div class='section'>",
                "<h2>💻 Code Execution Details</h2>",
            ])
            
            for i, code_exec in enumerate(self.code_executions):
                status_class = "success" if code_exec.success else "failure"
                status_text = "SUCCESS" if code_exec.success else "FAILED"
                
                html_parts.extend([
                    f"<div class='code-exec-box {status_class}'>",
                    f"<h3>Execution #{i+1} [{code_exec.id}] - {status_text}</h3>",
                    f"<p><strong>Purpose:</strong> {html.escape(code_exec.purpose)}</p>",
                    f"<p><strong>Attempt:</strong> #{code_exec.attempt_number}</p>",
                    f"<p><strong>Execution Time:</strong> {code_exec.execution_time_seconds:.2f}s</p>",
                    "<h4>Python Code</h4>",
                    f"<pre class='code-block'>{html.escape(code_exec.code)}</pre>",
                ])
                
                if code_exec.stdout:
                    stdout_display = code_exec.stdout[:3000]
                    if len(code_exec.stdout) > 3000:
                        stdout_display += f"\n... (truncated, {len(code_exec.stdout)} total chars)"
                    html_parts.extend([
                        "<h4>Standard Output</h4>",
                        f"<pre class='code-block'>{html.escape(stdout_display)}</pre>",
                    ])
                
                if not code_exec.success and code_exec.error_message:
                    html_parts.extend([
                        "<h4>Error Message</h4>",
                        f"<pre class='code-block' style='color: #e74c3c;'>{html.escape(code_exec.error_message)}</pre>",
                    ])
                
                if code_exec.fix_reasoning:
                    html_parts.extend([
                        "<h4>Fix Reasoning</h4>",
                        f"<p>{html.escape(code_exec.fix_reasoning)}</p>",
                    ])
                
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        # Supervisor Decisions
        if self.supervisor_decisions:
            html_parts.extend([
                "<div class='section'>",
                "<h2>🧠 Supervisor Decisions</h2>",
            ])
            
            for decision in self.supervisor_decisions:
                html_parts.extend([
                    "<div class='decision-box'>",
                    f"<h3>Decision [{decision.id}] - Iteration {decision.iteration}</h3>",
                    f"<p><strong>Action:</strong> <code>{html.escape(decision.action_chosen)}</code></p>",
                    f"<p><strong>State:</strong> {decision.hypotheses_count} hypotheses, "
                    f"{decision.experiments_count} experiments, {decision.findings_count} findings</p>",
                ])
                
                if decision.reasoning:
                    html_parts.extend([
                        "<h4>Reasoning</h4>",
                        f"<p>{html.escape(decision.reasoning)}</p>",
                    ])
                
                if decision.action_parameters:
                    html_parts.extend([
                        "<h4>Action Parameters</h4>",
                        f"<pre class='code-block'>{html.escape(json.dumps(decision.action_parameters, indent=2, default=str))}</pre>",
                    ])
                
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_parts.extend([
            "<script>",
            """
            document.querySelectorAll('.collapsible').forEach(btn => {
                btn.addEventListener('click', function() {
                    this.classList.toggle('active');
                    const content = this.nextElementSibling;
                    content.classList.toggle('show');
                });
            });
            """,
            "</script>",
            "</body>",
            "</html>",
        ])
        
        return "\n".join(html_parts)


# =============================================================================
# Global Trace Manager
# =============================================================================

class TraceManager:
    """Singleton manager for the current research trace.
    
    This provides a global access point to add events to the current trace
    from any node in the graph.
    """
    
    _instance: Optional['TraceManager'] = None
    _current_trace: Optional[ResearchTrace] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'TraceManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def start_trace(
        cls,
        research_query: str = "",
        config: Dict[str, Any] = None
    ) -> ResearchTrace:
        """Start a new research trace."""
        manager = cls.get_instance()
        manager._current_trace = ResearchTrace(
            research_query=research_query,
            config=config or {}
        )
        
        # Add start event
        manager._current_trace.add_event(
            event_type=TraceEventType.WORKFLOW_START,
            title="Research Workflow Started",
            description=f"Starting research: {research_query[:200]}..." if len(research_query) > 200 else f"Starting research: {research_query}",
            data={"config": config or {}},
            success=True
        )
        
        return manager._current_trace
    
    @classmethod
    def get_trace(cls) -> Optional[ResearchTrace]:
        """Get the current trace."""
        manager = cls.get_instance()
        return manager._current_trace
    
    @classmethod
    def set_trace(cls, trace: ResearchTrace):
        """Set the current trace (useful for restoring from state)."""
        manager = cls.get_instance()
        manager._current_trace = trace
    
    @classmethod
    def add_event(cls, **kwargs) -> Optional[TraceEvent]:
        """Add an event to the current trace."""
        manager = cls.get_instance()
        if manager._current_trace:
            return manager._current_trace.add_event(**kwargs)
        return None
    
    @classmethod
    def add_code_execution(cls, trace: CodeExecutionTrace):
        """Add a code execution trace."""
        manager = cls.get_instance()
        if manager._current_trace:
            manager._current_trace.add_code_execution(trace)
    
    @classmethod
    def add_supervisor_decision(cls, decision: SupervisorDecisionTrace):
        """Add a supervisor decision trace."""
        manager = cls.get_instance()
        if manager._current_trace:
            manager._current_trace.add_supervisor_decision(decision)
    
    @classmethod
    def finalize(cls, final_report: str = "") -> Optional[ResearchTrace]:
        """Finalize the current trace."""
        manager = cls.get_instance()
        if manager._current_trace:
            manager._current_trace.finalize(final_report)
            return manager._current_trace
        return None
    
    @classmethod
    def reset(cls):
        """Reset the trace manager."""
        manager = cls.get_instance()
        manager._current_trace = None


# =============================================================================
# Convenience Functions
# =============================================================================

def trace_node_enter(node_name: str, iteration: int = None, data: Dict[str, Any] = None):
    """Record entering a node."""
    TraceManager.add_event(
        event_type=TraceEventType.NODE_ENTER,
        title=f"Entering {node_name}",
        node_name=node_name,
        iteration=iteration,
        data=data or {}
    )


def trace_node_exit(node_name: str, iteration: int = None, success: bool = True, data: Dict[str, Any] = None):
    """Record exiting a node."""
    TraceManager.add_event(
        event_type=TraceEventType.NODE_EXIT,
        title=f"Exiting {node_name}",
        node_name=node_name,
        iteration=iteration,
        success=success,
        data=data or {}
    )


def trace_phase_start(
    phase_name: str,
    node_name: str = None,
    iteration: int = None,
    data: Dict[str, Any] = None,
):
    """Record the start of a timed phase."""
    TraceManager.add_event(
        event_type=TraceEventType.PHASE_START,
        title=f"Phase Started: {phase_name}",
        description=f"Started phase '{phase_name}'",
        node_name=node_name,
        iteration=iteration,
        data={"phase_name": phase_name, **(data or {})},
    )


def trace_phase_end(
    phase_name: str,
    duration_seconds: float,
    success: bool = True,
    node_name: str = None,
    iteration: int = None,
    data: Dict[str, Any] = None,
):
    """Record the end of a timed phase."""
    TraceManager.add_event(
        event_type=TraceEventType.PHASE_END,
        title=f"Phase Completed: {phase_name}",
        description=f"Completed phase '{phase_name}' in {duration_seconds:.2f}s",
        node_name=node_name,
        iteration=iteration,
        duration_seconds=duration_seconds,
        success=success,
        data={"phase_name": phase_name, **(data or {})},
    )


def trace_performance_metric(
    metric_name: str,
    duration_seconds: float,
    node_name: str = None,
    iteration: int = None,
    data: Dict[str, Any] = None,
):
    """Record granular performance metrics for profiling."""
    TraceManager.add_event(
        event_type=TraceEventType.PERFORMANCE_METRIC,
        title=f"Performance Metric: {metric_name}",
        description=f"{metric_name} took {duration_seconds:.2f}s",
        node_name=node_name,
        iteration=iteration,
        duration_seconds=duration_seconds,
        success=True,
        data={"metric_name": metric_name, **(data or {})},
    )


def trace_code_execution(
    code: str,
    purpose: str,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    error_message: str = None,
    execution_time: float = 0.0,
    attempt_number: int = 1,
    experiment_id: str = None,
    hypothesis_id: str = None,
    output_ids: List[str] = None,
    previous_attempt_id: str = None,
    fix_reasoning: str = None,
) -> CodeExecutionTrace:
    """Record a code execution and add to trace."""
    trace = CodeExecutionTrace(
        code=code,
        purpose=purpose,
        success=success,
        stdout=stdout,
        stderr=stderr,
        error_message=error_message,
        execution_time_seconds=execution_time,
        attempt_number=attempt_number,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        output_ids=output_ids or [],
        previous_attempt_id=previous_attempt_id,
        fix_reasoning=fix_reasoning,
    )
    TraceManager.add_code_execution(trace)
    
    # Also add as event
    event_type = TraceEventType.CODE_EXECUTION_SUCCESS if success else TraceEventType.CODE_EXECUTION_FAILURE
    TraceManager.add_event(
        event_type=event_type,
        title=f"Code Execution {'Succeeded' if success else 'Failed'} (Attempt #{attempt_number})",
        description=purpose,
        code_execution_id=trace.id,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        output_ids=output_ids or [],
        duration_seconds=execution_time,
        success=success,
        data={
            "code_length": len(code),
            "stdout_length": len(stdout),
            "error": error_message if not success else None,
        }
    )
    
    return trace


def trace_supervisor_decision(
    iteration: int,
    action: str,
    action_params: Dict[str, Any],
    state_summary: str,
    reasoning: str = None,
    tool_calls: List[Dict[str, Any]] = None,
    hypotheses_count: int = 0,
    experiments_count: int = 0,
    findings_count: int = 0,
) -> SupervisorDecisionTrace:
    """Record a supervisor decision and add to trace."""
    decision = SupervisorDecisionTrace(
        iteration=iteration,
        action_chosen=action,
        action_parameters=action_params,
        state_summary=state_summary,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        hypotheses_count=hypotheses_count,
        experiments_count=experiments_count,
        findings_count=findings_count,
    )
    TraceManager.add_supervisor_decision(decision)
    
    # Also add as event
    TraceManager.add_event(
        event_type=TraceEventType.SUPERVISOR_DECISION,
        title=f"Supervisor Decision: {action}",
        description=reasoning or f"Chose action: {action}",
        iteration=iteration,
        data={
            "action": action,
            "parameters": action_params,
            "state": {
                "hypotheses": hypotheses_count,
                "experiments": experiments_count,
                "findings": findings_count,
            }
        }
    )
    
    return decision


def trace_hypothesis_created(hypothesis_id: str, statement: str, rationale: str = ""):
    """Record hypothesis creation."""
    trace = TraceManager.get_trace()
    if trace:
        trace.total_hypotheses += 1
    
    TraceManager.add_event(
        event_type=TraceEventType.HYPOTHESIS_CREATED,
        title=f"Hypothesis Created: {statement[:80]}...",
        description=rationale,
        hypothesis_id=hypothesis_id,
        data={"statement": statement, "rationale": rationale}
    )


def trace_experiment_started(experiment_id: str, hypothesis_id: str, objective: str):
    """Record experiment start."""
    trace = TraceManager.get_trace()
    if trace:
        trace.total_experiments += 1
    
    TraceManager.add_event(
        event_type=TraceEventType.EXPERIMENT_STARTED,
        title=f"Experiment Started: {objective[:80]}...",
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        data={"objective": objective}
    )


def trace_finding_recorded(finding_id: str, statement: str, is_novel: bool = False):
    """Record a finding."""
    trace = TraceManager.get_trace()
    if trace:
        trace.total_findings += 1
    
    TraceManager.add_event(
        event_type=TraceEventType.FINDING_RECORDED,
        title=f"Finding: {statement[:80]}...",
        data={"statement": statement, "is_novel": is_novel}
    )


def trace_claim_created(claim_id: str, finding_id: str, claim_text: str):
    """Record claim creation for the evidence ledger."""
    TraceManager.add_event(
        event_type=TraceEventType.CLAIM_CREATED,
        title=f"Claim Created: {claim_text[:80]}...",
        data={"claim_id": claim_id, "finding_id": finding_id, "claim_text": claim_text[:500]}
    )


def trace_claim_validated(claim_id: str, reason: str, novelty_confidence: float):
    """Record claim validation success."""
    TraceManager.add_event(
        event_type=TraceEventType.CLAIM_VALIDATED,
        title=f"Claim Validated: {claim_id}",
        description=reason,
        data={"claim_id": claim_id, "novelty_confidence": novelty_confidence},
        success=True,
    )


def trace_claim_rejected(claim_id: str, reason: str, novelty_confidence: float):
    """Record claim rejection/inconclusive outcome."""
    TraceManager.add_event(
        event_type=TraceEventType.CLAIM_REJECTED,
        title=f"Claim Rejected: {claim_id}",
        description=reason,
        data={"claim_id": claim_id, "novelty_confidence": novelty_confidence},
        success=False,
    )


def trace_replication_check(claim_id: str, status: str, details: Dict[str, Any]):
    """Record replication lifecycle events for a claim."""
    event_type = (
        TraceEventType.REPLICATION_STARTED
        if status == "started"
        else TraceEventType.REPLICATION_COMPLETED
    )
    success = details.get("passed") if status != "started" else None
    TraceManager.add_event(
        event_type=event_type,
        title=f"Replication {status.title()}: {claim_id}",
        data={"claim_id": claim_id, **details},
        success=success,
    )


def trace_output_generated(output_id: str, output_type: str, description: str = ""):
    """Record output generation."""
    trace = TraceManager.get_trace()
    if trace:
        trace.total_outputs_generated += 1
    
    TraceManager.add_event(
        event_type=TraceEventType.OUTPUT_GENERATED,
        title=f"Output Generated: {output_type}",
        description=description,
        output_ids=[output_id],
        data={"output_type": output_type}
    )


def trace_web_retrieval(
    queries: List[str],
    mode: str,
    sources_accepted: int,
    sources_rejected: int,
    saved_path: Optional[str] = None,
    source_urls: Optional[List[str]] = None,
    status: str = "complete",
):
    """Record a web retrieval event (contextual_retrieve call).

    Args:
        queries: Search queries executed.
        mode: Retrieval mode (exploratory, verification, methods, novelty).
        sources_accepted: Number of sources that passed scoring.
        sources_rejected: Number of sources rejected by scoring.
        saved_path: Path where results were persisted (if any).
        source_urls: URLs of accepted sources.
        status: 'complete', 'null', or 'error'.
    """
    event_type_map = {
        "complete": TraceEventType.WEB_RETRIEVAL_COMPLETE,
        "null": TraceEventType.WEB_RETRIEVAL_NULL,
        "error": TraceEventType.WEB_RETRIEVAL_ERROR,
    }
    event_type = event_type_map.get(status, TraceEventType.WEB_RETRIEVAL_COMPLETE)

    title = f"Web Retrieval [{mode}]: {sources_accepted} accepted, {sources_rejected} rejected"
    if status == "null":
        title = f"Web Retrieval [{mode}]: No evidence found"
    elif status == "error":
        title = f"Web Retrieval [{mode}]: Error"

    TraceManager.add_event(
        event_type=event_type,
        title=title,
        data={
            "queries": queries,
            "mode": mode,
            "sources_accepted": sources_accepted,
            "sources_rejected": sources_rejected,
            "source_urls": source_urls or [],
            "saved_path": saved_path or "",
        },
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "TraceEventType",
    "TraceEvent",
    "CodeExecutionTrace",
    "SupervisorDecisionTrace",
    "ResearchTrace",
    "TraceManager",
    "trace_node_enter",
    "trace_node_exit",
    "trace_phase_start",
    "trace_phase_end",
    "trace_performance_metric",
    "trace_code_execution",
    "trace_supervisor_decision",
    "trace_hypothesis_created",
    "trace_experiment_started",
    "trace_finding_recorded",
    "trace_claim_created",
    "trace_claim_validated",
    "trace_claim_rejected",
    "trace_replication_check",
    "trace_output_generated",
    "trace_web_retrieval",
]
