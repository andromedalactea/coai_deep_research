#!/usr/bin/env python3
"""
Computational Scientific Discovery - Direct Runner

Run the discovery system directly from command line with progress monitoring.
No LangGraph Studio needed - runs entirely in Python.

Usage:
    # Run with a query string
    python run_discovery.py "Your research question here"
    
    # Run with a prompt file
    python run_discovery.py --file examples/biosignature-discovery-prompt.txt
    
    # Run with verbose output (see all steps)
    python run_discovery.py --verbose "Your query"
    
    # Run with custom iterations
    python run_discovery.py --iterations 5 "Your query"
    
    # Quick test mode (fewer iterations)
    python run_discovery.py --quick "Your query"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Import traceability system
from open_deep_research.computational.traceability import (
    TraceManager,
    ResearchTrace,
)


# =============================================================================
# Progress Callback - Monitor Discovery in Real-Time
# =============================================================================

class DiscoveryMonitor:
    """Monitor and display discovery progress in real-time."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = None
        self.current_node = None
        self.iterations = 0
        self.experiments_run = 0
        self.outputs_generated = 0
        self.last_action = ""
        
    def start(self):
        """Mark discovery start."""
        self.start_time = datetime.now()
        print("\n" + "="*70)
        print("   COMPUTATIONAL SCIENTIFIC DISCOVERY - RUNNING")
        print("="*70)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*70 + "\n")
        
    def update(self, event: dict):
        """Process a streaming event from the graph."""
        # Extract node name and data
        for node_name, node_data in event.items():
            if node_name == "__start__":
                continue
                
            self.current_node = node_name
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            # Track key metrics
            if node_name == "discovery_supervisor":
                self.iterations += 1
                
            if node_name == "run_experiment":
                self.experiments_run += 1
                
            # Count outputs
            if isinstance(node_data, dict):
                if "all_outputs" in node_data:
                    self.outputs_generated = len(node_data.get("all_outputs", {}))
            
            # Display progress
            self._display_progress(node_name, node_data, elapsed)
    
    def _display_progress(self, node_name: str, data: dict, elapsed: float):
        """Display current progress."""
        # Node emoji mapping
        emojis = {
            "clarify_discovery_query": "❓",
            "generate_research_brief": "📋",
            "discovery_supervisor": "🧠",
            "supervisor_tools": "🔧",
            "gather_knowledge": "📚",
            "run_experiment": "🔬",
            "analyze_results": "📊",
            "synthesize_findings": "📝",
        }
        
        emoji = emojis.get(node_name, "▶")
        
        # Status line
        status = (
            f"[{elapsed:6.1f}s] {emoji} {node_name:25s} | "
            f"Iter: {self.iterations} | "
            f"Experiments: {self.experiments_run} | "
            f"Outputs: {self.outputs_generated}"
        )
        print(status)
        
        # Verbose output
        if self.verbose and isinstance(data, dict):
            try:
                # Show key information from the data
                if "supervisor_messages" in data:
                    msgs = data.get("supervisor_messages", [])
                    # Handle both list and dict cases
                    if isinstance(msgs, list) and msgs:
                        last_msg = msgs[-1]
                        if hasattr(last_msg, 'content'):
                            content = str(last_msg.content)[:200]
                            print(f"         └─ {content}...")
                        
                if "_experiment_hypothesis" in data:
                    hyp = data.get('_experiment_hypothesis', '')
                    if hyp:
                        print(f"         └─ Testing: {str(hyp)[:100]}...")
                    
                if "computation_results" in data:
                    results = data.get("computation_results", [])
                    if isinstance(results, list) and results:
                        latest = results[-1]
                        if hasattr(latest, 'success'):
                            status = "✅ Success" if latest.success else "❌ Failed"
                            print(f"         └─ Code execution: {status}")
                            
                            # Show complete code executed
                            if hasattr(latest, 'code_executed') and latest.code_executed:
                                print(f"\n{'='*70}")
                                print("PYTHON CODE EXECUTED:")
                                print('='*70)
                                print(latest.code_executed)
                                print('='*70 + "\n")
                            
                            # Show error details if failed
                            if not latest.success and hasattr(latest, 'error_message') and latest.error_message:
                                print(f"\n{'='*70}")
                                print("ERROR MESSAGE:")
                                print('='*70)
                                print(latest.error_message)
                                print('='*70 + "\n")
                            
                            # Show stdout/stderr if available
                            if hasattr(latest, 'outputs') and latest.outputs:
                                for output in latest.outputs:
                                    if hasattr(output, 'output_type'):
                                        otype = output.output_type.value if hasattr(output.output_type, 'value') else str(output.output_type)
                                        if otype == 'text' or otype == 'log':
                                            if hasattr(output, 'text_content') and output.text_content:
                                                print(f"\n{'='*70}")
                                                print(f"OUTPUT - {output.description if hasattr(output, 'description') else 'Text Output'}:")
                                                print('='*70)
                                                print(output.text_content)
                                                print('='*70 + "\n")
                            
                            # Show output count and types summary
                            if hasattr(latest, 'outputs') and latest.outputs:
                                output_types = {}
                                vision_analyzed = 0
                                for o in latest.outputs:
                                    otype = getattr(o, 'output_type', 'unknown')
                                    otype_str = otype.value if hasattr(otype, 'value') else str(otype)
                                    output_types[otype_str] = output_types.get(otype_str, 0) + 1
                                    # Count vision-analyzed images
                                    if otype_str == 'image' and hasattr(o, 'interpretation') and o.interpretation:
                                        vision_analyzed += 1
                                type_summary = ", ".join([f"{k}: {v}" for k, v in output_types.items()])
                                print(f"         └─ Generated {len(latest.outputs)} outputs ({type_summary})")
                                
                                # Show vision analysis status
                                if vision_analyzed > 0:
                                    print(f"         └─ 🖼️ Vision analysis: {vision_analyzed} image(s) analyzed by AI")
                                
                                # Show brief vision analysis for each image
                                for o in latest.outputs:
                                    if hasattr(o, 'interpretation') and o.interpretation:
                                        label = getattr(o, 'citation_label', 'Image')
                                        # First line of interpretation
                                        first_line = o.interpretation.split('\n')[0][:100]
                                        print(f"             └─ {label}: {first_line}...")
                            
                if "findings" in data:
                    findings = data.get("findings", [])
                    if isinstance(findings, list) and findings:
                        latest = findings[-1]
                        if hasattr(latest, 'statement'):
                            print(f"         └─ Finding: {str(latest.statement)[:100]}...")
            except Exception as e:
                # Don't let verbose output crash the monitoring
                pass
    
    def finish(self, result: dict):
        """Mark discovery complete and show summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "-"*70)
        print("   DISCOVERY COMPLETE")
        print("-"*70)
        print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Iterations: {self.iterations}")
        print(f"Experiments run: {self.experiments_run}")
        print(f"Outputs generated: {self.outputs_generated}")
        
        # Count specific outputs
        all_outputs = result.get("all_outputs", {})
        # Handle both Pydantic objects and dicts (due to serialization)
        def is_image_output(o):
            if hasattr(o, 'output_type'):
                return o.output_type.value == 'image'
            elif isinstance(o, dict) and 'output_type' in o:
                ot = o['output_type']
                return (ot == 'image' or (hasattr(ot, 'value') and ot.value == 'image'))
            return False
        images = sum(1 for o in all_outputs.values() if is_image_output(o))
        print(f"  - Images/Figures: {images}")
        
        hypotheses = result.get("hypotheses", [])
        print(f"Hypotheses tested: {len(hypotheses)}")
        
        findings = result.get("findings", [])
        print(f"Key findings: {len(findings)}")
        
        print("-"*70)


# =============================================================================
# Main Runner
# =============================================================================

async def run_discovery_with_monitoring(
    query: str,
    config: dict,
    verbose: bool = False,
    output_dir: Path = None
):
    """Run discovery with real-time progress monitoring."""
    
    from open_deep_research.computational.discovery_graph import computational_discovery
    from langchain_core.messages import HumanMessage
    
    # ==========================================================================
    # TRACEABILITY: Initialize the research trace
    # ==========================================================================
    trace = TraceManager.start_trace(
        research_query=query,
        config=config
    )
    print(f"\n🔍 Trace ID: {trace.trace_id}")
    
    # Initialize monitor
    monitor = DiscoveryMonitor(verbose=verbose)
    monitor.start()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Stream the execution
    final_result = None
    
    try:
        async for event in computational_discovery.astream(
            initial_state,
            config,
            stream_mode="updates"
        ):
            monitor.update(event)
            
            # Keep track of the latest state for final result
            for node_name, node_data in event.items():
                if isinstance(node_data, dict):
                    if final_result is None:
                        final_result = {}
                    final_result.update(node_data)
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  Discovery interrupted by user")
        print("Partial results may be available.")
        
    except Exception as e:
        print(f"\n\n❌ Discovery failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Show completion summary
    if final_result:
        monitor.finish(final_result)
        
        # Save outputs
        if output_dir:
            await save_outputs(final_result, output_dir)
    
    return final_result


async def save_research_trace(result: dict, run_dir: Path, timestamp: str):
    """Save comprehensive research trace for traceability and reproducibility.
    
    Creates:
    - trace_{timestamp}.json: Complete JSON trace for programmatic access
    - trace_{timestamp}.md: Human-readable markdown trace
    - trace_{timestamp}.html: Interactive HTML trace viewer
    - code_executions/: Directory with individual code execution files
    """
    
    print(f"\n📜 Saving Research Trace (for human/AI review)...")
    
    # Build comprehensive trace from state data
    trace_data = {
        "trace_id": result.get("trace_id", ""),
        "timestamp": timestamp,
        "created_at": result.get("trace_started_at", datetime.now().isoformat()),
        "completed_at": datetime.now().isoformat(),
        
        # Research context
        "research_query": result.get("research_query", ""),
        "research_brief": result.get("research_brief", ""),
        
        # Timeline of events
        "events": result.get("trace_events", []),
        
        # Code execution traces (most important for reproducibility)
        "code_executions": result.get("code_execution_traces", []),
        
        # Supervisor decision traces
        "supervisor_decisions": result.get("supervisor_decision_traces", []),
        
        # Summary statistics
        "summary": {
            "total_iterations": result.get("discovery_iterations", 0),
            "total_code_executions": len(result.get("code_execution_traces", [])),
            "successful_code_executions": sum(
                1 for ce in result.get("code_execution_traces", []) 
                if ce.get("success", False)
            ),
            "total_hypotheses": len(result.get("hypotheses", [])),
            "total_experiments": len(result.get("experiments", [])),
            "total_findings": len(result.get("findings", [])),
            "total_outputs": len(result.get("all_outputs", {})),
        },
        
        # Detailed records
        "hypotheses": [],
        "experiments": [],
        "findings": [],
        
        # Final report
        "final_report": result.get("final_report", ""),
    }
    
    # Process hypotheses
    for h in result.get("hypotheses", []):
        if hasattr(h, 'statement'):
            trace_data["hypotheses"].append({
                "id": h.id,
                "statement": h.statement,
                "rationale": h.rationale,
                "status": h.status.value if hasattr(h.status, 'value') else str(h.status),
                "supporting_evidence": h.supporting_evidence,
            })
        elif isinstance(h, dict):
            trace_data["hypotheses"].append(h)
    
    # Process experiments
    for exp in result.get("experiments", []):
        if hasattr(exp, 'id'):
            exp_data = {
                "id": exp.id,
                "hypothesis_id": exp.hypothesis_id,
                "objective": exp.design.objective if exp.design else "",
                "methodology": exp.design.methodology if exp.design else "",
                "status": exp.status.value if hasattr(exp.status, 'value') else str(exp.status),
                "code_executed": exp.code_executed,
                "findings_summary": exp.findings_summary,
                "supports_hypothesis": exp.supports_hypothesis,
            }
            trace_data["experiments"].append(exp_data)
        elif isinstance(exp, dict):
            trace_data["experiments"].append(exp)
    
    # Process findings
    for f in result.get("findings", []):
        if hasattr(f, 'statement'):
            trace_data["findings"].append({
                "id": f.id,
                "statement": f.statement,
                "significance": f.significance,
                "is_novel": f.is_novel,
                "supporting_experiments": f.supporting_experiments,
                "visualization_ids": f.visualization_ids,
            })
        elif isinstance(f, dict):
            trace_data["findings"].append(f)
    
    # ==========================================================================
    # Save JSON trace (for programmatic access)
    # ==========================================================================
    json_trace_path = run_dir / f"trace_{timestamp}.json"
    with open(json_trace_path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)
    print(f"   📊 JSON Trace: {json_trace_path}")
    
    # ==========================================================================
    # Save Markdown trace (human-readable)
    # ==========================================================================
    md_trace = generate_markdown_trace(trace_data)
    md_trace_path = run_dir / f"trace_{timestamp}.md"
    with open(md_trace_path, "w") as f:
        f.write(md_trace)
    print(f"   📝 Markdown Trace: {md_trace_path}")
    
    # ==========================================================================
    # Save HTML trace (interactive viewer)
    # ==========================================================================
    html_trace = generate_html_trace(trace_data)
    html_trace_path = run_dir / f"trace_{timestamp}.html"
    with open(html_trace_path, "w") as f:
        f.write(html_trace)
    print(f"   🌐 HTML Trace: {html_trace_path}")
    
    # ==========================================================================
    # Save individual code execution files (for easy review/replication)
    # ==========================================================================
    code_dir = run_dir / "code_executions"
    code_dir.mkdir(exist_ok=True)
    
    code_executions = result.get("code_execution_traces", [])
    for i, ce in enumerate(code_executions):
        code_file = code_dir / f"execution_{i+1:02d}_{'success' if ce.get('success') else 'failed'}.py"
        
        # Build code file with metadata header
        code_content = f'''"""
Code Execution #{i+1}
==================
Timestamp: {ce.get('timestamp', 'N/A')}
Attempt: #{ce.get('attempt_number', 1)}
Purpose: {ce.get('purpose', 'N/A')}
Status: {'SUCCESS' if ce.get('success') else 'FAILED'}
Experiment ID: {ce.get('experiment_id', 'N/A')}
Hypothesis ID: {ce.get('hypothesis_id', 'N/A')}
Execution Time: {ce.get('execution_time_seconds', 0):.2f}s
'''
        
        if not ce.get('success') and ce.get('error_message'):
            code_content += f'''
Error Message:
{ce.get('error_message', '')}
'''
        
        code_content += f'''
Output IDs: {ce.get('output_ids', [])}
"""

# =============================================================================
# EXECUTED CODE
# =============================================================================

{ce.get('code', '# No code recorded')}
'''
        
        # Add stdout/stderr as comments at the end
        if ce.get('stdout'):
            stdout_preview = ce.get('stdout')[:3000]
            if len(ce.get('stdout', '')) > 3000:
                stdout_preview += f"\n... (truncated, {len(ce.get('stdout'))} total chars)"
            code_content += f'''

# =============================================================================
# STANDARD OUTPUT
# =============================================================================
"""
{stdout_preview}
"""
'''
        
        if ce.get('stderr'):
            code_content += f'''

# =============================================================================
# STANDARD ERROR
# =============================================================================
"""
{ce.get('stderr')[:2000]}
"""
'''
        
        with open(code_file, "w") as f:
            f.write(code_content)
    
    if code_executions:
        print(f"   💻 Code Executions: {code_dir}/ ({len(code_executions)} files)")


def generate_markdown_trace(trace_data: dict) -> str:
    """Generate a human-readable markdown trace document."""
    
    lines = [
        "# Research Traceability Report",
        "",
        f"**Trace ID:** `{trace_data.get('trace_id', 'N/A')}`",
        f"**Created:** {trace_data.get('created_at', 'N/A')}",
        f"**Completed:** {trace_data.get('completed_at', 'N/A')}",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    
    summary = trace_data.get("summary", {})
    lines.extend([
        f"| Total Iterations | {summary.get('total_iterations', 0)} |",
        f"| Total Code Executions | {summary.get('total_code_executions', 0)} |",
        f"| Successful Executions | {summary.get('successful_code_executions', 0)} |",
        f"| Total Hypotheses | {summary.get('total_hypotheses', 0)} |",
        f"| Total Experiments | {summary.get('total_experiments', 0)} |",
        f"| Total Findings | {summary.get('total_findings', 0)} |",
        f"| Total Outputs | {summary.get('total_outputs', 0)} |",
        "",
        "---",
        "",
        "## Research Query",
        "",
        trace_data.get("research_query", "*No query recorded*"),
        "",
    ])
    
    if trace_data.get("research_brief"):
        lines.extend([
            "## Research Brief",
            "",
            trace_data.get("research_brief"),
            "",
        ])
    
    # Hypotheses
    if trace_data.get("hypotheses"):
        lines.extend([
            "---",
            "",
            "## Hypotheses Tested",
            "",
        ])
        for h in trace_data.get("hypotheses", []):
            status = h.get("status", "unknown")
            status_emoji = {"supported": "✅", "refuted": "❌", "testing": "🔬"}.get(status, "❓")
            lines.extend([
                f"### {status_emoji} {h.get('statement', 'Unknown')[:100]}",
                "",
                f"**Status:** {status}",
                "",
                f"**Rationale:** {h.get('rationale', 'N/A')}",
                "",
            ])
    
    # Experiments
    if trace_data.get("experiments"):
        lines.extend([
            "---",
            "",
            "## Experiments Conducted",
            "",
        ])
        for exp in trace_data.get("experiments", []):
            status = exp.get("status", "unknown")
            lines.extend([
                f"### Experiment: {exp.get('objective', 'Unknown')[:80]}",
                "",
                f"**ID:** `{exp.get('id', 'N/A')}`",
                f"**Status:** {status}",
                f"**Supports Hypothesis:** {exp.get('supports_hypothesis', 'N/A')}",
                "",
                "**Methodology:**",
                exp.get('methodology', 'N/A')[:500],
                "",
                "**Findings Summary:**",
                exp.get('findings_summary', 'N/A')[:500] if exp.get('findings_summary') else 'N/A',
                "",
            ])
    
    # Code Executions
    if trace_data.get("code_executions"):
        lines.extend([
            "---",
            "",
            "## Code Execution History",
            "",
            "All code executions are saved in the `code_executions/` directory for review and replication.",
            "",
        ])
        for i, ce in enumerate(trace_data.get("code_executions", [])):
            success = ce.get("success", False)
            status_emoji = "✅" if success else "❌"
            lines.extend([
                f"### {status_emoji} Execution #{i+1} (Attempt #{ce.get('attempt_number', 1)})",
                "",
                f"**Timestamp:** {ce.get('timestamp', 'N/A')}",
                f"**Purpose:** {ce.get('purpose', 'N/A')[:200]}",
                f"**Status:** {'SUCCESS' if success else 'FAILED'}",
                f"**Execution Time:** {ce.get('execution_time_seconds', 0):.2f}s",
                "",
            ])
            
            if not success and ce.get("error_message"):
                lines.extend([
                    "**Error:**",
                    "```",
                    ce.get("error_message", "")[:500],
                    "```",
                    "",
                ])
            
            # Code preview (first 50 lines)
            code = ce.get("code", "")
            code_lines = code.split("\n")
            if len(code_lines) > 50:
                code_preview = "\n".join(code_lines[:50]) + f"\n... ({len(code_lines)} total lines)"
            else:
                code_preview = code
            
            lines.extend([
                "**Code:**",
                "```python",
                code_preview,
                "```",
                "",
            ])
    
    # Supervisor Decisions
    if trace_data.get("supervisor_decisions"):
        lines.extend([
            "---",
            "",
            "## Supervisor Decision Trail",
            "",
        ])
        for dec in trace_data.get("supervisor_decisions", []):
            lines.extend([
                f"### Iteration {dec.get('iteration', 'N/A')}: {dec.get('action_chosen', 'Unknown')}",
                "",
                f"**Timestamp:** {dec.get('timestamp', 'N/A')}",
                "",
            ])
            
            state_summary = dec.get("state_summary", {})
            if isinstance(state_summary, dict):
                lines.extend([
                    "**State at Decision:**",
                    f"- Hypotheses: {state_summary.get('hypotheses_count', 0)}",
                    f"- Experiments: {state_summary.get('experiments_count', 0)}",
                    f"- Findings: {state_summary.get('findings_count', 0)}",
                    "",
                ])
            
            if dec.get("reasoning"):
                lines.extend([
                    "**Reasoning:**",
                    dec.get("reasoning", "")[:500],
                    "",
                ])
    
    # Findings
    if trace_data.get("findings"):
        lines.extend([
            "---",
            "",
            "## Key Findings",
            "",
        ])
        for f in trace_data.get("findings", []):
            novelty = "🆕 " if f.get("is_novel") else ""
            lines.extend([
                f"### {novelty}{f.get('statement', 'Unknown')[:100]}",
                "",
                f"**Significance:** {f.get('significance', 'N/A')}",
                f"**Supporting Experiments:** {f.get('supporting_experiments', [])}",
                "",
            ])
    
    lines.extend([
        "---",
        "",
        "*This trace was automatically generated to enable human review and replication of the AI research process.*",
    ])
    
    return "\n".join(lines)


def generate_html_trace(trace_data: dict) -> str:
    """Generate an interactive HTML trace viewer."""
    import html as html_lib
    
    summary = trace_data.get("summary", {})
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>Research Trace - {trace_data.get('trace_id', 'Unknown')}</title>",
        "<style>",
        """
        :root { --primary: #4361ee; --success: #2ecc71; --danger: #e74c3c; --dark: #1a1a2e; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f6fa; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { color: var(--dark); border-bottom: 3px solid var(--primary); padding-bottom: 10px; }
        h2 { color: var(--dark); margin-top: 30px; display: flex; align-items: center; gap: 10px; }
        h2::before { content: ''; width: 4px; height: 24px; background: var(--primary); }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: linear-gradient(135deg, var(--primary), #3a0ca3); color: white; padding: 20px; border-radius: 10px; }
        .stat-card h3 { margin: 0; font-size: 2em; }
        .stat-card p { margin: 5px 0 0 0; opacity: 0.9; }
        .code-block { background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 8px; overflow-x: auto; 
                      font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; white-space: pre-wrap; }
        .event { background: #f8f9fa; border-left: 4px solid var(--primary); padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
        .event.success { border-left-color: var(--success); }
        .event.failure { border-left-color: var(--danger); }
        .event-header { display: flex; justify-content: space-between; align-items: center; }
        .event-title { font-weight: bold; color: var(--dark); }
        .event-time { color: #666; font-size: 0.85em; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; color: white; }
        .badge-success { background: var(--success); }
        .badge-danger { background: var(--danger); }
        .badge-primary { background: var(--primary); }
        .collapsible { cursor: pointer; padding: 15px; background: #e8e8e8; border: none; width: 100%; 
                       text-align: left; font-size: 1em; border-radius: 8px; margin: 5px 0; display: flex; 
                       justify-content: space-between; align-items: center; }
        .collapsible:hover { background: #ddd; }
        .collapsible::after { content: '+'; font-size: 1.5em; color: var(--primary); }
        .collapsible.active::after { content: '-'; }
        .content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }
        .tabs { display: flex; gap: 5px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab { padding: 10px 20px; cursor: pointer; background: #e8e8e8; border: none; border-radius: 8px 8px 0 0; }
        .tab.active { background: var(--primary); color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .hypothesis-card, .experiment-card, .finding-card { 
            background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 15px 0; }
        .status-supported { color: var(--success); }
        .status-refuted { color: var(--danger); }
        .timeline { position: relative; padding-left: 30px; }
        .timeline::before { content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 2px; background: var(--primary); }
        .timeline-item { position: relative; margin: 20px 0; }
        .timeline-item::before { content: ''; position: absolute; left: -24px; top: 5px; width: 10px; height: 10px; 
                                  background: var(--primary); border-radius: 50%; }
        """,
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        f"<h1>🔬 Research Traceability Report</h1>",
        f"<p><strong>Trace ID:</strong> <code>{trace_data.get('trace_id', 'N/A')}</code></p>",
        f"<p><strong>Created:</strong> {trace_data.get('created_at', 'N/A')}</p>",
        f"<p><strong>Completed:</strong> {trace_data.get('completed_at', 'N/A')}</p>",
        "</div>",
        
        # Stats Grid
        "<div class='container'>",
        "<h2>Summary Statistics</h2>",
        "<div class='stats-grid'>",
        f"<div class='stat-card'><h3>{summary.get('total_iterations', 0)}</h3><p>Total Iterations</p></div>",
        f"<div class='stat-card'><h3>{summary.get('total_code_executions', 0)}</h3><p>Code Executions</p></div>",
        f"<div class='stat-card' style='background: linear-gradient(135deg, #27ae60, #2ecc71);'>"
        f"<h3>{summary.get('successful_code_executions', 0)}</h3><p>Successful</p></div>",
        f"<div class='stat-card'><h3>{summary.get('total_hypotheses', 0)}</h3><p>Hypotheses</p></div>",
        f"<div class='stat-card'><h3>{summary.get('total_experiments', 0)}</h3><p>Experiments</p></div>",
        f"<div class='stat-card'><h3>{summary.get('total_findings', 0)}</h3><p>Findings</p></div>",
        "</div>",
        "</div>",
    ]
    
    # Tabs Navigation
    html_parts.extend([
        "<div class='container'>",
        "<div class='tabs'>",
        "<button class='tab active' onclick='showTab(\"query\")'>Query & Brief</button>",
        "<button class='tab' onclick='showTab(\"hypotheses\")'>Hypotheses</button>",
        "<button class='tab' onclick='showTab(\"experiments\")'>Experiments</button>",
        "<button class='tab' onclick='showTab(\"code\")'>Code Executions</button>",
        "<button class='tab' onclick='showTab(\"decisions\")'>Supervisor Decisions</button>",
        "<button class='tab' onclick='showTab(\"findings\")'>Findings</button>",
        "</div>",
    ])
    
    # Query Tab
    html_parts.extend([
        "<div id='query' class='tab-content active'>",
        "<h2>Research Query</h2>",
        f"<p>{html_lib.escape(trace_data.get('research_query', 'No query recorded'))}</p>",
    ])
    
    if trace_data.get("research_brief"):
        html_parts.extend([
            "<h2>Research Brief</h2>",
            f"<pre style='white-space: pre-wrap; background: #f5f5f5; padding: 15px; border-radius: 8px;'>"
            f"{html_lib.escape(trace_data.get('research_brief', ''))}</pre>",
        ])
    html_parts.append("</div>")
    
    # Hypotheses Tab
    html_parts.append("<div id='hypotheses' class='tab-content'>")
    for h in trace_data.get("hypotheses", []):
        status = h.get("status", "unknown")
        status_class = "status-supported" if status == "supported" else "status-refuted" if status == "refuted" else ""
        html_parts.extend([
            "<div class='hypothesis-card'>",
            f"<h3>{html_lib.escape(h.get('statement', 'Unknown')[:100])}</h3>",
            f"<p class='{status_class}'><strong>Status:</strong> {status.upper()}</p>",
            f"<p><strong>Rationale:</strong> {html_lib.escape(h.get('rationale', 'N/A'))}</p>",
            "</div>",
        ])
    html_parts.append("</div>")
    
    # Experiments Tab
    html_parts.append("<div id='experiments' class='tab-content'>")
    for exp in trace_data.get("experiments", []):
        html_parts.extend([
            "<div class='experiment-card'>",
            f"<h3>{html_lib.escape(exp.get('objective', 'Unknown')[:80])}</h3>",
            f"<p><strong>ID:</strong> <code>{exp.get('id', 'N/A')}</code></p>",
            f"<p><strong>Status:</strong> {exp.get('status', 'N/A')}</p>",
            f"<p><strong>Supports Hypothesis:</strong> {exp.get('supports_hypothesis', 'N/A')}</p>",
        ])
        
        if exp.get("code_executed"):
            code_preview = exp.get("code_executed", "")[:2000]
            html_parts.extend([
                "<button class='collapsible'>View Code</button>",
                "<div class='content'>",
                f"<pre class='code-block'>{html_lib.escape(code_preview)}</pre>",
                "</div>",
            ])
        
        if exp.get("findings_summary"):
            html_parts.append(f"<p><strong>Findings:</strong> {html_lib.escape(exp.get('findings_summary', '')[:500])}</p>")
        
        html_parts.append("</div>")
    html_parts.append("</div>")
    
    # Code Executions Tab
    html_parts.append("<div id='code' class='tab-content'>")
    html_parts.append("<div class='timeline'>")
    for i, ce in enumerate(trace_data.get("code_executions", [])):
        success = ce.get("success", False)
        event_class = "success" if success else "failure"
        badge_class = "badge-success" if success else "badge-danger"
        
        html_parts.extend([
            "<div class='timeline-item'>",
            f"<div class='event {event_class}'>",
            "<div class='event-header'>",
            f"<span class='event-title'>Execution #{i+1} (Attempt #{ce.get('attempt_number', 1)})</span>",
            f"<span class='badge {badge_class}'>{'SUCCESS' if success else 'FAILED'}</span>",
            "</div>",
            f"<p class='event-time'>{ce.get('timestamp', 'N/A')} | {ce.get('execution_time_seconds', 0):.2f}s</p>",
            f"<p>{html_lib.escape(ce.get('purpose', 'N/A')[:200])}</p>",
        ])
        
        if not success and ce.get("error_message"):
            html_parts.extend([
                "<p><strong>Error:</strong></p>",
                f"<pre class='code-block' style='background: #c0392b;'>{html_lib.escape(ce.get('error_message', '')[:500])}</pre>",
            ])
        
        code = ce.get("code", "")
        code_lines = code.split("\n")
        code_preview = "\n".join(code_lines[:30])
        if len(code_lines) > 30:
            code_preview += f"\n... ({len(code_lines)} total lines - see code_executions/ folder for full code)"
        
        html_parts.extend([
            "<button class='collapsible'>View Code</button>",
            "<div class='content'>",
            f"<pre class='code-block'>{html_lib.escape(code_preview)}</pre>",
            "</div>",
            "</div>",
            "</div>",
        ])
    html_parts.extend(["</div>", "</div>"])
    
    # Supervisor Decisions Tab
    html_parts.append("<div id='decisions' class='tab-content'>")
    for dec in trace_data.get("supervisor_decisions", []):
        html_parts.extend([
            "<div class='event'>",
            "<div class='event-header'>",
            f"<span class='event-title'>Iteration {dec.get('iteration', 'N/A')}: {dec.get('action_chosen', 'Unknown')}</span>",
            f"<span class='event-time'>{dec.get('timestamp', 'N/A')}</span>",
            "</div>",
        ])
        
        state = dec.get("state_summary", {})
        if isinstance(state, dict):
            html_parts.append(
                f"<p>State: {state.get('hypotheses_count', 0)} hypotheses, "
                f"{state.get('experiments_count', 0)} experiments, "
                f"{state.get('findings_count', 0)} findings</p>"
            )
        
        if dec.get("reasoning"):
            html_parts.append(f"<p><em>{html_lib.escape(dec.get('reasoning', '')[:300])}</em></p>")
        
        html_parts.append("</div>")
    html_parts.append("</div>")
    
    # Findings Tab
    html_parts.append("<div id='findings' class='tab-content'>")
    for f in trace_data.get("findings", []):
        novelty_badge = "<span class='badge badge-primary'>NOVEL</span> " if f.get("is_novel") else ""
        html_parts.extend([
            "<div class='finding-card'>",
            f"<h3>{novelty_badge}{html_lib.escape(f.get('statement', 'Unknown')[:100])}</h3>",
            f"<p><strong>Significance:</strong> {html_lib.escape(f.get('significance', 'N/A'))}</p>",
            f"<p><strong>Supporting Experiments:</strong> {f.get('supporting_experiments', [])}</p>",
            "</div>",
        ])
    html_parts.append("</div>")
    
    html_parts.append("</div>")  # Close container
    
    # JavaScript
    html_parts.extend([
        "<script>",
        """
        function showTab(tabId) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Collapsible functionality
        document.querySelectorAll('.collapsible').forEach(btn => {
            btn.addEventListener('click', function() {
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + 'px';
                }
            });
        });
        """,
        "</script>",
        "</body>",
        "</html>",
    ])
    
    return "\n".join(html_parts)


async def save_outputs(result: dict, output_dir: Path):
    """Save all outputs to files in a timestamped subdirectory.
    
    This function saves:
    1. Final report (report_{timestamp}.md)
    2. All images/figures (output_{id}.{format})
    3. Basic metadata (metadata_{timestamp}.json)
    4. FULL TRACEABILITY DATA:
       - trace_{timestamp}.json: Complete JSON trace for programmatic access
       - trace_{timestamp}.md: Human-readable markdown trace
       - trace_{timestamp}.html: Interactive HTML trace viewer
       - code_executions/: Individual code execution files
    """
    import base64
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Saving outputs to: {run_dir}")
    
    # Save final report
    report = result.get("final_report", "")
    if report:
        report_path = run_dir / f"report_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"   📄 Report: {report_path}")
    
    # Save images - handle both Pydantic objects and dicts (due to serialization)
    all_outputs = result.get("all_outputs", {})
    images_saved = 0
    for output_id, output in all_outputs.items():
        # Get image_base64 - works for both Pydantic objects and dicts
        if hasattr(output, 'image_base64'):
            img_base64 = output.image_base64
            img_format = getattr(output, 'image_format', 'png') or 'png'
        elif isinstance(output, dict):
            img_base64 = output.get('image_base64')
            img_format = output.get('image_format', 'png') or 'png'
        else:
            img_base64 = None
            img_format = 'png'
        
        if img_base64:
            img_path = run_dir / f"output_{output_id}.{img_format}"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_base64))
            print(f"   🖼️  Figure: {img_path}")
            images_saved += 1
    
    if images_saved > 0:
        print(f"   📊 Total images saved: {images_saved}")
    
    # ==========================================================================
    # TRACEABILITY: Save complete research trace
    # ==========================================================================
    await save_research_trace(result, run_dir, timestamp)
    
    # Save basic metadata (for backward compatibility)
    metadata = {
        "timestamp": timestamp,
        "trace_id": result.get("trace_id", ""),
        "hypotheses_count": len(result.get("hypotheses", [])),
        "experiments_count": len(result.get("experiments", [])),
        "findings_count": len(result.get("findings", [])),
        "outputs_count": len(all_outputs),
        "code_executions_count": len(result.get("code_execution_traces", [])),
    }
    
    # Add hypothesis details - handle both Pydantic objects and dicts
    if result.get("hypotheses"):
        hypotheses_list = []
        for h in result.get("hypotheses", []):
            if hasattr(h, 'statement'):
                statement = h.statement
                status = h.status.value if hasattr(h.status, 'value') else str(h.status)
            elif isinstance(h, dict):
                statement = h.get('statement', '')
                status = h.get('status', {})
                if hasattr(status, 'value'):
                    status = status.value
                elif isinstance(status, str):
                    pass  # Already a string
                else:
                    status = str(status)
            else:
                continue
            hypotheses_list.append({"statement": statement, "status": status})
        metadata["hypotheses"] = hypotheses_list
    
    # Add findings details - handle both Pydantic objects and dicts
    if result.get("findings"):
        findings_list = []
        for f in result.get("findings", []):
            if hasattr(f, 'statement'):
                findings_list.append({"statement": f.statement, "is_novel": f.is_novel})
            elif isinstance(f, dict):
                findings_list.append({
                    "statement": f.get('statement', ''),
                    "is_novel": f.get('is_novel', False)
                })
        metadata["findings"] = findings_list
    
    meta_path = run_dir / f"metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   📋 Metadata: {meta_path}")
    
    print(f"\n✅ All outputs saved to {run_dir}")


def check_environment():
    """Check that required environment variables are set."""
    print("📋 Environment Check:")
    
    e2b_key = os.getenv("E2B_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    vision_model = os.getenv("VISION_MODEL")
    
    if not e2b_key:
        print("   ❌ E2B_API_KEY not set")
        print("      Get from: https://e2b.dev/dashboard")
        return False
    else:
        print("   ✅ E2B_API_KEY is set")
    
    if not openai_key and not anthropic_key:
        print("   ❌ No LLM API key set (need OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return False
    else:
        if openai_key:
            print("   ✅ OPENAI_API_KEY is set")
        if anthropic_key:
            print("   ✅ ANTHROPIC_API_KEY is set")
    
    # Check vision model configuration
    print("\n🖼️  Vision Analysis Check:")
    if vision_model:
        print(f"   ✅ VISION_MODEL configured: {vision_model}")
        # Check if corresponding API key is available
        if "google" in vision_model.lower() or "gemini" in vision_model.lower():
            if google_key:
                print("   ✅ GOOGLE_API_KEY is set")
            else:
                print("   ⚠️  GOOGLE_API_KEY not set - vision analysis will fail")
        elif "openai" in vision_model.lower() or "gpt" in vision_model.lower():
            if openai_key:
                print("   ✅ Using OpenAI for vision")
        elif "anthropic" in vision_model.lower() or "claude" in vision_model.lower():
            if anthropic_key:
                print("   ✅ Using Anthropic for vision")
    else:
        # Check if any vision-capable key is available
        if google_key:
            print("   ✅ GOOGLE_API_KEY available - will use Gemini for vision")
        elif openai_key:
            print("   ⚠️  Will use OpenAI GPT-4V for vision (set VISION_MODEL=google:gemini-3-flash-preview for better results)")
        else:
            print("   ⚠️  No vision model configured - generated images won't be analyzed")
            print("      Set GOOGLE_API_KEY and VISION_MODEL=google:gemini-3-flash-preview for vision analysis")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Computational Scientific Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_discovery.py "Is there a correlation between X and Y?"
  python run_discovery.py --file examples/biosignature-discovery-prompt.txt
  python run_discovery.py --verbose --iterations 5 "Your query"
  python run_discovery.py --quick "Quick test query"
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (or use --file for a prompt file)"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a file containing the research prompt"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output (intermediate steps)"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=8,
        help="Maximum discovery iterations (default: 8)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: fewer iterations (3), faster but less thorough"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: from .env RESEARCH_MODEL or openai:gpt-4o)"
    )
    
    parser.add_argument(
        "--final-model",
        type=str,
        default=None,
        help="Model for final report (default: from .env FINAL_REPORT_MODEL or same as --model)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save outputs to files"
    )
    
    args = parser.parse_args()
    
    # Get query
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        with open(args.file) as f:
            query = f.read()
        print(f"📄 Loaded prompt from: {args.file}")
    elif args.query:
        query = args.query
    else:
        print("❌ Please provide a query or use --file")
        parser.print_help()
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\n❌ Missing required API keys. Please set them and try again.")
        sys.exit(1)
    
    # Get models from .env or command line args
    research_model = args.model or os.getenv("RESEARCH_MODEL", "openai:gpt-4o")
    final_report_model = args.final_model or os.getenv("FINAL_REPORT_MODEL", research_model)
    
    # Get recursion limit from environment (default: 100)
    recursion_limit = int(os.getenv("RECURSION_LIMIT", "100"))
    
    # Configure
    iterations = 3 if args.quick else args.iterations
    
    config = {
        "configurable": {
            "research_model": research_model,
            "final_report_model": final_report_model,
            "max_researcher_iterations": iterations,
            "allow_clarification": False,
            "scientific_domain": "astronomy",  # Can be made configurable
        },
        "recursion_limit": recursion_limit
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"   Research model: {research_model}")
    print(f"   Final report model: {final_report_model}")
    print(f"   Max iterations: {iterations}")
    print(f"   Recursion limit: {recursion_limit}")
    print(f"   Verbose: {args.verbose}")
    print(f"   Output dir: {args.output if not args.no_save else '(not saving)'}")
    
    # Show query preview
    print(f"\n📝 Query preview:")
    print(f"   {query[:200]}{'...' if len(query) > 200 else ''}")
    
    # Confirm
    print("\n" + "="*70)
    response = input("Start discovery? [Y/n]: ").strip().lower()
    if response == 'n':
        print("Cancelled.")
        sys.exit(0)
    
    # Run
    output_dir = Path(args.output) if not args.no_save else None
    
    result = asyncio.run(
        run_discovery_with_monitoring(
            query=query,
            config=config,
            verbose=args.verbose,
            output_dir=output_dir
        )
    )
    
    if result:
        # Print report summary
        report = result.get("final_report", "")
        if report:
            print("\n" + "="*70)
            print("   FINAL REPORT PREVIEW")
            print("="*70)
            print(report[:3000])
            if len(report) > 3000:
                print(f"\n... ({len(report) - 3000} more characters)")
                print(f"Full report saved to: {args.output}/report_*.md")
        
        print("\n✅ Discovery complete!")
        sys.exit(0)
    else:
        print("\n❌ Discovery failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
