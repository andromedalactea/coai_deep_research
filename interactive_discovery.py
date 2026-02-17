#!/usr/bin/env python3
"""
Interactive Computational Discovery

A simpler interface for running discovery in Jupyter notebooks or interactive Python.

Usage in Python/Jupyter:
    
    from interactive_discovery import discover, quick_discover
    
    # Simple usage
    result = await discover("Your research question")
    
    # Quick test (fewer iterations)
    result = await quick_discover("Quick test question")
    
    # With custom config
    result = await discover(
        "Your question",
        iterations=5,
        verbose=True
    )
    
    # Access results
    print(result.report)
    result.show_figures()
    result.summary()
"""

import asyncio
import base64
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class DiscoveryResult:
    """Container for discovery results with helpful methods."""
    
    report: str = ""
    hypotheses: List[Any] = field(default_factory=list)
    experiments: List[Any] = field(default_factory=list)
    findings: List[Any] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    computation_results: List[Any] = field(default_factory=list)
    raw_state: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    @property
    def figures(self) -> List[Any]:
        """Get all figure outputs."""
        return [o for o in self.outputs.values() 
                if hasattr(o, 'output_type') and o.output_type.value == 'image']
    
    @property
    def tables(self) -> List[Any]:
        """Get all table outputs."""
        return [o for o in self.outputs.values() 
                if hasattr(o, 'output_type') and o.output_type.value == 'table']
    
    @property
    def stats(self) -> List[Any]:
        """Get all statistical result outputs."""
        return [o for o in self.outputs.values() 
                if hasattr(o, 'output_type') and o.output_type.value == 'statistical_result']
    
    def summary(self):
        """Print a summary of the discovery results."""
        print("="*60)
        print("   DISCOVERY RESULTS SUMMARY")
        print("="*60)
        print(f"Duration: {self.duration_seconds:.1f}s ({self.duration_seconds/60:.1f} min)")
        print(f"Hypotheses tested: {len(self.hypotheses)}")
        print(f"Experiments run: {len(self.experiments)}")
        print(f"Findings: {len(self.findings)}")
        print(f"Total outputs: {len(self.outputs)}")
        print(f"  - Figures: {len(self.figures)}")
        print(f"  - Tables: {len(self.tables)}")
        print(f"  - Statistical results: {len(self.stats)}")
        print(f"Report length: {len(self.report)} chars")
        print("="*60)
        
        if self.hypotheses:
            print("\nHypotheses:")
            for h in self.hypotheses:
                status = h.status.value if hasattr(h, 'status') else 'unknown'
                print(f"  [{status}] {h.statement[:80]}...")
        
        if self.findings:
            print("\nKey Findings:")
            for f in self.findings:
                novel = "🆕" if f.is_novel else ""
                print(f"  {novel} {f.statement[:80]}...")
    
    def show_figures(self, max_display: int = 5):
        """Display figures (works in Jupyter notebooks)."""
        try:
            from IPython.display import display, Image
            
            figures = self.figures[:max_display]
            if not figures:
                print("No figures generated.")
                return
            
            print(f"Showing {len(figures)} of {len(self.figures)} figures:")
            for i, fig in enumerate(figures):
                print(f"\n--- {fig.citation_label or f'Figure {i+1}'} ---")
                if fig.description:
                    print(f"Description: {fig.description}")
                if fig.image_base64:
                    display(Image(data=base64.b64decode(fig.image_base64)))
                    
        except ImportError:
            print("IPython not available. Save figures to files instead:")
            print("  result.save_figures('output_dir')")
    
    def save_figures(self, output_dir: str = "outputs"):
        """Save all figures to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fig in self.figures:
            if fig.image_base64:
                fmt = fig.image_format or 'png'
                path = output_path / f"figure_{fig.id}.{fmt}"
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(fig.image_base64))
                print(f"Saved: {path}")
    
    def save_report(self, filename: str = None):
        """Save the report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovery_report_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write(self.report)
        print(f"Report saved to: {filename}")
    
    def print_report(self, max_chars: int = 5000):
        """Print the report (truncated if long)."""
        if len(self.report) <= max_chars:
            print(self.report)
        else:
            print(self.report[:max_chars])
            print(f"\n... ({len(self.report) - max_chars} more characters)")
            print("Use result.save_report() to save the full report.")


class ProgressPrinter:
    """Simple progress printer for discovery."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = datetime.now()
        self.current_node = None
        
    def update(self, event: dict):
        """Update progress display."""
        for node_name, data in event.items():
            if node_name == "__start__":
                continue
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            # Emoji for each node type
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
            print(f"[{elapsed:5.1f}s] {emoji} {node_name}")
            
            if self.verbose and isinstance(data, dict):
                if "computation_results" in data:
                    results = data["computation_results"]
                    if results:
                        latest = results[-1] if isinstance(results, list) else results
                        if hasattr(latest, 'success'):
                            print(f"         └─ Code: {'✅' if latest.success else '❌'}")


async def discover(
    query: str,
    iterations: int = 15,
    model: str = None,
    final_model: str = None,
    verbose: bool = False,
    show_progress: bool = True
) -> DiscoveryResult:
    """
    Run computational scientific discovery on a research query.
    
    Args:
        query: The research question or prompt
        iterations: Maximum discovery iterations (default: 15)
        model: LLM model to use (default: from .env RESEARCH_MODEL or openai:gpt-4o)
        final_model: Model for final report (default: from .env FINAL_REPORT_MODEL or same as model)
        verbose: Show detailed progress (default: False)
        show_progress: Show any progress at all (default: True)
    
    Returns:
        DiscoveryResult object with report, figures, and data
    
    Example:
        result = await discover("Is there a correlation between X and Y?")
        print(result.report)
        result.show_figures()
    """
    from open_deep_research.computational.discovery_graph import computational_discovery
    from langchain_core.messages import HumanMessage
    
    # Check environment
    if not os.getenv("E2B_API_KEY"):
        raise EnvironmentError("E2B_API_KEY not set. Get from https://e2b.dev/dashboard")
    compat_key_pattern = re.compile(r"^OPENAI_COMPAT_[A-Z0-9_]+_API_KEY$")
    has_openai_compat_alias_key = any(
        compat_key_pattern.match(key) and bool(value)
        for key, value in os.environ.items()
    )
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not has_openai_compat_alias_key:
        raise EnvironmentError(
            "No LLM API key set (OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_COMPAT_<ALIAS>_API_KEY)"
        )
    
    # Get models from .env if not provided
    research_model = model or os.getenv("RESEARCH_MODEL", "openai:gpt-4o")
    final_report_model = final_model or os.getenv("FINAL_REPORT_MODEL", research_model)
    
    # Config
    config = {
        "configurable": {
            "research_model": research_model,
            "final_report_model": final_report_model,
            "max_researcher_iterations": iterations,
            "max_discovery_iterations": iterations,
            "allow_clarification": False,
        }
    }
    
    # Create a sources directory for persisting downloaded articles
    sources_dir = Path("outputs") / "interactive_sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    config["configurable"]["sources_dir"] = str(sources_dir)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Progress tracking
    progress = ProgressPrinter(verbose) if show_progress else None
    start_time = datetime.now()
    
    if show_progress:
        print("\n🚀 Starting Computational Scientific Discovery...")
        print(f"   Max iterations: {iterations}")
        print(f"   Research model: {research_model}")
        print(f"   Final report model: {final_report_model}")
        print("-"*50)
    
    # Run with streaming
    final_state = {}
    
    try:
        async for event in computational_discovery.astream(
            initial_state,
            config,
            stream_mode="updates"
        ):
            if progress:
                progress.update(event)
            
            # Accumulate state
            for node_name, node_data in event.items():
                if isinstance(node_data, dict):
                    final_state.update(node_data)
                    
    except Exception as e:
        print(f"\n❌ Error during discovery: {e}")
        raise
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if show_progress:
        print("-"*50)
        print(f"✅ Complete in {duration:.1f}s")
    
    # Build result object
    result = DiscoveryResult(
        report=final_state.get("final_report", ""),
        hypotheses=final_state.get("hypotheses", []),
        experiments=final_state.get("experiments", []),
        findings=final_state.get("findings", []),
        outputs=final_state.get("all_outputs", {}),
        computation_results=final_state.get("computation_results", []),
        raw_state=final_state,
        duration_seconds=duration
    )
    
    return result


async def quick_discover(query: str, **kwargs) -> DiscoveryResult:
    """
    Quick discovery with fewer iterations (3).
    Good for testing or simple queries.
    
    Example:
        result = await quick_discover("Calculate mean of [1,2,3,4,5]")
    """
    kwargs.setdefault("iterations", 5)
    return await discover(query, **kwargs)


def run_sync(query: str, **kwargs) -> DiscoveryResult:
    """
    Synchronous wrapper for discover().
    Use this if you're not in an async context.
    
    Example:
        result = run_sync("Your research question")
    """
    return asyncio.run(discover(query, **kwargs))


# =============================================================================
# Convenience functions for common tasks
# =============================================================================

async def test_code_execution():
    """Quick test that E2B code execution works."""
    from open_deep_research.computational.code_interpreter import execute_code
    
    print("Testing E2B code execution...")
    
    result = await execute_code(
        code="""
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Test Plot: Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
display(plt.gcf())

print(f"Mean of sin(x): {np.mean(y):.4f}")
print("Test successful!")
""",
        purpose="Test E2B execution"
    )
    
    if result.success:
        print("✅ E2B code execution works!")
        print(f"   Generated {len(result.outputs)} output(s)")
        return True
    else:
        print(f"❌ E2B execution failed: {result.error_message}")
        return False


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Interactive Discovery Module")
    print("="*50)
    print("Usage in Python/Jupyter:")
    print()
    print("  from interactive_discovery import discover, quick_discover")
    print()
    print("  # Run discovery")
    print('  result = await discover("Your research question")')
    print()
    print("  # Quick test")
    print('  result = await quick_discover("Quick test")')
    print()
    print("  # View results")
    print("  result.summary()")
    print("  result.show_figures()")
    print("  print(result.report)")
    print()
    
    # Quick test
    print("Running quick E2B test...")
    asyncio.run(test_code_execution())
