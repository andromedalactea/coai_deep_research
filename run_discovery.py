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


async def save_outputs(result: dict, output_dir: Path):
    """Save all outputs to files."""
    import base64
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📁 Saving outputs to: {output_dir}")
    
    # Save final report
    report = result.get("final_report", "")
    if report:
        report_path = output_dir / f"report_{timestamp}.md"
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
            img_path = output_dir / f"output_{output_id}.{img_format}"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_base64))
            print(f"   🖼️  Figure: {img_path}")
            images_saved += 1
    
    if images_saved > 0:
        print(f"   📊 Total images saved: {images_saved}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "hypotheses_count": len(result.get("hypotheses", [])),
        "experiments_count": len(result.get("experiments", [])),
        "findings_count": len(result.get("findings", [])),
        "outputs_count": len(all_outputs),
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
    
    meta_path = output_dir / f"metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   📋 Metadata: {meta_path}")
    
    print(f"\n✅ All outputs saved to {output_dir}")


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
