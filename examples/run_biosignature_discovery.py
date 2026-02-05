"""
Run Novel Biosignature Discovery

This script demonstrates using the Computational Scientific Discovery System
to identify novel, non-Earth-centric biosignatures through automated:
- Literature review
- Thermodynamic modeling
- Spectral predictions
- Statistical analysis
- Iterative hypothesis refinement

Expected runtime: 50-75 minutes
Requires: E2B_API_KEY and OPENAI_API_KEY
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


async def run_biosignature_discovery():
    """Execute the novel biosignature discovery workflow."""
    from open_deep_research.computational import run_computational_discovery
    
    # Load the optimized prompt
    prompt_file = Path(__file__).parent / "biosignature-discovery-prompt.txt"
    with open(prompt_file) as f:
        query = f.read()
    
    print("="*70)
    print("   NOVEL BIOSIGNATURE DISCOVERY - COMPUTATIONAL SESSION")
    print("="*70)
    print("\nStarting computational scientific discovery...")
    print("Expected duration: 50-75 minutes")
    print("\nThe system will autonomously:")
    print("  1. Review ArXiv literature on biosignatures")
    print("  2. Generate testable hypotheses")
    print("  3. Model thermodynamic viability")
    print("  4. Predict spectral signatures")
    print("  5. Assess detectability")
    print("  6. Verify novelty")
    print("  7. Iterate and refine")
    print("  8. Generate comprehensive report with figures")
    print("\n" + "-"*70 + "\n")
    
    # Configure for thorough discovery
    config = {
        "configurable": {
            "research_model": "openai:gpt-4o",
            "final_report_model": "openai:gpt-4o",
            "max_researcher_iterations": 8,  # Allow thorough exploration
            "allow_clarification": False,
            "scientific_domain": "astronomy",
            "enable_telescope_archives": True,
            "enable_arxiv_deep_extraction": True,
            "significance_level": 0.05,
            "max_figures_in_report": 15,
            "save_computation_outputs": True,
            "interpret_images_with_ai": True,
        }
    }
    
    try:
        result = await run_computational_discovery(query, config)
        
        print("\n" + "="*70)
        print("   DISCOVERY COMPLETE")
        print("="*70)
        
        # Save the report
        output_dir = Path("examples/outputs")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "biosignature_discovery_report.md"
        with open(report_path, "w") as f:
            f.write(result.get("final_report", "No report generated"))
        
        print(f"\n📄 Report saved to: {report_path}")
        
        # Save computational outputs
        all_outputs = result.get("all_outputs", {})
        if all_outputs:
            print(f"\n📊 Generated {len(all_outputs)} computational outputs:")
            
            # Save images
            import base64
            for output_id, output in all_outputs.items():
                if output.image_base64 and output.image_format:
                    img_path = output_dir / f"biosig_output_{output_id}.{output.image_format}"
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(output.image_base64))
                    print(f"   - {output.citation_label or 'Figure'}: {img_path}")
        
        # Print summary statistics
        if result.get("hypotheses"):
            print(f"\n🔬 Tested {len(result['hypotheses'])} hypothesis(es)")
        
        if result.get("experiments"):
            print(f"🧪 Conducted {len(result['experiments'])} experiment(s)")
        
        if result.get("findings"):
            print(f"💡 Generated {len(result['findings'])} key finding(s)")
        
        # Print report preview
        print("\n" + "-"*70)
        print("REPORT PREVIEW:")
        print("-"*70)
        report = result.get("final_report", "")
        print(report[:2000])
        if len(report) > 2000:
            print(f"\n... ({len(report) - 2000} more characters)")
        print("\n" + "="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_quick_test():
    """Run a quick test of the computational system (5-10 minutes)."""
    from open_deep_research.computational import run_computational_discovery
    
    print("="*70)
    print("   QUICK TEST - Sulfur Biosignature Hypothesis")
    print("="*70)
    print("\nTesting computational discovery with a focused query...")
    print("Expected duration: 5-10 minutes\n")
    
    quick_query = """
    Test the hypothesis: Sulfur aerosols (S₈) could serve as a biosignature 
    on super-Earth exoplanets with thick hydrogen atmospheres orbiting M-dwarf stars.
    
    Tasks:
    1. Calculate thermodynamic viability of H₂S → S₈ + H₂ photolysis
    2. Model the spectral signature of S₈ aerosols in transmission
    3. Estimate detectability with JWST
    4. Compare to abiotic sulfur sources
    5. Search ArXiv to verify this specific combination is novel
    
    Generate a brief report with at least one figure showing the predicted spectrum.
    """
    
    config = {
        "configurable": {
            "research_model": "openai:gpt-4o",
            "max_researcher_iterations": 3,
            "allow_clarification": False,
        }
    }
    
    try:
        result = await run_computational_discovery(quick_query, config)
        
        print("\n✅ Quick test complete!")
        print(f"\nGenerated {len(result.get('all_outputs', {}))} outputs")
        print("\nReport preview:")
        print("-"*70)
        print(result.get("final_report", "No report")[:1000])
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run novel biosignature discovery"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (5-10 min) instead of full discovery (50-75 min)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        success = asyncio.run(run_quick_test())
    else:
        success = asyncio.run(run_biosignature_discovery())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
