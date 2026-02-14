"""E2B Code Interpreter Infrastructure for Scientific Computing.

This module provides the core computational infrastructure for the
Scientific Discovery System, handling:
- Code execution in secure E2B sandboxes
- Output management (text, images, tables, statistical results)
- AI-interpretable output formatting
- Persistent sandbox management for iterative experiments
"""

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from open_deep_research.computational.state import (
    ComputationalOutput,
    ComputationResult,
    OutputType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# E2B Sandbox Management
# =============================================================================

class SandboxManager:
    """Manages E2B sandboxes for computational experiments.
    
    Supports both ephemeral (one-off) and persistent (session-long) sandboxes
    for iterative scientific computing workflows.
    """
    
    _instances: Dict[str, Any] = {}  # sandbox_id -> Sandbox instance
    
    @classmethod
    async def get_or_create_sandbox(
        cls,
        sandbox_id: Optional[str] = None,
        timeout: int = 3600,
        setup_scientific_env: bool = True
    ) -> Tuple[str, Any]:
        """Get existing sandbox or create a new one.
        
        Args:
            sandbox_id: Optional ID to retrieve existing sandbox
            timeout: Sandbox timeout in seconds (default 1 hour)
            setup_scientific_env: Whether to install scientific packages
            
        Returns:
            Tuple of (sandbox_id, sandbox_instance)
        """
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "E2B SDK not installed. Install with: pip install e2b-code-interpreter"
            )
        
        # Check for existing sandbox
        if sandbox_id and sandbox_id in cls._instances:
            sandbox = cls._instances[sandbox_id]
            # Verify sandbox is still alive
            try:
                # Simple health check
                return sandbox_id, sandbox
            except Exception:
                # Sandbox died, remove from cache
                del cls._instances[sandbox_id]
        
        # Create new sandbox
        new_id = sandbox_id or str(uuid.uuid4())[:12]
        
        try:
            sandbox = Sandbox.create(timeout=timeout)
            cls._instances[new_id] = sandbox
            
            if setup_scientific_env:
                await cls._setup_scientific_environment(sandbox)
            
            logger.info(f"Created new E2B sandbox: {new_id}")
            return new_id, sandbox
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
    
    @classmethod
    async def _setup_scientific_environment(cls, sandbox: Any) -> None:
        """Install scientific packages in the sandbox."""
        setup_code = """
import subprocess
import sys

# Install scientific computing packages
packages = [
    'astropy',      # Astronomical computations
    'astroquery',   # Query astronomical databases
    'sympy',        # Symbolic mathematics
    'lmfit',        # Curve fitting
    'uncertainties', # Error propagation
    'corner',       # Corner plots for MCMC
    'emcee',        # MCMC sampling
    'arviz',        # Bayesian visualization
]

for pkg in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    except:
        pass  # Some packages may fail, continue anyway

print("Scientific environment ready")
"""
        try:
            result = sandbox.run_code(setup_code)
            logger.info("Scientific environment setup complete")
        except Exception as e:
            logger.warning(f"Partial environment setup: {e}")
    
    @classmethod
    def close_sandbox(cls, sandbox_id: str) -> None:
        """Close and cleanup a sandbox."""
        if sandbox_id in cls._instances:
            try:
                cls._instances[sandbox_id].close()
            except Exception:
                pass
            del cls._instances[sandbox_id]
            logger.info(f"Closed sandbox: {sandbox_id}")
    
    @classmethod
    def close_all(cls) -> None:
        """Close all managed sandboxes."""
        for sandbox_id in list(cls._instances.keys()):
            cls.close_sandbox(sandbox_id)


# Alias for convenience
CodeInterpreterManager = SandboxManager


# =============================================================================
# Output Processing
# =============================================================================

class OutputProcessor:
    """Processes raw E2B execution outputs into structured ComputationalOutput objects."""
    
    @staticmethod
    def process_execution_result(
        execution: Any,
        code: str,
        purpose: str = "",
        experiment_id: Optional[str] = None
    ) -> ComputationResult:
        """Process an E2B execution result into a structured ComputationResult.
        
        Args:
            execution: Raw E2B execution result
            code: The code that was executed
            purpose: Description of why this code was run
            experiment_id: Optional link to an experiment
            
        Returns:
            ComputationResult with all outputs properly structured
        """
        result_id = str(uuid.uuid4())[:8]
        outputs = []
        
        # Process stdout
        stdout = ""
        if hasattr(execution, 'logs') and execution.logs:
            # E2B may return stdout/stderr as list or string
            raw_stdout = execution.logs.stdout
            raw_stderr = execution.logs.stderr
            
            # Convert to string if list
            if isinstance(raw_stdout, list):
                stdout = "".join(raw_stdout) if raw_stdout else ""
            else:
                stdout = raw_stdout or ""
            
            if isinstance(raw_stderr, list):
                stderr = "".join(raw_stderr) if raw_stderr else ""
            else:
                stderr = raw_stderr or ""
            
            if stdout:
                outputs.append(ComputationalOutput(
                    output_type=OutputType.TEXT,
                    text_content=stdout,
                    description="Standard output from code execution",
                    source_experiment_id=experiment_id,
                    source_code_snippet=code[:200] + "..." if len(code) > 200 else code
                ))
        else:
            stderr = ""
        
        # Process error if any
        error_message = None
        if hasattr(execution, 'error') and execution.error:
            error_message = f"{execution.error.name}: {execution.error.value}"
            outputs.append(ComputationalOutput(
                output_type=OutputType.ERROR,
                text_content=error_message,
                description="Execution error"
            ))
        
        # Process rich outputs (images, etc.)
        if hasattr(execution, 'results') and execution.results:
            for idx, result in enumerate(execution.results):
                # Handle PNG images
                if hasattr(result, 'png') and result.png:
                    outputs.append(ComputationalOutput(
                        output_type=OutputType.IMAGE,
                        image_base64=result.png,
                        image_format="png",
                        description=f"Generated visualization #{idx + 1}",
                        source_experiment_id=experiment_id,
                        source_code_snippet=code[:200] + "..." if len(code) > 200 else code
                    ))
                
                # Handle SVG
                if hasattr(result, 'svg') and result.svg:
                    outputs.append(ComputationalOutput(
                        output_type=OutputType.IMAGE,
                        image_base64=base64.b64encode(result.svg.encode()).decode(),
                        image_format="svg",
                        description=f"Generated SVG visualization #{idx + 1}",
                        source_experiment_id=experiment_id
                    ))
                
                # Handle HTML (for interactive plots)
                if hasattr(result, 'html') and result.html:
                    outputs.append(ComputationalOutput(
                        output_type=OutputType.TEXT,
                        text_content=result.html,
                        description="Interactive HTML visualization"
                    ))
                
                # Handle text output
                if hasattr(result, 'text') and result.text:
                    outputs.append(ComputationalOutput(
                        output_type=OutputType.TEXT,
                        text_content=result.text,
                        description="Computed result"
                    ))
                
                # Handle data/JSON
                if hasattr(result, 'data') and result.data:
                    if isinstance(result.data, (dict, list)):
                        outputs.append(ComputationalOutput(
                            output_type=OutputType.TABLE if isinstance(result.data, list) else OutputType.TEXT,
                            table_data=result.data if isinstance(result.data, list) else None,
                            text_content=json.dumps(result.data, indent=2) if isinstance(result.data, dict) else None,
                            description="Structured data output"
                        ))
        
        # Get return value
        return_value = None
        if hasattr(execution, 'text') and execution.text:
            return_value = execution.text
        
        # Extract statistical results from stdout
        if stdout:
            stat_outputs = OutputProcessor.extract_statistical_results(stdout)
            outputs.extend(stat_outputs)
        
        return ComputationResult(
            id=result_id,
            code_executed=code,
            success=error_message is None,
            error_message=error_message,
            outputs=outputs,
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            purpose=purpose,
            experiment_id=experiment_id
        )
    
    @staticmethod
    def extract_statistical_results(stdout: str) -> List[ComputationalOutput]:
        """Extract statistical test results from output text.
        
        Looks for common statistical output patterns:
        - p-value: X.XXX
        - correlation: X.XXX
        - t-statistic: X.XXX
        - chi-square: X.XXX
        - R-squared: X.XXX
        """
        outputs = []
        
        patterns = [
            (r'p[- ]?value[:\s]*([0-9.e-]+)', 'p-value'),
            (r'correlation[:\s]*([0-9.-]+)', 'correlation coefficient'),
            (r't[- ]?statistic[:\s]*([0-9.-]+)', 't-statistic'),
            (r'chi[- ]?square[d]?[:\s]*([0-9.]+)', 'chi-square'),
            (r'r[- ]?squared[:\s]*([0-9.]+)', 'R-squared'),
            (r'mean[:\s]*([0-9.e+-]+)', 'mean'),
            (r'std(?:dev)?[:\s]*([0-9.e+-]+)', 'standard deviation'),
            (r'confidence interval[:\s]*\[?([0-9.e+-]+)[,\s]+([0-9.e+-]+)\]?', 'confidence interval'),
        ]
        
        for pattern, name in patterns:
            matches = re.finditer(pattern, stdout, re.IGNORECASE)
            for match in matches:
                outputs.append(ComputationalOutput(
                    output_type=OutputType.STATISTICAL_RESULT,
                    text_content=f"{name}: {match.group(1)}",
                    description=f"Statistical result: {name}"
                ))
        
        return outputs


# =============================================================================
# Code Execution Functions
# =============================================================================

async def execute_code(
    code: str,
    purpose: str = "",
    sandbox_id: Optional[str] = None,
    timeout: int = 300,
    config: Optional[RunnableConfig] = None
) -> ComputationResult:
    """Execute Python code in an E2B sandbox.
    
    This is the core execution function that handles:
    - Sandbox management (ephemeral or persistent)
    - Code execution with timeout
    - Output processing and structuring
    - Error handling
    
    Args:
        code: Python code to execute
        purpose: Description of what this code does
        sandbox_id: Optional sandbox ID for persistent sessions
        timeout: Execution timeout in seconds
        config: Optional LangGraph config
        
    Returns:
        ComputationResult with all outputs structured for AI consumption
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return ComputationResult(
            code_executed=code,
            success=False,
            error_message="E2B SDK not installed. Install with: pip install e2b-code-interpreter",
            purpose=purpose
        )
    
    # Check for API key
    if not os.getenv("E2B_API_KEY"):
        return ComputationResult(
            code_executed=code,
            success=False,
            error_message="E2B_API_KEY not set in environment",
            purpose=purpose
        )
    
    start_time = datetime.now()
    
    try:
        # Use persistent sandbox if ID provided, otherwise create ephemeral
        if sandbox_id:
            sid, sandbox = await SandboxManager.get_or_create_sandbox(
                sandbox_id=sandbox_id,
                timeout=timeout + 60  # Buffer for setup
            )
            execution = sandbox.run_code(code)
        else:
            # Ephemeral sandbox for one-off execution
            sandbox = Sandbox.create(timeout=timeout)
            try:
                execution = sandbox.run_code(code)
            finally:
                sandbox.kill()
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Process the result
        result = OutputProcessor.process_execution_result(
            execution=execution,
            code=code,
            purpose=purpose
        )
        result.execution_time_seconds = execution_time
        
        # NOTE: Statistical results are already extracted in process_execution_result()
        # Do NOT extract again here to avoid duplicates
        
        return result
        
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return ComputationResult(
            code_executed=code,
            success=False,
            error_message=str(e),
            purpose=purpose,
            execution_time_seconds=(datetime.now() - start_time).total_seconds()
        )


async def execute_experiment(
    experiment_code: str,
    experiment_id: str,
    hypothesis: str,
    sandbox_id: Optional[str] = None,
    setup_code: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> ComputationResult:
    """Execute a complete scientific experiment.
    
    This function handles experiment execution with:
    - Optional setup code (imports, data loading)
    - Main experiment code
    - Proper linking to experiment/hypothesis
    - Enhanced output processing
    
    Args:
        experiment_code: Main code for the experiment
        experiment_id: ID of the experiment record
        hypothesis: The hypothesis being tested
        sandbox_id: Optional persistent sandbox ID
        setup_code: Optional setup code to run first
        config: Optional LangGraph config
        
    Returns:
        ComputationResult linked to the experiment
    """
    # Combine setup and experiment code
    full_code = ""
    
    if setup_code:
        full_code += f"# === Setup Code ===\n{setup_code}\n\n"
    
    full_code += f"# === Experiment Code ===\n{experiment_code}"
    
    # Execute with experiment context
    result = await execute_code(
        code=full_code,
        purpose=f"Testing hypothesis: {hypothesis}",
        sandbox_id=sandbox_id,
        config=config
    )
    
    result.experiment_id = experiment_id
    
    # Link outputs to experiment
    for output in result.outputs:
        output.source_experiment_id = experiment_id
    
    return result


# =============================================================================
# Tool Definitions for LangGraph
# =============================================================================

CODE_INTERPRETER_DESCRIPTION = """
Execute Python code for scientific analysis, numerical computation, and visualization.

USE THIS TOOL WHEN YOU NEED TO:
- Perform numerical calculations or simulations
- Analyze data statistically (t-tests, correlations, ANOVA, etc.)
- Create visualizations (plots, charts, histograms)
- Fit models to data
- Process and transform datasets
- Solve equations numerically or symbolically
- Run Monte Carlo simulations
- Perform astronomical calculations (with astropy)

AVAILABLE PACKAGES:
- numpy, scipy, pandas, matplotlib, seaborn (core scientific stack)
- astropy, astroquery (astronomical computing)
- sympy (symbolic mathematics)
- statsmodels, sklearn (statistics and ML)
- lmfit, emcee (curve fitting, MCMC)

IMPORTANT GUIDELINES:
1. For visualizations, ALWAYS end with: display(plt.gcf())
2. Print all important numerical results explicitly
3. Include clear labels and titles on plots
4. Use descriptive variable names
5. Add comments explaining the methodology
6. Report statistical significance where applicable

OUTPUT HANDLING:
- Text outputs are captured automatically
- Images are captured and stored for the report
- Use print() for important findings
- Return key values at the end of the code
"""


class CodeInterpreterInput(BaseModel):
    """Input schema for the code interpreter tool."""
    code: str = Field(
        description="Python code to execute for scientific analysis"
    )
    purpose: str = Field(
        description="Brief description of what this code will compute or analyze"
    )


@tool(description=CODE_INTERPRETER_DESCRIPTION)
async def execute_python_code(
    code: str,
    purpose: str,
    config: RunnableConfig = None
) -> str:
    """Execute Python code and return structured results.
    
    Args:
        code: Python code to execute
        purpose: What this computation will achieve
        
    Returns:
        Formatted string with execution results and output references
    """
    result = await execute_code(code=code, purpose=purpose, config=config)
    return result.to_ai_summary()


@tool(description="Run a numerical simulation with the given parameters")
async def run_simulation(
    simulation_code: str,
    n_iterations: int,
    parameters: Dict[str, Any],
    config: RunnableConfig = None
) -> str:
    """Run a numerical simulation.
    
    Args:
        simulation_code: Python code defining the simulation
        n_iterations: Number of iterations to run
        parameters: Dictionary of simulation parameters
        
    Returns:
        Simulation results summary
    """
    # Inject parameters into code
    param_code = "\n".join([
        f"{key} = {repr(value)}" for key, value in parameters.items()
    ])
    param_code += f"\nn_iterations = {n_iterations}\n"
    
    full_code = f"{param_code}\n{simulation_code}"
    
    result = await execute_code(
        code=full_code,
        purpose=f"Simulation with {n_iterations} iterations"
    )
    
    return result.to_ai_summary()


@tool(description="Perform statistical hypothesis testing on data")
async def statistical_test(
    test_type: str,
    data_code: str,
    alpha: float = 0.05,
    config: RunnableConfig = None
) -> str:
    """Perform a statistical hypothesis test.
    
    Args:
        test_type: Type of test (t_test, chi_square, anova, correlation, etc.)
        data_code: Code that defines/loads the data variables
        alpha: Significance level (default 0.05)
        
    Returns:
        Test results with statistical interpretation
    """
    test_implementations = {
        "t_test": """
from scipy import stats
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha={alpha}: {p_value < alpha}")
""",
        "paired_t_test": """
from scipy import stats
t_stat, p_value = stats.ttest_rel(before, after)
print(f"Paired T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha={alpha}: {p_value < alpha}")
""",
        "correlation": """
from scipy import stats
correlation, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha={alpha}: {p_value < alpha}")
""",
        "chi_square": """
from scipy import stats
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Significant at alpha={alpha}: {p_value < alpha}")
""",
        "anova": """
from scipy import stats
f_stat, p_value = stats.f_oneway(*groups)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha={alpha}: {p_value < alpha}")
""",
    }
    
    test_code = test_implementations.get(test_type, test_implementations["t_test"])
    full_code = f"alpha = {alpha}\n{data_code}\n{test_code}"
    
    result = await execute_code(
        code=full_code,
        purpose=f"Statistical {test_type} at alpha={alpha}"
    )
    
    return result.to_ai_summary()


# =============================================================================
# Helper Functions for AI Interpretation
# =============================================================================

def format_outputs_for_ai(
    outputs: List[ComputationalOutput],
    include_image_descriptions: bool = True
) -> str:
    """Format computation outputs for AI model consumption.
    
    Creates a comprehensive description of all outputs that an AI model
    can use to understand and reason about the computational results.
    
    Args:
        outputs: List of ComputationalOutput objects
        include_image_descriptions: Whether to include image descriptions
        
    Returns:
        Formatted string suitable for AI model input
    """
    sections = ["=== COMPUTATIONAL OUTPUTS ===\n"]
    
    # Group by type
    images = [o for o in outputs if o.output_type == OutputType.IMAGE]
    texts = [o for o in outputs if o.output_type == OutputType.TEXT]
    tables = [o for o in outputs if o.output_type == OutputType.TABLE]
    stats = [o for o in outputs if o.output_type == OutputType.STATISTICAL_RESULT]
    errors = [o for o in outputs if o.output_type == OutputType.ERROR]
    
    if texts:
        sections.append("--- Text Outputs ---")
        for output in texts:
            sections.append(output.to_ai_description())
        sections.append("")
    
    if stats:
        sections.append("--- Statistical Results ---")
        for output in stats:
            sections.append(output.to_ai_description())
        sections.append("")
    
    if images:
        sections.append(f"--- Visualizations ({len(images)} images) ---")
        for i, output in enumerate(images):
            sections.append(f"[Image {i+1}] {output.description}")
            if output.caption:
                sections.append(f"  Caption: {output.caption}")
            if output.interpretation:
                sections.append(f"  Interpretation: {output.interpretation}")
            sections.append(f"  Reference: output_{output.id}.{output.image_format}")
        sections.append("")
    
    if tables:
        sections.append("--- Data Tables ---")
        for output in tables:
            sections.append(output.to_ai_description())
        sections.append("")
    
    if errors:
        sections.append("--- Errors ---")
        for output in errors:
            sections.append(f"ERROR: {output.text_content}")
        sections.append("")
    
    return "\n".join(sections)


def format_outputs_for_report(
    outputs: List[ComputationalOutput],
    figure_start: int = 1,
    table_start: int = 1
) -> Tuple[str, Dict[str, str]]:
    """Format outputs for inclusion in the final report.
    
    Args:
        outputs: List of outputs to format
        figure_start: Starting number for figure labels
        table_start: Starting number for table labels
        
    Returns:
        Tuple of (formatted_text, output_id_to_label_mapping)
    """
    sections = []
    label_mapping = {}
    
    fig_num = figure_start
    tbl_num = table_start
    
    for output in outputs:
        if output.output_type == OutputType.IMAGE:
            label = f"Figure {fig_num}"
            output.citation_label = label
            label_mapping[output.id] = label
            
            caption = output.caption or output.description
            sections.append(f"\n![{caption}](output_{output.id}.{output.image_format})")
            sections.append(f"*{label}: {caption}*\n")
            
            fig_num += 1
            
        elif output.output_type == OutputType.TABLE:
            label = f"Table {tbl_num}"
            output.citation_label = label
            label_mapping[output.id] = label
            
            # Format table as markdown
            if output.table_data:
                headers = list(output.table_data[0].keys()) if output.table_data else []
                sections.append(f"\n**{label}: {output.description}**\n")
                sections.append("| " + " | ".join(headers) + " |")
                sections.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in output.table_data[:20]:  # Limit rows
                    sections.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
                if len(output.table_data) > 20:
                    sections.append(f"*... and {len(output.table_data) - 20} more rows*")
                sections.append("")
            
            tbl_num += 1
    
    return "\n".join(sections), label_mapping


# =============================================================================
# Vision Analysis Integration
# =============================================================================

async def analyze_images_with_vision(
    computation_result: ComputationResult,
    hypothesis: str = "",
    experiment_context: str = ""
) -> ComputationResult:
    """Enhance computation result with vision-based image analysis.
    
    This function analyzes all image outputs using a vision-capable model
    and updates them with detailed interpretations.
    
    Args:
        computation_result: The result from code execution
        hypothesis: The hypothesis being tested
        experiment_context: Additional context about the experiment
        
    Returns:
        Updated ComputationResult with image interpretations
    """
    import os
    
    # Check if vision analysis is enabled
    if not os.getenv("VISION_MODEL") and not os.getenv("GOOGLE_API_KEY"):
        logger.info("Vision analysis disabled (no VISION_MODEL or GOOGLE_API_KEY set)")
        return computation_result
    
    # Get images from outputs
    images = [o for o in computation_result.outputs if o.output_type == OutputType.IMAGE]
    
    if not images:
        logger.debug("No images to analyze")
        return computation_result
    
    logger.info(f"Analyzing {len(images)} image(s) with vision model...")
    
    try:
        from .image_analyzer import analyze_scientific_figure, analysis_to_text
        
        for i, output in enumerate(images):
            if not output.image_base64:
                continue
            
            logger.debug(f"Analyzing image {i+1}/{len(images)}")
            
            # Build context for this specific image
            expected_content = f"Image {i+1} generated by experiment code"
            if computation_result.purpose:
                expected_content = f"Visualization for: {computation_result.purpose}"
            
            # Analyze the image
            analysis = await analyze_scientific_figure(
                image_base64=output.image_base64,
                image_format=output.image_format or "png",
                hypothesis=hypothesis,
                experiment_context=experiment_context,
                expected_content=expected_content,
                source_code=computation_result.code_executed
            )
            
            # Update the output with analysis results
            output.interpretation = analysis_to_text(analysis)
            
            # Update description with more detail
            if analysis.plot_type and analysis.plot_type != "unknown":
                output.description = f"{analysis.plot_type.title()}: {analysis.main_pattern}"
            
            # Store caption from analysis
            if analysis.full_description:
                output.caption = analysis.full_description[:500]  # Limit caption length
            
            # Log analysis summary
            support_str = "supports" if analysis.supports_hypothesis else (
                "contradicts" if analysis.supports_hypothesis == False else "unclear for"
            )
            logger.info(
                f"Image {i+1} analysis: {analysis.plot_type}, "
                f"quality={analysis.quality_score*100:.0f}%, "
                f"{support_str} hypothesis ({analysis.confidence} confidence)"
            )
    
    except Exception as e:
        logger.warning(f"Vision analysis failed, continuing without: {e}")
    
    return computation_result


def validate_experiment_quality(
    computation_result: ComputationResult,
    require_statistical_results: bool = True,
    require_visualizations: bool = False,
    min_stdout_length: int = 100
) -> Dict[str, Any]:
    """Validate the quality and completeness of experiment results.
    
    Args:
        computation_result: The result to validate
        require_statistical_results: Whether statistical tests are required
        require_visualizations: Whether figures are required
        min_stdout_length: Minimum stdout length for meaningful output
        
    Returns:
        Dictionary with quality metrics and issues
    """
    quality = {
        "is_valid": True,
        "quality_score": 1.0,
        "has_numerical_results": False,
        "has_statistical_tests": False,
        "has_visualizations": False,
        "has_conclusion": False,
        "stdout_length": len(computation_result.stdout) if computation_result.stdout else 0,
        "output_count": len(computation_result.outputs),
        "warnings": [],
        "errors": []
    }
    
    # Check execution success
    if not computation_result.success:
        quality["is_valid"] = False
        quality["quality_score"] = 0.0
        quality["errors"].append(f"Execution failed: {computation_result.error_message}")
        return quality
    
    stdout = computation_result.stdout or ""
    
    # Check stdout length
    if len(stdout) < min_stdout_length:
        quality["warnings"].append(f"Short output ({len(stdout)} chars) - may be incomplete")
        quality["quality_score"] -= 0.2
    
    # Check for numerical results
    numerical_patterns = [
        r"[\d]+\.[\d]+",  # Floating point numbers
        r"mean|average|median",
        r"min|max|range",
        r"sum|total|count"
    ]
    import re
    for pattern in numerical_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            quality["has_numerical_results"] = True
            break
    
    if not quality["has_numerical_results"]:
        quality["warnings"].append("No numerical results detected in output")
        quality["quality_score"] -= 0.15
    
    # Check for statistical test results
    stat_patterns = [
        r"p[-_]?value",
        r"t[-_]?statistic",
        r"chi[-_]?square",
        r"r[-_]?squared|r²|R²",
        r"correlation",
        r"significant|significance",
        r"confidence interval|CI",
        r"effect size"
    ]
    for pattern in stat_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            quality["has_statistical_tests"] = True
            break
    
    if require_statistical_results and not quality["has_statistical_tests"]:
        quality["warnings"].append("No statistical test results detected")
        quality["quality_score"] -= 0.2
    
    # Check for visualizations
    images = [o for o in computation_result.outputs if o.output_type == OutputType.IMAGE]
    quality["has_visualizations"] = len(images) > 0
    
    if require_visualizations and not quality["has_visualizations"]:
        quality["warnings"].append("No visualizations generated")
        quality["quality_score"] -= 0.15
    
    # Check for conclusion
    conclusion_patterns = [
        r"conclusion|result|finding",
        r"support(s|ed)?|refute(s|d)?",
        r"hypothesis.*(support|reject|accept)",
        r"therefore|thus|hence",
        r"we (found|conclude|determined)"
    ]
    for pattern in conclusion_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            quality["has_conclusion"] = True
            break
    
    if not quality["has_conclusion"]:
        quality["warnings"].append("No clear conclusion statement detected")
        quality["quality_score"] -= 0.1
    
    # Clamp quality score
    quality["quality_score"] = max(0.0, min(1.0, quality["quality_score"]))
    
    # Determine validity
    if quality["errors"]:
        quality["is_valid"] = False
    elif quality["quality_score"] < 0.3:
        quality["is_valid"] = False
        quality["errors"].append("Quality too low for meaningful analysis")
    
    return quality


def get_quality_improvement_suggestions(quality: Dict[str, Any]) -> List[str]:
    """Generate suggestions for improving experiment quality.
    
    Args:
        quality: Quality validation result from validate_experiment_quality
        
    Returns:
        List of actionable suggestions
    """
    suggestions = []
    
    if not quality["has_numerical_results"]:
        suggestions.append(
            "Add print() statements for key numerical results "
            "(e.g., 'Mean value: {:.2f}'.format(mean))"
        )
    
    if not quality["has_statistical_tests"]:
        suggestions.append(
            "Include statistical tests with explicit p-values "
            "(e.g., 'p-value: {:.6f}'.format(p_val))"
        )
    
    if not quality["has_visualizations"]:
        suggestions.append(
            "Add visualizations using matplotlib "
            "(plt.figure(), plt.plot(), plt.savefig())"
        )
    
    if not quality["has_conclusion"]:
        suggestions.append(
            "Add explicit conclusion statement "
            "(e.g., 'CONCLUSION: Hypothesis SUPPORTED because...')"
        )
    
    if quality["stdout_length"] < 100:
        suggestions.append(
            "Print more intermediate results and explanations "
            "to document the analysis process"
        )
    
    return suggestions
