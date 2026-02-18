"""Configuration for the Computational Scientific Discovery System.

This module extends the base configuration with settings specific to
computational discovery, including:
- E2B code interpreter settings
- Scientific data source configurations
- Discovery iteration parameters
- Output management settings
"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration as BaseConfiguration


class ScientificDomain(str, Enum):
    """Scientific domains with specialized tools and data sources."""
    ASTRONOMY = "astronomy"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    CLIMATE = "climate"
    GENERAL = "general"


class ComputationalConfiguration(BaseConfiguration):
    """Extended configuration for computational scientific discovery.
    
    Inherits all settings from the base Configuration and adds
    settings specific to the computational discovery workflow.
    """
    
    # ==========================================================================
    # Novelty Engine Settings
    # ==========================================================================

    enable_novelty_engine: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable multi-round prior-art novelty checks before validating high-impact claims."
            }
        }
    )

    novelty_engine: str = Field(
        default="semantic_scholar",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "semantic_scholar",
                "description": "Primary literature search engine for novelty checks.",
                "options": [
                    {"label": "Semantic Scholar", "value": "semantic_scholar"},
                    {"label": "OpenAlex", "value": "openalex"},
                ]
            }
        }
    )

    novelty_max_rounds: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 4,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum query-reflection rounds for novelty checks."
            }
        }
    )

    novelty_min_confidence_to_block: float = Field(
        default=0.75,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.75,
                "min": 0.0,
                "max": 1.0,
                "description": "If non-novel confidence meets this threshold, block claim validation as novel."
            }
        }
    )

    novelty_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional dedicated model for novelty checks. Falls back to worker/research model."
            }
        }
    )

    # ==========================================================================
    # Reflection Loop Settings
    # ==========================================================================

    enable_hypothesis_reflection: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable iterative hypothesis reflection/refinement before experiment execution."
            }
        }
    )

    reflection_max_rounds: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 8,
                "step": 1,
                "description": "Maximum reflection rounds for each hypothesis."
            }
        }
    )

    reflection_done_keyword: str = Field(
        default="I AM DONE",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "I AM DONE",
                "description": "Keyword used by reflection model to indicate convergence."
            }
        }
    )

    reflection_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional dedicated model for hypothesis reflection."
            }
        }
    )

    # ==========================================================================
    # Experiment Orchestrator Settings
    # ==========================================================================

    orchestrator_max_runs: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 8,
                "step": 1,
                "description": "Maximum planned runs per orchestrated experiment cycle."
            }
        }
    )

    orchestrator_max_fix_attempts: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 2,
                "min": 0,
                "max": 6,
                "step": 1,
                "description": "Maximum fix/retry attempts when experiment code fails."
            }
        }
    )

    orchestrator_require_explicit_completion: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Require explicit completion criteria to stop run orchestration."
            }
        }
    )

    # ==========================================================================
    # Paper Pipeline Settings
    # ==========================================================================

    enable_paper_pipeline: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Generate a structured paper draft and perform an automated review pass."
            }
        }
    )

    writeup_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional dedicated model for draft writing."
            }
        }
    )

    review_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Optional dedicated model for paper reviews."
            }
        }
    )

    review_reflection_rounds: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 2,
                "min": 1,
                "max": 5,
                "step": 1,
                "description": "Number of review improvement loops over generated draft."
            }
        }
    )

    # ==========================================================================
    # Experiment Pack Settings
    # ==========================================================================

    experiment_pack_id: str = Field(
        default="astronomy_default",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "astronomy_default",
                "description": "Experiment pack profile to guide orchestration (domain heuristics, metrics, checks)."
            }
        }
    )

    enforce_experiment_pack_contract: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "If enabled, planner must satisfy pack-required metrics and outputs."
            }
        }
    )

    # ==========================================================================
    # E2B Code Interpreter Settings
    # ==========================================================================
    
    enable_code_interpreter: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable E2B code interpreter for computational experiments"
            }
        }
    )
    
    sandbox_timeout_seconds: int = Field(
        default=3600,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3600,
                "min": 300,
                "max": 7200,
                "description": "Maximum sandbox lifetime in seconds (default 1 hour)"
            }
        }
    )
    
    code_execution_timeout: int = Field(
        default=300,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 300,
                "min": 60,
                "max": 1800,
                "description": "Maximum time for a single code execution in seconds"
            }
        }
    )
    
    use_persistent_sandbox: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Use a persistent sandbox for iterative experiments (recommended)"
            }
        }
    )
    
    setup_scientific_environment: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Pre-install scientific packages (astropy, etc.) in sandbox"
            }
        }
    )
    
    # ==========================================================================
    # Discovery Workflow Settings
    # ==========================================================================
    
    max_discovery_iterations: int = Field(
        default=15,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 15,
                "min": 3,
                "max": 30,
                "step": 1,
                "description": "Maximum number of hypothesis-experiment-analyze cycles"
            }
        }
    )
    
    max_experiments_per_hypothesis: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum experiments to run for a single hypothesis"
            }
        }
    )
    
    require_statistical_evidence: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Require statistical significance for hypothesis acceptance"
            }
        }
    )
    
    significance_level: float = Field(
        default=0.05,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.05,
                "min": 0.001,
                "max": 0.1,
                "description": "Statistical significance level (alpha) for hypothesis testing"
            }
        }
    )

    # ==========================================================================
    # Claim Validation / Replication Gate
    # ==========================================================================

    claim_validation_min_novelty_confidence: float = Field(
        default=0.6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.6,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum novelty confidence for a claim to be eligible for validated status."
            }
        }
    )

    require_replication_for_novel_claims: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Require a replication check for high-impact novel claims before validating them."
            }
        }
    )

    max_replication_attempts: int = Field(
        default=1,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 1,
                "min": 1,
                "max": 3,
                "step": 1,
                "description": "Maximum replication attempts per claim when replication is required."
            }
        }
    )
    
    # ==========================================================================
    # Scientific Domain Settings
    # ==========================================================================
    
    scientific_domain: ScientificDomain = Field(
        default=ScientificDomain.GENERAL,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "general",
                "description": "Primary scientific domain for specialized tools and prompts",
                "options": [
                    {"label": "General", "value": ScientificDomain.GENERAL.value},
                    {"label": "Astronomy", "value": ScientificDomain.ASTRONOMY.value},
                    {"label": "Physics", "value": ScientificDomain.PHYSICS.value},
                    {"label": "Biology", "value": ScientificDomain.BIOLOGY.value},
                    {"label": "Chemistry", "value": ScientificDomain.CHEMISTRY.value},
                    {"label": "Climate Science", "value": ScientificDomain.CLIMATE.value},
                ]
            }
        }
    )
    
    enable_telescope_archives: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable access to telescope archives (NASA MAST, SDSS, etc.)"
            }
        }
    )
    
    enable_exoplanet_archive: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable access to NASA Exoplanet Archive"
            }
        }
    )
    
    enable_arxiv_deep_extraction: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable deep extraction of equations and data from ArXiv papers"
            }
        }
    )
    
    # ==========================================================================
    # Output Management Settings
    # ==========================================================================
    
    save_computation_outputs: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Save all computational outputs (figures, data) for the report"
            }
        }
    )
    
    max_figures_in_report: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "max": 50,
                "description": "Maximum number of figures to include in the final report"
            }
        }
    )
    
    interpret_images_with_ai: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Use AI to interpret and caption generated visualizations"
            }
        }
    )
    
    # ==========================================================================
    # Model Settings for Discovery
    # ==========================================================================
    
    hypothesis_model: str = Field(
        default="openai:gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4o",
                "description": "Model for generating and refining hypotheses"
            }
        }
    )
    
    experiment_design_model: str = Field(
        default="openai:gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4o",
                "description": "Model for designing computational experiments"
            }
        }
    )
    
    analysis_model: str = Field(
        default="openai:gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4o",
                "description": "Model for analyzing experiment results"
            }
        }
    )
    
    @classmethod
    def from_runnable_config(
        cls,
        config: Optional[RunnableConfig] = None
    ) -> "ComputationalConfiguration":
        """Create a ComputationalConfiguration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


# =============================================================================
# Helper Functions
# =============================================================================

def get_domain_specific_packages(domain: ScientificDomain) -> List[str]:
    """Get recommended Python packages for a scientific domain."""
    base_packages = [
        "numpy", "scipy", "pandas", "matplotlib", "seaborn",
        "statsmodels", "scikit-learn"
    ]
    
    domain_packages = {
        ScientificDomain.ASTRONOMY: [
            "astropy", "astroquery", "photutils", "specutils",
            "reproject", "regions"
        ],
        ScientificDomain.PHYSICS: [
            "sympy", "uncertainties", "pint", "lmfit"
        ],
        ScientificDomain.BIOLOGY: [
            "biopython", "networkx", "lifelines"
        ],
        ScientificDomain.CHEMISTRY: [
            "rdkit", "openbabel", "chempy"
        ],
        ScientificDomain.CLIMATE: [
            "xarray", "cartopy", "netCDF4", "cftime"
        ],
        ScientificDomain.GENERAL: []
    }
    
    return base_packages + domain_packages.get(domain, [])


def get_domain_data_sources(domain: ScientificDomain) -> List[str]:
    """Get relevant data sources for a scientific domain."""
    sources = {
        ScientificDomain.ASTRONOMY: [
            "NASA MAST Archive",
            "NASA Exoplanet Archive", 
            "SDSS",
            "VizieR",
            "SIMBAD"
        ],
        ScientificDomain.PHYSICS: [
            "arXiv (physics)",
            "NIST databases",
            "HEPData"
        ],
        ScientificDomain.BIOLOGY: [
            "PubMed",
            "UniProt",
            "NCBI",
            "Gene Ontology"
        ],
        ScientificDomain.CHEMISTRY: [
            "PubChem",
            "ChEMBL",
            "NIST Chemistry"
        ],
        ScientificDomain.CLIMATE: [
            "NOAA",
            "NASA GISS",
            "ERA5",
            "CMIP6"
        ],
        ScientificDomain.GENERAL: [
            "arXiv",
            "Google Scholar",
            "Web Search"
        ]
    }
    
    return sources.get(domain, sources[ScientificDomain.GENERAL])


def get_domain_prompt_context(domain: ScientificDomain) -> str:
    """Get domain-specific guidance text for injection into prompts.
    
    This replaces hardcoded astronomy references with domain-appropriate
    data sources, example queries, and methodology guidance.
    """
    contexts = {
        ScientificDomain.ASTRONOMY: """
**Domain: Astronomy & Astrophysics**

Available real data sources via **astroquery**:
- **NASA Exoplanet Archive**: Confirmed exoplanets (mass, radius, orbital parameters, stellar properties)
- **SIMBAD**: Stellar classifications, coordinates, proper motions
- **VizieR**: MILLIONS of catalogs — Gaia DR3, atmospheric retrievals, spectral surveys, chemical abundances, and much more. Use `Vizier.find_catalogs("your topic")` to discover!
- **MAST**: HST, JWST, Kepler, TESS observations and data products
- **SDSS**: Galaxy surveys, spectroscopic data
- **Lightkurve**: Time-series photometry from Kepler/TESS
- **HITRAN/Splatalogue**: Molecular spectral line databases
- **IRSA**: NASA/IPAC Infrared Science Archive
- **And many more**: Use ExploreData to discover all available astroquery modules!

**IMPORTANT**: Before simulating data, ALWAYS search VizieR and MAST for existing observations.
VizieR alone hosts millions of catalogs including published atmospheric retrievals and spectral surveys.

Install astroquery in the sandbox: `pip install astroquery astropy lightkurve`

Key packages: astropy, astroquery, lightkurve, rebound, specutils
""",
        ScientificDomain.PHYSICS: """
**Domain: Physics**

Available data and computation tools:
- **NIST databases**: Physical constants, atomic spectra, material properties
- **arXiv**: Latest physics papers and preprints
- **HEPData**: High energy physics experimental data
- Simulation tools: scipy.integrate, sympy, numba for computational physics

Key packages: sympy, uncertainties, pint, lmfit, scipy
""",
        ScientificDomain.BIOLOGY: """
**Domain: Biology & Bioinformatics**

Available data sources:
- **UniProt**: Protein sequences and functional information
- **NCBI/GenBank**: Genomic sequences and annotations
- **PubMed**: Biomedical literature
- **Gene Ontology**: Functional gene annotations
- **Biopython**: Sequence analysis tools

Key packages: biopython, scikit-bio, networkx, lifelines
""",
        ScientificDomain.CHEMISTRY: """
**Domain: Chemistry**

Available data sources:
- **PubChem**: Chemical compound data, bioactivity
- **ChEMBL**: Drug-like molecules and bioactivity data
- **NIST Chemistry WebBook**: Thermodynamic data, spectra
- Computational chemistry: molecular descriptors, reaction energetics

Key packages: rdkit, ase, pymatgen, chempy, cclib
""",
        ScientificDomain.CLIMATE: """
**Domain: Climate Science**

Available data sources:
- **NOAA**: Climate observations, weather data
- **NASA GISS**: Global temperature records
- **ERA5**: Reanalysis climate data
- **CMIP6**: Climate model outputs

Key packages: xarray, cartopy, netCDF4, cftime, cfgrib
""",
        ScientificDomain.GENERAL: """
**Domain: General Scientific Research**

Available tools:
- **arXiv**: Academic papers across all sciences
- **Web APIs**: Various scientific data APIs via fetch_scientific_api
- Statistical analysis: scipy.stats, statsmodels, scikit-learn
- Visualization: matplotlib, seaborn, plotly

Key packages: numpy, scipy, pandas, matplotlib, scikit-learn, statsmodels
"""
    }
    
    return contexts.get(domain, contexts[ScientificDomain.GENERAL])
