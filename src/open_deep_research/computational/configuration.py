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
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 10,
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
