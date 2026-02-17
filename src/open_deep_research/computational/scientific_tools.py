"""Scientific Data Access Tools for Computational Discovery.

This module provides tools for accessing scientific data sources:
- ArXiv papers with equation/data extraction
- Astronomical databases (NASA MAST, Exoplanet Archive, SDSS)
- PubMed papers
- General scientific APIs

These tools enable the discovery system to gather real scientific data
for computational experiments and hypothesis testing.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import aiohttp
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from open_deep_research.computational.state import (
    DataSourceType,
    ExtractedDataTable,
    ExtractedEquation,
    PaperData,
    ScientificDataSource,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sources Directory Helper
# =============================================================================

def _get_sources_dir(config: RunnableConfig = None) -> Optional[str]:
    """Extract the sources directory path from the runtime config.
    
    The sources directory is where downloaded articles and documents
    are persisted for traceability and reproducibility.
    
    Args:
        config: Runtime configuration that may contain sources_dir
        
    Returns:
        Path to the sources directory, or None if not configured
    """
    if config is None:
        return None
    configurable = config.get("configurable", {})
    return configurable.get("sources_dir", None)


# =============================================================================
# Persistent ArXiv Retriever (keeps downloaded PDFs)
# =============================================================================

def _persistent_arxiv_lazy_load(
    wrapper,
    query: str,
    sources_dir: Optional[str] = None
) -> Iterator[Document]:
    """Load ArXiv papers, keeping downloaded PDFs in sources directory.
    
    This replaces the default ArxivAPIWrapper.lazy_load() which deletes
    PDFs immediately after reading. Instead, PDFs are saved to the
    sources directory for traceability and reproducibility.
    
    Args:
        wrapper: An ArxivAPIWrapper instance (or subclass like ArxivRetriever)
        query: Search query string
        sources_dir: Directory to persist downloaded PDFs (None = delete after read)
        
    Yields:
        Document objects with paper text and metadata
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF package not found, please install it with "
            "`pip install pymupdf`"
        )

    try:
        query = query.replace(":", "").replace("-", "")
        results = wrapper._fetch_results(query)
    except wrapper.arxiv_exceptions as ex:
        logger.debug("Error on arxiv: %s", ex)
        return

    for result in results:
        doc_file_name = None
        try:
            if sources_dir:
                os.makedirs(sources_dir, exist_ok=True)
                arxiv_id = result.entry_id.split("/")[-1]
                safe_id = re.sub(r'[^\w.-]', '_', arxiv_id)
                pdf_filename = f"{safe_id}.pdf"
                dest_path = os.path.join(sources_dir, pdf_filename)

                if os.path.exists(dest_path):
                    doc_file_name = dest_path
                else:
                    doc_file_name = result.download_pdf(
                        dirpath=sources_dir,
                        filename=pdf_filename,
                    )
                logger.info(f"PDF saved to sources: {doc_file_name}")
            else:
                doc_file_name = result.download_pdf()

            with fitz.open(doc_file_name) as doc_file:
                text: str = "".join(page.get_text() for page in doc_file)

        except FileNotFoundError as f_ex:
            logger.debug(f_ex)
            continue
        except Exception as e:
            if wrapper.continue_on_failure:
                logger.error(e)
                continue
            else:
                raise e

        if wrapper.load_all_available_meta:
            extra_metadata = {
                "entry_id": result.entry_id,
                "published_first_time": str(result.published.date()),
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
                "primary_category": result.primary_category,
                "categories": result.categories,
                "links": [link.href for link in result.links],
            }
        else:
            extra_metadata = {}

        metadata = {
            "Published": str(result.updated.date()),
            "Title": result.title,
            "Authors": ", ".join(a.name for a in result.authors),
            "Summary": result.summary,
            **extra_metadata,
        }

        yield Document(
            page_content=(
                text[: wrapper.doc_content_chars_max]
                if wrapper.doc_content_chars_max
                else text
            ),
            metadata=metadata,
        )

        # Only delete the temp file when there is no sources directory
        if not sources_dir and doc_file_name:
            try:
                os.remove(doc_file_name)
            except OSError:
                pass


# =============================================================================
# ArXiv Paper Tools with Data Extraction
# =============================================================================

ARXIV_SEARCH_DESCRIPTION = """
Search ArXiv for academic papers and extract computational data.

This tool searches the ArXiv preprint repository and extracts:
- Paper metadata (title, authors, abstract)
- Mathematical equations in LaTeX format
- Key parameters and methodologies
- Data tables when available

USE THIS TOOL WHEN:
- You need to find recent research on a scientific topic
- You want to extract equations for computational verification
- You need methodology details for reproducing experiments
- You want to find data published in papers

EXAMPLE QUERIES:
- "exoplanet orbital period stellar metallicity"
- "gravitational wave detection LIGO"
- "neural network architecture transformers"
"""


class ArxivSearchInput(BaseModel):
    """Input for ArXiv search."""
    query: str = Field(description="Search query for ArXiv")
    max_results: int = Field(default=5, description="Maximum papers to return")
    extract_equations: bool = Field(default=True, description="Extract mathematical equations")
    extract_methodology: bool = Field(default=True, description="Extract methodology details")


@tool(description=ARXIV_SEARCH_DESCRIPTION)
async def search_arxiv_papers(
    query: str,
    max_results: int = 5,
    extract_equations: bool = True,
    config: RunnableConfig = None
) -> str:
    """Search ArXiv and extract computational data from papers.
    
    Args:
        query: Search query
        max_results: Maximum papers to retrieve
        extract_equations: Whether to extract equations
        
    Returns:
        Formatted string with paper data and extracted information
    """
    try:
        from langchain_community.retrievers import ArxivRetriever
    except ImportError:
        return "ArXiv retriever not available. Install langchain-community."
    
    try:
        # Create retriever
        retriever = ArxivRetriever(
            load_max_docs=max_results,
            get_full_documents=True,
            load_all_available_meta=True
        )
        
        # Use persistent loader that keeps PDFs in sources directory
        sources_dir = _get_sources_dir(config)
        
        # Run in thread pool (synchronous API)
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: list(_persistent_arxiv_lazy_load(retriever, query, sources_dir))
        )
        
        if not docs:
            return f"No papers found for query: {query}"
        
        results = []
        for doc in docs:
            metadata = doc.metadata
            paper_info = {
                "title": metadata.get("Title", "Unknown"),
                "authors": metadata.get("Authors", "Unknown"),
                "arxiv_id": metadata.get("entry_id", ""),
                "published": str(metadata.get("Published", "")),
                "abstract": metadata.get("Summary", ""),
                "categories": metadata.get("categories", []),
            }
            
            # Extract equations if requested
            if extract_equations and doc.page_content:
                equations = extract_latex_equations(doc.page_content)
                paper_info["equations"] = equations[:10]  # Limit to 10
            
            # Extract key parameters
            if doc.page_content:
                params = extract_key_parameters(doc.page_content)
                paper_info["key_parameters"] = params
            
            results.append(paper_info)
        
        # Format output
        output_parts = [f"=== ArXiv Search Results for: {query} ===\n"]
        output_parts.append(f"Found {len(results)} papers\n")
        
        for i, paper in enumerate(results, 1):
            output_parts.append(f"\n--- Paper {i}: {paper['title']} ---")
            output_parts.append(f"Authors: {paper['authors']}")
            output_parts.append(f"ArXiv ID: {paper['arxiv_id']}")
            output_parts.append(f"Published: {paper['published']}")
            output_parts.append(f"Categories: {', '.join(paper.get('categories', []))}")
            output_parts.append(f"\nAbstract: {paper['abstract'][:500]}...")
            
            if paper.get("equations"):
                output_parts.append(f"\nExtracted Equations ({len(paper['equations'])}):")
                for eq in paper["equations"][:5]:
                    output_parts.append(f"  - {eq}")
            
            if paper.get("key_parameters"):
                output_parts.append(f"\nKey Parameters: {json.dumps(paper['key_parameters'], indent=2)}")
            
            output_parts.append("")
        
        return "\n".join(output_parts)
        
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return f"ArXiv search failed: {str(e)}"


def extract_latex_equations(text: str) -> List[str]:
    """Extract LaTeX equations from paper text."""
    equations = []
    
    # Match display equations
    display_patterns = [
        r'\$\$(.*?)\$\$',  # $$...$$
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
        r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}',
        r'\\\[(.*?)\\\]',  # \[...\]
    ]
    
    for pattern in display_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        equations.extend([m.strip() for m in matches if len(m.strip()) > 3])
    
    # Match inline equations (be selective to avoid noise)
    inline_matches = re.findall(r'\$([^$]{5,100})\$', text)
    for match in inline_matches:
        # Filter out simple expressions
        if any(op in match for op in ['=', '\\frac', '\\int', '\\sum', '\\partial']):
            equations.append(match.strip())
    
    return list(set(equations))[:20]  # Deduplicate and limit


def extract_key_parameters(text: str) -> Dict[str, Any]:
    """Extract key numerical parameters from paper text."""
    params = {}
    
    # Common parameter patterns
    patterns = [
        (r'([a-zA-Z_]+)\s*=\s*([0-9.e+-]+)\s*(?:±|\\pm|\\pm)\s*([0-9.e+-]+)', 'value_with_error'),
        (r'([a-zA-Z_]+)\s*=\s*([0-9.e+-]+)', 'simple_value'),
        (r'mass[:\s]+([0-9.e+-]+)\s*(?:M_?\{?(?:sun|⊙|\\odot)\}?)?', 'mass'),
        (r'radius[:\s]+([0-9.e+-]+)\s*(?:R_?\{?(?:sun|⊙|\\odot|earth|\\oplus)\}?)?', 'radius'),
        (r'period[:\s]+([0-9.e+-]+)\s*(?:days?|d)?', 'period'),
        (r'temperature[:\s]+([0-9.e+-]+)\s*K?', 'temperature'),
    ]
    
    for pattern, param_type in patterns[:2]:  # Limit to avoid noise
        matches = re.findall(pattern, text[:5000], re.IGNORECASE)
        for match in matches[:5]:  # Limit matches
            if isinstance(match, tuple):
                if param_type == 'value_with_error':
                    params[match[0]] = {"value": match[1], "error": match[2]}
                else:
                    params[match[0]] = match[1]
            else:
                params[param_type] = match
    
    return params


# =============================================================================
# Astronomical Database Tools
# =============================================================================

NASA_EXOPLANET_DESCRIPTION = """
Query the NASA Exoplanet Archive for confirmed exoplanet data.

This tool accesses the official NASA database of confirmed exoplanets,
providing data for computational analysis including:
- Planetary parameters (mass, radius, orbital period, semi-major axis)
- Host star properties (mass, radius, temperature, metallicity)
- Discovery information
- System configurations

USE THIS TOOL WHEN:
- Analyzing exoplanet populations
- Testing relationships between planetary/stellar properties
- Gathering real data for statistical analysis
- Validating models against observed systems

EXAMPLE QUERIES:
- Get all confirmed exoplanets
- Filter by discovery method
- Get planets in specific mass/radius ranges
"""


@tool(description=NASA_EXOPLANET_DESCRIPTION)
async def query_nasa_exoplanet_archive(
    columns: List[str] = None,
    where_clause: str = None,
    limit: int = 100,
    config: RunnableConfig = None
) -> str:
    """Query NASA Exoplanet Archive via TAP API.
    
    Args:
        columns: Columns to retrieve (default: common planetary parameters)
        where_clause: SQL WHERE clause for filtering (e.g., "pl_bmassj > 1")
        limit: Maximum rows to return
        
    Returns:
        Formatted data from the archive
    """
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Default columns if not specified
    if columns is None:
        columns = [
            "pl_name",           # Planet name
            "hostname",          # Host star name
            "discoverymethod",   # Discovery method
            "disc_year",         # Discovery year
            "pl_orbper",         # Orbital period (days)
            "pl_bmassj",         # Planet mass (Jupiter masses)
            "pl_radj",           # Planet radius (Jupiter radii)
            "pl_orbsmax",        # Semi-major axis (AU)
            "st_mass",           # Stellar mass (solar masses)
            "st_rad",            # Stellar radius (solar radii)
            "st_teff",           # Stellar effective temperature (K)
            "st_met",            # Stellar metallicity [Fe/H]
        ]
    
    # Build query
    columns_str = ", ".join(columns)
    query = f"SELECT {columns_str} FROM ps"
    
    if where_clause:
        query += f" WHERE {where_clause}"
    
    query += f" ORDER BY disc_year DESC LIMIT {limit}"
    
    params = {
        "query": query,
        "format": "json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data:
                        return "No exoplanets found matching criteria."
                    
                    # Format output
                    output_parts = [
                        f"=== NASA Exoplanet Archive Query Results ===",
                        f"Query: {query}",
                        f"Found {len(data)} planets\n",
                    ]
                    
                    # Summary statistics
                    if len(data) > 0:
                        output_parts.append("Sample Data (first 10 rows):")
                        for i, planet in enumerate(data[:10]):
                            output_parts.append(f"\n{i+1}. {planet.get('pl_name', 'Unknown')}")
                            output_parts.append(f"   Host: {planet.get('hostname', 'Unknown')}")
                            output_parts.append(f"   Period: {planet.get('pl_orbper', 'N/A')} days")
                            output_parts.append(f"   Mass: {planet.get('pl_bmassj', 'N/A')} M_Jup")
                            output_parts.append(f"   Radius: {planet.get('pl_radj', 'N/A')} R_Jup")
                            output_parts.append(f"   Star [Fe/H]: {planet.get('st_met', 'N/A')}")
                    
                    # Add data as JSON for computation
                    output_parts.append(f"\n\n=== Full JSON Data ({len(data)} rows) ===")
                    output_parts.append("```json")
                    output_parts.append(json.dumps(data[:50], indent=2))  # Limit for readability
                    if len(data) > 50:
                        output_parts.append(f"... and {len(data) - 50} more rows")
                    output_parts.append("```")
                    
                    return "\n".join(output_parts)
                else:
                    return f"NASA API error: {response.status}"
                    
    except Exception as e:
        logger.error(f"NASA Exoplanet Archive query failed: {e}")
        return f"Query failed: {str(e)}"


MAST_ARCHIVE_DESCRIPTION = """
Query NASA MAST (Mikulski Archive for Space Telescopes) for astronomical observations.

MAST hosts data from major space telescopes including:
- Hubble Space Telescope (HST)
- James Webb Space Telescope (JWST)
- TESS (Transiting Exoplanet Survey Satellite)
- Kepler/K2
- GALEX, Swift, and more

USE THIS TOOL WHEN:
- Looking for observations of specific astronomical objects
- Gathering photometric or spectroscopic data
- Finding time-series data for variable objects
- Accessing high-resolution images
"""


@tool(description=MAST_ARCHIVE_DESCRIPTION)
async def query_mast_archive(
    target: str,
    radius: float = 0.2,
    mission: str = None,
    config: RunnableConfig = None
) -> str:
    """Query MAST archive for observations of a target.
    
    Args:
        target: Target name (e.g., "M31", "Kepler-186") or coordinates
        radius: Search radius in degrees
        mission: Specific mission to search (HST, JWST, TESS, etc.)
        
    Returns:
        Available observations and data products
    """
    base_url = "https://mast.stsci.edu/api/v0/invoke"
    
    # Build request
    request_data = {
        "service": "Mast.Caom.Cone",
        "params": {
            "ra": None,
            "dec": None,
            "radius": radius
        },
        "format": "json"
    }
    
    # Resolve target name to coordinates
    resolve_url = "https://mast.stsci.edu/api/v0/invoke"
    resolve_data = {
        "service": "Mast.Name.Lookup",
        "params": {"input": target, "format": "json"}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # First resolve the target name
            async with session.post(
                resolve_url,
                json=resolve_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            ) as resolve_response:
                if resolve_response.status == 200:
                    resolve_result = await resolve_response.json()
                    if resolve_result.get("resolvedCoordinate"):
                        coords = resolve_result["resolvedCoordinate"][0]
                        request_data["params"]["ra"] = coords["ra"]
                        request_data["params"]["dec"] = coords["decl"]
                    else:
                        return f"Could not resolve target: {target}"
                else:
                    return f"Target resolution failed: {resolve_response.status}"
            
            # Now query the archive
            async with session.post(
                base_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data.get("data"):
                        return f"No observations found for {target}"
                    
                    observations = data["data"]
                    
                    # Filter by mission if specified
                    if mission:
                        observations = [
                            obs for obs in observations
                            if mission.upper() in obs.get("obs_collection", "").upper()
                        ]
                    
                    # Format output
                    output_parts = [
                        f"=== MAST Archive Results for: {target} ===",
                        f"Coordinates: RA={request_data['params']['ra']:.4f}, Dec={request_data['params']['dec']:.4f}",
                        f"Search radius: {radius}°",
                        f"Found {len(observations)} observations\n",
                    ]
                    
                    # Group by mission
                    missions = {}
                    for obs in observations:
                        mission_name = obs.get("obs_collection", "Unknown")
                        if mission_name not in missions:
                            missions[mission_name] = []
                        missions[mission_name].append(obs)
                    
                    output_parts.append("Observations by Mission:")
                    for mission_name, obs_list in sorted(missions.items()):
                        output_parts.append(f"\n{mission_name}: {len(obs_list)} observations")
                        for obs in obs_list[:3]:  # Show first 3
                            output_parts.append(f"  - {obs.get('obs_id', 'N/A')}: {obs.get('target_name', 'N/A')}")
                            output_parts.append(f"    Instrument: {obs.get('instrument_name', 'N/A')}")
                            output_parts.append(f"    Exposure: {obs.get('t_exptime', 'N/A')}s")
                    
                    return "\n".join(output_parts)
                else:
                    return f"MAST API error: {response.status}"
                    
    except Exception as e:
        logger.error(f"MAST query failed: {e}")
        return f"MAST query failed: {str(e)}"


# =============================================================================
# SDSS (Sloan Digital Sky Survey) Tools
# =============================================================================

SDSS_DESCRIPTION = """
Query the Sloan Digital Sky Survey (SDSS) database.

SDSS is one of the most successful surveys in astronomy, providing:
- Photometry for hundreds of millions of objects
- Spectra for millions of galaxies and quasars
- Detailed catalogs of stars, galaxies, and quasars

USE THIS TOOL WHEN:
- Analyzing large samples of galaxies or stars
- Getting spectroscopic classifications
- Studying galaxy properties and distributions
- Statistical analysis of astronomical populations
"""


@tool(description=SDSS_DESCRIPTION)
async def query_sdss_database(
    sql_query: str = None,
    object_type: str = "galaxy",
    limit: int = 100,
    config: RunnableConfig = None
) -> str:
    """Query SDSS database using SQL or preset queries.
    
    Args:
        sql_query: Custom SQL query (optional)
        object_type: Type of object ('galaxy', 'star', 'quasar')
        limit: Maximum rows to return
        
    Returns:
        Query results with astronomical data
    """
    base_url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
    
    # Preset queries for common object types
    preset_queries = {
        "galaxy": f"""
SELECT TOP {limit}
    p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
    s.z as redshift, s.class, s.subclass
FROM PhotoObj p
JOIN SpecObj s ON p.objid = s.bestObjID
WHERE s.class = 'GALAXY'
ORDER BY s.z
""",
        "star": f"""
SELECT TOP {limit}
    p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
    s.class, s.subclass
FROM PhotoObj p
JOIN SpecObj s ON p.objid = s.bestObjID
WHERE s.class = 'STAR'
""",
        "quasar": f"""
SELECT TOP {limit}
    p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
    s.z as redshift, s.class
FROM PhotoObj p
JOIN SpecObj s ON p.objid = s.bestObjID
WHERE s.class = 'QSO'
ORDER BY s.z DESC
""",
    }
    
    query = sql_query if sql_query else preset_queries.get(object_type, preset_queries["galaxy"])
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "cmd": query,
                "format": "json"
            }
            
            async with session.get(base_url, params=params, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if not result:
                        return "No SDSS objects found."
                    
                    # Handle SDSS response format
                    data = result[0].get("Rows", []) if isinstance(result, list) else []
                    
                    output_parts = [
                        f"=== SDSS Query Results ===",
                        f"Object type: {object_type}",
                        f"Found {len(data)} objects\n",
                    ]
                    
                    if data:
                        output_parts.append("Sample objects:")
                        for i, obj in enumerate(data[:10]):
                            output_parts.append(f"\n{i+1}. Object {obj.get('objid', 'N/A')}")
                            output_parts.append(f"   RA, Dec: {obj.get('ra', 'N/A')}, {obj.get('dec', 'N/A')}")
                            output_parts.append(f"   Magnitudes (u,g,r,i,z): {obj.get('u', 'N/A')}, {obj.get('g', 'N/A')}, {obj.get('r', 'N/A')}, {obj.get('i', 'N/A')}, {obj.get('z', 'N/A')}")
                            if obj.get('redshift'):
                                output_parts.append(f"   Redshift: {obj.get('redshift', 'N/A')}")
                    
                    return "\n".join(output_parts)
                else:
                    return f"SDSS query error: {response.status}"
                    
    except Exception as e:
        logger.error(f"SDSS query failed: {e}")
        return f"SDSS query failed: {str(e)}"


# =============================================================================
# General Scientific Data Tools
# =============================================================================

@tool(description="Fetch data from any scientific API endpoint")
async def fetch_scientific_api(
    url: str,
    method: str = "GET",
    params: Dict[str, Any] = None,
    headers: Dict[str, str] = None,
    config: RunnableConfig = None
) -> str:
    """Fetch data from a scientific API.
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET or POST)
        params: Query parameters or request body
        headers: HTTP headers
        
    Returns:
        API response data
    """
    try:
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, params=params, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if "json" in content_type:
                            data = await response.json()
                            return json.dumps(data, indent=2)[:10000]  # Limit output
                        else:
                            text = await response.text()
                            return text[:10000]
                    else:
                        return f"API error: {response.status}"
            else:
                async with session.post(url, json=params, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        return json.dumps(data, indent=2)[:10000]
                    else:
                        return f"API error: {response.status}"
                        
    except Exception as e:
        return f"API request failed: {str(e)}"


# =============================================================================
# Data Extraction Helpers
# =============================================================================

async def extract_paper_data(
    arxiv_id: str,
    config: RunnableConfig = None
) -> Optional[PaperData]:
    """Extract comprehensive data from an ArXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        PaperData object with extracted information
    """
    try:
        from langchain_community.retrievers import ArxivRetriever
        
        retriever = ArxivRetriever(
            load_max_docs=1,
            get_full_documents=True,
            load_all_available_meta=True
        )
        
        # Use persistent loader that keeps PDFs in sources directory
        sources_dir = _get_sources_dir(config)
        
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: list(_persistent_arxiv_lazy_load(retriever, arxiv_id, sources_dir))
        )
        
        if not docs:
            return None
        
        doc = docs[0]
        metadata = doc.metadata
        
        # Extract equations
        equations = []
        if doc.page_content:
            latex_eqs = extract_latex_equations(doc.page_content)
            for eq in latex_eqs:
                equations.append(ExtractedEquation(
                    latex=eq,
                    source_paper_id=arxiv_id
                ))
        
        # Extract parameters
        params = extract_key_parameters(doc.page_content) if doc.page_content else {}
        
        return PaperData(
            title=metadata.get("Title", "Unknown"),
            authors=metadata.get("Authors", "").split(", "),
            arxiv_id=arxiv_id,
            url=metadata.get("entry_id", ""),
            abstract=metadata.get("Summary", ""),
            full_text=doc.page_content,
            equations=equations,
            key_parameters=params
        )
        
    except Exception as e:
        logger.error(f"Paper extraction failed: {e}")
        return None


# =============================================================================
# Tool Collection for Discovery System
# =============================================================================

def get_scientific_tools() -> List:
    """Get all scientific data access tools."""
    return [
        search_arxiv_papers,
        query_nasa_exoplanet_archive,
        query_mast_archive,
        query_sdss_database,
        fetch_scientific_api,
    ]
