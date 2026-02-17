"""Context module for providing AI agents with pre-loaded scientific database schemas.

This module loads context files that contain metadata about astronomical databases,
so the AI knows what columns exist without needing to explore at runtime.

It also provides DATA DISCOVERY utilities: executable code templates that the AI
can run inside E2B to dynamically discover what data is available in astroquery,
enabling it to find datasets it didn't know existed.
"""

import os
from pathlib import Path
from functools import lru_cache


CONTEXT_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def get_astroquery_context() -> str:
    """Load the astroquery context file.
    
    Returns:
        The full astroquery reference markdown content
    """
    context_file = CONTEXT_DIR / "astroquery_context.md"
    if context_file.exists():
        return context_file.read_text()
    return ""


@lru_cache(maxsize=1)
def get_astroquery_quick_reference() -> str:
    """Get a condensed quick reference for astroquery.
    
    This is designed to be included directly in prompts without
    overwhelming the context window.
    
    Returns:
        A condensed reference string with essential column names
    """
    return """
## ASTROQUERY QUICK REFERENCE - USE THESE EXACT COLUMN NAMES!

### NASA Exoplanet Archive (table="ps")
**PLANET COLUMNS:**
| Column | Description | Unit |
|--------|-------------|------|
| `pl_name` | Planet name | - |
| `pl_rade` | Planet radius | Earth radii |
| `pl_bmasse` | Planet mass | Earth masses |
| `pl_orbper` | Orbital period | days |
| `pl_orbsmax` | Semi-major axis | AU |
| `pl_eqt` | Equilibrium temperature | K |
| `pl_dens` | Planet density | g/cm³ |
| `pl_insol` | Insolation flux | Earth flux |
| `pl_trandur` | Transit duration | hours |
| `pl_tranmid` | Transit midpoint | BJD |

**STELLAR COLUMNS:**
| Column | Description | Unit |
|--------|-------------|------|
| `hostname` | Host star name | - |
| `st_teff` | Stellar temperature | K |
| `st_rad` | Stellar radius | Solar radii |
| `st_mass` | Stellar mass | Solar masses |
| `st_met` | Metallicity [Fe/H] | dex |
| `st_logg` | Surface gravity | log(cgs) |
| `st_age` | Stellar age | Gyr |

**DISCOVERY COLUMNS:**
| Column | Description | Values |
|--------|-------------|--------|
| `discoverymethod` | How planet was found | Transit, Radial Velocity, Imaging, Microlensing |
| `disc_year` | Year discovered | - |
| `disc_facility` | Discovery facility | - |

**IMPORTANT FLAGS:**
| Column | Description |
|--------|-------------|
| `default_flag` | =1 for best data per planet (USE THIS!) |

### VizieR - Gaia DR3 (catalog='I/355/gaiadr3')
| Column | Description | Unit |
|--------|-------------|------|
| `Source` | Unique source ID | - |
| `RA_ICRS` | Right Ascension | deg |
| `DE_ICRS` | Declination | deg |
| `Plx` | Parallax | mas |
| `pmRA` | Proper motion RA | mas/yr |
| `pmDE` | Proper motion Dec | mas/yr |
| `Gmag` | G magnitude | mag |
| `BP-RP` | Color index | mag |
| `Teff` | Temperature | K |

**IMPORTANT:** Set `Vizier.ROW_LIMIT = -1` to get all results!

### SIMBAD
**Default fields:** MAIN_ID, RA, DEC, COO_QUAL
**Add with add_votable_fields():** plx, rv_value, fe_h, sptype, pm, flux(V), otype

### MAST Collections
HST, JWST, TESS, Kepler, K2, GALEX, PS1

### TYPE CONVERSION (ALWAYS DO THIS!)
```python
import pandas as pd
if hasattr(result, 'to_pandas'):
    df = result.to_pandas()  # Astropy Table
elif isinstance(result, pd.DataFrame):
    df = result
else:
    df = pd.DataFrame(result)
```
"""


@lru_cache(maxsize=1)
def get_data_discovery_guidance() -> str:
    """Get guidance text about the data discovery self-reflection system.
    
    This is injected into supervisor and experiment design prompts to encourage
    the AI to discover data sources rather than assuming or simulating.
    
    Returns:
        Guidance text about data discovery capabilities
    """
    return """
## DATA DISCOVERY SELF-REFLECTION SYSTEM

You have a powerful capability: you can DISCOVER what data exists in scientific
databases before deciding to simulate or use theoretical calculations.

### BEFORE SIMULATING, ALWAYS ASK:
1. "Could this data exist in a database or catalog I haven't explored?"
2. "Has anyone published measurements, surveys, or retrievals related to my research?"
3. "Are there relevant observations available in any archive I can query?"
4. "Have I searched with broad enough keywords across multiple data services?"

### HOW TO DISCOVER DATA:
Use the **ExploreData** tool. Derive your keywords and targets from the research
question -- do NOT rely on pre-set examples. The system's strength is autonomous discovery.

**Strategy 1: Catalog Search** (searches millions of published catalogs)
```
ExploreData(
    data_source="VizieR",
    exploration_goal="Search for catalogs related to [DERIVE FROM YOUR RESEARCH QUESTION]. Use Vizier.find_catalogs() with keywords you extract from your hypothesis. Inspect the top results."
)
```

**Strategy 2: Data Service Discovery** (find databases you didn't know existed)
```
ExploreData(
    data_source="astroquery modules",
    exploration_goal="List all available astroquery submodules to find specialized data services relevant to my research."
)
```

**Strategy 3: Multi-Source Search** (cast a wide net)
```
ExploreData(
    data_source="multiple sources",
    exploration_goal="Search across available databases for [DERIVE DATA TYPE FROM YOUR RESEARCH] related to [DERIVE TARGETS FROM YOUR HYPOTHESIS]. Report what is available."
)
```

### CRITICAL RULE:
If you find yourself generating synthetic data (np.random, simulated observations),
STOP and run a data discovery exploration FIRST. The data you need may exist in a
catalog you simply haven't searched yet.

### KEY PRINCIPLE:
Derive ALL search terms, keywords, and targets from your own research question.
Do not assume you know what databases contain -- discover it programmatically.
"""


def get_context_for_domain(domain: str) -> str:
    """Get context for a specific scientific domain.
    
    Args:
        domain: The domain (e.g., "astronomy", "chemistry", "biology")
    
    Returns:
        Context string for that domain
    """
    if domain.lower() in ["astronomy", "astro", "exoplanet", "stellar"]:
        return get_astroquery_quick_reference()
    
    # Add other domains as needed
    return ""
