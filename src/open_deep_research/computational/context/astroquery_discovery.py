"""Astroquery Data Discovery Code Templates.

This module provides executable code templates that the AI can run inside E2B
to dynamically discover what data is available in astroquery databases.

The key idea: instead of hard-coding what's available, the system runs discovery
code to find catalogs, columns, and datasets relevant to its research question.
This enables the AI to find data it didn't know existed.
"""


# =============================================================================
# DEEP DISCOVERY: Explore all astroquery modules and their capabilities
# =============================================================================

ASTROQUERY_MODULE_DISCOVERY_CODE = '''#!/usr/bin/env python3
"""
Astroquery Module Discovery
Purpose: Discover ALL available modules and data services in astroquery.
This helps the AI understand what data sources exist beyond the commonly used ones.
"""

import subprocess
import sys

packages = ['astroquery', 'astropy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import astroquery
import pkgutil
import importlib

print("=" * 80)
print("ASTROQUERY MODULE DISCOVERY")
print(f"astroquery version: {astroquery.__version__}")
print("=" * 80)

print("\\nAll available astroquery submodules:")
print("-" * 60)

modules_found = []
for importer, modname, ispkg in pkgutil.walk_packages(
    astroquery.__path__, prefix="astroquery."
):
    if ispkg and modname.count('.') == 1:
        modules_found.append(modname)

for mod in sorted(modules_found):
    short_name = mod.replace("astroquery.", "")
    try:
        m = importlib.import_module(mod)
        doc = (m.__doc__ or "").strip().split("\\n")[0][:100] if m.__doc__ else "No description"
        print(f"  {short_name:35s} | {doc}")
    except Exception:
        print(f"  {short_name:35s} | (could not load)")

print(f"\\nTotal modules discovered: {len(modules_found)}")
print("\\n" + "=" * 80)
print("KEY MODULES FOR ASTRONOMICAL RESEARCH:")
print("=" * 80)

key_modules = {
    "nasa_exoplanet_archive": "Confirmed exoplanets with properties (mass, radius, orbit, atmosphere)",
    "vizier": "Millions of catalogs - Gaia, 2MASS, SDSS, atmospheric retrievals, spectral surveys",
    "simbad": "Stellar classifications, coordinates, proper motions, spectral types",
    "mast": "HST, JWST, TESS, Kepler observations and data products",
    "sdss": "Sloan Digital Sky Survey - galaxy/star/quasar spectra and photometry",
    "irsa": "NASA/IPAC Infrared Science Archive - infrared surveys",
    "esa.hubble": "ESA Hubble archive",
    "gaia": "Direct Gaia archive access",
    "alma": "ALMA radio/submm observations",
    "nrao": "NRAO radio telescope data",
    "ned": "NASA Extragalactic Database",
    "heasarc": "High Energy Astrophysics Science Archive",
    "skyview": "Multi-wavelength sky images",
    "jplhorizons": "Solar system ephemerides",
    "jplsbdb": "JPL Small-Body Database",
    "splatalogue": "Molecular spectral line database",
    "atomic": "Atomic line data",
    "linelists.cdms": "Cologne Database for Molecular Spectroscopy",
    "hitran": "HITRAN molecular spectroscopy database",
    "ipac.nexsci": "NExScI archives (exoplanet + stellar data)",
}

for mod_name, description in key_modules.items():
    status = "AVAILABLE" if any(mod_name in m for m in modules_found) else "NOT FOUND"
    print(f"  [{status:9s}] {mod_name:30s} | {description}")

print("\\n" + "=" * 80)
print("RECOMMENDATION: Use VizieR catalog search to find specialized datasets!")
print("VizieR hosts millions of catalogs including atmospheric retrievals,")
print("spectral surveys, chemical abundances, and more.")
print("=" * 80)
'''


# =============================================================================
# VIZIER CATALOG DISCOVERY: Find catalogs by research topic
# =============================================================================

VIZIER_CATALOG_DISCOVERY_CODE = '''#!/usr/bin/env python3
"""
VizieR Catalog Discovery
Purpose: Search VizieR for catalogs related to a specific research topic.
VizieR hosts millions of catalogs - this discovers what's available.
"""

import subprocess
import sys

packages = ['astroquery', 'astropy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from astroquery.vizier import Vizier

SEARCH_KEYWORDS = {search_keywords}

print("=" * 80)
print("VIZIER CATALOG DISCOVERY")
print(f"Searching for: {{SEARCH_KEYWORDS}}")
print("=" * 80)

all_catalogs = {{}}
for keyword in SEARCH_KEYWORDS:
    print(f"\\n--- Searching: '{{keyword}}' ---")
    try:
        catalog_list = Vizier.find_catalogs(keyword)
        for cat_id, cat_info in catalog_list.items():
            if cat_id not in all_catalogs:
                all_catalogs[cat_id] = {{
                    "id": cat_id,
                    "description": str(getattr(cat_info, 'description', 'N/A'))[:200],
                    "keyword": keyword,
                }}
                print(f"  FOUND: {{cat_id}} | {{str(getattr(cat_info, 'description', 'N/A'))[:120]}}")
    except Exception as e:
        print(f"  Error searching '{{keyword}}': {{e}}")

print(f"\\n{{'-' * 80}}")
print(f"Total unique catalogs found: {{len(all_catalogs)}}")
print(f"{{'-' * 80}}")

# For each promising catalog, try to peek at its columns
print("\\n" + "=" * 80)
print("INSPECTING TOP CATALOGS (first 5)")
print("=" * 80)

Vizier.ROW_LIMIT = 3
for i, (cat_id, info) in enumerate(list(all_catalogs.items())[:5]):
    print(f"\\n--- Catalog: {{cat_id}} ---")
    print(f"Description: {{info['description']}}")
    try:
        tables = Vizier.get_catalogs(cat_id)
        if tables:
            for j, table in enumerate(tables):
                print(f"  Table {{j+1}}: {{len(table)}} rows, {{len(table.colnames)}} columns")
                print(f"  Columns: {{', '.join(table.colnames[:20])}}")
                if len(table.colnames) > 20:
                    print(f"           ... and {{len(table.colnames) - 20}} more columns")
                if len(table) > 0:
                    print(f"  Sample row: ", end="")
                    for col in table.colnames[:5]:
                        try:
                            print(f"{{col}}={{table[col][0]}} ", end="")
                        except:
                            pass
                    print()
    except Exception as e:
        print(f"  Could not inspect: {{e}}")

print("\\n" + "=" * 80)
print("SUMMARY: Use these catalog IDs with Vizier.query_constraints() or Vizier.get_catalogs()")
print("to access the actual data for your experiments.")
print("=" * 80)
'''


# =============================================================================
# NASA EXOPLANET ARCHIVE SCHEMA DISCOVERY
# =============================================================================

NASA_EXOPLANET_SCHEMA_DISCOVERY_CODE = '''#!/usr/bin/env python3
"""
NASA Exoplanet Archive Schema Discovery
Purpose: Discover ALL available tables and columns via TAP_SCHEMA.
"""

import subprocess
import sys

packages = ['astroquery', 'astropy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd

print("=" * 80)
print("NASA EXOPLANET ARCHIVE - SCHEMA DISCOVERY")
print("=" * 80)

# Step 1: Discover available tables
print("\\n--- Step 1: Available Tables ---")
try:
    tables_query = "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE schema_name='dbo'"
    tables_result = NasaExoplanetArchive.query_criteria(tables_query, format="table")
    if tables_result is not None and len(tables_result) > 0:
        df_tables = tables_result.to_pandas() if hasattr(tables_result, 'to_pandas') else pd.DataFrame(tables_result)
        print(f"Found {{len(df_tables)}} tables:")
        for _, row in df_tables.iterrows():
            print(f"  {{row.iloc[0]:30s}} | {{str(row.iloc[1])[:80] if len(row) > 1 else 'N/A'}}")
except Exception as e:
    print(f"TAP_SCHEMA query failed: {{e}}")
    print("Falling back to known tables...")
    known_tables = ["ps", "pscomppars", "stellarhosts", "toi", "koi"]
    for t in known_tables:
        print(f"  {{t}}")

# Step 2: Discover columns for {target_table}
TARGET_TABLE = "{target_table}"
print(f"\\n--- Step 2: Columns in '{{TARGET_TABLE}}' ---")

try:
    col_query = f"SELECT column_name, description, unit FROM TAP_SCHEMA.columns WHERE table_name='{{TARGET_TABLE}}'"
    col_result = NasaExoplanetArchive.query_criteria(col_query, format="table")
    if col_result is not None and len(col_result) > 0:
        df_cols = col_result.to_pandas() if hasattr(col_result, 'to_pandas') else pd.DataFrame(col_result)
        print(f"Found {{len(df_cols)}} columns:")
        for _, row in df_cols.iterrows():
            col_name = str(row.iloc[0])
            desc = str(row.iloc[1])[:60] if len(row) > 1 else "N/A"
            unit = str(row.iloc[2]) if len(row) > 2 else ""
            print(f"  {{col_name:35s}} | {{desc:60s}} | {{unit}}")
    else:
        print("No columns returned from TAP_SCHEMA, trying wildcard query...")
        raise Exception("Empty result")
except Exception as e:
    print(f"TAP_SCHEMA column query failed: {{e}}")
    print("Falling back to wildcard query...")
    try:
        test = NasaExoplanetArchive.query_criteria(
            table=TARGET_TABLE, select="*", where="rownum < 2"
        )
        if test is not None and len(test) > 0:
            print(f"Found {{len(test.colnames)}} columns via wildcard:")
            for i, col in enumerate(test.colnames):
                try:
                    val = test[col][0]
                    print(f"  {{i+1:3d}}. {{col:35s}} = {{val}}")
                except:
                    print(f"  {{i+1:3d}}. {{col:35s}} = [error reading]")
    except Exception as e2:
        print(f"Wildcard query also failed: {{e2}}")

# Step 3: Quick data availability check
print(f"\\n--- Step 3: Data availability in {{TARGET_TABLE}} ---")
try:
    count = NasaExoplanetArchive.query_criteria(
        table=TARGET_TABLE, select="count(*) as total"
    )
    if count is not None and len(count) > 0:
        total = count[0][0] if hasattr(count[0], '__getitem__') else count['total'][0]
        print(f"Total rows: {{total}}")
except Exception as e:
    print(f"Count query failed: {{e}}")

print("\\n" + "=" * 80)
print("USE THESE EXACT COLUMN NAMES IN YOUR EXPERIMENT QUERIES!")
print("=" * 80)
'''


# =============================================================================
# TARGETED DATA AVAILABILITY SEARCH
# =============================================================================

TARGETED_DATA_SEARCH_CODE = '''#!/usr/bin/env python3
"""
Targeted Data Availability Search
Purpose: Given a specific research need, search across multiple astroquery sources
to find if the required data exists anywhere.

Research Need: {research_need}
"""

import subprocess
import sys

packages = ['astroquery', 'astropy', 'pandas', 'numpy']
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TARGETED DATA AVAILABILITY SEARCH")
print(f"Research Need: {research_need}")
print("=" * 80)

data_found = []
data_not_found = []

# =============================================================================
# 1. Search NASA Exoplanet Archive
# =============================================================================
print("\\n--- 1. Searching NASA Exoplanet Archive ---")
try:
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

    # Check what atmosphere-related columns exist
    test = NasaExoplanetArchive.query_criteria(
        table="ps", select="*", where="rownum < 2"
    )
    atmosphere_cols = [c for c in test.colnames if any(
        kw in c.lower() for kw in ['atm', 'spec', 'tran', 'emission', 'chem', 'mol']
    )]
    if atmosphere_cols:
        print(f"  Atmosphere-related columns found: {{atmosphere_cols}}")
        data_found.append(("NASA Exoplanet Archive", atmosphere_cols))
    else:
        print("  No dedicated atmosphere columns in main table")

    # Check supplemental tables
    for table in ["ps", "pscomppars", "stellarhosts"]:
        try:
            cols_query = f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='{{table}}'"
            cols = NasaExoplanetArchive.query_criteria(cols_query, format="table")
            if cols is not None:
                all_cols = [str(c) for c in cols['column_name']]
                relevant = [c for c in all_cols if any(
                    kw in c.lower() for kw in {search_column_keywords}
                )]
                if relevant:
                    print(f"  Table '{{table}}': Relevant columns = {{relevant[:15]}}")
                    data_found.append((f"NExA/{{table}}", relevant))
        except:
            pass
except Exception as e:
    print(f"  NASA Exoplanet Archive search failed: {{e}}")

# =============================================================================
# 2. Search VizieR for specialized catalogs
# =============================================================================
print("\\n--- 2. Searching VizieR catalogs ---")
try:
    from astroquery.vizier import Vizier

    vizier_keywords = {vizier_search_terms}
    for keyword in vizier_keywords:
        print(f"  Searching VizieR for '{{keyword}}'...")
        try:
            catalogs = Vizier.find_catalogs(keyword)
            if catalogs:
                for cat_id, cat_info in list(catalogs.items())[:3]:
                    desc = str(getattr(cat_info, 'description', 'N/A'))[:150]
                    print(f"    FOUND: {{cat_id}} | {{desc}}")
                    data_found.append((f"VizieR/{{cat_id}}", desc))

                    # Peek at columns
                    try:
                        Vizier.ROW_LIMIT = 2
                        peek = Vizier.get_catalogs(cat_id)
                        if peek:
                            for t in peek:
                                print(f"      Columns: {{', '.join(t.colnames[:10])}}")
                                if len(t.colnames) > 10:
                                    print(f"      ... and {{len(t.colnames) - 10}} more")
                    except:
                        pass
        except Exception as e:
            print(f"    Error: {{e}}")
except Exception as e:
    print(f"  VizieR search failed: {{e}}")

# =============================================================================
# 3. Search MAST for relevant observations
# =============================================================================
print("\\n--- 3. Searching MAST ---")
try:
    from astroquery.mast import Observations

    # Search for spectroscopic observations of relevant targets
    sample_targets = {sample_targets}
    for target in sample_targets[:3]:
        try:
            obs = Observations.query_criteria(
                objectname=target,
                dataproduct_type="spectrum",
                obs_collection=["JWST", "HST"]
            )
            if obs is not None and len(obs) > 0:
                print(f"  {{target}}: {{len(obs)}} spectroscopic observations found")
                missions = list(set(obs['obs_collection']))
                instruments = list(set(obs['instrument_name']))[:5]
                print(f"    Missions: {{missions}}")
                print(f"    Instruments: {{instruments}}")
                data_found.append((f"MAST/{{target}}", f"{{len(obs)}} spectra"))
            else:
                print(f"  {{target}}: No spectroscopic data found")
                data_not_found.append((f"MAST/{{target}}", "No spectra"))
        except Exception as e:
            print(f"  {{target}}: Error - {{e}}")
except Exception as e:
    print(f"  MAST search failed: {{e}}")

# =============================================================================
# 4. Check SIMBAD for object information
# =============================================================================
print("\\n--- 4. Checking SIMBAD ---")
try:
    from astroquery.simbad import Simbad
    custom = Simbad()
    custom.add_votable_fields('sptype', 'fe_h', 'plx', 'rv_value', 'otype')

    for target in sample_targets[:3]:
        try:
            result = custom.query_object(target)
            if result is not None:
                print(f"  {{target}}: Found in SIMBAD")
                print(f"    Fields: {{result.colnames}}")
                data_found.append((f"SIMBAD/{{target}}", "Object data available"))
        except:
            pass
except Exception as e:
    print(f"  SIMBAD search failed: {{e}}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\\n" + "=" * 80)
print("DATA AVAILABILITY SUMMARY")
print("=" * 80)

print(f"\\nData sources found: {{len(data_found)}}")
for source, info in data_found:
    print(f"  [AVAILABLE] {{source}}: {{info}}")

print(f"\\nData not found: {{len(data_not_found)}}")
for source, info in data_not_found:
    print(f"  [MISSING]   {{source}}: {{info}}")

if data_found:
    print("\\nRECOMMENDATION: Use the available data sources above in your experiment!")
    print("Query the specific catalogs/tables identified to get REAL data.")
else:
    print("\\nWARNING: No directly relevant data found. Consider:")
    print("  1. Broadening your search keywords")
    print("  2. Using theoretical calculations with known physics")
    print("  3. Checking the specific VizieR catalogs manually")

print("\\n" + "=" * 80)
'''


# =============================================================================
# SIMBAD FIELD DISCOVERY
# =============================================================================

SIMBAD_FIELD_DISCOVERY_CODE = '''#!/usr/bin/env python3
"""
SIMBAD Field Discovery
Purpose: Discover all queryable fields in SIMBAD.
"""

import subprocess
import sys

packages = ['astroquery', 'astropy']
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from astroquery.simbad import Simbad

print("=" * 80)
print("SIMBAD FIELD DISCOVERY")
print("=" * 80)

print("\\nAll available VOTable fields:")
print("-" * 60)

fields = Simbad.list_votable_fields()
if hasattr(fields, 'to_pandas'):
    df = fields.to_pandas()
    for _, row in df.iterrows():
        print(f"  {row.iloc[0]:30s} | {str(row.iloc[1])[:60] if len(row) > 1 else ''}")
elif hasattr(fields, '__iter__'):
    for f in fields:
        print(f"  {f}")
else:
    print(fields)

print("\\n" + "=" * 80)
print("To use these fields:")
print("  custom = Simbad()")
print("  custom.add_votable_fields('plx', 'rv_value', 'fe_h', 'sptype')")
print("  result = custom.query_object('YOUR_TARGET_NAME')")
print("=" * 80)
'''


# =============================================================================
# SIMULATION DETECTION: Patterns that indicate the AI is simulating
# =============================================================================

SIMULATION_DETECTION_PATTERNS = [
    "np.random.normal",
    "np.random.uniform",
    "np.random.choice",
    "np.random.rand(",
    "np.random.randn(",
    "random.gauss",
    "random.uniform",
    "random.choice",
    "simulated_data",
    "fake_data",
    "synthetic_data",
    "generate_mock",
    "mock_observations",
    "simulated based on",
    "physically-motivated probabilities",
    "heuristic detection",
]

SIMULATION_LEGITIMATE_CONTEXTS = [
    "monte carlo",
    "bootstrap",
    "parameter sweep",
    "uncertainty propagation",
    "error analysis",
    "noise injection",
    "physical simulation",
    "n-body",
    "molecular dynamics",
    "resampling",
    "permutation test",
    "null distribution",
    "power analysis",
]


def get_discovery_code_for_topic(
    research_need: str,
    search_keywords: list[str],
    vizier_terms: list[str] | None = None,
    column_keywords: list[str] | None = None,
    sample_targets: list[str] | None = None,
    target_table: str = "ps",
) -> dict[str, str]:
    """Generate a set of discovery codes tailored to a specific research need.

    All parameters should be derived from the research question at hand.
    No defaults point to specific astronomical objects -- the system must
    discover relevant targets and data autonomously.

    Returns a dictionary mapping discovery type to executable Python code.
    """
    if vizier_terms is None:
        vizier_terms = search_keywords
    if column_keywords is None:
        column_keywords = []  # Must be derived from research context
    if sample_targets is None:
        sample_targets = []  # Must be derived from research context

    codes = {
        "module_discovery": ASTROQUERY_MODULE_DISCOVERY_CODE,
        "vizier_catalog_search": VIZIER_CATALOG_DISCOVERY_CODE.replace(
            "{search_keywords}", repr(search_keywords)
        ),
        "exoplanet_archive_schema": NASA_EXOPLANET_SCHEMA_DISCOVERY_CODE.replace(
            "{target_table}", target_table
        ),
        "targeted_data_search": TARGETED_DATA_SEARCH_CODE.format(
            research_need=research_need,
            search_column_keywords=repr(column_keywords),
            vizier_search_terms=repr(vizier_terms),
            sample_targets=repr(sample_targets),
        ),
        "simbad_fields": SIMBAD_FIELD_DISCOVERY_CODE,
    }

    return codes


def check_code_for_simulation_patterns(code: str) -> dict:
    """Analyze generated experiment code to detect simulation/synthetic data usage.

    Returns a dict with:
    - has_simulation: bool
    - simulation_patterns: list of detected patterns
    - is_legitimate: bool (True if simulation appears to be for a legitimate purpose)
    - recommendation: str
    """
    code_lower = code.lower()

    detected = []
    for pattern in SIMULATION_DETECTION_PATTERNS:
        if pattern.lower() in code_lower:
            detected.append(pattern)

    legitimate_context = False
    for context in SIMULATION_LEGITIMATE_CONTEXTS:
        if context in code_lower:
            legitimate_context = True
            break

    has_simulation = len(detected) > 0

    if not has_simulation:
        return {
            "has_simulation": False,
            "simulation_patterns": [],
            "is_legitimate": True,
            "recommendation": "No simulation patterns detected. Code appears to use real data.",
        }

    if legitimate_context:
        return {
            "has_simulation": True,
            "simulation_patterns": detected,
            "is_legitimate": True,
            "recommendation": (
                "Simulation detected but appears legitimate (Monte Carlo, bootstrap, etc.). "
                "Verify that it uses real data as input and simulates a physical process."
            ),
        }

    return {
        "has_simulation": True,
        "simulation_patterns": detected,
        "is_legitimate": False,
        "recommendation": (
            "WARNING: Code appears to generate synthetic/simulated data instead of using "
            "real observations. BEFORE running this experiment, use ExploreData to search "
            "for real data in: VizieR catalogs, NASA Exoplanet Archive, MAST, or SIMBAD. "
            "The data you need may exist in a catalog you haven't explored yet."
        ),
    }
