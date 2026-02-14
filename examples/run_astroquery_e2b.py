"""
Run Astroquery Examples in E2B Sandbox

This script demonstrates using astroquery (astronomical database queries)
inside an E2B code interpreter sandbox. It shows how to:
- Query various astronomical catalogs (SIMBAD, VizieR, NASA Exoplanet Archive)
- Retrieve and process astronomical data
- Generate visualizations

Requires: E2B_API_KEY in .env file
Get your key at: https://e2b.dev/dashboard
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Setup Code - Install packages in E2B sandbox
# =============================================================================

SETUP_CODE = '''
import subprocess
import sys

# Install required packages
packages = ['astroquery', 'astropy', 'matplotlib', 'numpy']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Packages installed successfully")
'''


# =============================================================================
# Astroquery Example Codes
# =============================================================================

SIMBAD_QUERY_CODE = '''
"""Query SIMBAD for information about a celestial object."""
from astroquery.simbad import Simbad

# Query SIMBAD for the star Betelgeuse
result = Simbad.query_object("Betelgeuse")
print("=== SIMBAD Query: Betelgeuse ===")
print(result)
print()

# Get more details - add optional fields
custom_simbad = Simbad()
custom_simbad.add_votable_fields('ra(d)', 'dec(d)', 'sptype', 'distance')
result_detailed = custom_simbad.query_object("Betelgeuse")
print("=== Detailed Query ===")
print(result_detailed)
print()

# Query multiple objects
objects = ["Sirius", "Vega", "Polaris", "Proxima Centauri"]
results = Simbad.query_objects(objects)
print("=== Multiple Stars Query ===")
print(results)
'''

EXOPLANET_ARCHIVE_CODE = '''
"""Query NASA Exoplanet Archive for confirmed exoplanets."""
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import matplotlib.pyplot as plt
import numpy as np

# Query confirmed planets around M-dwarf stars
query = NasaExoplanetArchive.query_critaeria(
    table="ps",
    select="pl_name,hostname,pl_rade,pl_bmasse,pl_orbper,st_teff,st_rad",
    where="st_teff < 3900 AND pl_rade < 2.0 AND pl_bmasse IS NOT NULL",
    order="pl_bmasse ASC"
)
print("=== NASA Exoplanet Archive: Small Planets around M-dwarfs ===")
print(f"Found {len(query)} planets")
print()
print(query[:10])  # Show first 10
print()

# Extract data for plotting
radii = query['pl_rade'].value  # Earth radii
masses = query['pl_bmasse'].value  # Earth masses
periods = query['pl_orbper'].value  # days

# Create mass-radius diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mass-Radius plot
ax1 = axes[0]
scatter = ax1.scatter(masses, radii, c=periods, cmap='viridis', 
                       alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Planet Mass (Earth masses)', fontsize=12)
ax1.set_ylabel('Planet Radius (Earth radii)', fontsize=12)
ax1.set_title('Mass-Radius Diagram: M-dwarf Exoplanets', fontsize=14)
ax1.set_xscale('log')
plt.colorbar(scatter, ax=ax1, label='Orbital Period (days)')

# Add Earth and Mars for reference
ax1.plot(1.0, 1.0, 'g*', markersize=15, label='Earth', markeredgecolor='black')
ax1.plot(0.107, 0.532, 'r*', markersize=12, label='Mars', markeredgecolor='black')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Period histogram
ax2 = axes[1]
ax2.hist(periods[~np.isnan(periods)], bins=30, color='steelblue', 
         edgecolor='black', alpha=0.7)
ax2.set_xlabel('Orbital Period (days)', fontsize=12)
ax2.set_ylabel('Number of Planets', fontsize=12)
ax2.set_title('Orbital Period Distribution', fontsize=14)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/exoplanet_mass_radius.png', dpi=150, bbox_inches='tight')
display(plt.gcf())

print("\\n=== Statistics ===")
print(f"Median radius: {np.nanmedian(radii):.2f} Earth radii")
print(f"Median mass: {np.nanmedian(masses):.2f} Earth masses")
print(f"Median period: {np.nanmedian(periods):.2f} days")
'''

VIZIER_QUERY_CODE = '''
"""Query VizieR for catalog data - Gaia DR3 nearby stars."""
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Configure Vizier to return unlimited rows
Vizier.ROW_LIMIT = 1000

# Query Gaia DR3 catalog for stars within 25 pc
# Catalog: "I/355/gaiadr3" is Gaia DR3 main source
result = Vizier.query_constraints(
    catalog='I/355/gaiadr3',
    Plx=">40",  # Parallax > 40 mas (within ~25 pc)
    Gmag="<10"  # Brighter than 10th magnitude
)

if result:
    gaia_data = result[0]
    print("=== Gaia DR3 Nearby Stars (d < 25 pc) ===")
    print(f"Found {len(gaia_data)} stars")
    print()
    print(gaia_data['Source', 'RA_ICRS', 'DE_ICRS', 'Plx', 'Gmag', 'BP-RP'][:10])
    print()
    
    # Calculate distances from parallax
    parallax = gaia_data['Plx'].value  # mas
    distance_pc = 1000 / parallax  # parsecs
    
    # Color-Magnitude diagram
    bp_rp = gaia_data['BP-RP'].value  # Color index
    g_mag = gaia_data['Gmag'].value  # Apparent magnitude
    abs_g = g_mag - 5 * np.log10(distance_pc) + 5  # Absolute magnitude
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bp_rp, abs_g, c=distance_pc, cmap='plasma_r', 
                          alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    plt.colorbar(scatter, label='Distance (pc)')
    plt.xlabel('BP-RP Color (mag)', fontsize=12)
    plt.ylabel('Absolute G Magnitude', fontsize=12)
    plt.title('HR Diagram: Nearby Stars from Gaia DR3', fontsize=14)
    plt.gca().invert_yaxis()  # Brighter stars at top
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/tmp/gaia_hr_diagram.png', dpi=150, bbox_inches='tight')
    display(plt.gcf())
    
    print("\\n=== Color Distribution ===")
    print(f"Bluest star (BP-RP): {np.nanmin(bp_rp):.2f}")
    print(f"Reddest star (BP-RP): {np.nanmax(bp_rp):.2f}")
    print(f"Closest star: {np.nanmin(distance_pc):.2f} pc")
else:
    print("No results found")
'''

MAST_QUERY_CODE = '''
"""Query MAST for HST observations of a target."""
from astroquery.mast import Observations
import matplotlib.pyplot as plt

# Search for HST observations of the Orion Nebula
obs = Observations.query_criteria(
    objectname="Orion Nebula",
    obs_collection="HST",
    dataproduct_type="image"
)

print("=== MAST Query: HST Images of Orion Nebula ===")
print(f"Found {len(obs)} observations")
print()

if len(obs) > 0:
    # Show sample of observations
    print(obs['obs_id', 'target_name', 'filters', 'instrument_name', 't_exptime'][:15])
    print()
    
    # Analyze filter usage
    filters = obs['filters'].value
    unique_filters, counts = np.unique(filters, return_counts=True)
    
    # Plot filter distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Sort by count
    sorted_idx = np.argsort(counts)[::-1][:20]  # Top 20 filters
    ax.barh(range(len(sorted_idx)), counts[sorted_idx], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(unique_filters[sorted_idx])
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_title('HST Filter Usage for Orion Nebula Observations', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/tmp/hst_filters.png', dpi=150, bbox_inches='tight')
    display(plt.gcf())
    
    # Exposure time statistics
    exp_times = obs['t_exptime'].value
    print("\\n=== Exposure Time Statistics ===")
    print(f"Min exposure: {np.nanmin(exp_times):.1f} seconds")
    print(f"Max exposure: {np.nanmax(exp_times):.1f} seconds")
    print(f"Median exposure: {np.nanmedian(exp_times):.1f} seconds")
'''

CONE_SEARCH_CODE = '''
"""Perform a cone search around a specific coordinate."""
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Define center of search: M31 (Andromeda Galaxy)
center = SkyCoord("00h42m44.3s", "+41d16m09s", frame='icrs')
print(f"Search center: M31 (Andromeda)")
print(f"Coordinates: RA = {center.ra.deg:.4f}°, Dec = {center.dec.deg:.4f}°")
print()

# Configure SIMBAD query
custom_simbad = Simbad()
custom_simbad.add_votable_fields('ra(d)', 'dec(d)', 'otype', 'flux(V)')

# Query objects within 30 arcminutes of M31
result = custom_simbad.query_region(center, radius=30*u.arcmin)

print(f"=== Objects within 30 arcmin of M31 ===")
print(f"Found {len(result)} objects")
print()
print(result[:15])
print()

# Analyze object types
otypes = result['OTYPE'].value
unique_types, type_counts = np.unique(otypes, return_counts=True)

# Sort by count and take top 15
sorted_idx = np.argsort(type_counts)[::-1][:15]
top_types = unique_types[sorted_idx]
top_counts = type_counts[sorted_idx]

# Plot object type distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
ax1 = axes[0]
ax1.pie(top_counts, labels=top_types, autopct='%1.1f%%', startangle=90)
ax1.set_title('Object Types near M31', fontsize=14)

# Spatial distribution
ax2 = axes[1]
ras = result['RA_d'].value
decs = result['DEC_d'].value
ax2.scatter(ras, decs, alpha=0.5, s=10, c='steelblue', edgecolors='none')
ax2.scatter([center.ra.deg], [center.dec.deg], c='red', s=200, marker='*', 
            label='M31 Center', zorder=10)
ax2.set_xlabel('Right Ascension (deg)', fontsize=12)
ax2.set_ylabel('Declination (deg)', fontsize=12)
ax2.set_title('Spatial Distribution of Objects', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('/tmp/m31_cone_search.png', dpi=150, bbox_inches='tight')
display(plt.gcf())

print("\\n=== Object Type Summary ===")
for otype, count in zip(top_types[:10], top_counts[:10]):
    print(f"  {otype}: {count} objects")
'''


# =============================================================================
# E2B Execution Functions
# =============================================================================

async def run_astroquery_example(
    example_name: str,
    code: str,
    save_outputs: bool = True
) -> dict:
    """Run an astroquery example in E2B sandbox.
    
    Args:
        example_name: Name of the example for output naming
        code: Python code to execute
        save_outputs: Whether to save generated images
        
    Returns:
        Dictionary with execution results
    """
    from open_deep_research.computational.code_interpreter import execute_code
    
    print(f"\n{'='*60}")
    print(f"  Running: {example_name}")
    print(f"{'='*60}")
    
    # Prepend setup code to install required packages
    full_code = SETUP_CODE + "\n\n" + code
    
    # Execute code in E2B
    result = await execute_code(
        code=full_code,
        purpose=f"Astroquery example: {example_name}"
    )
    
    # Print results
    if result.success:
        print("\n✅ Execution successful!")
        print(f"⏱️  Execution time: {result.execution_time_seconds:.2f}s")
        
        if result.stdout:
            print("\n--- Output ---")
            print(result.stdout)
    else:
        print(f"\n❌ Execution failed: {result.error_message}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    
    # Save images if generated
    outputs_saved = []
    if save_outputs and result.outputs:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        for output in result.outputs:
            if output.image_base64 and output.image_format:
                filename = f"astroquery_{example_name}_{output.id}.{output.image_format}"
                filepath = output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(output.image_base64))
                outputs_saved.append(filepath)
                print(f"\n📊 Saved image: {filepath}")
    
    return {
        "success": result.success,
        "execution_time": result.execution_time_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error_message,
        "outputs_saved": outputs_saved,
        "num_images": len([o for o in result.outputs if o.image_base64])
    }


async def run_all_examples():
    """Run all astroquery examples."""
    examples = [
        ("simbad", SIMBAD_QUERY_CODE),
        ("exoplanet_archive", EXOPLANET_ARCHIVE_CODE),
        ("vizier_gaia", VIZIER_QUERY_CODE),
        ("mast_hst", MAST_QUERY_CODE),
        ("cone_search", CONE_SEARCH_CODE),
    ]
    
    print("="*70)
    print("   ASTROQUERY E2B EXAMPLES")
    print("="*70)
    print("\nThis will run several astronomical database queries in E2B sandbox.")
    print("Each example demonstrates different astroquery capabilities.\n")
    
    results = {}
    for name, code in examples:
        try:
            results[name] = await run_astroquery_example(name, code)
        except Exception as e:
            print(f"\n❌ Example '{name}' failed with exception: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("   SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results.values() if r.get("success"))
    print(f"\n✅ {successful}/{len(examples)} examples completed successfully")
    
    for name, result in results.items():
        status = "✅" if result.get("success") else "❌"
        time_str = f"{result.get('execution_time', 0):.2f}s" if result.get("execution_time") else "N/A"
        images = result.get("num_images", 0)
        print(f"  {status} {name}: {time_str}, {images} images")
    
    return results


async def run_single_example(example_name: str):
    """Run a single astroquery example by name."""
    examples = {
        "simbad": SIMBAD_QUERY_CODE,
        "exoplanet": EXOPLANET_ARCHIVE_CODE,
        "vizier": VIZIER_QUERY_CODE,
        "mast": MAST_QUERY_CODE,
        "cone": CONE_SEARCH_CODE,
    }
    
    if example_name not in examples:
        print(f"Unknown example: {example_name}")
        print(f"Available: {', '.join(examples.keys())}")
        return None
    
    return await run_astroquery_example(example_name, examples[example_name])


async def run_custom_code(code: str):
    """Run custom astroquery code in E2B sandbox."""
    return await run_astroquery_example("custom", code)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run astroquery examples in E2B sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_astroquery_e2b.py                  # Run all examples
  python run_astroquery_e2b.py --example simbad # Run SIMBAD example only
  python run_astroquery_e2b.py --example exoplanet
  python run_astroquery_e2b.py --example vizier
  python run_astroquery_e2b.py --example mast
  python run_astroquery_e2b.py --example cone
  
Available examples:
  simbad    - Query SIMBAD for star information
  exoplanet - Query NASA Exoplanet Archive
  vizier    - Query VizieR for Gaia DR3 data
  mast      - Query MAST for HST observations
  cone      - Cone search around M31
"""
    )
    
    parser.add_argument(
        "--example",
        type=str,
        choices=["simbad", "exoplanet", "vizier", "mast", "cone"],
        help="Run a specific example (default: run all)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output images to disk"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("E2B_API_KEY"):
        print("❌ E2B_API_KEY not found in environment")
        print("Get your key at: https://e2b.dev/dashboard")
        print("Add it to your .env file: E2B_API_KEY=your_key_here")
        sys.exit(1)
    
    # Run examples
    if args.example:
        asyncio.run(run_single_example(args.example))
    else:
        asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
