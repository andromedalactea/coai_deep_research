# Astroquery Reference for AI Scientific Discovery

**Generated:** 2026-02-05 18:06:50
**Purpose:** Complete reference for querying astronomical databases via astroquery

This document provides the AI with exact column names, query syntax, and examples
for accessing real astronomical data. Use this information to construct valid queries.

---

## Table of Contents
1. [NASA Exoplanet Archive](#1-nasa-exoplanet-archive)
2. [VizieR Catalogs (Gaia, 2MASS, etc.)](#2-vizier-catalogs)
3. [SIMBAD (Stellar Objects)](#3-simbad)
4. [MAST (Space Telescope Archives)](#4-mast)
5. [Query Examples](#5-query-examples)
6. [Common Pitfalls](#6-common-pitfalls)

---


## 1. NASA Exoplanet Archive

The NASA Exoplanet Archive contains data on **5,700+ confirmed exoplanets**.

### 1.1 How to Query

```python
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Basic query - get all confirmed planets
planets = NasaExoplanetArchive.query_criteria(
    table="ps",  # Planetary Systems table
    select="pl_name,pl_rade,pl_bmasse,pl_orbper,pl_eqt,st_teff,discoverymethod",
    where="default_flag=1"  # Best data per planet
)
```

### 1.2 Available Tables

| Table | Description | Use Case |
|-------|-------------|----------|
| `ps` | Planetary Systems (one row per planet per reference) | Most queries |
| `pscomppars` | Composite Parameters (one row per planet, best values) | Statistical analysis |
| `stellarhosts` | Host star properties only | Stellar population studies |
| `toi` | TESS Objects of Interest | Candidate planets |
| `koi` | Kepler Objects of Interest | Kepler candidates |

### 1.3 CRITICAL: Correct Column Names

**Total columns available: 356**

**Planet Properties (156 columns):**
| Column | Description | Unit |
|--------|-------------|------|
| `pl_name` | Name |  |
| `pl_letter` | Letter |  |
| `pl_refname` | Refname |  |
| `pl_orbper` | Orbper | d |
| `pl_orbpererr1` | Orbper Upper Error | d |
| `pl_orbpererr2` | Orbper Lower Error | d |
| `pl_orbperlim` | Orbperlim |  |
| `pl_orbperstr` | Orbperstr |  |
| `pl_orblpererr1` | Orblper Upper Error | deg |
| `pl_orblper` | Orblper | deg |
| `pl_orblpererr2` | Orblper Lower Error | deg |
| `pl_orblperlim` | Orblperlim |  |
| `pl_orblperstr` | Orblperstr |  |
| `pl_orbsmax` | Orbsmax | AU |
| `pl_orbsmaxerr1` | Orbsmax Upper Error | AU |
| `pl_orbsmaxerr2` | Orbsmax Lower Error | AU |
| `pl_orbsmaxlim` | Orbsmaxlim |  |
| `pl_orbsmaxstr` | Orbsmaxstr |  |
| `pl_orbincl` | Orbincl | deg |
| `pl_orbinclerr1` | Orbincl Upper Error | deg |
| ... | *136 more planet columns* | |

**Stellar Properties (61 columns):**
| Column | Description | Unit |
|--------|-------------|------|
| `st_metratio` | Metratio |  |
| `st_spectype` | Spectype |  |
| `st_rotp` | Rotp | d |
| `st_rotperr1` | Rotperr1 | d |
| `st_rotperr2` | Rotperr2 | d |
| `st_rotplim` | Rotplim |  |
| `st_rotpstr` | Rotpstr |  |
| `st_refname` | Refname |  |
| `st_teff` | Teff | K |
| `st_tefferr1` | Tefferr1 | K |
| `st_tefferr2` | Tefferr2 | K |
| `st_tefflim` | Tefflim |  |
| `st_teffstr` | Teffstr |  |
| `st_met` | Met | dex |
| `st_meterr1` | Meterr1 | dex |
| ... | *46 more stellar columns* | |

**Discovery Information:**
| Column | Description |
|--------|-------------|
| `discoverymethod` | Discovery method (Transit, Radial Velocity, Imaging, Microlensing) |
| `disc_year` | Year of discovery |
| `disc_facility` | Discovery facility |
| `disc_locale` | Discovery location (Ground/Space) |
| `disc_telescope` | Discovery telescope |
| `disc_instrument` | Discovery instrument |

**Discovery Method Values:**
- `Transit` - Planet transits in front of star
- `Radial Velocity` - Doppler wobble detection  
- `Imaging` - Direct imaging
- `Microlensing` - Gravitational lensing
- `Eclipse Timing Variations` - Binary eclipse timing
- `Pulsation Timing Variations` - Pulsar timing
- `Astrometry` - Stellar wobble position

**Key Column Names (static reference):**

**Planet Properties:**
| Column | Description |
|--------|-------------|
| `pl_name` | Planet name |
| `pl_rade` | Planet radius [Earth radii] |
| `pl_bmasse` | Planet mass [Earth masses] |
| `pl_orbper` | Orbital period [days] |
| `pl_orbsmax` | Semi-major axis [AU] |
| `pl_eqt` | Equilibrium temperature [K] |
| `pl_dens` | Planet density [g/cm³] |
| `pl_insol` | Insolation flux [Earth flux] |

**Stellar Properties:**
| Column | Description |
|--------|-------------|
| `hostname` | Host star name |
| `st_teff` | Stellar temperature [K] |
| `st_rad` | Stellar radius [Solar radii] |
| `st_mass` | Stellar mass [Solar masses] |
| `st_met` | Stellar metallicity [dex] |
| `st_logg` | Stellar surface gravity [log(cgs)] |
| `st_age` | Stellar age [Gyr] |

**Discovery Information:**
| Column | Description |
|--------|-------------|
| `discoverymethod` | Discovery method (Transit, RV, etc.) |
| `disc_year` | Year of discovery |
| `disc_facility` | Discovery facility |

**Discovery Method Values:**
- `Transit` - Planet transits in front of star
- `Radial Velocity` - Doppler wobble detection
- `Imaging` - Direct imaging
- `Microlensing` - Gravitational lensing


## 2. VizieR Catalogs

VizieR hosts **millions of astronomical catalogs**. Key catalogs:

### 2.1 Popular Catalog IDs

| Catalog ID | Name | Description |
|------------|------|-------------|
| `I/355/gaiadr3` | Gaia DR3 | 1.8 billion stars with positions, parallaxes, photometry |
| `II/246/out` | 2MASS | Near-infrared JHK photometry |
| `V/147/sdss12` | SDSS DR12 | Optical ugriz photometry |
| `J/A+A/616/A1` | Gaia DR2 | Previous Gaia release |
| `II/311/wise` | AllWISE | Mid-infrared photometry |

### 2.2 How to Query VizieR

```python
from astroquery.vizier import Vizier

# Set row limit (default is 50!)
Vizier.ROW_LIMIT = 10000

# Query by constraints
result = Vizier.query_constraints(
    catalog='I/355/gaiadr3',
    Plx=">40",      # Parallax > 40 mas (nearby stars)
    Gmag="<10"      # Bright stars
)

# Query by region
from astropy.coordinates import SkyCoord
import astropy.units as u
center = SkyCoord(ra=180*u.deg, dec=45*u.deg, frame='icrs')
result = Vizier.query_region(center, radius=10*u.arcmin, catalog='I/355/gaiadr3')
```

### 2.3 Gaia DR3 Key Columns

| Column | Description | Unit |
|--------|-------------|------|
| `RA_ICRS` | RA_ICRS | deg |
| `DE_ICRS` | DE_ICRS | deg |
| `Source` | Source | - |
| `e_RA_ICRS` | e_RA_ICRS | mas |
| `e_DE_ICRS` | e_DE_ICRS | mas |
| `Plx` | Plx | mas |
| `e_Plx` | e_Plx | mas |
| `PM` | PM | mas / yr |
| `pmRA` | pmRA | mas / yr |
| `e_pmRA` | e_pmRA | mas / yr |
| `pmDE` | pmDE | mas / yr |
| `e_pmDE` | e_pmDE | mas / yr |
| `RUWE` | RUWE | - |
| `FG` | FG | - |
| `e_FG` | e_FG | - |

| Column | Description | Unit |
|--------|-------------|------|
| `Source` | Unique Gaia source ID | - |
| `RA_ICRS` | Right Ascension | deg |
| `DE_ICRS` | Declination | deg |
| `Plx` | Parallax | mas |
| `e_Plx` | Parallax error | mas |
| `pmRA` | Proper motion in RA | mas/yr |
| `pmDE` | Proper motion in Dec | mas/yr |
| `Gmag` | G-band magnitude | mag |
| `BPmag` | BP magnitude | mag |
| `RPmag` | RP magnitude | mag |
| `BP-RP` | Color index | mag |
| `RV` | Radial velocity | km/s |
| `Teff` | Effective temperature | K |
| `logg` | Surface gravity | log(cgs) |
| `[Fe/H]` | Metallicity | dex |

**Distance calculation:** `distance_pc = 1000 / Plx`


## 3. SIMBAD

SIMBAD is the astronomical database for **stellar objects**.

### 3.1 How to Query

```python
from astroquery.simbad import Simbad

# Query single object
result = Simbad.query_object("Vega")

# Query multiple objects
result = Simbad.query_objects(["Sirius", "Betelgeuse", "Proxima Centauri"])

# Cone search around coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u
center = SkyCoord("00h42m44.3s", "+41d16m09s", frame='icrs')
result = Simbad.query_region(center, radius=30*u.arcmin)

# Add custom fields
custom = Simbad()
custom.add_votable_fields('plx', 'rv_value', 'fe_h', 'sptype')
result = custom.query_object("Vega")
```

### 3.2 Default Fields Returned

| Field | Description |
|-------|-------------|
| `MAIN_ID` | Primary object identifier |
| `RA` | Right Ascension (sexagesimal) |
| `DEC` | Declination (sexagesimal) |
| `RA_PREC` | RA precision |
| `DEC_PREC` | Dec precision |
| `COO_ERR_MAJA` | Position error major axis |
| `COO_ERR_MINA` | Position error minor axis |
| `COO_ERR_ANGLE` | Position error angle |
| `COO_QUAL` | Coordinate quality |
| `COO_WAVELENGTH` | Wavelength of coordinate measurement |
| `COO_BIBCODE` | Reference for coordinates |

### 3.3 Additional Fields (add with add_votable_fields)

| Field | Description |
|-------|-------------|
| `plx` | Parallax (mas) |
| `plx_error` | Parallax error |
| `rv_value` | Radial velocity (km/s) |
| `fe_h` | Metallicity [Fe/H] |
| `sptype` | Spectral type |
| `pm` | Proper motion |
| `flux(V)` | V-band magnitude |
| `flux(B)` | B-band magnitude |
| `distance` | Distance (pc) |
| `otype` | Object type |


## 4. MAST (Space Telescope Archives)

MAST provides access to **HST, JWST, Kepler, TESS**, and other missions.

### 4.1 How to Query

```python
from astroquery.mast import Observations, Catalogs

# Search by object name
obs = Observations.query_object("M31", radius="5 arcmin")

# Search by criteria
obs = Observations.query_criteria(
    obs_collection="JWST",
    dataproduct_type="image",
    filters="F444W"
)

# Search TESS Input Catalog
from astroquery.mast import Catalogs
tic = Catalogs.query_object("TIC 261136679", catalog="TIC")
```

### 4.2 Mission Collections

| Collection | Mission | Data Types |
|------------|---------|------------|
| `HST` | Hubble Space Telescope | Images, Spectra |
| `JWST` | James Webb Space Telescope | Images, Spectra |
| `TESS` | TESS | Light curves |
| `Kepler` | Kepler | Light curves |
| `K2` | K2 | Light curves |
| `GALEX` | GALEX | UV images |
| `PS1` | Pan-STARRS | Optical images |

### 4.3 Data Product Types

| Type | Description |
|------|-------------|
| `image` | 2D images |
| `spectrum` | 1D spectra |
| `timeseries` | Light curves |
| `cube` | 3D data cubes |
| `measurements` | Catalog measurements |


## 5. Query Examples

### 5.1 Find Earth-like Exoplanets

```python
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Query for small planets in habitable zone temperature range
planets = NasaExoplanetArchive.query_criteria(
    table="ps",
    select="pl_name,hostname,pl_rade,pl_bmasse,pl_orbper,pl_eqt,st_teff,discoverymethod",
    where="pl_rade < 1.5 AND pl_eqt BETWEEN 200 AND 350 AND default_flag=1"
)

# Convert to pandas
import pandas as pd
if hasattr(planets, 'to_pandas'):
    df = planets.to_pandas()
else:
    df = pd.DataFrame(planets)

print(f"Found {len(df)} potentially habitable planets")
```

### 5.2 Find Nearby Stars from Gaia

```python
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = 5000

# Stars within 25 parsecs (parallax > 40 mas)
result = Vizier.query_constraints(
    catalog='I/355/gaiadr3',
    Plx=">40",
    Gmag="<12"
)

if result:
    gaia = result[0]
    # Calculate distances
    distances = 1000 / gaia['Plx'].value
    print(f"Found {len(gaia)} nearby stars")
```

### 5.3 Get TESS Light Curve

```python
import lightkurve as lk

# Search for TESS observations
search = lk.search_lightcurve('TIC 261136679', mission='TESS')
print(f"Found {len(search)} light curves")

# Download and plot
if len(search) > 0:
    lc = search[0].download()
    lc.plot()
```


## 6. Common Pitfalls

### 6.1 Column Name Errors

**WRONG:**
```python
# These column names are INCORRECT and will fail!
planets = NasaExoplanetArchive.query_criteria(
    table="ps",
    where="disc_method = 'Transit'"  # ❌ WRONG! It's 'discoverymethod'
)
```

**CORRECT:**
```python
# Use exact column names from this reference
planets = NasaExoplanetArchive.query_criteria(
    table="ps",
    where="discoverymethod = 'Transit'"  # ✅ CORRECT
)
```

### 6.2 VizieR Row Limit

**WRONG:**
```python
# Default limit is 50 rows - you'll miss most data!
from astroquery.vizier import Vizier
result = Vizier.query_constraints(catalog='I/355/gaiadr3', Plx=">40")
# Only returns 50 rows!
```

**CORRECT:**
```python
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1  # No limit, or set to specific number
result = Vizier.query_constraints(catalog='I/355/gaiadr3', Plx=">40")
```

### 6.3 Type Conversion

**WRONG:**
```python
# Calling to_pandas() on something that's already a DataFrame
df = result.to_pandas()  # ❌ Fails if result is already DataFrame
```

**CORRECT:**
```python
import pandas as pd

# Check type before converting
if hasattr(result, 'to_pandas'):
    df = result.to_pandas()  # Astropy Table
elif isinstance(result, pd.DataFrame):
    df = result  # Already DataFrame
else:
    df = pd.DataFrame(result)
```

### 6.4 Missing Package Installation

**Always install required packages at the start:**
```python
import subprocess
import sys

packages = ['astroquery', 'astropy', 'pandas', 'matplotlib', 'lightkurve']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

---

*End of Astroquery Reference*