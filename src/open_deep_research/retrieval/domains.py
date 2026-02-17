"""High-authority domain registry and trust scoring for scientific retrieval.

Provides domain-level authority classification so that the retrieval system
can decide when to invoke Deep Read (raw scraping) vs lightweight
summarization.  Scores are deterministic and do not require an LLM call.
"""

import re
from typing import Optional
from urllib.parse import urlparse


# ── Tier 1: domains that almost always publish peer-reviewed or
#    institutional-quality scientific content.
_TIER1_EXACT: set[str] = {
    "arxiv.org",
    "adsabs.harvard.edu",
    "ui.adsabs.harvard.edu",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "iopscience.iop.org",
    "academic.oup.com",
    "journals.aps.org",
    "aanda.org",
    "iopscience.iop.org",
    "link.springer.com",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
    "scholar.google.com",
}

# ── Tier 1 suffix patterns (e.g. *.nasa.gov, *.edu)
_TIER1_SUFFIX_PATTERNS: list[str] = [
    ".nasa.gov",
    ".esa.int",
    ".edu",
    ".ac.uk",
]

# ── Tier 2: reputable but less rigorous aggregators / preprint mirrors.
_TIER2_EXACT: set[str] = {
    "en.wikipedia.org",
    "github.com",
    "stackoverflow.com",
    "medium.com",
    "researchgate.net",
    "semanticscholar.org",
    "paperswithcode.com",
    "huggingface.co",
    "kaggle.com",
}

# ── Trust scores by tier
_TRUST_TIER1 = 0.95
_TRUST_TIER2 = 0.60
_TRUST_DEFAULT = 0.35


def _extract_domain(url: str) -> Optional[str]:
    """Extract the hostname from a URL, stripping 'www.' prefix."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def is_high_authority(url: str) -> bool:
    """Return True when *url* belongs to a tier-1 scientific domain."""
    domain = _extract_domain(url)
    if domain is None:
        return False

    if domain in _TIER1_EXACT:
        return True

    for suffix in _TIER1_SUFFIX_PATTERNS:
        if domain.endswith(suffix):
            return True

    return False


def get_domain_trust_score(url: str) -> float:
    """Return a deterministic 0-1 trust score based on the URL domain.

    * 0.95 – peer-reviewed journals, institutional archives (.edu, .nasa.gov)
    * 0.60 – reputable aggregators (Wikipedia, GitHub, ResearchGate …)
    * 0.35 – everything else
    """
    domain = _extract_domain(url)
    if domain is None:
        return _TRUST_DEFAULT

    # Check tier-1 exact
    if domain in _TIER1_EXACT:
        return _TRUST_TIER1

    # Check tier-1 suffix patterns
    for suffix in _TIER1_SUFFIX_PATTERNS:
        if domain.endswith(suffix):
            return _TRUST_TIER1

    # Check tier-2 exact
    if domain in _TIER2_EXACT:
        return _TRUST_TIER2

    return _TRUST_DEFAULT


def get_authority_label(url: str) -> str:
    """Human-readable label: 'high', 'medium', or 'low'."""
    score = get_domain_trust_score(url)
    if score >= 0.9:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
