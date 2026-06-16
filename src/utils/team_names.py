"""
Team-name standardization.

Different sources spell the same nation differently (``Faroe Islands`` vs
``Faroe_Islands``, ``Czech Republic`` vs ``Czechia``, ``IR Iran`` vs ``Iran``).
We normalise everything to a single canonical display name so match results,
Flashscore stats, FIFA ranks and Elo ratings all join on the same key.
"""

from __future__ import annotations

import unicodedata

from .config import CODE2NAME

# Explicit aliases -> canonical name. Extend as new sources are added.
_ALIASES = {
    "czech republic": "Czechia",
    "faroe islands": "Faroe Islands",
    "faroe_islands": "Faroe Islands",
    "ir iran": "Iran",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "republic of ireland": "Ireland",
    "usa": "United States",
    "united states of america": "United States",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "bosnia and herzegovina": "Bosnia",
    "bosnia & herzegovina": "Bosnia",
    "cape verde islands": "Cape Verde",
    "cote d'ivoire": "Cote d Ivoire",
    "ivory coast": "Cote d Ivoire",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "guinea bissau": "Guinea-Bissau",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
}

# Canonical names that the eloratings code table stores with underscores.
_CANONICAL = {name.replace("_", " ") for name in CODE2NAME.values()}


def standardize_team_name(name: str) -> str:
    """Map any source spelling to the canonical display name.

    Steps: strip/clean, replace underscores, strip accents, apply the alias map,
    then title-case as a fallback. Idempotent.
    """
    if name is None:
        return name
    raw = str(name).strip().replace("_", " ")
    raw = " ".join(raw.split())  # collapse internal whitespace
    key = raw.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Accent-fold, then re-check the alias map.
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    if folded.lower() in _ALIASES:
        return _ALIASES[folded.lower()]
    # If it already matches a canonical eloratings name, keep that spelling.
    for canon in _CANONICAL:
        if folded.lower() == canon.lower():
            return canon
    return folded if folded else raw


def code_to_name(code: str) -> str:
    """eloratings 2-letter code -> canonical display name."""
    return standardize_team_name(CODE2NAME.get(code, code))
