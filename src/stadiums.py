"""NFL stadium metadata: outdoor vs enclosed, keyed by DraftKings TeamAbbrev.

Weather adjustments use the **home** team's stadium. Enclosed / retractable
roofs are treated as non-outdoor so wind downgrades are skipped (assume roof
closed for DFS purposes). SoFi (LAR/LAC) has a fixed translucent roof canopy
and is ``fixed_roof``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Stadium:
    """One NFL home venue."""

    name: str
    team: str
    city: str
    lat: float
    lon: float
    is_outdoor: bool
    enclosure: str  # "outdoor" | "dome" | "fixed_roof" | "retractable"


# Home stadiums keyed by DraftKings TeamAbbrev.
STADIUMS: Dict[str, Stadium] = {
    "ARI": Stadium(
        "State Farm Stadium",
        "ARI",
        "Glendale, AZ",
        33.5276,
        -112.2626,
        False,
        "retractable",
    ),
    "ATL": Stadium(
        "Mercedes-Benz Stadium",
        "ATL",
        "Atlanta, GA",
        33.7554,
        -84.4010,
        False,
        "retractable",
    ),
    "BAL": Stadium(
        "M&T Bank Stadium",
        "BAL",
        "Baltimore, MD",
        39.2780,
        -76.6227,
        True,
        "outdoor",
    ),
    "BUF": Stadium(
        "Highmark Stadium",
        "BUF",
        "Orchard Park, NY",
        42.77306,
        -78.79222,
        True,
        "outdoor",
    ),
    "CAR": Stadium(
        "Bank of America Stadium",
        "CAR",
        "Charlotte, NC",
        35.2258,
        -80.8528,
        True,
        "outdoor",
    ),
    "CHI": Stadium(
        "Soldier Field",
        "CHI",
        "Chicago, IL",
        41.8623,
        -87.6167,
        True,
        "outdoor",
    ),
    "CIN": Stadium(
        "Paycor Stadium",
        "CIN",
        "Cincinnati, OH",
        39.0954,
        -84.5160,
        True,
        "outdoor",
    ),
    "CLE": Stadium(
        "Huntington Bank Field",
        "CLE",
        "Cleveland, OH",
        41.5061,
        -81.6995,
        True,
        "outdoor",
    ),
    "DAL": Stadium(
        "AT&T Stadium",
        "DAL",
        "Arlington, TX",
        32.7473,
        -97.0945,
        False,
        "retractable",
    ),
    "DEN": Stadium(
        "Empower Field at Mile High",
        "DEN",
        "Denver, CO",
        39.7439,
        -105.0201,
        True,
        "outdoor",
    ),
    "DET": Stadium(
        "Ford Field",
        "DET",
        "Detroit, MI",
        42.3400,
        -83.0456,
        False,
        "dome",
    ),
    "GB": Stadium(
        "Lambeau Field",
        "GB",
        "Green Bay, WI",
        44.5013,
        -88.0622,
        True,
        "outdoor",
    ),
    "HOU": Stadium(
        "Reliant Stadium",
        "HOU",
        "Houston, TX",
        29.6847,
        -95.4107,
        False,
        "retractable",
    ),
    "IND": Stadium(
        "Lucas Oil Stadium",
        "IND",
        "Indianapolis, IN",
        39.7601,
        -86.1639,
        False,
        "retractable",
    ),
    "JAX": Stadium(
        "EverBank Stadium",
        "JAX",
        "Jacksonville, FL",
        30.3239,
        -81.6373,
        True,
        "outdoor",
    ),
    "KC": Stadium(
        "Arrowhead Stadium",
        "KC",
        "Kansas City, MO",
        39.0489,
        -94.4839,
        True,
        "outdoor",
    ),
    "LAC": Stadium(
        "SoFi Stadium",
        "LAC",
        "Inglewood, CA",
        33.9535,
        -118.3392,
        False,
        "fixed_roof",
    ),
    "LAR": Stadium(
        "SoFi Stadium",
        "LAR",
        "Inglewood, CA",
        33.9535,
        -118.3392,
        False,
        "fixed_roof",
    ),
    "LV": Stadium(
        "Allegiant Stadium",
        "LV",
        "Las Vegas, NV",
        36.0909,
        -115.1833,
        False,
        "fixed_roof",
    ),
    "MIA": Stadium(
        "Hard Rock Stadium",
        "MIA",
        "Miami Gardens, FL",
        25.9580,
        -80.2389,
        True,
        "outdoor",
    ),
    "MIN": Stadium(
        "U.S. Bank Stadium",
        "MIN",
        "Minneapolis, MN",
        44.9738,
        -93.2581,
        False,
        "fixed_roof",
    ),
    "NE": Stadium(
        "Gillette Stadium",
        "NE",
        "Foxborough, MA",
        42.0909,
        -71.2643,
        True,
        "outdoor",
    ),
    "NO": Stadium(
        "Caesars Superdome",
        "NO",
        "New Orleans, LA",
        29.9511,
        -90.0812,
        False,
        "dome",
    ),
    "NYG": Stadium(
        "MetLife Stadium",
        "NYG",
        "East Rutherford, NJ",
        40.8128,
        -74.0742,
        True,
        "outdoor",
    ),
    "NYJ": Stadium(
        "MetLife Stadium",
        "NYJ",
        "East Rutherford, NJ",
        40.8128,
        -74.0742,
        True,
        "outdoor",
    ),
    "PHI": Stadium(
        "Lincoln Financial Field",
        "PHI",
        "Philadelphia, PA",
        39.9008,
        -75.1675,
        True,
        "outdoor",
    ),
    "PIT": Stadium(
        "Acrisure Stadium",
        "PIT",
        "Pittsburgh, PA",
        40.4468,
        -80.0158,
        True,
        "outdoor",
    ),
    "SEA": Stadium(
        "Lumen Field",
        "SEA",
        "Seattle, WA",
        47.5952,
        -122.3316,
        True,
        "outdoor",
    ),
    "SF": Stadium(
        "Levi's Stadium",
        "SF",
        "Santa Clara, CA",
        37.4030,
        -121.9700,
        True,
        "outdoor",
    ),
    "TB": Stadium(
        "Raymond James Stadium",
        "TB",
        "Tampa, FL",
        27.9759,
        -82.5033,
        True,
        "outdoor",
    ),
    "TEN": Stadium(
        "Nissan Stadium",
        "TEN",
        "Nashville, TN",
        36.1665,
        -86.7713,
        True,
        "outdoor",
    ),
    "WAS": Stadium(
        "Northwest Stadium",
        "WAS",
        "Landover, MD",
        38.9077,
        -76.8645,
        True,
        "outdoor",
    ),
}

# Explicit indoor / non-outdoor set for quick checks and tests.
INDOOR_TEAM_ABBREVS = frozenset(
    team for team, stadium in STADIUMS.items() if not stadium.is_outdoor
)


def get_stadium(team_abbrev: str) -> Optional[Stadium]:
    """Return stadium metadata for a DraftKings team abbreviation."""
    key = str(team_abbrev or "").strip().upper()
    return STADIUMS.get(key)


def is_outdoor_stadium(team_abbrev: str) -> bool:
    """
    True when the team's home venue is outdoor (weather applies).

    Unknown abbreviations default to outdoor so we do not silently skip wind.
    """
    stadium = get_stadium(team_abbrev)
    if stadium is None:
        return True
    return stadium.is_outdoor
