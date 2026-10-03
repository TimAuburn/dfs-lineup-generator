"""DraftKings NFL Classic / Showdown lineup optimizer package."""

from .exporter import lineups_to_dk_upload_csv
from .optimizer import NFLClassicOptimizer, NFLShowdownOptimizer, get_nfl_optimizer
from .parser import REQUIRED_COLUMNS, parse_dk_csv
from .presets import ContestFormat, ContestPreset, apply_preset

__all__ = [
    "parse_dk_csv",
    "REQUIRED_COLUMNS",
    "NFLClassicOptimizer",
    "NFLShowdownOptimizer",
    "get_nfl_optimizer",
    "lineups_to_dk_upload_csv",
    "ContestFormat",
    "ContestPreset",
    "apply_preset",
]
