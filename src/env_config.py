"""Load API keys from a local ``.env`` file (never from the Streamlit UI)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# Canonical names used across the app and documented in ``.env.example``.
ODDS_API_KEY_ENV = "ODDS_API_KEY"
WEATHER_API_KEY_ENV = "WEATHER_API_KEY"

_LOADED = False


def load_env(dotenv_path: Optional[Path] = None, *, override: bool = False) -> Path:
    """
    Load ``KEY=value`` pairs from ``.env`` into ``os.environ``.

    Returns the path that was attempted. Safe to call multiple times; subsequent
    calls are no-ops unless ``override=True`` or a different path is passed.
    """
    global _LOADED
    path = Path(dotenv_path) if dotenv_path is not None else DEFAULT_ENV_PATH
    load_dotenv(dotenv_path=path, override=override)
    _LOADED = True
    return path


def _ensure_loaded() -> None:
    if not _LOADED:
        load_env()


def get_env_key(name: str, *, dotenv_path: Optional[Path] = None) -> Optional[str]:
    """Return a stripped env value, or ``None`` when missing/blank."""
    if dotenv_path is not None:
        load_env(dotenv_path, override=False)
    else:
        _ensure_loaded()
    value = (os.environ.get(name) or "").strip()
    return value or None


def get_odds_api_key(*, dotenv_path: Optional[Path] = None) -> Optional[str]:
    """The Odds API key (``ODDS_API_KEY``)."""
    return get_env_key(ODDS_API_KEY_ENV, dotenv_path=dotenv_path)


def get_weather_api_key(*, dotenv_path: Optional[Path] = None) -> Optional[str]:
    """Weather Company / weather.com API key (``WEATHER_API_KEY``)."""
    return get_env_key(WEATHER_API_KEY_ENV, dotenv_path=dotenv_path)


def key_status(name: str, *, dotenv_path: Optional[Path] = None) -> str:
    """Short UI caption: loaded vs missing (never reveals the secret)."""
    if get_env_key(name, dotenv_path=dotenv_path):
        return f"{name}: loaded from .env"
    return f"{name}: missing (set in .env)"
