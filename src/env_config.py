"""Load API keys from the process environment, a ``.env`` file, or Streamlit secrets.

The value is never logged. Callers that only need a status line should use
``key_status``, which names the source and omits the secret.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping, Optional

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# Canonical names used across the app and documented in ``.env.example``.
ODDS_API_KEY_ENV = "ODDS_API_KEY"

SOURCE_ENVIRONMENT = "environment"
SOURCE_DOTENV = "dotenv"
SOURCE_SECRETS = "secrets"
SOURCE_MISSING = "missing"

# Values ``load_env`` actually wrote from a ``.env`` file. A key that was
# already set in the process is not recorded here.
_DOTENV_APPLIED: dict[str, str] = {}
_LOADED = False

SecretReader = Callable[[str], Optional[str]]


def load_env(dotenv_path: Optional[Path] = None, *, override: bool = False) -> Path:
    """
    Load ``KEY=value`` pairs from ``.env`` into ``os.environ``.

    Returns the path that was attempted. Safe to call multiple times. A later
    call leaves existing variables in place unless ``override=True``.
    """
    global _LOADED
    path = Path(dotenv_path) if dotenv_path is not None else DEFAULT_ENV_PATH
    before = {key: value for key, value in os.environ.items()}
    parsed = dotenv_values(path)
    load_dotenv(dotenv_path=path, override=override)
    if parsed:
        for key, raw in parsed.items():
            file_value = _clean(raw)
            prior = _clean(before.get(key))
            if not file_value:
                if override:
                    _DOTENV_APPLIED.pop(key, None)
                continue
            if override or prior is None:
                _DOTENV_APPLIED[key] = file_value
                continue
            # Already set. Keep the dotenv mark only when this process still
            # holds the value an earlier load applied.
            if _DOTENV_APPLIED.get(key) == prior:
                continue
            _DOTENV_APPLIED.pop(key, None)
    _LOADED = True
    return path


def _ensure_loaded() -> None:
    if not _LOADED:
        load_env()


def _clean(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_exec_environment() -> Optional[dict[str, str]]:
    """Environment variables from process start, when the OS still exposes them.

    Linux keeps the original ``execve`` environment in ``/proc/self/environ``.
    Assignments made later by Python, python-dotenv, or Streamlit are absent.
    Returns ``None`` when that snapshot cannot be read.
    """
    path = Path("/proc/self/environ")
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    parsed: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key_bytes, value_bytes = entry.split(b"=", 1)
        key = key_bytes.decode("utf-8", errors="replace")
        if not key:
            continue
        parsed[key] = value_bytes.decode("utf-8", errors="replace")
    return parsed


def _already_loaded_secret(name: str) -> Optional[str]:
    """Read a secret Streamlit has already parsed, without parsing files.

    Touching ``st.secrets`` parses ``secrets.toml`` and copies string values
    into ``os.environ``. Skip that when a value is already in the environment
    so a caption check cannot replace it.
    """
    try:
        from streamlit.runtime.secrets import secrets_singleton
    except ImportError:
        return None
    parsed = getattr(secrets_singleton, "_secrets", None)
    if not isinstance(parsed, Mapping):
        return None
    value = parsed.get(name)
    if not isinstance(value, str):
        return None
    return _clean(value)


def _read_streamlit_secret(name: str) -> Optional[str]:
    """Return ``st.secrets[name]`` when the environment variable is missing.

    Streamlit raises ``StreamlitSecretNotFoundError`` (a ``FileNotFoundError``)
    when ``secrets.toml`` is absent. A missing key raises ``KeyError``. Both,
    and importing Streamlit when it is not installed, yield ``None``.
    """
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None
    try:
        if not st.secrets.load_if_toml_exists():
            return None
        value = st.secrets[name]
    except (StreamlitSecretNotFoundError, KeyError, FileNotFoundError):
        return None
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return _clean(value)


def _secret_for(
    name: str,
    reader: Optional[SecretReader],
    *,
    allow_load: bool,
) -> Optional[str]:
    if reader is not None:
        value = reader(name)
        if not isinstance(value, str):
            return None
        return _clean(value)
    if allow_load:
        return _read_streamlit_secret(name)
    return _already_loaded_secret(name)


def resolve_key(
    name: str,
    *,
    dotenv_path: Optional[Path] = None,
    secret_reader: Optional[SecretReader] = None,
    process_env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[str], str]:
    """Return ``(value, source)``.

    ``source`` is ``environment``, ``dotenv``, ``secrets``, or ``missing``.
    ``value`` is ``None`` when the key is absent. Captions must not include it.

    An external environment variable is one present at process start. A value
    python-dotenv wrote is ``dotenv``. Streamlit Community Cloud copies
    ``secrets.toml`` into ``os.environ`` before the script runs; that value is
    ``secrets`` when it was not already in the process-start environment.
    ``st.secrets`` is read only when the variable is still missing.
    """
    if dotenv_path is not None:
        load_env(dotenv_path, override=False)
    else:
        _ensure_loaded()

    current = _clean(os.environ.get(name))
    if current and _DOTENV_APPLIED.get(name) == current:
        return current, SOURCE_DOTENV

    if process_env is None:
        baseline = read_exec_environment()
    else:
        baseline = process_env
    exec_value = None if baseline is None else _clean(baseline.get(name))
    if current and exec_value and current == exec_value:
        return current, SOURCE_ENVIRONMENT

    if current:
        # Set after process start. Streamlit's secrets loader does this when
        # it promotes secrets.toml. A test or shell assignment that did not
        # come from secrets stays labeled as the process environment.
        secret = _secret_for(name, secret_reader, allow_load=False)
        if secret and secret == current:
            return current, SOURCE_SECRETS
        return current, SOURCE_ENVIRONMENT

    secret = _secret_for(name, secret_reader, allow_load=True)
    if secret:
        return secret, SOURCE_SECRETS
    return None, SOURCE_MISSING


def get_env_key(
    name: str,
    *,
    dotenv_path: Optional[Path] = None,
    secret_reader: Optional[SecretReader] = None,
    process_env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a stripped env value, or ``None`` when missing/blank."""
    value, _source = resolve_key(
        name,
        dotenv_path=dotenv_path,
        secret_reader=secret_reader,
        process_env=process_env,
    )
    return value


def get_odds_api_key(
    *,
    dotenv_path: Optional[Path] = None,
    secret_reader: Optional[SecretReader] = None,
    process_env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """The Odds API key (``ODDS_API_KEY``)."""
    return get_env_key(
        ODDS_API_KEY_ENV,
        dotenv_path=dotenv_path,
        secret_reader=secret_reader,
        process_env=process_env,
    )


def key_status(
    name: str,
    *,
    dotenv_path: Optional[Path] = None,
    secret_reader: Optional[SecretReader] = None,
    process_env: Optional[Mapping[str, str]] = None,
) -> str:
    """Short UI caption naming the source. Never includes the secret value."""
    _value, source = resolve_key(
        name,
        dotenv_path=dotenv_path,
        secret_reader=secret_reader,
        process_env=process_env,
    )
    if source == SOURCE_ENVIRONMENT:
        return f"{name}: loaded from the process environment"
    if source == SOURCE_DOTENV:
        return f"{name}: loaded from .env"
    if source == SOURCE_SECRETS:
        return f"{name}: loaded from Streamlit secrets"
    return f"{name}: missing (not in the environment, .env, or Streamlit secrets)"
