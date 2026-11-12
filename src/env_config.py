"""Load API keys and name which source supplied them.

Under ``streamlit run``, including Streamlit Community Cloud, a ``secrets.toml``
file is parsed before the app script runs. Root-level strings, ints, and
floats are copied into ``os.environ`` by
``streamlit.runtime.secrets.Secrets._maybe_set_environment_variable``, which
assigns ``os.environ[name] = str(value)`` and therefore replaces a process
environment value of the same name. ``load_dotenv`` does not override, so a
``.env`` file cannot replace that copy either. Precedence is Streamlit
secrets, then the process environment, then ``.env``.

Outside ``streamlit run`` nothing copies secrets first. The process
environment wins, then ``.env``, and ``st.secrets`` is read only when the
name is still missing.

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


def _read_public_secret(name: str, *, restore_env: bool) -> Optional[str]:
    """Return ``st.secrets[name]`` when it is a string, else ``None``.

    Membership uses the public ``in`` check. A missing or broken secrets file
    raises ``StreamlitSecretNotFoundError``. Any other failure, including a
    future Streamlit API change, is caught too. None of those crash the app
    or mark the source as secrets.

    Touching ``st.secrets`` parses ``secrets.toml`` and copies root values
    into ``os.environ``. When ``restore_env`` is true, ``name`` is put back
    so a caption check cannot replace a value that was already set.
    """
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None

    had_key = name in os.environ
    previous = os.environ.get(name) if had_key else None
    value: object = None
    try:
        try:
            if name not in st.secrets:
                value = None
            else:
                value = st.secrets[name]
        except (StreamlitSecretNotFoundError, KeyError, FileNotFoundError):
            value = None
        except Exception:
            value = None
    finally:
        if restore_env:
            if had_key:
                os.environ[name] = "" if previous is None else previous
            else:
                os.environ.pop(name, None)
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
    # A value already in the environment must not be replaced by the parse.
    # When the name is still missing, leave Streamlit's copy in place.
    return _read_public_secret(name, restore_env=not allow_load)


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

    Under ``streamlit run`` the returned value is whatever Streamlit left in
    the environment. A root secret wins over the process environment and over
    ``.env``. This function labels that value ``secrets`` when it matches
    ``st.secrets`` and was not already present at process start. A value
    python-dotenv wrote is ``dotenv``. A value present at process start, and
    a later assignment that is not a secret, is ``environment``. ``st.secrets``
    supplies the value itself only when the name is still missing, which is
    the path outside ``streamlit run``.
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
