"""Odds API key source captions. The secret value is never part of the text."""

from __future__ import annotations

import logging

import pytest
from streamlit.errors import StreamlitSecretNotFoundError

from src.env_config import (
    ODDS_API_KEY_ENV,
    SOURCE_DOTENV,
    SOURCE_ENVIRONMENT,
    SOURCE_MISSING,
    SOURCE_SECRETS,
    get_odds_api_key,
    key_status,
    load_env,
    read_exec_environment,
    resolve_key,
)

PROCESS_CAPTION = f"{ODDS_API_KEY_ENV}: loaded from the process environment"
DOTENV_CAPTION = f"{ODDS_API_KEY_ENV}: loaded from .env"
SECRETS_CAPTION = f"{ODDS_API_KEY_ENV}: loaded from Streamlit secrets"
MISSING_CAPTION = (
    f"{ODDS_API_KEY_ENV}: missing (not in the environment, .env, or Streamlit secrets)"
)

_PROCESS_VALUE = "process-odds-key-9f3a"
_DOTENV_VALUE = "dotenv-odds-key-9f3a"
_SECRETS_VALUE = "secrets-odds-key-9f3a"


def _no_secret(_name: str) -> None:
    return None


def _refuse_secret(_name: str) -> None:
    raise AssertionError("st.secrets should not be read when the env var is set")


@pytest.fixture(autouse=True)
def _isolate_key_state(monkeypatch):
    """Keep dotenv bookkeeping and Streamlit's secrets cache inside this test."""
    import src.env_config as env_config
    from streamlit import config
    from streamlit.runtime.secrets import secrets_singleton

    saved_applied = dict(env_config._DOTENV_APPLIED)
    saved_loaded = env_config._LOADED
    saved_secrets = secrets_singleton._secrets
    saved_watchers = secrets_singleton._file_watchers_installed
    saved_files = list(config.get_option("secrets.files"))
    env_config._DOTENV_APPLIED.clear()
    env_config._LOADED = False
    monkeypatch.delenv(ODDS_API_KEY_ENV, raising=False)
    yield
    env_config._DOTENV_APPLIED.clear()
    env_config._DOTENV_APPLIED.update(saved_applied)
    env_config._LOADED = saved_loaded
    secrets_singleton._secrets = saved_secrets
    secrets_singleton._file_watchers_installed = saved_watchers
    config.set_option("secrets.files", saved_files)


def _empty_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def _point_secrets_at(path) -> None:
    from streamlit import config
    from streamlit.runtime.secrets import secrets_singleton

    config.set_option("secrets.files", [str(path)])
    secrets_singleton._secrets = None
    secrets_singleton._file_watchers_installed = False


def test_exec_environment_ignores_variables_set_after_start(monkeypatch):
    monkeypatch.setenv("DFS_EXEC_ENV_PROBE", "set-after-start")
    baseline = read_exec_environment()
    if baseline is None:
        pytest.skip("/proc/self/environ is not available")
    assert "DFS_EXEC_ENV_PROBE" not in baseline
    assert "PATH" in baseline


def test_process_environment_beats_dotenv_and_does_not_reveal_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv(ODDS_API_KEY_ENV, _PROCESS_VALUE)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{ODDS_API_KEY_ENV}={_DOTENV_VALUE}\n", encoding="utf-8")
    load_env(env_file, override=False)

    value, source = resolve_key(
        ODDS_API_KEY_ENV,
        secret_reader=_refuse_secret,
        process_env={ODDS_API_KEY_ENV: _PROCESS_VALUE},
    )
    caption = key_status(
        ODDS_API_KEY_ENV,
        secret_reader=_refuse_secret,
        process_env={ODDS_API_KEY_ENV: _PROCESS_VALUE},
    )
    assert value == _PROCESS_VALUE
    assert source == SOURCE_ENVIRONMENT
    assert caption == PROCESS_CAPTION
    assert _PROCESS_VALUE not in caption
    assert _DOTENV_VALUE not in caption
    assert get_odds_api_key(
        secret_reader=_refuse_secret,
        process_env={ODDS_API_KEY_ENV: _PROCESS_VALUE},
    ) == _PROCESS_VALUE


def test_dotenv_file_is_the_source_when_the_process_env_is_absent(tmp_path, capsys, caplog):
    env_file = tmp_path / ".env"
    env_file.write_text(f'{ODDS_API_KEY_ENV}="{_DOTENV_VALUE}"\n', encoding="utf-8")
    caplog.set_level(logging.DEBUG)
    load_env(env_file, override=True)

    value, source = resolve_key(ODDS_API_KEY_ENV, secret_reader=_no_secret, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, secret_reader=_no_secret, process_env={})
    captured = capsys.readouterr()
    assert value == _DOTENV_VALUE
    assert source == SOURCE_DOTENV
    assert caption == DOTENV_CAPTION
    assert _DOTENV_VALUE not in caption
    assert "secret" not in caption
    assert _DOTENV_VALUE not in captured.out
    assert _DOTENV_VALUE not in captured.err
    assert _DOTENV_VALUE not in caplog.text


def test_streamlit_secrets_supply_the_key_when_env_and_dotenv_are_absent(tmp_path, capsys, caplog):
    env_file = _empty_env(tmp_path)
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(f'{ODDS_API_KEY_ENV} = "{_SECRETS_VALUE}"\n', encoding="utf-8")
    _point_secrets_at(secrets_file)
    caplog.set_level(logging.DEBUG)
    load_env(env_file, override=True)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    captured = capsys.readouterr()
    assert value == _SECRETS_VALUE
    assert source == SOURCE_SECRETS
    assert caption == SECRETS_CAPTION
    assert _SECRETS_VALUE not in caption
    assert _SECRETS_VALUE not in captured.out
    assert _SECRETS_VALUE not in captured.err
    assert _SECRETS_VALUE not in caplog.text
    assert get_odds_api_key(process_env={}) == _SECRETS_VALUE


def test_secret_promoted_into_the_environment_is_still_streamlit_secrets(tmp_path):
    """Streamlit copies secrets.toml into os.environ before the script runs."""
    import os

    from streamlit.runtime.secrets import secrets_singleton

    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(f'{ODDS_API_KEY_ENV} = "{_SECRETS_VALUE}"\n', encoding="utf-8")
    _point_secrets_at(secrets_file)
    assert secrets_singleton.load_if_toml_exists() is True
    assert os.environ.get(ODDS_API_KEY_ENV) == _SECRETS_VALUE

    load_env(_empty_env(tmp_path), override=False)
    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    assert value == _SECRETS_VALUE
    assert source == SOURCE_SECRETS
    assert caption == SECRETS_CAPTION
    assert _SECRETS_VALUE not in caption


def test_missing_secrets_file_does_not_raise(tmp_path):
    import streamlit as st

    missing = tmp_path / "does-not-exist.toml"
    _point_secrets_at(missing)
    load_env(_empty_env(tmp_path), override=True)

    with pytest.raises(StreamlitSecretNotFoundError):
        st.secrets[ODDS_API_KEY_ENV]

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    assert value is None
    assert source == SOURCE_MISSING
    assert caption == MISSING_CAPTION
    assert get_odds_api_key(process_env={}) is None


def test_key_absent_from_env_dotenv_and_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("OTHER_KEY", raising=False)
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text('OTHER_KEY = "not-the-odds-key"\n', encoding="utf-8")
    _point_secrets_at(secrets_file)
    load_env(_empty_env(tmp_path), override=True)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    assert value is None
    assert source == SOURCE_MISSING
    assert caption == MISSING_CAPTION
    assert "not-the-odds-key" not in caption


def test_in_process_env_var_is_labeled_environment_without_proc(tmp_path, monkeypatch):
    monkeypatch.setenv(ODDS_API_KEY_ENV, _PROCESS_VALUE)
    monkeypatch.setattr("src.env_config.read_exec_environment", lambda: None)
    load_env(_empty_env(tmp_path), override=False)

    value, source = resolve_key(ODDS_API_KEY_ENV, secret_reader=_no_secret)
    caption = key_status(ODDS_API_KEY_ENV, secret_reader=_no_secret)
    assert value == _PROCESS_VALUE
    assert source == SOURCE_ENVIRONMENT
    assert caption == PROCESS_CAPTION
    assert _PROCESS_VALUE not in caption


class _RaisingSecrets:
    """Stand-in for a secrets object whose public access fails closed."""

    def __contains__(self, key: object) -> bool:
        raise RuntimeError("st.secrets is unavailable")

    def __getitem__(self, key: object) -> str:
        raise RuntimeError("st.secrets is unavailable")


def test_secrets_access_raising_does_not_crash_or_mislabel(tmp_path, monkeypatch, capsys, caplog):
    """A broken or renamed secrets API must not crash or claim the secret source."""
    import os

    monkeypatch.setattr("streamlit.secrets", _RaisingSecrets())
    monkeypatch.setenv(ODDS_API_KEY_ENV, _PROCESS_VALUE)
    caplog.set_level(logging.DEBUG)
    load_env(_empty_env(tmp_path), override=False)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    captured = capsys.readouterr()
    assert value == _PROCESS_VALUE
    assert source == SOURCE_ENVIRONMENT
    assert caption == PROCESS_CAPTION
    assert _PROCESS_VALUE not in caption
    assert _PROCESS_VALUE not in captured.out
    assert _PROCESS_VALUE not in captured.err
    assert _PROCESS_VALUE not in caplog.text
    assert os.environ.get(ODDS_API_KEY_ENV) == _PROCESS_VALUE


def test_secrets_access_raising_with_nothing_else_set_is_missing(tmp_path, monkeypatch, capsys, caplog):
    monkeypatch.setattr("streamlit.secrets", _RaisingSecrets())
    caplog.set_level(logging.DEBUG)
    load_env(_empty_env(tmp_path), override=True)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    captured = capsys.readouterr()
    assert value is None
    assert source == SOURCE_MISSING
    assert caption == MISSING_CAPTION
    assert _SECRETS_VALUE not in caption
    assert _SECRETS_VALUE not in captured.out
    assert _SECRETS_VALUE not in captured.err
    assert _SECRETS_VALUE not in caplog.text


def test_broken_secrets_file_does_not_crash_and_keeps_dotenv(tmp_path, monkeypatch, capsys, caplog):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text("this is not toml [\n", encoding="utf-8")
    _point_secrets_at(secrets_file)
    monkeypatch.delenv(ODDS_API_KEY_ENV, raising=False)
    load_env(_empty_env(tmp_path), override=True)
    absent, absent_source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    assert absent is None
    assert absent_source == SOURCE_MISSING

    env_file = tmp_path / ".env"
    env_file.write_text(f"{ODDS_API_KEY_ENV}={_DOTENV_VALUE}\n", encoding="utf-8")
    caplog.set_level(logging.DEBUG)
    load_env(env_file, override=True)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    captured = capsys.readouterr()
    assert value == _DOTENV_VALUE
    assert source == SOURCE_DOTENV
    assert caption == DOTENV_CAPTION
    assert _DOTENV_VALUE not in caption
    assert _DOTENV_VALUE not in captured.out
    assert _DOTENV_VALUE not in captured.err
    assert _DOTENV_VALUE not in caplog.text


def test_caption_check_does_not_replace_an_in_process_key(tmp_path, monkeypatch):
    """Reading st.secrets to label a later assignment must not clobber that value."""
    import os

    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(f'{ODDS_API_KEY_ENV} = "{_SECRETS_VALUE}"\n', encoding="utf-8")
    _point_secrets_at(secrets_file)
    monkeypatch.setenv(ODDS_API_KEY_ENV, _PROCESS_VALUE)
    load_env(_empty_env(tmp_path), override=False)

    value, source = resolve_key(ODDS_API_KEY_ENV, process_env={})
    caption = key_status(ODDS_API_KEY_ENV, process_env={})
    assert value == _PROCESS_VALUE
    assert source == SOURCE_ENVIRONMENT
    assert caption == PROCESS_CAPTION
    assert os.environ.get(ODDS_API_KEY_ENV) == _PROCESS_VALUE
    assert _PROCESS_VALUE not in caption
    assert _SECRETS_VALUE not in caption


def test_env_only_does_not_read_a_different_secret(tmp_path, monkeypatch, capsys, caplog):
    """A process-start value stays put even when secrets.toml has another key."""
    import os

    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(f'{ODDS_API_KEY_ENV} = "{_SECRETS_VALUE}"\n', encoding="utf-8")
    _point_secrets_at(secrets_file)
    monkeypatch.setenv(ODDS_API_KEY_ENV, _PROCESS_VALUE)
    caplog.set_level(logging.DEBUG)
    load_env(_empty_env(tmp_path), override=False)

    value, source = resolve_key(
        ODDS_API_KEY_ENV,
        process_env={ODDS_API_KEY_ENV: _PROCESS_VALUE},
    )
    caption = key_status(
        ODDS_API_KEY_ENV,
        process_env={ODDS_API_KEY_ENV: _PROCESS_VALUE},
    )
    captured = capsys.readouterr()
    assert value == _PROCESS_VALUE
    assert source == SOURCE_ENVIRONMENT
    assert caption == PROCESS_CAPTION
    assert os.environ.get(ODDS_API_KEY_ENV) == _PROCESS_VALUE
    assert _PROCESS_VALUE not in caption
    assert _SECRETS_VALUE not in caption
    assert _SECRETS_VALUE not in captured.out
    assert _SECRETS_VALUE not in captured.err
    assert _SECRETS_VALUE not in caplog.text
    assert _PROCESS_VALUE not in caplog.text
