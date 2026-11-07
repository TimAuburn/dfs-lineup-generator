"""Lifecycle checks for launcher/idle_watch.py.

These run Streamlit on Linux. They do not execute the Windows launcher window.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
APP = ROOT / "app.py"


def _load_idle_watch():
    spec = importlib.util.spec_from_file_location(
        "idle_watch", ROOT / "launcher" / "idle_watch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


idle_watch = _load_idle_watch()


def test_parse_active_sessions_ignores_health_and_other_metrics():
    body = "\n".join(
        [
            "# TYPE active_sessions gauge",
            "# HELP active_sessions Current number of active sessions.",
            "active_sessions 0",
            "session_events_total{type=\"connect\"} 3",
            "active_sessions_total 9",
        ]
    )
    assert idle_watch.parse_active_sessions(body) == 0
    assert idle_watch.parse_active_sessions("active_sessions 2\n") == 2
    assert idle_watch.parse_active_sessions("ok\n") is None
    assert idle_watch.parse_active_sessions("") is None


def test_note_session_count_resets_for_connected_tabs_and_ignores_unknown():
    note = idle_watch.note_session_count
    # First confirmed empty reading starts the clock and does not stop.
    zero_since, stop = note(0, None, now=10.0, timeout=4.0)
    assert zero_since == 10.0
    assert stop is False
    # A connected tab, including a quiet one, clears the clock.
    zero_since, stop = note(1, zero_since, now=13.0, timeout=4.0)
    assert zero_since is None
    assert stop is False
    # A failed reading does not stop and does not look like a new zero.
    kept, stop = note(None, 10.0, now=20.0, timeout=4.0)
    assert kept == 10.0
    assert stop is False
    # Only a later confirmed zero past the timeout stops.
    _, stop = note(0, 10.0, now=14.0, timeout=4.0)
    assert stop is True


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(tmp: Path, port: int, timeout: float, poll: float) -> tuple[subprocess.Popen[str], Path]:
    if not PY.is_file():
        pytest.skip("project .venv is not installed")
    log = tmp / "launch.log"
    reason = tmp / "idle-shutdown.txt"
    command = [
        str(PY),
        str(ROOT / "launcher" / "idle_watch.py"),
        "--timeout-seconds",
        str(timeout),
        "--port",
        str(port),
        "--poll-seconds",
        str(poll),
        "--output-log",
        str(log),
        "--reason-file",
        str(reason),
        "--",
        str(PY),
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, reason


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _wait_health(port: int, timeout: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.read().strip() == b"ok":
                    return True
        except (OSError, urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    return False


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
    return True


def test_idle_server_exits_while_health_is_polled(tmp_path: Path):
    port = _free_port()
    proc, reason = _start(tmp_path, port, timeout=4, poll=0.4)
    hits = 0
    try:
        assert _wait_health(port), "streamlit did not become healthy"
        started = time.monotonic()
        while proc.poll() is None and time.monotonic() - started < 25:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=1
                ) as response:
                    response.read()
                hits += 1
            except (OSError, urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.2)
        assert proc.poll() is not None
        assert proc.returncode == 0
        assert hits >= 3
        assert time.monotonic() - started >= 3
        assert "no connected browser session" in reason.read_text(encoding="utf-8")
        assert not _port_open(port)
    finally:
        _stop(proc)


def test_relaunch_after_idle_exit(tmp_path: Path):
    port = _free_port()
    first, reason = _start(tmp_path, port, timeout=3, poll=0.4)
    try:
        assert _wait_health(port)
        assert first.wait(timeout=25) == 0
    finally:
        _stop(first)
    assert reason.is_file()
    assert not _port_open(port)

    second, _second_reason = _start(tmp_path / "again", port, timeout=30, poll=1)
    try:
        assert _wait_health(port)
        assert second.poll() is None
    finally:
        _stop(second)
