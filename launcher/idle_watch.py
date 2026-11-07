"""Run Streamlit and exit after connected browser sessions stay gone.

Streamlit 1.67 counts live browser websockets in the ``active_sessions`` gauge
at ``/_stcore/metrics``. A connected tab stays in that count until the
websocket closes. That includes a tab where nobody is clicking and a tab whose
script is rerunning, so lineup generation is not cut off. This watcher only
reads that gauge. It does not call ``/_stcore/health``, and a health check is
not a browser session.

``server.disconnectedSessionTTL`` only drops stored state after a disconnect.
It does not exit the process. Counting TCP connections was rejected because a
health probe, a metrics read, or a failed socket listing can look like a
browser. If the gauge cannot be read, the server is left running.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_active_sessions(body: str) -> int | None:
    """Return the active_sessions gauge, or None when the payload has none."""
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(" ", 1)[0].split("{", 1)[0]
        if name != "active_sessions":
            continue
        try:
            return int(float(line.split()[-1]))
        except (ValueError, IndexError):
            return None
    return None


def note_session_count(
    count: int | None,
    zero_since: float | None,
    now: float,
    timeout: float,
) -> tuple[float | None, bool]:
    """Decide whether a gauge reading should stop the server.

    ``None`` means the gauge could not be read. That does not stop the server
    and does not pretend the reading was zero or that a browser is connected.
    A count of one or more clears the idle clock. Zero starts it, and only a
    later zero past ``timeout`` stops the server.
    """
    if count is None or count >= 1:
        if count is None:
            return zero_since, False
        return None, False
    if zero_since is None:
        return now, False
    return zero_since, (now - zero_since) >= timeout


def read_active_sessions(url: str, timeout: float) -> int | None:
    """Read the gauge. None means observation failed and must not stop the app."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return None
            payload = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return None
    return parse_active_sessions(payload)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Streamlit and exit after no browser session stays connected."
    )
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--reason-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("idle_watch: missing streamlit command", file=sys.stderr)
        return 2
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        print("idle_watch: timeout and poll must be positive", file=sys.stderr)
        return 2

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    metrics_url = (
        f"http://127.0.0.1:{args.port}/_stcore/metrics?families=active_sessions"
    )
    warned = False
    zero_since: float | None = None
    with args.output_log.open("ab") as log_handle:
        proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)

        def _on_term(_signum: int, _frame: object) -> None:
            _terminate(proc)
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _on_term)
        try:
            while True:
                if proc.poll() is not None:
                    return int(proc.returncode or 0)
                time.sleep(args.poll_seconds)
                if proc.poll() is not None:
                    return int(proc.returncode or 0)
                count = read_active_sessions(
                    metrics_url, timeout=min(2.0, args.poll_seconds)
                )
                if count is None and not warned:
                    print(
                        "idle_watch: active_sessions gauge unavailable; not shutting down",
                        file=sys.stderr,
                    )
                    warned = True
                if count is not None:
                    warned = False
                now = time.monotonic()
                zero_since, should_stop = note_session_count(
                    count, zero_since, now, args.timeout_seconds
                )
                if not should_stop:
                    continue
                reason = (
                    "no connected browser session for "
                    f"{int(args.timeout_seconds)} seconds"
                )
                args.reason_file.parent.mkdir(parents=True, exist_ok=True)
                args.reason_file.write_text(reason + "\n", encoding="utf-8")
                _terminate(proc)
                return 0
        except KeyboardInterrupt:
            _terminate(proc)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
