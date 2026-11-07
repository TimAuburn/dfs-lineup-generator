"""Regression checks for the Windows desktop launcher.

These tests lock the safety contract in the launcher sources and exercise the
pure Python install-stamp snippets. They do not execute cmd.exe, WScript.Shell,
or OneDrive Desktop redirection, so they are not evidence that a Windows
double-click works.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"

BAT_FILES = ("launch_dfs.bat",)
PS1_FILES = (
    "create_shortcut.ps1",
    "windows_helpers.ps1",
)
SCRIPTS = BAT_FILES + PS1_FILES

# Command-shaped patterns. A sentence such as "not a git repository" must not
# trip the git-checkout check, and os.environ must not look like a .env write.
FORBIDDEN = (
    r"(?:^|[^a-z])git\s+stash\b",
    r"(?:^|[^a-z])git\s+reset\b",
    r"(?:^|[^a-z])git\s+clean\b",
    r"(?:^|[^a-z])git\s+checkout\b",
    r"(?:^|[^a-z])git\s+restore\b",
    r"(?:^|[^a-z])git\s+rebase\b",
    r"(?:^|[^a-z])git\s+merge\b",
    r"taskkill\s+/im\b",
    r"stop-process\b",
    r"activate\.bat\b",
    r"activate\.ps1\b",
)


def _read(name: str) -> str:
    return (LAUNCHER / name).read_text(encoding="utf-8")


def _code_lines(name: str) -> str:
    """Drop batch REM lines and PowerShell comments before forbidden-token scans."""
    kept = []
    for line in _read(name).splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("rem ") or stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept).lower()


def _python_snippets(name: str) -> list[str]:
    return re.findall(r'-c "([^"]+)"', _read(name))


def test_launcher_files_exist():
    for name in SCRIPTS:
        assert (LAUNCHER / name).is_file(), name
    assert (LAUNCHER / ".gitattributes").is_file()
    assert not (LAUNCHER / "dfs.ico").exists()


def test_windows_scripts_are_crlf_without_bom():
    for name in SCRIPTS:
        data = (LAUNCHER / name).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), name
        assert b"\r\n" in data, name
        assert b"\n" not in data.replace(b"\r\n", b""), name


def test_gitattributes_force_crlf_on_launcher_scripts():
    text = (LAUNCHER / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in text
    assert "*.ps1 text eol=crlf" in text


def test_launch_commands_resolve_root_then_update_then_start():
    text = _read("launch_dfs.bat")
    # Labels are not in execution order. The block before the first handler
    # is the double-click path: root, venv, git, pip, then start or open.
    main = text.split("\n:start_server", 1)[0]
    assert 'cd /d "%LAUNCHER_DIR%\\.."' in main
    assert "%~dp0" in main
    assert main.index("call :select_python") < main.index('if not "%PYTHON_OK%"=="1" goto :missing_venv')
    assert main.index('if not "%PYTHON_OK%"=="1" goto :missing_venv') < main.index('call :log "Using Python: %PY%"')
    assert main.index('call :log "Using Python: %PY%"') < main.index("call :git_update")
    assert main.index("call :git_update") < main.index("call :maybe_pip")
    assert main.index("call :maybe_pip") < main.index("goto :start_server")
    assert 'if not exist "%PY%" goto :missing_venv' not in text
    git_body = text.split("\n:git_update", 1)[1].split("\n:git_cd_failed", 1)[0]
    assert git_body.index("git fetch origin main") < git_body.index("git pull --ff-only origin main")
    assert "git status --porcelain" in git_body
    pip_body = text.split("\n:maybe_pip", 1)[1]
    assert pip_body.index("-m pip install -r requirements.txt") < pip_body.index("\n:no_req")
    start = text.split("\n:start_server", 1)[1].split("\n:open_and_exit", 1)[0]
    assert '"%PY%" -m streamlit run app.py' in start
    pip_cmd = text.split("\n:pip_needed", 1)[1].split("\n:no_req", 1)[0]
    assert '"%PY%" -m pip install -r requirements.txt' in pip_cmd
    missing = text.split("\n:missing_venv", 1)[1].split("\n:port_blocked", 1)[0]
    assert r"%ROOT%\.venv\Scripts\python.exe" in missing
    assert r"%ROOT%\venv\Scripts\python.exe" in missing
    select = text.split("\n:select_python", 1)[1].split("\n:acquire_or_wait", 1)[0]
    assert "-Command resolve-python" in select
    assert len(re.findall(r"(?m)^git pull --ff-only origin main >>", text)) == 1
    assert len(re.findall(r"(?m)^git fetch origin main >>", text)) == 1
    assert 'mkdir "%LOCKDIR%"' in text
    assert "pause" in text
    assert "launch.log" in text
    assert "--server.port=8501" in text
    assert "ExecutionPolicy Bypass" in text


def test_launch_skips_pull_unless_clean_main_and_warns():
    text = _read("launch_dfs.bat")
    assert "git status --porcelain" in text
    assert 'if /I not "%BRANCH%"=="main"' in text
    assert "working tree is not clean" in text
    assert "Launching local code without updating." in text
    assert "Launching local code anyway." in text
    assert "requirements.sha256" in text
    assert "The requirements stamp was not updated" in text


def test_launcher_does_not_destroy_work_or_kill_broadly():
    for name in SCRIPTS:
        code = _code_lines(name)
        for pattern in FORBIDDEN:
            assert re.search(pattern, code) is None, f"{name} matches {pattern}"
    launch = _read("launch_dfs.bat").lower()
    helper = _read("windows_helpers.ps1").lower()
    assert "taskkill" not in launch
    assert "taskkill" not in helper
    assert "del /s" not in launch
    assert not (LAUNCHER / "stop_dfs.bat").exists()


def test_port_helper_refuses_prefix_and_unrelated_listeners():
    text = _read("windows_helpers.ps1")
    assert "Test-CommandIsOurApp" in text
    assert "Test-PathHasRootBoundary" in text
    assert "Get-NetTCPConnection" in text
    assert "UNKNOWN" in text
    assert "_stcore/health" in text
    assert "Start-Process $Url" in text
    assert "Invoke-Expression" not in text
    assert "app.py" in text
    assert "streamlit" in text


def test_shortcut_script_uses_known_desktop_and_only_its_names():
    text = _read("create_shortcut.ps1")
    assert "[Environment]::GetFolderPath('Desktop')" in text
    assert "DFS Lineup Generator.lnk" in text
    assert "-WindowStyle 7" in text
    assert "-WindowStyle 1" not in text
    assert "WScript.Shell" in text
    assert "WorkingDirectory" in text
    assert "dfs.ico" in text
    assert "Remove-Item -LiteralPath $legacyStop" in text
    assert "$legacyStop = Join-Path $desktop 'Stop DFS Lineup Generator.lnk'" in text
    lowered = text.lower()
    assert "get-childitem" not in lowered
    assert "*.lnk" not in lowered
    assert "gvandeveer" not in lowered


def test_no_hardcoded_user_path_or_env_writes():
    for name in SCRIPTS:
        text = _read(name).lower()
        assert "gvandeveer" not in text
        assert re.search(r"(?<![a-z])\.env(?![a-z])", text) is None


def test_launcher_runtime_is_gitignored_and_scripts_are_not():
    ignored = (
        "launcher/logs/launch.log",
        "launcher/logs/idle-shutdown.txt",
        "launcher/logs/requirements.sha256",
        "launcher/logs/launch.lockdir/owner.pid",
        "launcher/logs/port-state.txt",
        "launcher/logs/git-status.txt",
        "launcher/logs/launcher.pid",
    )
    for rel in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "-v", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, rel
        assert "launcher/logs/" in result.stdout
    for rel in (
        "launcher/launch_dfs.bat",
        "launcher/stop_dfs.bat",
        "launcher/create_shortcut.ps1",
        "launcher/windows_helpers.ps1",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode != 0, rel


def test_requirements_stamp_snippets_round_trip(tmp_path: Path):
    snippets = _python_snippets("launch_dfs.bat")
    assert len(snippets) == 2
    compare, write = snippets
    project = tmp_path / "dfs-lineup-generator"
    (project / "launcher" / "logs").mkdir(parents=True)
    req = project / "requirements.txt"
    req.write_text("streamlit\n", encoding="utf-8")
    env = os.environ.copy()
    env["DFS_ROOT"] = str(project)
    env["DFS_PYTHON"] = r"C:\proj\.venv\Scripts\python.exe"

    def run(snippet: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    missing = run(compare)
    assert missing.returncode == 20, missing.stderr

    written = run(write)
    assert written.returncode == 0, written.stderr
    stamp = project / "launcher" / "logs" / "requirements.sha256"
    assert stamp.is_file()
    assert not list((project / "launcher" / "logs").glob("*.tmp"))

    same = run(compare)
    assert same.returncode == 0, same.stderr
    assert r"C:\proj\.venv\Scripts\python.exe" in stamp.read_text(encoding="utf-8")

    env["DFS_PYTHON"] = r"C:\proj\venv\Scripts\python.exe"
    switched = run(compare)
    assert switched.returncode == 20, switched.stderr
    assert r"C:\proj\.venv\Scripts\python.exe" in stamp.read_text(encoding="utf-8")

    req.write_text("streamlit\npandas\n", encoding="utf-8")
    changed = run(compare)
    assert changed.returncode == 20, changed.stderr

    # A failed install must leave the previous stamp untouched. The compare
    # snippet only reads, and the writer is a separate command the bat skips
    # when pip fails.
    assert stamp.read_text(encoding="utf-8").strip()
    stale = run(compare)
    assert stale.returncode == 20


def test_requirements_stamp_compare_fails_closed_without_requirements(tmp_path: Path):
    compare = _python_snippets("launch_dfs.bat")[0]
    project = tmp_path / "empty"
    project.mkdir()
    env = os.environ.copy()
    env["DFS_ROOT"] = str(project)
    env["DFS_PYTHON"] = r"C:\proj\venv\Scripts\python.exe"
    result = subprocess.run(
        [sys.executable, "-c", compare],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def _powershell() -> str | None:
    for candidate in ("pwsh", "powershell", "powershell.exe"):
        found = subprocess.run(
            ["bash", "-lc", f"command -v {candidate}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    bundled = Path("/tmp/pwsh/pwsh")
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return None


def _env_folder(selected: str) -> str:
    parts = selected.replace("\\", "/").split("/")
    assert parts[-1] == "python.exe"
    assert parts[-2] == "Scripts"
    return parts[-3]


def _resolve_python(root: Path, out: Path) -> str:
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is not installed; interpreter selection was not executed")
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(LAUNCHER / "windows_helpers.ps1"),
            "-Command",
            "resolve-python",
            "-ProjectRoot",
            str(root),
            "-OutFile",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return out.read_text(encoding="utf-8").strip()


def test_resolve_python_dot_venv_venv_both_and_neither(tmp_path: Path):
    """Run the helper the bat calls. This does not execute launch_dfs.bat."""
    root = tmp_path / "Python Scripts" / "dfs-lineup-generator"
    out = tmp_path / "python-path.txt"

    assert _resolve_python(root, out) == "MISSING"

    dot = root / ".venv" / "Scripts" / "python.exe"
    dot.parent.mkdir(parents=True)
    dot.write_bytes(b"")
    assert _env_folder(_resolve_python(root, out)) == ".venv"

    dot.unlink()
    plain = root / "venv" / "Scripts" / "python.exe"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(b"")
    selected = _resolve_python(root, out)
    assert _env_folder(selected) == "venv"
    assert Path(selected).is_file()

    dot.write_bytes(b"")
    assert _env_folder(_resolve_python(root, out)) == ".venv"


def test_venv_and_dot_venv_are_gitignored():
    for rel in (".venv/Scripts/python.exe", "venv/Scripts/python.exe"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, rel


def test_launch_logging_and_idle_exit_release_the_lock():
    launch = _read("launch_dfs.bat")
    assert 'set "LOG=%LOGDIR%\\launch.log"' in launch
    assert '>>"%LOG%" echo' not in launch
    assert "-Command append-log" in launch
    assert "idle_watch.py" in launch
    assert '"%PY%" -m streamlit run app.py' in launch
    assert "--output-log" in launch
    assert "DFS_IDLE_SECONDS=7200" in launch
    idle = launch.split("\n:idle_exit", 1)[1].split("\n:open_and_exit", 1)[0]
    assert 'call :log "Idle shutdown: %IDLE_REASON%"' in idle
    assert "call :release_lock" in idle
    assert "exit /b 0" in idle
    for line in launch.splitlines():
        if "wait-and-open" in line:
            assert "-LogPath" not in line


_RAW_IO_ERRORS = (
    "being used by another process",
    "cannot access the file",
    "IOException",
    "UnauthorizedAccess",
)


def _append_log(log: Path, message: str) -> subprocess.CompletedProcess[str]:
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is not installed; log append was not executed")
    env = os.environ.copy()
    env["DFS_LOG_MSG"] = message
    return subprocess.run(
        [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(LAUNCHER / "windows_helpers.ps1"),
            "-Command",
            "append-log",
            "-LogPath",
            str(log),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_log_io_stays_off_console(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    blob = result.stdout + result.stderr
    for marker in _RAW_IO_ERRORS:
        assert marker not in blob


def _hold_log(log: Path, seconds: float) -> subprocess.Popen[str]:
    executable = _powershell()
    assert executable is not None
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    script = (
        "$fs = New-Object System.IO.FileStream("
        "$env:LOCK_PATH, "
        "[System.IO.FileMode]::OpenOrCreate, "
        "[System.IO.FileAccess]::ReadWrite, "
        "[System.IO.FileShare]::None); "
        "[Console]::Out.WriteLine('HELD'); "
        "[Console]::Out.Flush(); "
        f"Start-Sleep -Milliseconds {int(seconds * 1000)}; "
        "$fs.Dispose()"
    )
    env = os.environ.copy()
    env["LOCK_PATH"] = str(log)
    holder = subprocess.Popen(
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    ready = holder.stdout.readline()
    assert "HELD" in ready, holder.stderr.read() if holder.stderr else ready
    return holder


def test_append_log_writes_when_the_file_is_free(tmp_path: Path):
    log = tmp_path / "Python Scripts" / "stop.log"
    first = _append_log(log, "No listener on port 8501. Nothing was stopped.")
    _assert_log_io_stays_off_console(first)
    second = _append_log(log, "Stopped the DFS Lineup Generator Streamlit server.")
    _assert_log_io_stays_off_console(second)
    text = log.read_text(encoding="utf-8")
    assert "Nothing was stopped." in text
    assert "Stopped the DFS Lineup Generator Streamlit server." in text


def test_append_log_locked_file_degrades_without_raw_errors(tmp_path: Path):
    """A long-lived exclusive holder must not surface a sharing violation.

    PowerShell's file APIs are what the bats call. This is not cmd.exe.
    """
    log = tmp_path / "launch.log"
    holder = _hold_log(log, 3.0)
    try:
        started = time.monotonic()
        result = _append_log(log, "---- stop ----")
        elapsed = time.monotonic() - started
        _assert_log_io_stays_off_console(result)
        assert elapsed < 2.5
        assert "---- stop ----" not in log.read_text(encoding="utf-8")
    finally:
        holder.wait(timeout=5)
    recovered = _append_log(log, "---- stop ----")
    _assert_log_io_stays_off_console(recovered)
    assert "---- stop ----" in log.read_text(encoding="utf-8")


def test_append_log_retries_a_brief_lock(tmp_path: Path):
    log = tmp_path / "stop.log"
    holder = _hold_log(log, 0.35)
    try:
        result = _append_log(log, "port check finished")
    finally:
        holder.wait(timeout=5)
    _assert_log_io_stays_off_console(result)
    assert "port check finished" in log.read_text(encoding="utf-8")


def test_append_log_unavailable_path_degrades_quietly(tmp_path: Path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    result = _append_log(blocker / "stop.log", "should not crash")
    _assert_log_io_stays_off_console(result)
    assert not (blocker / "stop.log").exists()


def test_helper_self_test_when_powershell_exists():
    executable = _powershell()
    if executable is None:
        pytest.skip("PowerShell is not installed; self-test was not executed")
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(LAUNCHER / "windows_helpers.ps1"),
            "-Command",
            "self-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "self-test ok" in result.stdout
