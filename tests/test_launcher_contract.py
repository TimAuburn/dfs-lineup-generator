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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"

BAT_FILES = (
    "launch_dfs.bat",
    "stop_dfs.bat",
)
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
    r"remove-item\b",
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
    assert r".venv\Scripts\python.exe" in main
    assert main.index('cd /d "%LAUNCHER_DIR%\\.."') < main.index(r".venv\Scripts\python.exe")
    assert main.index("call :git_update") < main.index("call :maybe_pip")
    assert main.index("call :maybe_pip") < main.index("goto :start_server")
    assert main.index('if not exist "%PY%" goto :missing_venv') < main.index("call :git_update")
    git_body = text.split("\n:git_update", 1)[1].split("\n:git_cd_failed", 1)[0]
    assert git_body.index("git fetch origin main") < git_body.index("git pull --ff-only origin main")
    assert "git status --porcelain" in git_body
    pip_body = text.split("\n:maybe_pip", 1)[1]
    assert pip_body.index("-m pip install -r requirements.txt") < pip_body.index("\n:no_req")
    start = text.split("\n:start_server", 1)[1].split("\n:open_and_exit", 1)[0]
    assert "-m streamlit run app.py" in start
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
    stop = _read("stop_dfs.bat").lower()
    helper = _read("windows_helpers.ps1").lower()
    assert "taskkill" not in launch
    assert "taskkill" not in helper
    assert "taskkill /pid" in stop
    assert "taskkill /f /pid" in stop
    assert "/im " not in stop
    assert "del /s" not in launch
    assert "del /s" not in stop


def test_stop_identifies_our_server_before_taskkill():
    text = _read("stop_dfs.bat")
    ours = text.index('if /I "%PORT_STATE%"=="OURS" goto :do_stop')
    validate = text.index('call :digits_only "%OLD_PID%"')
    kill = text.index("taskkill /PID %OLD_PID% /T")
    assert ours < validate < kill
    assert "Nothing was stopped." in text
    assert "different program" in text
    assert "Other Python programs were not stopped." in text


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
    assert "Stop DFS Lineup Generator.lnk" in text
    assert "-WindowStyle 7" in text
    assert "-WindowStyle 1" in text
    assert "WScript.Shell" in text
    assert "WorkingDirectory" in text
    assert "dfs.ico" in text
    lowered = text.lower()
    assert "get-childitem" not in lowered
    assert "remove-item" not in lowered
    assert "gvandeveer" not in lowered


def test_no_hardcoded_user_path_or_env_writes():
    for name in SCRIPTS:
        text = _read(name).lower()
        assert "gvandeveer" not in text
        assert re.search(r"(?<![a-z])\.env(?![a-z])", text) is None


def test_launcher_runtime_is_gitignored_and_scripts_are_not():
    ignored = (
        "launcher/logs/launch.log",
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
    result = subprocess.run(
        [sys.executable, "-c", compare],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_helper_self_test_when_powershell_exists():
    executable = None
    for candidate in ("pwsh", "powershell", "powershell.exe"):
        found = subprocess.run(
            ["bash", "-lc", f"command -v {candidate}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode == 0 and found.stdout.strip():
            executable = found.stdout.strip()
            break
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
