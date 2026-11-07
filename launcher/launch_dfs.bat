@echo off
setlocal EnableExtensions

rem Local desktop launcher for the Streamlit app.
rem The project root is the parent of this script, not a hard-coded path.
rem Dirty trees, other branches, and fetch or pull failures only warn.
rem Local edits are left in place. This file does not stop listeners.
rem An unknown or unrelated process on port 8501 is left running.

set "LAUNCHER_DIR=%~dp0"
if "%LAUNCHER_DIR:~-1%"=="\" set "LAUNCHER_DIR=%LAUNCHER_DIR:~0,-1%"
cd /d "%LAUNCHER_DIR%\.."
if errorlevel 1 goto :cd_failed
set "ROOT=%CD%"
set "LOGDIR=%ROOT%\launcher\logs"
set "LOG=%LOGDIR%\launch.log"
set "LOCKDIR=%LOGDIR%\launch.lockdir"
set "PY="
set "HELPER=%ROOT%\launcher\windows_helpers.ps1"
set "LOCK_HELD=0"
set "LAUNCH_PID="
set "DFS_ROOT=%ROOT%"
set "DFS_LAUNCH_LOG=%LOG%"
set "GIT_TERMINAL_PROMPT=0"
set "GCM_INTERACTIVE=Never"

title DFS Lineup Generator

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if errorlevel 1 goto :logdir_failed

call :log "---- launch %DATE% %TIME% ----"
call :log "Project root: %ROOT%"

if not exist "%HELPER%" goto :missing_helper
call :read_launch_pid
if not defined LAUNCH_PID goto :no_powershell

call :acquire_or_wait
if "%LAUNCH_ACTION%"=="browser" goto :open_and_exit
if "%LAUNCH_ACTION%"=="busy" goto :peer_busy
if not "%LAUNCH_ACTION%"=="owner" goto :lock_failed

call :select_python
if not "%PYTHON_OK%"=="1" goto :missing_venv
set "DFS_PYTHON=%PY%"
call :log "Using Python: %PY%"

call :git_update
call :maybe_pip

call :query_port
if /I "%PORT_STATE%"=="OURS" goto :open_and_exit
if /I "%PORT_STATE%"=="FREE" goto :start_server
goto :port_blocked

:start_server
cd /d "%ROOT%"
if errorlevel 1 goto :cd_failed
if not defined DFS_IDLE_SECONDS set "DFS_IDLE_SECONDS=7200"
call :digits_only "%DFS_IDLE_SECONDS%"
if not "%DIGITS_OK%"=="1" set "DFS_IDLE_SECONDS=7200"
set "IDLE_REASON_FILE=%LOGDIR%\idle-shutdown.txt"
if exist "%IDLE_REASON_FILE%" del /q "%IDLE_REASON_FILE%"
call :log "Starting: %PY% -m streamlit run app.py --server.port=8501 --server.address=localhost --server.headless=true --browser.gatherUsageStats=false"
call :log "Idle shutdown after %DFS_IDLE_SECONDS% seconds with no connected browser session. Close this window to stop sooner."
echo.
echo DFS Lineup Generator is starting.
echo   App URL: http://localhost:8501/
echo   Log: %LOG%
echo   This window stays open so errors remain readable. The desktop shortcut minimizes it.
echo   Close this window to stop the app.
echo   If no browser stays connected, it exits after %DFS_IDLE_SECONDS% seconds.
echo.
start "DFSLineupOpenBrowser" /MIN powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command wait-and-open -Url "http://localhost:8501/" -Port 8501
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"
"%PY%" "%ROOT%\launcher\idle_watch.py" --timeout-seconds %DFS_IDLE_SECONDS% --port 8501 --poll-seconds 15 --output-log "%LOG%" --reason-file "%IDLE_REASON_FILE%" -- "%PY%" -m streamlit run app.py --server.port=8501 --server.address=localhost --server.headless=true --browser.gatherUsageStats=false
set "RC=%ERRORLEVEL%"
if exist "%IDLE_REASON_FILE%" goto :idle_exit
call :log "Streamlit exited with code %RC%."
call :release_lock
if "%RC%"=="0" exit /b 0
echo.
echo Streamlit exited with code %RC%.
echo Last lines of %LOG%:
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath $env:DFS_LAUNCH_LOG) { Get-Content -LiteralPath $env:DFS_LAUNCH_LOG -Tail 40 }"
echo.
pause
exit /b %RC%

:idle_exit
set "IDLE_REASON="
for /f "usebackq delims=" %%R in ("%IDLE_REASON_FILE%") do set "IDLE_REASON=%%R"
del /q "%IDLE_REASON_FILE%"
call :log "Idle shutdown: %IDLE_REASON%"
call :release_lock
exit /b 0

:open_and_exit
call :log "Opening http://localhost:8501/ because this app is already on port 8501. PID %PORT_PID%."
start "DFSLineupOpenBrowser" /MIN powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command wait-and-open -Url "http://localhost:8501/" -Port 8501 -TimeoutSeconds 15
call :release_lock
exit /b 0

:missing_venv
call :log "Python virtual environment was not found. Checked .venv\Scripts\python.exe and venv\Scripts\python.exe."
echo.
echo The DFS Lineup Generator virtual environment was not found.
echo Checked both of these:
echo   %ROOT%\.venv\Scripts\python.exe
echo   %ROOT%\venv\Scripts\python.exe
echo.
echo From this project folder, create one of them and install dependencies.
echo When both exist, .venv is used:
echo   py -3 -m venv .venv
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
echo or:
echo   py -3 -m venv venv
echo   venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Then start DFS Lineup Generator again. Setup steps are in README.md.
echo.
call :release_lock
pause
exit /b 1

:port_blocked
call :log "WARN: port 8501 is not available for this app. State %PORT_STATE% PID %PORT_PID%. Not starting another server and not stopping the current listener."
echo.
echo Port 8501 cannot be used for a new DFS Lineup Generator server.
echo State: %PORT_STATE%
echo PID: %PORT_PID%
echo.
echo If another program is using that port, this launcher leaves it running.
echo If this app is already open, use the existing window at http://localhost:8501/
echo Details: %LOG%
echo.
call :release_lock
pause
exit /b 1

:peer_busy
call :log "WARN: another launch is still in progress. Not starting a second server."
echo.
echo Another DFS Lineup Generator launch is still in progress.
echo A second server was not started.
echo If this warning stays after the app has quit, delete this folder and try again:
echo   %LOCKDIR%
echo.
pause
exit /b 1

:cd_failed
echo Could not change to the project folder that contains this launcher.
echo Launcher folder: %LAUNCHER_DIR%
echo.
pause
exit /b 1

:logdir_failed
echo Could not create the log folder:
echo   %LOGDIR%
echo.
pause
exit /b 1

:missing_helper
echo Launcher helper script is missing:
echo   %HELPER%
echo.
pause
exit /b 1

:no_powershell
call :log "WARN: PowerShell could not report this window's process id. Not starting, so two double-clicks cannot overlap."
echo.
echo PowerShell is required to coordinate the launcher.
echo Windows PowerShell could not be run, so nothing was started.
echo.
call :release_lock
pause
exit /b 1

:lock_failed
call :log "WARN: could not acquire the launch lock. Not starting a second server."
echo.
echo Could not acquire the launch lock. A second server was not started.
echo If no app is running, delete this folder and try again:
echo   %LOCKDIR%
echo.
pause
exit /b 1

goto :eof

:log
set "LOG_RC=%ERRORLEVEL%"
set "MSG=%~1"
echo %MSG%
set "DFS_LOG_MSG=%MSG%"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command append-log -LogPath "%LOG%" >nul 2>&1
exit /b %LOG_RC%

:digits_only
set "DIGITS_OK=0"
if "%~1"=="" goto :eof
set "NONDIGIT="
for /f "delims=0123456789" %%N in ("%~1") do set "NONDIGIT=1"
if defined NONDIGIT goto :eof
set "DIGITS_OK=1"
goto :eof

:read_launch_pid
set "LAUNCH_PID="
set "PIDFILE=%LOGDIR%\launcher.pid"
if exist "%PIDFILE%" del /q "%PIDFILE%"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command parent-pid -OutFile "%PIDFILE%"
if not exist "%PIDFILE%" goto :eof
for /f "usebackq delims=" %%P in ("%PIDFILE%") do set "LAUNCH_PID=%%P"
call :digits_only "%LAUNCH_PID%"
if not "%DIGITS_OK%"=="1" set "LAUNCH_PID="
goto :eof

:select_python
set "PYTHON_OK=0"
set "PY="
set "PYFILE=%LOGDIR%\python-path.txt"
if exist "%PYFILE%" del /q "%PYFILE%"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command resolve-python -ProjectRoot "%ROOT%" -OutFile "%PYFILE%" -LogPath "%LOG%"
if not exist "%PYFILE%" goto :eof
set "PYLINE="
for /f "usebackq delims=" %%P in ("%PYFILE%") do set "PYLINE=%%P"
if not defined PYLINE goto :eof
if /I "%PYLINE%"=="MISSING" goto :eof
set "PY=%PYLINE%"
if not exist "%PY%" goto :eof
set "PYTHON_OK=1"
goto :eof

:acquire_or_wait
set "LAUNCH_ACTION="
set "MISSING_PID=0"
set "STALE_TRIES=0"
:lock_try
mkdir "%LOCKDIR%" 2>nul
if errorlevel 1 goto :lock_held_by_other
>"%LOCKDIR%\owner.pid" echo %LAUNCH_PID%
set "LOCK_HELD=1"
set "LAUNCH_ACTION=owner"
goto :eof

:lock_held_by_other
set "WAIT_TRIES=0"
:lock_wait
call :read_owner
if not defined OWNER goto :lock_owner_missing
set "MISSING_PID=0"
call :owner_is_alive
if "%OWNER_ALIVE%"=="0" goto :lock_stale
call :query_port
if /I "%PORT_STATE%"=="OURS" goto :lock_found_server
set /a WAIT_TRIES+=1
if %WAIT_TRIES% GEQ 30 goto :lock_still_busy
ping -n 3 127.0.0.1 >nul 2>nul
goto :lock_wait

:lock_found_server
set "LAUNCH_ACTION=browser"
goto :eof

:lock_still_busy
set "LAUNCH_ACTION=busy"
goto :eof

:lock_owner_missing
set /a MISSING_PID+=1
if %MISSING_PID% GEQ 5 goto :lock_stale
ping -n 3 127.0.0.1 >nul 2>nul
goto :lock_wait

:lock_stale
set /a STALE_TRIES+=1
if %STALE_TRIES% GEQ 3 goto :lock_still_busy
call :log "WARN: removing a stale launcher lock."
rmdir /s /q "%LOCKDIR%" 2>nul
goto :lock_try

:read_owner
set "OWNER="
if not exist "%LOCKDIR%\owner.pid" goto :eof
for /f "usebackq delims=" %%O in ("%LOCKDIR%\owner.pid") do set "OWNER=%%O"
goto :eof

:owner_is_alive
set "OWNER_ALIVE=0"
if not defined OWNER goto :eof
call :digits_only "%OWNER%"
if not "%DIGITS_OK%"=="1" goto :eof
tasklist /FI "PID eq %OWNER%" /FO CSV /NH > "%LOGDIR%\tasklist.txt" 2>nul
findstr /R /C:"[^0-9]%OWNER%[^0-9]" "%LOGDIR%\tasklist.txt" >nul
if not errorlevel 1 set "OWNER_ALIVE=1"
goto :eof

:release_lock
if not "%LOCK_HELD%"=="1" goto :eof
rmdir /s /q "%LOCKDIR%" 2>nul
set "LOCK_HELD=0"
goto :eof

:query_port
set "PORT_STATE=UNKNOWN"
set "PORT_PID="
set "PORT_FILE=%LOGDIR%\port-state.txt"
if exist "%PORT_FILE%" del /q "%PORT_FILE%"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%HELPER%" -Command port-state -ProjectRoot "%ROOT%" -Port 8501 -OutFile "%PORT_FILE%" -LogPath "%LOG%"
if errorlevel 1 goto :query_port_log
if not exist "%PORT_FILE%" goto :query_port_log
for /f "usebackq tokens=1,2 delims= " %%A in ("%PORT_FILE%") do (
  set "PORT_STATE=%%A"
  set "PORT_PID=%%B"
)
:query_port_log
call :log "Port 8501 state: %PORT_STATE% %PORT_PID%"
goto :eof

:git_update
cd /d "%ROOT%"
if errorlevel 1 goto :git_cd_failed
where git >nul 2>&1
if errorlevel 1 goto :git_missing
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 goto :git_missing
call :log "Running: git fetch origin main"
git fetch origin main >>"%LOG%" 2>&1
if errorlevel 1 goto :fetch_failed
git status --porcelain > "%LOGDIR%\git-status.txt" 2>>"%LOG%"
if errorlevel 1 goto :status_failed
set "DIRTY=0"
for /f "usebackq delims=" %%S in ("%LOGDIR%\git-status.txt") do set "DIRTY=1"
if "%DIRTY%"=="1" goto :dirty_tree
set "BRANCH="
for /f "usebackq delims=" %%B in (`git rev-parse --abbrev-ref HEAD`) do set "BRANCH=%%B"
if not defined BRANCH goto :not_main
if /I not "%BRANCH%"=="main" goto :not_main
call :log "Working tree is clean on main. Running: git pull --ff-only origin main"
git pull --ff-only origin main >>"%LOG%" 2>&1
if errorlevel 1 goto :pull_failed
call :log "git pull finished."
goto :eof

:git_cd_failed
call :log "WARN: could not enter the project folder before git. Launching local code without updating."
goto :eof

:git_missing
call :log "WARN: git is unavailable or this folder is not a git repository. Launching local code without updating."
goto :eof

:fetch_failed
call :log "WARN: git fetch origin main failed. Offline, no origin, or credentials. Launching local code without updating."
goto :eof

:status_failed
call :log "WARN: git status failed. Skipping git pull so local files are left untouched."
goto :eof

:dirty_tree
call :log "WARN: working tree is not clean. Skipping git pull so local changes are left untouched."
goto :eof

:not_main
call :log "WARN: branch is %BRANCH%, not main. Skipping git pull."
goto :eof

:pull_failed
call :log "WARN: git pull --ff-only origin main failed. Launching local code without resetting anything."
goto :eof

:maybe_pip
if not exist "%ROOT%\requirements.txt" goto :no_req
cd /d "%ROOT%"
if errorlevel 1 goto :pip_cd_failed
"%PY%" -c "import hashlib,os,pathlib,sys; root=pathlib.Path(os.environ['DFS_ROOT']); py=os.environ['DFS_PYTHON']; req=root/'requirements.txt'; stamp=root/'launcher'/'logs'/'requirements.sha256'; digest=hashlib.sha256(req.read_bytes()).hexdigest(); lines=[line.strip() for line in (stamp.read_text(encoding='utf-8').splitlines() if stamp.is_file() else []) if line.strip()]; sys.exit(0 if len(lines)>=2 and lines[0]==digest and lines[1]==py else 20)"
if errorlevel 20 goto :pip_needed
if errorlevel 1 goto :pip_hash_failed
call :log "requirements.txt is unchanged for this Python since the last successful install."
goto :eof

:pip_needed
call :log "requirements.txt changed or this Python has not been installed yet. Running: %PY% -m pip install -r requirements.txt"
"%PY%" -m pip install -r requirements.txt >>"%LOG%" 2>&1
if errorlevel 1 goto :pip_failed
"%PY%" -c "import hashlib,os,pathlib; root=pathlib.Path(os.environ['DFS_ROOT']); py=os.environ['DFS_PYTHON']; req=root/'requirements.txt'; stamp=root/'launcher'/'logs'/'requirements.sha256'; stamp.parent.mkdir(parents=True, exist_ok=True); tmp=stamp.with_suffix('.sha256.tmp'); tmp.write_text(hashlib.sha256(req.read_bytes()).hexdigest()+'\n'+py+'\n', encoding='utf-8'); tmp.replace(stamp)"
if errorlevel 1 goto :pip_stamp_failed
call :log "Dependencies installed. Recorded the requirements stamp under launcher\logs."
goto :eof

:no_req
call :log "WARN: requirements.txt is missing. Skipping pip install."
goto :eof

:pip_cd_failed
call :log "WARN: could not enter the project folder before pip. Launching local code without installing."
goto :eof

:pip_hash_failed
call :log "WARN: could not compare requirements.txt. Skipping pip install and launching local code."
goto :eof

:pip_failed
call :log "WARN: pip install failed. The requirements stamp was not updated, so the next launch will retry. Launching local code anyway."
goto :eof

:pip_stamp_failed
call :log "WARN: dependencies installed but the stamp could not be written. The next launch will run pip again."
goto :eof
