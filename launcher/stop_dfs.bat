@echo off
setlocal EnableExtensions

rem Stop only this project's Streamlit server on port 8501.
rem The listener must be identified as this app before taskkill runs.
rem This script does not stop other Python programs.

set "LAUNCHER_DIR=%~dp0"
if "%LAUNCHER_DIR:~-1%"=="\" set "LAUNCHER_DIR=%LAUNCHER_DIR:~0,-1%"
cd /d "%LAUNCHER_DIR%\.."
if errorlevel 1 goto :cd_failed
set "ROOT=%CD%"
set "LOGDIR=%ROOT%\launcher\logs"
set "LOG=%LOGDIR%\launch.log"
set "HELPER=%ROOT%\launcher\windows_helpers.ps1"
set "DFS_LAUNCH_LOG=%LOG%"

title Stop DFS Lineup Generator

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if errorlevel 1 goto :logdir_failed
if not exist "%HELPER%" goto :missing_helper

call :log "---- stop %DATE% %TIME% ----"
call :log "Project root: %ROOT%"
call :query_port

if /I "%PORT_STATE%"=="FREE" goto :already_stopped
if /I "%PORT_STATE%"=="OURS" goto :do_stop
if /I "%PORT_STATE%"=="OTHER" goto :refused
goto :unknown

:do_stop
set "OLD_PID=%PORT_PID%"
call :digits_only "%OLD_PID%"
if not "%DIGITS_OK%"=="1" goto :bad_pid
call :log "Stopping this app's Streamlit server. PID %OLD_PID%."
taskkill /PID %OLD_PID% /T
set "KILL_RC=%ERRORLEVEL%"
if not "%KILL_RC%"=="0" goto :kill_graceful_failed
ping -n 4 127.0.0.1 >nul 2>nul
call :query_port
if /I not "%PORT_STATE%"=="OURS" goto :stopped
if not "%PORT_PID%"=="%OLD_PID%" goto :stopped
goto :force_stop

:kill_graceful_failed
call :log "WARN: graceful stop returned %KILL_RC%. Checking whether PID %OLD_PID% is still this app."
ping -n 3 127.0.0.1 >nul 2>nul
call :query_port
if /I not "%PORT_STATE%"=="OURS" goto :stopped
if not "%PORT_PID%"=="%OLD_PID%" goto :stopped
goto :force_stop

:force_stop
call :log "Streamlit PID %OLD_PID% is still listening. Forcing that process tree to exit."
taskkill /F /PID %OLD_PID% /T
set "KILL_RC=%ERRORLEVEL%"
if not "%KILL_RC%"=="0" goto :kill_failed
ping -n 3 127.0.0.1 >nul 2>nul
call :query_port
if /I "%PORT_STATE%"=="OURS" if "%PORT_PID%"=="%OLD_PID%" goto :still_running
goto :stopped

:stopped
call :log "Stopped the DFS Lineup Generator Streamlit server. PID %OLD_PID%."
echo.
echo Stopped the DFS Lineup Generator Streamlit server.
echo PID: %OLD_PID%
echo Other Python programs were not stopped.
echo.
pause
exit /b 0

:already_stopped
call :log "No listener on port 8501. Nothing was stopped."
echo.
echo No DFS Lineup Generator server is listening on port 8501.
echo Nothing was stopped.
echo.
pause
exit /b 0

:refused
call :log "WARN: port 8501 belongs to a different program. PID %PORT_PID%. It was not stopped."
echo.
echo Port 8501 is in use by a different program.
echo PID: %PORT_PID%
echo It was not stopped.
echo.
pause
exit /b 1

:unknown
call :log "WARN: could not confirm that port 8501 belongs to this app. Nothing was stopped."
echo.
echo Could not confirm that port 8501 belongs to this app.
echo Nothing was stopped.
echo Details: %LOG%
echo.
pause
exit /b 1

:bad_pid
call :log "WARN: refused to stop because the process id was not a number. PID %OLD_PID%."
echo.
echo Refused to stop a process because the id was not a number.
echo Nothing was stopped.
echo.
pause
exit /b 1

:kill_failed
call :log "WARN: could not stop PID %OLD_PID%. Code %KILL_RC%."
echo.
echo Could not stop PID %OLD_PID%.
echo Nothing else was stopped.
echo Details: %LOG%
echo.
pause
exit /b 1

:still_running
call :log "WARN: PID %OLD_PID% is still listening after the stop attempt."
echo.
echo PID %OLD_PID% is still listening on port 8501.
echo No other programs were stopped.
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

goto :eof

:log
set "LOG_RC=%ERRORLEVEL%"
set "MSG=%~1"
echo %MSG%
>>"%LOG%" echo [%DATE% %TIME%] %MSG%
exit /b %LOG_RC%

:digits_only
set "DIGITS_OK=0"
if "%~1"=="" goto :eof
set "NONDIGIT="
for /f "delims=0123456789" %%N in ("%~1") do set "NONDIGIT=1"
if defined NONDIGIT goto :eof
set "DIGITS_OK=1"
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
