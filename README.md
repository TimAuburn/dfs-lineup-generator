# DraftKings NFL Lineup Optimizer

Streamlit + `pydfs-lineup-optimizer` tool for DraftKings **NFL Classic** (9-player) and **Showdown Captain Mode** (6-player) lineups.

## Features

- Contest format toggle: Classic vs Showdown
- Contest presets that configure solver rules:
  - Classic: Cash, Small-Field GPP, Large-Field Multi-Entry GPP
  - Showdown: Showdown GPP, Showdown Cash
- Sleeper weekly PPR projections and injury status (no API key); OUT, IR, and Doubtful players are removed
- Projection jitter before optimize (reproducible default seed `42`; pass `None` for entropy)
- DK upload CSV export (Classic name+ID slots; Showdown numeric CPT/FLEX IDs)
- Vegas lines from The Odds API (key in `.env`) with ITT projection tweaks
- National Weather Service wind checks for outdoor games (no API key);
  wind over 18 mph downgrades pass-game / kicker projections before optimize

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill ODDS_API_KEY
streamlit run app.py
```

API keys are read from `.env` only (not from the Streamlit UI). `.venv/` or
`venv/` is the virtualenv — do not store secrets there. See `.env.example` for
variable names. The desktop launcher uses `.venv` when both folders exist.

Windows can start this same local app from a Desktop shortcut. See below.

## Windows desktop shortcut

`launcher\launch_dfs.bat` starts the local Streamlit app without Cursor or a
terminal the operator has to keep typing in. It finds the project folder from
its own location (`cd /d` to the parent of `launcher\`), so the path is not
hard-coded and folders with spaces still work.

The app does not load a salary file by itself. After it opens, upload a
DraftKings `DKSalaries.csv`. This repo includes two local samples:
`tests\mock_dk_salaries.csv` (Classic) and `tests\mock_dk_showdown.csv`
(Showdown). Live Vegas lines need `ODDS_API_KEY` in `.env`. The window still
opens without that key. Sleeper and weather do not need a key; they are
contacted only after a CSV is loaded or a lineup is generated.

### One-time setup

1. Install Git for Windows and Python 3, with `git` and the `py` launcher on `PATH`.
2. In the project folder (the directory that contains `app.py`), create a
   virtualenv and install dependencies. Either folder name works:

   ```bat
   py -3 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   or:

   ```bat
   py -3 -m venv venv
   venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Optional: `copy .env.example .env` and fill `ODDS_API_KEY`.
4. Create the Desktop shortcuts. Desktop is
   `[Environment]::GetFolderPath('Desktop')`, so an OneDrive-redirected Desktop
   is used when Windows has moved that folder:

   ```bat
   powershell -ExecutionPolicy Bypass -File launcher\create_shortcut.ps1
   ```

   The script writes only `DFS Lineup Generator.lnk` and, if you run it again,
   overwrites only that shortcut. If an older copy of this project left
   `Stop DFS Lineup Generator.lnk` on the Desktop, that one file is removed.
   No other Desktop shortcut is touched.

   - `DFS Lineup Generator.lnk` — starts the app. The shortcut is saved with
     Run: Minimized. Its target is `launcher\launch_dfs.bat` and its working
     directory is the project root.

   An icon is optional. If `launcher\dfs.ico` exists, the shortcuts use it.

### What a launch does

Double-click **DFS Lineup Generator**.

1. The launcher `cd /d`s to the project root and looks for Python in this
   order: `.venv\Scripts\python.exe`, then `venv\Scripts\python.exe`. When both
   exist, `.venv` is used for pip and Streamlit. The chosen path is written to
   `launcher\logs\launch.log`. If neither file exists, the window names both
   paths and waits. It does not create a virtualenv for you.
2. It runs `git fetch origin main`.
3. It runs `git pull --ff-only origin main` only when that fetch succeeded, the
   working tree is clean, and the current branch is `main`. A dirty tree,
   any other branch, a failed fetch (including no network), or a failed pull
   is a warning. The copy already on disk still starts. The launcher does not
   stash, reset, or discard local work.
4. If `requirements.txt` changed, or the selected Python is not the one
   recorded for the last successful install, it runs that Python with
   `-m pip install -r requirements.txt` and does not activate the virtualenv.
   The stamp is `launcher\logs\requirements.sha256` and includes the
   interpreter path. A failed install does not update the stamp, so the next
   launch tries again, and the app still attempts to start.
5. It runs the selected Python with `-m streamlit run app.py` on port 8501,
   bound to localhost. `launcher\idle_watch.py` is the foreground process so
   the minimized window stays open with the server. If this app is already
   listening there, the launcher opens the browser and does not start a second
   server. If something else is listening, or the owner cannot be confirmed,
   it warns and leaves that process alone.
6. A second double-click while the first launch is still updating waits, then
   opens the browser once this server is up.
7. While at least one browser tab is connected, the app keeps running. That
   includes a tab where you are not clicking and a tab that is building
   lineups. Streamlit reports those tabs as `active_sessions`. After every
   connected tab has been gone for the idle timeout, the server and this
   window exit. The default is 7200 seconds (2 hours). Set `DFS_IDLE_SECONDS`
   before launch to use another positive number of seconds. The timeout is not
   a fixed clock from startup: a connected tab resets it. A failed reading of
   the session gauge does not shut the app down. Opening `/_stcore/health`
   does not count as someone using the app.

The shortcut keeps the window minimized. It is not hidden. The window title is
**DFS Lineup Generator**. Startup messages stay in that window. Launch output,
including Streamlit's own console output, is appended to
`launcher\logs\launch.log`. Short status lines go through a small append that
retries a few times if OneDrive has the file open, and then skips the write
instead of printing a file-access error. The window still shows the launch
result. If the site does not open, restore the launcher window from the
taskbar. A failed start leaves the log tail on screen until you press a key.
An idle exit is also written to that log after the server releases the file,
with the reason `no connected browser session`.

### Stopping

Close the minimized **DFS Lineup Generator** window. That stops the server.
There is no separate Stop shortcut. Running `launcher\create_shortcut.ps1`
again removes a leftover `Stop DFS Lineup Generator.lnk` from the Desktop and
does not remove any other shortcut.

If a launch keeps reporting that another launch is in progress after the app
has quit, delete `launcher\logs\launch.lockdir` and try again. That folder is
only the single-launch lock. Deleting it does not change your code. An idle
exit removes it as well, so the next double-click can start again.

## Tests

```bash
pytest -q
```
