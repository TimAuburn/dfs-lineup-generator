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

API keys are read from `.env` only (not from the Streamlit UI). `.venv/` is the
virtualenv — do not store secrets there. See `.env.example` for variable names.

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
2. In the project folder (the directory that contains `app.py`), create the
   virtualenv this repo already uses and install dependencies:

   ```bat
   py -3 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Optional: `copy .env.example .env` and fill `ODDS_API_KEY`.
4. Create the Desktop shortcuts. Desktop is
   `[Environment]::GetFolderPath('Desktop')`, so an OneDrive-redirected Desktop
   is used when Windows has moved that folder:

   ```bat
   powershell -ExecutionPolicy Bypass -File launcher\create_shortcut.ps1
   ```

   The script writes only these two shortcuts and, if you run it again,
   overwrites only those two:

   - `DFS Lineup Generator.lnk` — starts the app. The shortcut is saved with
     Run: Minimized. Its target is `launcher\launch_dfs.bat` and its working
     directory is the project root.
   - `Stop DFS Lineup Generator.lnk` — stops this app's server. The window is
     normal so the result stays on screen.

   An icon is optional. If `launcher\dfs.ico` exists, the shortcuts use it.

### What a launch does

Double-click **DFS Lineup Generator**.

1. The launcher `cd /d`s to the project root and requires
   `.venv\Scripts\python.exe`. If the virtualenv is missing, the window prints
   the setup commands and waits. It does not create the virtualenv for you.
2. It runs `git fetch origin main`.
3. It runs `git pull --ff-only origin main` only when that fetch succeeded, the
   working tree is clean, and the current branch is `main`. A dirty tree,
   any other branch, a failed fetch (including no network), or a failed pull
   is a warning. The copy already on disk still starts. The launcher does not
   stash, reset, or discard local work.
4. If `requirements.txt` differs from the last successful install, it runs
   `.venv\Scripts\python.exe -m pip install -r requirements.txt` without
   activating the virtualenv. The stamp is `launcher\logs\requirements.sha256`.
   A failed install does not update the stamp, so the next launch tries again,
   and the app still attempts to start.
5. It runs `.venv\Scripts\python.exe -m streamlit run app.py` on port 8501,
   bound to localhost. If this app is already listening there, the launcher
   opens the browser and does not start a second server. If something else is
   listening, or the owner cannot be confirmed, it warns and leaves that
   process alone.
6. A second double-click while the first launch is still updating waits, then
   opens the browser once this server is up.

The shortcut keeps the window minimized. It is not hidden. The window title is
**DFS Lineup Generator**. Startup messages stay in that window, and the full
session is appended to `launcher\logs\launch.log` (gitignored, along with the
rest of `launcher\logs\`). If the site does not open, restore the window from
the taskbar. A failed start leaves the log tail on screen until you press a key.

If a launch keeps reporting that another launch is in progress after the app
has quit, delete `launcher\logs\launch.lockdir` and try again. That folder is
only the single-launch lock. Deleting it does not change your code.

### Stopping

Use **Stop DFS Lineup Generator**, or close the minimized launcher window.
Closing the window stops the server because Streamlit is running in it.

The stop shortcut ends a process only when all of these are true:

- it is listening on port 8501
- its command line is this project's `streamlit run app.py` (the project path
  has to match as a whole folder, so a similarly named folder is not treated
  as this app)

It then stops that process tree only. It does not kill other Python programs.
If port 8501 is free, it reports that and exits. If the listener is some other
program, or the check is inconclusive, it reports that and does not stop it.

## Tests

```bash
pytest -q
```
