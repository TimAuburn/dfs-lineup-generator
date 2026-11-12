# Streamlit Community Cloud

Deploy this repository from the TimAuburn GitHub account as one private app.
The free tier provides one private app.

## App settings

- Repository: `TimAuburn/dfs-lineup-generator`
- Branch: `main`
- Main file path: `app.py`
- Python version: 3.12

`app.py` is the Streamlit entrypoint. It is the only script that calls
`st.set_page_config`. The README and `launcher/launch_dfs.bat` both start it
with `streamlit run app.py`.

Community Cloud installs `requirements.txt`. Test-only packages are listed in
`requirements-dev.txt` and are not installed on Cloud. Do not add
`packages.txt`. `pydfs-lineup-optimizer` 3.6.1 requires `PuLP==2.4`, and that
release's Linux build includes the CBC solver binary this app uses.

No `.streamlit/config.toml` is included. Salary and contest uploads are CSV or
ZIP files, and Streamlit's default upload limit is 200 MB.

## Odds API key

In the app's **Advanced settings**, open **Secrets** and add:

```toml
ODDS_API_KEY = "..."
```

The app reads `st.secrets["ODDS_API_KEY"]` when that variable is not already
in the environment. The sidebar caption names the source — the process
environment, a `.env` file, or Streamlit secrets — and does not show the key.

## Data and runtime

`data/dfs_tracker.db` is the SQLite contest archive. The Community Cloud
filesystem is ephemeral, so that file is temporary and is reset when the app
restarts or is redeployed.

Free-tier apps sleep after 12 hours without traffic. The next visit starts
the app again.

A 150-lineup Classic GPP run can take 8-10+ minutes. Leave the browser tab
open until it finishes.
