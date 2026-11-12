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

`streamlit run` copies that root-level value into the environment before the
app starts. The copy replaces a process environment value of the same name,
and a `.env` file does not override it. The sidebar caption names the source
— Streamlit secrets, the process environment, or a `.env` file — and does
not show the key.

## Data and runtime

`data/dfs_tracker.db` is the SQLite contest archive. The Community Cloud
filesystem is ephemeral, so that file is temporary and is reset when the app
restarts or is redeployed.

Free-tier apps sleep after 12 hours without traffic. The next visit starts
the app again.

A 150-lineup Classic GPP run can take 8-10+ minutes. Leave the browser tab
open until it finishes.
