# DraftKings NFL Lineup Optimizer

Streamlit + `pydfs-lineup-optimizer` tool for DraftKings **NFL Classic** (9-player) and **Showdown Captain Mode** (6-player) lineups.

## Features

- Contest format toggle: Classic vs Showdown
- Contest presets that configure solver rules:
  - Classic: Cash, Small-Field GPP, Large-Field Multi-Entry GPP
  - Showdown: Showdown GPP, Showdown Cash
- Projection jitter before optimize (reproducible default seed `42`; pass `None` for entropy)
- DK upload CSV export (Classic name+ID slots; Showdown numeric CPT/FLEX IDs)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```
