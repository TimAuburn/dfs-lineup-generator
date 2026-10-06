# DraftKings NFL Lineup Optimizer

Streamlit + `pydfs-lineup-optimizer` tool for DraftKings **NFL Classic** (9-player) and **Showdown Captain Mode** (6-player) lineups.

## Features

- Contest format toggle: Classic vs Showdown
- Contest presets that configure solver rules:
  - Classic: Cash, Small-Field GPP, Large-Field Multi-Entry GPP
  - Showdown: Showdown GPP, Showdown Cash
- Projection jitter before optimize (reproducible default seed `42`; pass `None` for entropy)
- DK upload CSV export (Classic name+ID slots; Showdown numeric CPT/FLEX IDs)
- Vegas lines from The Odds API (key in `.env`) with ITT projection tweaks
- Weather Company / weather.com wind checks for outdoor games (key in `.env`);
  wind over 18 mph downgrades pass-game / kicker projections before optimize

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill ODDS_API_KEY and WEATHER_API_KEY
streamlit run app.py
```

API keys are read from `.env` only (not from the Streamlit UI). `.venv/` is the
virtualenv — do not store secrets there. See `.env.example` for variable names.

## Tests

```bash
pytest -q
```
