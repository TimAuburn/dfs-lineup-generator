"""Streamlit UI for the DraftKings NFL Classic / Showdown Lineup Optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exporter import lineups_to_dk_upload_csv, upload_columns_for
from src.optimizer import DK_SALARY_CAP, get_nfl_optimizer
from src.parser import DKCSVValidationError, parse_dk_csv
from src.presets import (
    ContestFormat,
    ContestPreset,
    PRESET_LABELS,
    apply_preset,
    get_preset_config,
    presets_for_format,
)

st.set_page_config(
    page_title="DK NFL Lineup Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FORMAT_LABELS = {
    ContestFormat.CLASSIC: "NFL Classic (9 Players)",
    ContestFormat.SHOWDOWN: "NFL Showdown Captain Mode (6 Players)",
}


def _init_state() -> None:
    defaults = {
        "player_pool": None,
        "lineups": [],
        "error": None,
        "flash_success": None,
        "contest_format": ContestFormat.CLASSIC.value,
        "contest_preset": ContestPreset.CLASSIC_SMALL_GPP.value,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _on_format_change() -> None:
    fmt = ContestFormat(st.session_state.contest_format)
    presets = presets_for_format(fmt)
    st.session_state.contest_preset = presets[0].value
    st.session_state.player_pool = None
    st.session_state.lineups = []
    st.session_state.error = None


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — format, preset, and compact optimizer settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Contest Format")
    st.radio(
        "Contest Format",
        options=[ContestFormat.CLASSIC.value, ContestFormat.SHOWDOWN.value],
        format_func=lambda v: FORMAT_LABELS[ContestFormat(v)],
        key="contest_format",
        on_change=_on_format_change,
        label_visibility="collapsed",
    )

    contest_format = ContestFormat(st.session_state.contest_format)
    available_presets = presets_for_format(contest_format)
    if st.session_state.contest_preset not in {p.value for p in available_presets}:
        st.session_state.contest_preset = available_presets[0].value

    st.markdown("### Contest Preset")
    st.selectbox(
        "Contest Preset",
        options=[p.value for p in available_presets],
        format_func=lambda v: PRESET_LABELS[ContestPreset(v)],
        key="contest_preset",
        label_visibility="collapsed",
    )
    preset = ContestPreset(st.session_state.contest_preset)
    preset_cfg = get_preset_config(preset)
    st.caption(preset_cfg.description)

    lo, hi = preset_cfg.lineup_count_range
    lineup_count = st.number_input(
        "Lineups",
        min_value=int(lo),
        max_value=int(hi),
        value=int(min(max(preset_cfg.default_lineup_count, lo), hi)),
        step=1,
        key=f"lineup_count_{preset.value}",
    )

    with st.expander("Advanced overrides", expanded=False):
        jitter_seed = st.number_input(
            "Jitter seed",
            min_value=0,
            max_value=10_000_000,
            value=42,
            help="Projection jitter uses this seed (reproducible). Not used when jitter is 0%.",
        )
        use_max_exposure = st.checkbox("Limit max player exposure", value=False)
        max_exposure_pct = st.slider(
            "Max exposure %", 1, 100, 40, disabled=not use_max_exposure
        )
        use_min_exposure = st.checkbox("Require min player exposure", value=False)
        min_exposure_pct = st.slider(
            "Min exposure %", 0, 100, 0, disabled=not use_min_exposure
        )

# ---------------------------------------------------------------------------
# Main — title + primary action bar (upload / generate / download)
# ---------------------------------------------------------------------------
is_showdown = contest_format is ContestFormat.SHOWDOWN
title = (
    "DraftKings NFL Showdown Optimizer"
    if is_showdown
    else "DraftKings NFL Classic Lineup Optimizer"
)
st.title(title)
upload_help = (
    "Official DraftKings NFL Showdown / Captain Mode DKSalaries.csv "
    "(CPT + FLEX rows per player)."
    if is_showdown
    else "Official DraftKings NFL Classic salary export (DKSalaries.csv)."
)
st.caption(
    "Load a DK salary CSV, pick a contest preset, and export multi-entry lineups."
)

export_headers = ",".join(upload_columns_for(contest_format))
st.caption(f"Export template: `{export_headers}`")

lineups = st.session_state.lineups
csv_text = (
    lineups_to_dk_upload_csv(lineups, contest_format=contest_format) if lineups else ""
)
download_name = (
    "dk_nfl_showdown_upload.csv" if is_showdown else "dk_nfl_classic_upload.csv"
)

upload_col, generate_col, download_col = st.columns([3, 1.2, 1.6], gap="medium")
with upload_col:
    uploaded = st.file_uploader(
        "DraftKings player pool CSV",
        type=["csv"],
        help=upload_help,
        key=f"uploader_{contest_format.value}",
    )
with generate_col:
    st.write("")
    generate = st.button(
        "Generate Lineups",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.player_pool is None and uploaded is None,
    )
with download_col:
    st.write("")
    st.download_button(
        "Download DK Upload CSV",
        data=csv_text,
        file_name=download_name,
        mime="text/csv",
        type="secondary",
        use_container_width=True,
        disabled=not bool(lineups),
    )

if uploaded is not None:
    try:
        st.session_state.player_pool = parse_dk_csv(
            uploaded, contest_format=contest_format
        )
        st.session_state.error = None
    except DKCSVValidationError as exc:
        st.session_state.error = str(exc)
        st.session_state.player_pool = None

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.flash_success:
    st.success(st.session_state.flash_success)
    st.session_state.flash_success = None

pool: pd.DataFrame | None = st.session_state.player_pool

if pool is None:
    st.info(
        "Upload a DraftKings NFL Showdown DKSalaries.csv (CPT/FLEX rows) to begin."
        if is_showdown
        else "Upload a DraftKings NFL Classic DKSalaries.csv to begin."
    )
    st.stop()

roster_n = 6 if is_showdown else 9
st.subheader("Player pool")
st.caption(f"{len(pool)} rows loaded · {roster_n}-player roster · salary cap ${DK_SALARY_CAP:,}")
editable_cols = ["Name", "Position", "TeamAbbrev", "Salary", "AvgPointsPerGame", "Projection"]
display_extra = [
    c
    for c in ["ID", "Name + ID", "Roster Position", "RosterSlot", "Game Info", "Ownership"]
    if c in pool.columns
]
edited = st.data_editor(
    pool[editable_cols + display_extra],
    use_container_width=True,
    hide_index=True,
    height=320,
    disabled=[c for c in pool.columns if c != "Projection"],
    column_config={
        "Projection": st.column_config.NumberColumn(
            "Projection", help="Editable custom projection (FPPG)"
        ),
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
    },
    key=f"player_editor_{contest_format.value}",
)

working_pool = pool.copy()
working_pool["Projection"] = pd.to_numeric(edited["Projection"], errors="coerce").fillna(
    working_pool["AvgPointsPerGame"]
).values

if generate:
    progress = st.progress(0, text="Building optimizer…")
    try:
        optimizer = get_nfl_optimizer(contest_format)
        optimizer.load_from_dataframe(working_pool)
        apply_preset(
            optimizer,
            preset,
            lineup_count=int(lineup_count),
            jitter_seed=int(jitter_seed),
        )
        optimizer.set_exposure_limits(
            min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
            max_exposure_pct=float(max_exposure_pct) if use_max_exposure else None,
        )
        progress.progress(30, text=f"Generating {lineup_count} lineups…")
        generated = optimizer.generate_lineups()
        progress.progress(100, text="Done")
        st.session_state.lineups = generated
        st.session_state.flash_success = (
            f"Generated {len(generated)} lineup(s) with preset "
            f"{PRESET_LABELS[preset]}."
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - show optimizer failures in UI
        st.session_state.lineups = []
        st.error(f"Optimization failed: {exc}")
    finally:
        progress.empty()

lineups = st.session_state.lineups
if lineups:
    st.subheader(f"Generated lineups ({len(lineups)})")

    options = list(range(len(lineups)))

    def _lineup_label(i: int) -> str:
        lu = lineups[i]
        salary = lu.attrs.get("salary", int(lu["Salary"].sum()))
        projection = lu.attrs.get("projection", float(lu["Projection"].sum()))
        return f"Lineup {i + 1} · ${salary:,} · {projection:.2f} proj"

    selected = st.selectbox("View lineup", options=options, format_func=_lineup_label)
    lineup = lineups[selected]
    st.dataframe(
        lineup[["Slot", "Name", "Team", "Salary", "Projection"]],
        use_container_width=True,
        hide_index=True,
        height=320,
    )
