"""Streamlit UI for the DraftKings NFL Classic Lineup Optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exporter import lineups_to_dk_upload_csv
from src.optimizer import DK_SALARY_CAP, NFLClassicOptimizer
from src.parser import DKCSVValidationError, parse_dk_csv

st.set_page_config(
    page_title="DK NFL Classic Lineup Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults = {
        "player_pool": None,
        "lineups": [],
        "error": None,
        "flash_success": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — compact optimizer settings only (upload/actions live on main)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    lineup_count = st.number_input("Lineups", min_value=1, max_value=150, value=5, step=1)
    max_repeating = st.slider(
        "Max repeating players",
        min_value=1,
        max_value=8,
        value=7,
        help="Maximum players that may appear in consecutive lineups (uniqueness).",
    )

    with st.expander("Exposure", expanded=False):
        use_max_exposure = st.checkbox("Limit max player exposure", value=False)
        max_exposure_pct = st.slider(
            "Max exposure %", 1, 100, 40, disabled=not use_max_exposure
        )
        use_min_exposure = st.checkbox("Require min player exposure", value=False)
        min_exposure_pct = st.slider(
            "Min exposure %", 0, 100, 0, disabled=not use_min_exposure
        )

    with st.expander("Stacking", expanded=False):
        enable_qb_stack = st.checkbox("QB + WR/TE stack", value=True)
        stack_catchers = st.selectbox(
            "Pass catchers with QB",
            options=[1, 2],
            index=0,
            disabled=not enable_qb_stack,
        )
        enable_bring_back = st.checkbox(
            "Opposing bring-back",
            value=False,
            disabled=not enable_qb_stack,
            help="Force a WR/TE from the team opposing the stacked QB.",
        )
        bring_back_pos = st.selectbox(
            "Bring-back position",
            options=["WR", "TE"],
            index=0,
            disabled=not (enable_qb_stack and enable_bring_back),
        )

# ---------------------------------------------------------------------------
# Main — title + primary action bar (upload / generate / download)
# ---------------------------------------------------------------------------
st.title("DraftKings NFL Classic Lineup Optimizer")
st.caption("Load a DK salary CSV, tune stack/exposure rules, and export multi-entry lineups.")

lineups = st.session_state.lineups
csv_text = lineups_to_dk_upload_csv(lineups) if lineups else ""

upload_col, generate_col, download_col = st.columns([3, 1.2, 1.6], gap="medium")
with upload_col:
    uploaded = st.file_uploader(
        "DraftKings player pool CSV",
        type=["csv"],
        help="Official DraftKings NFL Classic salary export.",
    )
with generate_col:
    st.write("")  # align with uploader controls
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
        file_name="dk_nfl_classic_upload.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=True,
        disabled=not bool(lineups),
    )

if uploaded is not None:
    try:
        st.session_state.player_pool = parse_dk_csv(uploaded)
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
    st.info("Upload a DraftKings NFL Classic salary CSV to begin.")
    st.stop()

st.subheader("Player pool")
st.caption(f"{len(pool)} players loaded · salary cap ${DK_SALARY_CAP:,}")
editable_cols = ["Name", "Position", "TeamAbbrev", "Salary", "AvgPointsPerGame", "Projection"]
edited = st.data_editor(
    pool[
        editable_cols
        + [c for c in ["ID", "Name + ID", "Roster Position", "Game Info"] if c in pool.columns]
    ],
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
    key="player_editor",
)

# Merge edited projections back onto the full pool.
working_pool = pool.copy()
working_pool["Projection"] = pd.to_numeric(edited["Projection"], errors="coerce").fillna(
    working_pool["AvgPointsPerGame"]
).values

if generate:
    progress = st.progress(0, text="Building optimizer…")
    try:
        optimizer = NFLClassicOptimizer()
        optimizer.load_from_dataframe(working_pool)
        optimizer.set_lineup_count(int(lineup_count))
        optimizer.set_max_repeating_players(int(max_repeating))
        optimizer.set_exposure_limits(
            min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
            max_exposure_pct=float(max_exposure_pct) if use_max_exposure else None,
        )
        optimizer.set_qb_stack(
            enabled=enable_qb_stack,
            pass_catchers=int(stack_catchers),
            bring_back=enable_bring_back,
            bring_back_position=bring_back_pos,
        )
        progress.progress(30, text=f"Generating {lineup_count} lineups…")
        generated = optimizer.generate_lineups()
        progress.progress(100, text="Done")
        st.session_state.lineups = generated
        st.session_state.flash_success = f"Generated {len(generated)} lineup(s)."
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
