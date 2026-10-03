"""Streamlit UI for the DraftKings NFL Classic / Showdown Lineup Optimizer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exporter import lineups_to_dk_upload_csv, upload_columns_for
from src.optimizer import (
    DK_SALARY_CAP,
    STANDARD_TIER_FLOOR,
    get_nfl_optimizer,
    optimize_tiered_salaries,
    tiered_lineup_counts,
)
from src.parser import DKCSVValidationError, parse_dk_csv
from src.presets import (
    ContestFormat,
    ContestPreset,
    PRESET_LABELS,
    PresetConfig,
    apply_preset,
    config_with_overrides,
    format_settings_comparison,
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
        "generated_run": None,
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
    st.session_state.generated_run = None
    st.session_state.error = None


def _preset_key(preset: ContestPreset, name: str) -> str:
    """Widget keys include the preset so controls reset when the preset changes."""
    return f"{name}_{preset.value}"


def _render_settings_comparison(rows: List[Tuple[str, str, str]]) -> None:
    summary_df = pd.DataFrame(rows, columns=["Setting", "Preset default", "Actual"])
    st.dataframe(
        summary_df,
        width="stretch",
        hide_index=True,
        height=min(38 + 35 * len(summary_df), 420),
    )


def _extra_setting_rows(
    *,
    jitter_seed: int,
    use_max_exposure: bool,
    max_exposure_pct: int,
    use_min_exposure: bool,
    min_exposure_pct: int,
) -> List[Tuple[str, str, str]]:
    """UI-only settings that are not part of a contest preset."""
    rows: List[Tuple[str, str, str]] = []
    if int(jitter_seed) != 42:
        rows.append(("Jitter seed", "42", str(int(jitter_seed))))
    if use_max_exposure or use_min_exposure:
        rows.append(
            (
                "Max exposure",
                "Off",
                f"{int(max_exposure_pct)}%" if use_max_exposure else "Off",
            )
        )
        rows.append(
            (
                "Min exposure",
                "Off",
                f"{int(min_exposure_pct)}%" if use_min_exposure else "Off",
            )
        )
    return rows


def _collect_strategy_overrides(
    contest_format: ContestFormat,
    preset: ContestPreset,
    preset_cfg: PresetConfig,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Render editable strategy controls seeded from the active preset.

    Returns a dict of override values to apply after ``apply_preset``, plus a
    list of human-readable labels for fields that differ from preset defaults.
    """
    overrides: Dict[str, Any] = {}
    changed: List[str] = []

    st.caption(f"Defaults from: **{preset_cfg.label}** — edit any value to deviate.")

    preset_jitter = preset_cfg.jitter_pct or (0.0, 0.0)
    preset_j_lo = int(round(float(preset_jitter[0]) * 100))
    preset_j_hi = int(round(float(preset_jitter[1]) * 100))
    use_position_sds = bool(preset_cfg.position_jitter_sds)
    enable_jitter = st.checkbox(
        "Projection jitter",
        value=preset_j_hi > 0 or use_position_sds,
        key=_preset_key(preset, "enable_jitter"),
        help=(
            "Large-Field GPP uses position-specific SDs by default; "
            "uncheck to disable, or set a uniform % band to override."
        ),
    )
    jitter_band = st.slider(
        "Jitter band %",
        min_value=0,
        max_value=30,
        value=(max(preset_j_lo, 0), max(preset_j_hi, 1) if preset_j_hi == 0 else preset_j_hi)
        if not use_position_sds
        else (12, 18),
        disabled=not enable_jitter,
        key=_preset_key(preset, "jitter_band"),
    )
    keep_position_sds = st.checkbox(
        "Use position-specific jitter SDs",
        value=use_position_sds,
        disabled=not enable_jitter or not use_position_sds,
        key=_preset_key(preset, "keep_position_sds"),
        help="Only available when the preset defines position SDs (Large-Field GPP).",
    )
    if not enable_jitter:
        overrides["jitter_pct"] = (0.0, 0.0)
        overrides["position_jitter_sds"] = None
    elif keep_position_sds and use_position_sds:
        overrides["jitter_pct"] = preset_cfg.jitter_pct
        overrides["position_jitter_sds"] = dict(preset_cfg.position_jitter_sds or {})
    else:
        overrides["jitter_pct"] = (
            float(jitter_band[0]) / 100.0,
            float(jitter_band[1]) / 100.0,
        )
        overrides["position_jitter_sds"] = None

    preset_jitter_on = preset_j_hi > 0 or use_position_sds
    if bool(enable_jitter) != preset_jitter_on:
        changed.append("Jitter")
    elif enable_jitter and use_position_sds and not keep_position_sds:
        changed.append("Jitter")
    elif (
        enable_jitter
        and not use_position_sds
        and (
            int(jitter_band[0]) != preset_j_lo or int(jitter_band[1]) != preset_j_hi
        )
    ):
        changed.append("Jitter")

    if contest_format is ContestFormat.CLASSIC:
        enable_qb_stack = st.checkbox(
            "QB stack",
            value=preset_cfg.enable_qb_stack,
            key=_preset_key(preset, "enable_qb_stack"),
        )
        stack_pass_catchers = st.number_input(
            "Pass-catchers in stack",
            min_value=1,
            max_value=3,
            value=int(preset_cfg.stack_pass_catchers),
            step=1,
            disabled=not enable_qb_stack,
            key=_preset_key(preset, "stack_pass_catchers"),
        )
        enable_bring_back = st.checkbox(
            "Opposing bring-back",
            value=preset_cfg.enable_bring_back,
            disabled=not enable_qb_stack,
            key=_preset_key(preset, "enable_bring_back"),
        )
        bring_back_position = st.selectbox(
            "Bring-back position",
            options=["WR", "TE"],
            index=0 if preset_cfg.bring_back_position == "WR" else 1,
            disabled=not (enable_qb_stack and enable_bring_back),
            key=_preset_key(preset, "bring_back_position"),
        )
        exclude_vs_dst = st.checkbox(
            "Exclude offense vs DST",
            value=preset_cfg.exclude_vs_dst,
            key=_preset_key(preset, "exclude_vs_dst"),
        )

        overrides.update(
            enable_qb_stack=bool(enable_qb_stack),
            stack_pass_catchers=int(stack_pass_catchers),
            enable_bring_back=bool(enable_bring_back),
            bring_back_position=str(bring_back_position),
            exclude_vs_dst=bool(exclude_vs_dst),
        )
        if bool(enable_qb_stack) != preset_cfg.enable_qb_stack:
            changed.append("QB stack")
        elif enable_qb_stack and int(stack_pass_catchers) != preset_cfg.stack_pass_catchers:
            changed.append("Pass-catchers")
        if bool(enable_bring_back) != preset_cfg.enable_bring_back:
            changed.append("Bring-back")
        elif (
            enable_bring_back
            and str(bring_back_position) != preset_cfg.bring_back_position
        ):
            changed.append("Bring-back position")
        if bool(exclude_vs_dst) != preset_cfg.exclude_vs_dst:
            changed.append("Exclude vs DST")
    else:
        min_default = int(preset_cfg.min_salary_cap or 0)
        max_default = int(preset_cfg.max_salary_cap or DK_SALARY_CAP)
        use_salary_window = st.checkbox(
            "Enforce salary window",
            value=preset_cfg.min_salary_cap is not None
            or preset_cfg.max_salary_cap is not None,
            key=_preset_key(preset, "use_salary_window"),
        )
        salary_lo, salary_hi = st.slider(
            "Salary window",
            min_value=40_000,
            max_value=DK_SALARY_CAP,
            value=(
                min_default if min_default > 0 else 48_000,
                max_default if max_default > 0 else DK_SALARY_CAP,
            ),
            step=100,
            disabled=not use_salary_window,
            key=_preset_key(preset, "salary_window"),
        )
        require_both_teams = st.checkbox(
            "Require both teams",
            value=bool(preset_cfg.min_teams and preset_cfg.min_teams >= 2),
            key=_preset_key(preset, "require_both_teams"),
        )
        game_script = st.checkbox(
            "Game-script correlation (CPT stacks)",
            value=preset_cfg.game_script_correlation,
            key=_preset_key(preset, "game_script"),
        )
        cpt_qb_pass_catchers = st.number_input(
            "CPT QB pass-catchers",
            min_value=1,
            max_value=3,
            value=int(preset_cfg.cpt_qb_pass_catchers),
            step=1,
            disabled=not game_script,
            key=_preset_key(preset, "cpt_qb_pass_catchers"),
        )
        limit_k_dst = st.checkbox(
            "Limit combined K + DST",
            value=preset_cfg.max_k_dst_combined is not None,
            key=_preset_key(preset, "limit_k_dst"),
        )
        max_k_dst = st.number_input(
            "Max K + DST combined",
            min_value=0,
            max_value=6,
            value=int(
                preset_cfg.max_k_dst_combined
                if preset_cfg.max_k_dst_combined is not None
                else 2
            ),
            step=1,
            disabled=not limit_k_dst,
            key=_preset_key(preset, "max_k_dst"),
        )

        overrides.update(
            min_salary_cap=int(salary_lo) if use_salary_window else None,
            max_salary_cap=int(salary_hi) if use_salary_window else None,
            min_teams=2 if require_both_teams else None,
            game_script_correlation=bool(game_script),
            cpt_qb_pass_catchers=int(cpt_qb_pass_catchers),
            max_k_dst_combined=int(max_k_dst) if limit_k_dst else None,
        )
        preset_uses_window = (
            preset_cfg.min_salary_cap is not None or preset_cfg.max_salary_cap is not None
        )
        if bool(use_salary_window) != preset_uses_window:
            changed.append("Salary window")
        elif use_salary_window and (
            int(salary_lo) != int(preset_cfg.min_salary_cap or salary_lo)
            or int(salary_hi) != int(preset_cfg.max_salary_cap or salary_hi)
        ):
            changed.append("Salary window")
        preset_both = bool(preset_cfg.min_teams and preset_cfg.min_teams >= 2)
        if bool(require_both_teams) != preset_both:
            changed.append("Min teams")
        if bool(game_script) != preset_cfg.game_script_correlation:
            changed.append("Game-script")
        elif game_script and int(cpt_qb_pass_catchers) != preset_cfg.cpt_qb_pass_catchers:
            changed.append("CPT QB pass-catchers")
        preset_limit_k = preset_cfg.max_k_dst_combined is not None
        if bool(limit_k_dst) != preset_limit_k:
            changed.append("Max K+DST")
        elif limit_k_dst and int(max_k_dst) != int(preset_cfg.max_k_dst_combined or 0):
            changed.append("Max K+DST")

    limit_repeating = st.checkbox(
        "Limit max repeating players",
        value=preset_cfg.max_repeating_players is not None,
        key=_preset_key(preset, "limit_repeating"),
    )
    default_repeats = (
        int(preset_cfg.max_repeating_players)
        if preset_cfg.max_repeating_players is not None
        else 6
    )
    max_repeating = st.number_input(
        "Max repeating players",
        min_value=0,
        max_value=9 if contest_format is ContestFormat.CLASSIC else 6,
        value=default_repeats,
        step=1,
        disabled=not limit_repeating,
        key=_preset_key(preset, "max_repeating"),
    )

    limit_ownership = st.checkbox(
        "Apply cumulative ownership cap",
        value=preset_cfg.max_cumulative_ownership is not None,
        key=_preset_key(preset, "limit_ownership"),
    )
    default_own_pct = int(
        round((preset_cfg.max_cumulative_ownership or 1.25) * 100)
    )
    ownership_pct = st.number_input(
        "Ownership cap % (cumulative)",
        min_value=50,
        max_value=900,
        value=default_own_pct,
        step=5,
        disabled=not limit_ownership,
        key=_preset_key(preset, "ownership_pct"),
        help="Sum of ownership fractions across the roster (125 = 125%).",
    )

    overrides["max_repeating_players"] = (
        int(max_repeating) if limit_repeating else None
    )
    overrides["max_cumulative_ownership"] = (
        float(ownership_pct) / 100.0 if limit_ownership else None
    )

    preset_limit_rep = preset_cfg.max_repeating_players is not None
    if bool(limit_repeating) != preset_limit_rep:
        changed.append("Max repeating")
    elif limit_repeating and int(max_repeating) != int(
        preset_cfg.max_repeating_players or 0
    ):
        changed.append("Max repeating")

    preset_limit_own = preset_cfg.max_cumulative_ownership is not None
    if bool(limit_ownership) != preset_limit_own:
        changed.append("Ownership cap")
    elif limit_ownership and abs(
        float(ownership_pct) / 100.0 - float(preset_cfg.max_cumulative_ownership or 0)
    ) > 1e-9:
        changed.append("Ownership cap")

    return overrides, changed


def _apply_strategy_overrides(optimizer: Any, overrides: Dict[str, Any]) -> None:
    """Apply UI strategy deviations on top of ``apply_preset``."""
    if "jitter_pct" in overrides:
        optimizer.jitter_pct = overrides["jitter_pct"]
        optimizer.position_jitter_sds = overrides.get("position_jitter_sds")

    if "enable_qb_stack" in overrides and hasattr(optimizer, "set_qb_stack"):
        optimizer.set_qb_stack(
            enabled=overrides["enable_qb_stack"],
            pass_catchers=overrides["stack_pass_catchers"],
            bring_back=overrides["enable_bring_back"],
            bring_back_position=overrides["bring_back_position"],
        )
    if "exclude_vs_dst" in overrides:
        optimizer.exclude_vs_dst = overrides["exclude_vs_dst"]

    if "min_salary_cap" in overrides and hasattr(optimizer, "configure_showdown_rules"):
        optimizer.configure_showdown_rules(
            min_salary_cap=overrides["min_salary_cap"],
            max_salary_cap=overrides["max_salary_cap"],
            min_teams=overrides["min_teams"],
            game_script_correlation=overrides["game_script_correlation"],
            cpt_qb_pass_catchers=overrides["cpt_qb_pass_catchers"],
            max_k_dst_combined=overrides["max_k_dst_combined"],
        )

    optimizer.set_max_repeating_players(overrides.get("max_repeating_players"))
    optimizer.max_cumulative_ownership = overrides.get("max_cumulative_ownership")


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — format, preset, and editable overrides
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

    lo, hi = preset_cfg.lineup_count_range
    lineup_count = st.number_input(
        "Lineups",
        min_value=int(lo),
        max_value=int(hi),
        value=int(min(max(preset_cfg.default_lineup_count, lo), hi)),
        step=1,
        key=_preset_key(preset, "lineup_count"),
        help=f"Preset default: {preset_cfg.default_lineup_count} (range {lo}–{hi}).",
    )
    lineup_overridden = int(lineup_count) != int(preset_cfg.default_lineup_count)
    if lineup_overridden:
        st.caption(
            f"Override: lineups {int(lineup_count)} "
            f"(preset default {preset_cfg.default_lineup_count})."
        )

    with st.expander("Adjust strategy", expanded=True):
        strategy_overrides, strategy_changed = _collect_strategy_overrides(
            contest_format, preset, preset_cfg
        )

    st.markdown("### Salary settings")
    enable_tiered_salary = st.toggle(
        "Enable Tiered Salary Distribution",
        value=False,
        key="enable_tiered_salary",
        help=(
            "Spend most lineups near the cap, and let a set share "
            "finish under a lower salary ceiling."
        ),
    )
    leave_money_pct = st.slider(
        "Lineups Leaving Money on Table (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        disabled=not enable_tiered_salary,
        key="leave_money_pct",
    )
    discount_cap = st.number_input(
        "Discount Cap Ceiling",
        min_value=48_000,
        max_value=DK_SALARY_CAP,
        value=49_500,
        step=100,
        disabled=not enable_tiered_salary,
        key="discount_cap",
        help="Maximum salary for lineups that leave money on the table.",
    )
    if enable_tiered_salary:
        full_cap_n, discount_n = tiered_lineup_counts(
            int(lineup_count), float(leave_money_pct) / 100.0
        )
        full_label = "lineup" if full_cap_n == 1 else "lineups"
        discount_label = "lineup" if discount_n == 1 else "lineups"
        st.caption(
            f"{full_cap_n} {full_label} at ${STANDARD_TIER_FLOOR:,}–${DK_SALARY_CAP:,}; "
            f"{discount_n} {discount_label} at $48,000–${int(discount_cap):,}."
        )

    with st.expander("Advanced overrides", expanded=False):
        jitter_seed = st.number_input(
            "Jitter seed",
            min_value=0,
            max_value=10_000_000,
            value=42,
            help="Projection jitter uses this seed (reproducible). Not used when jitter is 0%.",
            key=_preset_key(preset, "jitter_seed"),
        )
        use_max_exposure = st.checkbox(
            "Limit max player exposure",
            value=False,
            key=_preset_key(preset, "use_max_exposure"),
        )
        max_exposure_pct = st.slider(
            "Max exposure %",
            1,
            100,
            40,
            disabled=not use_max_exposure,
            key=_preset_key(preset, "max_exposure_pct"),
        )
        use_min_exposure = st.checkbox(
            "Require min player exposure",
            value=False,
            key=_preset_key(preset, "use_min_exposure"),
        )
        min_exposure_pct = st.slider(
            "Min exposure %",
            0,
            100,
            0,
            disabled=not use_min_exposure,
            key=_preset_key(preset, "min_exposure_pct"),
        )

    all_changed = list(strategy_changed)
    if lineup_overridden:
        all_changed.insert(0, "Lineups")
    if use_max_exposure or use_min_exposure:
        all_changed.append("Exposure")
    if enable_tiered_salary:
        all_changed.append("Tiered salary")
    if all_changed:
        st.info("Overrides active: " + ", ".join(all_changed))
    else:
        st.caption("Using preset defaults (no overrides).")

actual_cfg = config_with_overrides(
    preset_cfg,
    lineup_count=int(lineup_count),
    overrides=strategy_overrides,
)
extra_setting_rows = _extra_setting_rows(
    jitter_seed=int(jitter_seed),
    use_max_exposure=bool(use_max_exposure),
    max_exposure_pct=int(max_exposure_pct),
    use_min_exposure=bool(use_min_exposure),
    min_exposure_pct=int(min_exposure_pct),
)
if enable_tiered_salary:
    full_cap_n, discount_n = tiered_lineup_counts(
        int(lineup_count), float(leave_money_pct) / 100.0
    )
    extra_setting_rows.append(
        (
            "Tiered salary",
            "Off",
            (
                f"{int(leave_money_pct)}% leave money "
                f"({discount_n} under ${int(discount_cap):,}; "
                f"{full_cap_n} at ${STANDARD_TIER_FLOOR:,}–${DK_SALARY_CAP:,})"
            ),
        )
    )
live_setting_rows = format_settings_comparison(
    preset_cfg,
    actual_cfg,
    extra_setting_rows,
)
generated_run = st.session_state.generated_run
settings_match_run = bool(
    generated_run
    and generated_run.get("preset") == preset.value
    and generated_run.get("rows") == live_setting_rows
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

if settings_match_run and st.session_state.lineups:
    settings_note = "Actual is what these lineups were generated with."
elif st.session_state.lineups and generated_run:
    settings_note = (
        "Actual is what Generate Lineups will use now. "
        "The lineups below were built with different settings."
    )
else:
    settings_note = "Actual is what Generate Lineups will use."

with st.container(border=True):
    st.markdown(f"#### Lineup settings — {preset_cfg.label}")
    st.caption(preset_cfg.description)
    st.caption(settings_note)
    _render_settings_comparison(live_setting_rows)

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
        _apply_strategy_overrides(optimizer, strategy_overrides)
        optimizer.set_exposure_limits(
            min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
            max_exposure_pct=float(max_exposure_pct) if use_max_exposure else None,
        )
        progress.progress(30, text=f"Generating {lineup_count} lineups…")
        if enable_tiered_salary:
            generated = optimize_tiered_salaries(
                optimizer,
                total_n=int(lineup_count),
                leave_money_ratio=float(leave_money_pct) / 100.0,
                standard_cap=DK_SALARY_CAP,
                discount_cap=int(discount_cap),
            )
        else:
            generated = optimizer.generate_lineups()
        progress.progress(100, text="Done")
        st.session_state.lineups = generated
        st.session_state.generated_run = {
            "preset": preset.value,
            "preset_label": preset_cfg.label,
            "rows": live_setting_rows,
        }
        override_note = (
            f" (overrides: {', '.join(all_changed)})" if all_changed else ""
        )
        st.session_state.flash_success = (
            f"Generated {len(generated)} lineup(s) with preset "
            f"{PRESET_LABELS[preset]}{override_note}."
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - show optimizer failures in UI
        st.session_state.lineups = []
        st.session_state.generated_run = None
        st.error(f"Optimization failed: {exc}")
    finally:
        progress.empty()

lineups = st.session_state.lineups
if lineups:
    used = st.session_state.generated_run
    if used and used.get("rows") != live_setting_rows:
        with st.container(border=True):
            st.markdown(f"#### Settings used for these lineups — {used['preset_label']}")
            st.caption(
                "Sidebar settings have changed since this generate. "
                "This table is what the lineups below were built with."
            )
            _render_settings_comparison(used["rows"])

    st.subheader(f"Generated lineups ({len(lineups)})")

    options = list(range(len(lineups)))

    def _spent(lu: pd.DataFrame) -> int:
        return int(lu.attrs.get("salary", int(lu["Salary"].sum())))

    def _remaining(lu: pd.DataFrame) -> int:
        if "salary_remaining" in lu.attrs:
            return int(lu.attrs["salary_remaining"])
        return DK_SALARY_CAP - _spent(lu)

    preview = pd.DataFrame(
        {
            "Lineup": [i + 1 for i in options],
            "Salary": [_spent(lu) for lu in lineups],
            "Salary Remaining ($)": [_remaining(lu) for lu in lineups],
            "Projection": [
                float(lu.attrs.get("projection", float(lu["Projection"].sum())))
                for lu in lineups
            ],
        }
    )
    st.dataframe(
        preview,
        width="stretch",
        hide_index=True,
        column_config={
            "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "Salary Remaining ($)": st.column_config.NumberColumn(
                "Salary Remaining ($)",
                format="$%d",
                help="Dollars left under the $50,000 DraftKings cap.",
            ),
            "Projection": st.column_config.NumberColumn("Projection", format="%.2f"),
        },
    )

    def _lineup_label(i: int) -> str:
        lu = lineups[i]
        projection = lu.attrs.get("projection", float(lu["Projection"].sum()))
        return (
            f"Lineup {i + 1} · ${_spent(lu):,} · "
            f"${_remaining(lu):,} left · {projection:.2f} proj"
        )

    selected = st.selectbox("View lineup", options=options, format_func=_lineup_label)
    lineup = lineups[selected]
    lineup_view = lineup[["Slot", "Name", "Team", "Salary", "Projection"]].copy()
    lineup_view["Salary Remaining ($)"] = _remaining(lineup)
    st.dataframe(
        lineup_view,
        width="stretch",
        hide_index=True,
        height=320,
        column_config={
            "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "Salary Remaining ($)": st.column_config.NumberColumn(
                "Salary Remaining ($)",
                format="$%d",
            ),
            "Projection": st.column_config.NumberColumn("Projection", format="%.2f"),
        },
    )
