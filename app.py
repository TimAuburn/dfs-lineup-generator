"""Streamlit UI for the DraftKings NFL Classic / Showdown Lineup Optimizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    KICKER_CAPTAIN_MAX_EXPOSURE,
    apply_projection_delta,
    frame_to_lines,
    lines_to_frame,
    low_scoring_showdown,
    projection_adjustment_labels,
)
from src.env_config import (
    ODDS_API_KEY_ENV,
    get_odds_api_key,
    key_status,
    load_env,
)
from src.exporter import lineups_to_dk_upload_csv, upload_columns_for
from src.optimizer import (
    DK_SALARY_CAP,
    SHOWDOWN_QB_MIN_PROJECTION,
    STANDARD_TIER_FLOOR,
    GameScriptBucket,
    classic_gpp_rule_check,
    default_starting_quarterbacks,
    drop_backup_players,
    even_lineup_counts,
    game_script_summary,
    games_from_pool,
    get_nfl_optimizer,
    optimize_game_scripts,
    optimize_tiered_salaries,
    player_exposure_table,
    quarterback_label,
    tiered_lineup_counts,
    unique_quarterbacks,
)
from src.parser import DKCSVValidationError, is_inactive_injury, parse_dk_csv
from src.sleeper import overlay_sleeper_projections, sleeper_status_caption
from src.vegas_cache import (
    coverage_warnings,
    fetch_vegas_lines,
    format_cache_status,
    normalize_vegas_memory,
)
from src.weather import (
    WIND_THRESHOLD_MPH,
    fetch_slate_weather,
    format_weather_cache_status,
    weather_adjustment_labels,
)
from src.presets import (
    ContestFormat,
    ContestPreset,
    PRESET_LABELS,
    PresetConfig,
    apply_preset,
    config_with_overrides,
    FLEX_POSITION_LABELS,
    flex_position_label,
    flex_positions_for_label,
    format_settings_comparison,
    get_preset_config,
    presets_for_format,
)

load_env()

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
        "vegas_editor_version": 0,
        "vegas_source_json": None,
        "vegas_frame": None,
        "weather_by_home": {},
        "weather_source_json": None,
        "sleeper_summary": None,
        "sleeper_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Drop an in-memory odds frame from before the implied-total correction.
    # The editor version bump changes the data_editor widget key.
    migrated = normalize_vegas_memory(
        {
            "vegas_schema": st.session_state.get("vegas_schema"),
            "vegas_frame": st.session_state.get("vegas_frame"),
            "vegas_source_json": st.session_state.get("vegas_source_json"),
            "vegas_editor_version": st.session_state.get("vegas_editor_version", 0),
        }
    )
    for key, value in migrated.items():
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
    if use_min_exposure:
        rows.append(("Min exposure", "Off", f"{int(min_exposure_pct)}%"))
    return rows


def _game_script_buckets(preset: ContestPreset, lineup_count: int) -> List[GameScriptBucket]:
    """Editable per-game lineup counts and salary caps for a loaded Classic pool."""
    pool = st.session_state.get("player_pool")
    games = games_from_pool(pool) if isinstance(pool, pd.DataFrame) else []
    if not games:
        st.caption("Upload a salary file to assign lineups to games.")
        return []
    signature = st.session_state.get("pool_signature") or "none"
    default = pd.DataFrame(
        {
            "Game": games,
            "Lineups": even_lineup_counts(len(games), int(lineup_count)),
            "Salary cap": [DK_SALARY_CAP] * len(games),
        }
    )
    edited = st.data_editor(
        default,
        hide_index=True,
        width="stretch",
        disabled=["Game"],
        num_rows="fixed",
        column_config={
            "Game": st.column_config.TextColumn("Game"),
            "Lineups": st.column_config.NumberColumn(
                "Lineups",
                min_value=0,
                max_value=150,
                step=1,
                format="%d",
            ),
            "Salary cap": st.column_config.NumberColumn(
                "Salary cap",
                min_value=1,
                max_value=DK_SALARY_CAP,
                step=100,
                format="$%d",
                help=(
                    "Maximum salary for lineups in this game. "
                    "A lower cap is the discount tier."
                ),
            ),
        },
        key=f"game_script_alloc_{preset.value}_{signature}",
    )
    counts = pd.to_numeric(edited["Lineups"], errors="coerce")
    caps = pd.to_numeric(edited["Salary cap"], errors="coerce")
    if counts.isna().any() or caps.isna().any() or (caps.fillna(0) <= 0).any():
        st.error("Enter a lineup count and a salary cap for every game.")
        return [
            GameScriptBucket(str(game), 0, DK_SALARY_CAP) for game in games
        ]
    assigned = int(counts.sum())
    if assigned != int(lineup_count):
        st.error(
            f"Lineup counts add up to {assigned}. "
            f"They need to add up to {int(lineup_count)}."
        )
    else:
        st.caption("Lineups share at most 4 players across these games.")
    return [
        GameScriptBucket(str(game), int(count), int(cap))
        for game, count, cap in zip(edited["Game"], counts, caps)
    ]


def _collect_strategy_overrides(
    contest_format: ContestFormat,
    preset: ContestPreset,
    preset_cfg: PresetConfig,
    lineup_count: int,
) -> Tuple[Dict[str, Any], List[str], Optional[List[GameScriptBucket]]]:
    """
    Render editable strategy controls seeded from the active preset.

    Returns override values to apply after ``apply_preset``, labels for fields
    that differ from the preset, and game-script buckets when that allocation
    is on. The bucket list is None when the toggle is off.
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
        bring_back_options = ["WR", "TE", "SKILL"]
        bring_back_position = st.selectbox(
            "Bring-back position",
            options=bring_back_options,
            index=(
                bring_back_options.index(preset_cfg.bring_back_position)
                if preset_cfg.bring_back_position in bring_back_options
                else 0
            ),
            disabled=not (enable_qb_stack and enable_bring_back),
            key=_preset_key(preset, "bring_back_position"),
            help="SKILL is any opposing WR, TE, or RB from the quarterback's game.",
        )
        exclude_vs_dst = st.checkbox(
            "Exclude offense vs DST",
            value=preset_cfg.exclude_vs_dst,
            key=_preset_key(preset, "exclude_vs_dst"),
        )
        flex_labels = [label for label, _positions in FLEX_POSITION_LABELS]
        preset_flex = flex_position_label(preset_cfg.flex_positions)
        flex_choice = st.radio(
            "Flex position",
            options=flex_labels,
            index=flex_labels.index(preset_flex) if preset_flex in flex_labels else 0,
            key=_preset_key(preset, "flex_position"),
            help=(
                "WR or RB fills the flex with a running back or wide receiver. "
                "The tight end stays in the TE spot."
            ),
        )

        overrides.update(
            enable_qb_stack=bool(enable_qb_stack),
            stack_pass_catchers=int(stack_pass_catchers),
            enable_bring_back=bool(enable_bring_back),
            bring_back_position=str(bring_back_position),
            exclude_vs_dst=bool(exclude_vs_dst),
        )
        if flex_choice is not None:
            overrides["flex_positions"] = flex_positions_for_label(str(flex_choice))
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
        if flex_choice is not None and str(flex_choice) != flex_position_label(
            preset_cfg.flex_positions
        ):
            changed.append("Flex")
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
            key=_preset_key(preset, f"salary_window_{min_default}_{max_default}"),
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
        allow_kicker_captain = st.checkbox(
            "Allow kicker captain",
            value=bool(preset_cfg.allow_kicker_captain),
            key=_preset_key(preset, "allow_kicker_captain"),
            help="Off keeps CPT to QB, RB, WR, and TE. A game total of 38.5 or lower turns this on.",
        )
        limit_kickers = st.checkbox(
            "Allow at most 1 Kicker per lineup",
            value=preset_cfg.max_kickers == 1,
            key=_preset_key(preset, "limit_kickers"),
            help="Checked lineups include at most one kicker, so both teams' kickers cannot be paired.",
        )

        overrides.update(
            min_salary_cap=int(salary_lo) if use_salary_window else None,
            max_salary_cap=int(salary_hi) if use_salary_window else None,
            min_teams=2 if require_both_teams else None,
            game_script_correlation=bool(game_script),
            cpt_qb_pass_catchers=int(cpt_qb_pass_catchers),
            max_k_dst_combined=int(max_k_dst) if limit_k_dst else None,
            max_kickers=1 if limit_kickers else None,
            allow_kicker_captain=bool(allow_kicker_captain),
            captain_positions=preset_cfg.captain_positions,
            captain_min_salary=preset_cfg.captain_min_salary,
            flex_min_salary=preset_cfg.flex_min_salary,
            max_dst=preset_cfg.max_dst,
            ban_dst=preset_cfg.ban_dst,
            max_below_salary_players=preset_cfg.max_below_salary_players,
            below_salary_threshold=preset_cfg.below_salary_threshold,
            max_rbs_per_team=preset_cfg.max_rbs_per_team,
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
        if bool(allow_kicker_captain) != bool(preset_cfg.allow_kicker_captain):
            changed.append("Kicker captain")
        if bool(limit_kickers) != (preset_cfg.max_kickers == 1):
            changed.append("Max kickers")

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

    game_script_buckets: Optional[List[GameScriptBucket]] = None
    if contest_format is ContestFormat.CLASSIC:
        enable_game_script = st.checkbox(
            "Enable Game-Script Portfolio Allocation",
            value=False,
            key=_preset_key(preset, "game_script_allocation"),
            help=(
                "Split this batch across games. Each game uses that game's "
                "quarterback, one same-team pass catcher, and an opposing "
                "skill player. Flex is WR or RB, and no skill player faces "
                "the defense. Lineups share at most 4 players."
            ),
        )
        if enable_game_script:
            game_script_buckets = _game_script_buckets(preset, int(lineup_count))

    return overrides, changed, game_script_buckets


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
    if "flex_positions" in overrides and hasattr(optimizer, "flex_positions"):
        optimizer.flex_positions = overrides["flex_positions"]

    if "min_salary_cap" in overrides and hasattr(optimizer, "configure_showdown_rules"):
        optimizer.configure_showdown_rules(
            min_salary_cap=overrides["min_salary_cap"],
            max_salary_cap=overrides["max_salary_cap"],
            min_teams=overrides["min_teams"],
            game_script_correlation=overrides["game_script_correlation"],
            cpt_qb_pass_catchers=overrides["cpt_qb_pass_catchers"],
            max_k_dst_combined=overrides["max_k_dst_combined"],
            max_kickers=overrides.get("max_kickers", 1),
            allow_kicker_captain=bool(overrides.get("allow_kicker_captain", False)),
            captain_positions=overrides.get("captain_positions"),
            captain_min_salary=overrides.get("captain_min_salary"),
            flex_min_salary=overrides.get("flex_min_salary"),
            max_dst=overrides.get("max_dst"),
            ban_dst=bool(overrides.get("ban_dst", False)),
            max_below_salary_players=overrides.get("max_below_salary_players"),
            below_salary_threshold=overrides.get("below_salary_threshold"),
            max_rbs_per_team=overrides.get("max_rbs_per_team"),
        )

    optimizer.set_max_repeating_players(overrides.get("max_repeating_players"))
    optimizer.max_cumulative_ownership = overrides.get("max_cumulative_ownership")


_init_state()

# None until a Showdown pool is loaded. An empty list excludes every quarterback.
confirmed_qb_keys: List[Tuple[str, str]] | None = None

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

    st.markdown("### Injuries")
    injury_choice = st.radio(
        "Players who will not play",
        options=["exclude", "include"],
        format_func=lambda v: (
            "Exclude OUT, IR, and Doubtful"
            if v == "exclude"
            else "Include everyone"
        ),
        index=0,
        key="injury_choice",
        help="Questionable and probable players stay in the pool either way.",
    )
    exclude_inactive = injury_choice == "exclude"
    loaded_pool = st.session_state.player_pool
    if loaded_pool is not None and "Injury" in loaded_pool.columns:
        inactive_names = (
            loaded_pool.loc[loaded_pool["Injury"].map(is_inactive_injury), "Name"]
            .astype(str)
            .drop_duplicates()
        )
        if exclude_inactive and len(inactive_names):
            preview = ", ".join(inactive_names.head(4))
            extra = f" +{len(inactive_names) - 4}" if len(inactive_names) > 4 else ""
            st.caption(f"Leaving out {len(inactive_names)}: {preview}{extra}")
        elif exclude_inactive:
            st.caption("No OUT, IR, or Doubtful players in this pool.")

    if (
        contest_format is ContestFormat.SHOWDOWN
        and loaded_pool is not None
        and "RosterSlot" in loaded_pool.columns
    ):
        qb_table = unique_quarterbacks(loaded_pool)
        if not qb_table.empty:
            st.markdown("### Confirmed starting QBs")
            labels = [
                quarterback_label(row.Name, row.TeamAbbrev, row.Projection)
                for row in qb_table.itertuples(index=False)
            ]
            label_to_key = {
                quarterback_label(row.Name, row.TeamAbbrev, row.Projection): (
                    str(row.Name),
                    str(row.TeamAbbrev),
                )
                for row in qb_table.itertuples(index=False)
            }
            starter_keys = set(default_starting_quarterbacks(loaded_pool))
            default_labels = [
                label for label in labels if label_to_key[label] in starter_keys
            ]
            slate_key = hashlib.sha1("\n".join(labels).encode()).hexdigest()[:12]
            selected_labels = st.multiselect(
                "Confirmed starting QBs",
                options=labels,
                default=default_labels,
                key=f"starting_qbs_{slate_key}",
                label_visibility="collapsed",
                help=(
                    "Starts with the highest-projected quarterback on each team. "
                    "Any quarterback you leave out is excluded. "
                    "Quarterbacks under 12.0 points are removed automatically."
                ),
            )
            confirmed_qb_keys = [label_to_key[label] for label in selected_labels]
            left_out = [label for label in labels if label not in set(selected_labels)]
            under_floor = (
                qb_table.loc[
                    qb_table["Projection"] < SHOWDOWN_QB_MIN_PROJECTION, "Name"
                ]
                .astype(str)
                .tolist()
            )
            if left_out:
                st.caption("Leaving out " + ", ".join(left_out) + ".")
            if under_floor:
                st.caption(
                    "Under 12.0 points, removed: "
                    + ", ".join(dict.fromkeys(under_floor))
                    + "."
                )

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
        strategy_overrides, strategy_changed, game_script_buckets = (
            _collect_strategy_overrides(
                contest_format, preset, preset_cfg, lineup_count=int(lineup_count)
            )
        )

    st.markdown("### Salary settings")
    enable_tiered_salary = st.toggle(
        "Enable Tiered Salary Distribution",
        value=bool(preset_cfg.tiered_salary),
        key=_preset_key(preset, "enable_tiered_salary"),
        help=(
            "Spend most lineups near the cap, and let a set share "
            "finish under a lower salary ceiling."
        ),
    )
    leave_money_pct = st.slider(
        "Lineups Leaving Money on Table (%)",
        min_value=0,
        max_value=100,
        value=int(round(float(preset_cfg.tiered_leave_ratio) * 100)),
        step=5,
        disabled=not enable_tiered_salary,
        key=_preset_key(preset, "leave_money_pct"),
    )
    discount_floor = st.number_input(
        "Discount floor",
        min_value=40_000,
        max_value=DK_SALARY_CAP,
        value=int(preset_cfg.tiered_discount_floor),
        step=100,
        disabled=not enable_tiered_salary,
        key=_preset_key(preset, "discount_floor"),
        help="Minimum salary for lineups that leave money on the table.",
    )
    discount_cap = st.number_input(
        "Discount Cap Ceiling",
        min_value=40_000,
        max_value=DK_SALARY_CAP,
        value=int(preset_cfg.tiered_discount_cap),
        step=100,
        disabled=not enable_tiered_salary,
        key=_preset_key(preset, "discount_cap"),
        help="Maximum salary for lineups that leave money on the table.",
    )
    if game_script_buckets is not None and enable_tiered_salary:
        st.caption(
            "Game-script allocation uses each game's salary cap. "
            "Tiered salary is left off for this run."
        )
    if enable_tiered_salary and game_script_buckets is None:
        full_cap_n, discount_n = tiered_lineup_counts(
            int(lineup_count), float(leave_money_pct) / 100.0
        )
        full_label = "lineup" if full_cap_n == 1 else "lineups"
        discount_label = "lineup" if discount_n == 1 else "lineups"
        st.caption(
            f"{full_cap_n} {full_label} at ${STANDARD_TIER_FLOOR:,}–${DK_SALARY_CAP:,}; "
            f"{discount_n} {discount_label} at ${int(discount_floor):,}–${int(discount_cap):,}."
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
            value=preset_cfg.max_player_exposure is not None,
            key=_preset_key(preset, "use_max_exposure"),
        )
        max_exposure_pct = st.slider(
            "Max exposure %",
            1,
            100,
            int(round((preset_cfg.max_player_exposure or 0.40) * 100)),
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

    st.markdown("### API keys")
    st.caption(key_status(ODDS_API_KEY_ENV))
    st.caption("The odds key loads from a local `.env` (see `.env.example`). Weather uses the National Weather Service and does not need a key.")

    st.markdown("### Vegas odds")
    odds_api_key = get_odds_api_key()
    force_vegas_refresh = st.button("Force Refresh Vegas Odds", key="force_vegas_refresh")
    fetched_lines = fetch_vegas_lines(
        api_key=odds_api_key,
        force_refresh=bool(force_vegas_refresh),
    )
    vegas_warnings = coverage_warnings()
    signature = json.dumps(fetched_lines, sort_keys=True, default=str)
    if (
        force_vegas_refresh
        or st.session_state.vegas_frame is None
        or signature != st.session_state.vegas_source_json
    ):
        st.session_state.vegas_source_json = signature
        st.session_state.vegas_frame = lines_to_frame(fetched_lines)
        if force_vegas_refresh:
            st.session_state.vegas_editor_version = int(st.session_state.vegas_editor_version) + 1
    st.caption(format_cache_status())
    for warning in vegas_warnings:
        st.warning(warning)
    edited_vegas = st.data_editor(
        st.session_state.vegas_frame,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        height=240,
        key=f"vegas_editor_{st.session_state.vegas_editor_version}",
        column_config={
            "Game": st.column_config.TextColumn("Game", width="small"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Spread": st.column_config.NumberColumn("Spread", format="%.1f", width="small"),
            "Over/Under": st.column_config.NumberColumn(
                "Over/Under", format="%.1f", width="small"
            ),
            "ITT": st.column_config.NumberColumn(
                "ITT",
                format="%.1f",
                width="small",
                help=(
                    "(Over/Under − signed spread) / 2. "
                    "A negative spread is the favorite, so that team is higher."
                ),
            ),
        },
    )

    st.markdown("### Weather")
    force_weather_refresh = st.button("Force Refresh Weather", key="force_weather_refresh")
    weather_pool = st.session_state.player_pool
    fetched_weather = fetch_slate_weather(
        weather_pool,
        force_refresh=bool(force_weather_refresh),
    )
    weather_sig = json.dumps(fetched_weather, sort_keys=True, default=str)
    if (
        force_weather_refresh
        or st.session_state.weather_by_home is None
        or weather_sig != st.session_state.weather_source_json
    ):
        st.session_state.weather_source_json = weather_sig
        st.session_state.weather_by_home = fetched_weather
    st.caption(format_weather_cache_status())
    st.caption(
        f"Outdoor games with sustained wind over {WIND_THRESHOLD_MPH:.0f} mph "
        "downgrade QB/WR/TE (−10%) and K (−15%) before optimize; RB/DST unchanged. "
        "Indoor / enclosed home stadiums skip weather."
    )

    strategy_overrides["tiered_salary"] = bool(enable_tiered_salary)
    strategy_overrides["tiered_leave_ratio"] = float(leave_money_pct) / 100.0
    strategy_overrides["tiered_discount_floor"] = int(discount_floor)
    strategy_overrides["tiered_discount_cap"] = int(discount_cap)
    strategy_overrides["max_player_exposure"] = (
        float(max_exposure_pct) / 100.0 if use_max_exposure else None
    )

    all_changed = list(strategy_changed)
    if lineup_overridden:
        all_changed.insert(0, "Lineups")
    preset_exposure_on = preset_cfg.max_player_exposure is not None
    preset_exposure_pct = int(round((preset_cfg.max_player_exposure or 0) * 100))
    exposure_changed = bool(use_max_exposure) != preset_exposure_on or (
        use_max_exposure and int(max_exposure_pct) != preset_exposure_pct
    )
    if exposure_changed or use_min_exposure:
        all_changed.append("Exposure")
    tiered_changed = bool(enable_tiered_salary) != bool(preset_cfg.tiered_salary)
    if enable_tiered_salary and not tiered_changed:
        tiered_changed = (
            int(leave_money_pct) != int(round(float(preset_cfg.tiered_leave_ratio) * 100))
            or int(discount_floor) != int(preset_cfg.tiered_discount_floor)
            or int(discount_cap) != int(preset_cfg.tiered_discount_cap)
        )
    if tiered_changed:
        all_changed.append("Tiered salary")
    if not exclude_inactive:
        all_changed.append("Injuries")
    if all_changed:
        st.info("Overrides active: " + ", ".join(all_changed))
    else:
        st.caption("Using preset defaults (no overrides).")

vegas_lines = frame_to_lines(edited_vegas)
slate_teams: set[str] = set()
if st.session_state.player_pool is not None and "TeamAbbrev" in st.session_state.player_pool.columns:
    slate_teams = set(
        st.session_state.player_pool["TeamAbbrev"].astype(str).str.strip().str.upper()
    )
low_scoring_game = contest_format is ContestFormat.SHOWDOWN and low_scoring_showdown(
    vegas_lines, slate_teams
)
if low_scoring_game:
    strategy_overrides["allow_kicker_captain"] = True

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
extra_setting_rows.append(
    (
        "Inactive players",
        "Excluded",
        "Excluded" if exclude_inactive else "Included",
    )
)
itt_labels = projection_adjustment_labels(st.session_state.player_pool, vegas_lines)
if itt_labels:
    extra_setting_rows.append(("Vegas ITT", "No adjustment", ", ".join(itt_labels)))
weather_by_home = st.session_state.weather_by_home or {}
wind_labels = weather_adjustment_labels(st.session_state.player_pool, weather_by_home)
if wind_labels:
    extra_setting_rows.append(("Wind weather", "No adjustment", ", ".join(wind_labels)))
if low_scoring_game:
    extra_setting_rows.append(
        (
            "Kicker exposure",
            "No cap",
            "10% (game total ≤ 38.5)",
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
for warning in vegas_warnings:
    st.warning(warning)
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
        width="stretch",
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
        width="stretch",
        disabled=not bool(lineups),
    )

if uploaded is not None:
    try:
        parsed_pool = parse_dk_csv(uploaded, contest_format=contest_format)
        force_sleeper = bool(st.session_state.pop("sleeper_force", False))
        parsed_pool, sleeper_summary = overlay_sleeper_projections(
            parsed_pool,
            force_refresh=force_sleeper,
        )
        if force_sleeper:
            st.session_state.sleeper_version = int(st.session_state.sleeper_version) + 1
        signature = hashlib.sha1(uploaded.getvalue()).hexdigest()
        pool_changed = st.session_state.get("pool_signature") != signature
        st.session_state.player_pool = parsed_pool
        st.session_state.sleeper_summary = sleeper_summary
        st.session_state.pool_signature = signature
        st.session_state.error = None
        # Sidebar reads the pool before this upload runs, so refresh once.
        if pool_changed:
            st.rerun()
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
    if low_scoring_game:
        st.caption(
            "Game total is 38.5 or lower, so a kicker can be captain with a 10% exposure cap."
        )
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
inactive_in_pool = (
    int(pool["Injury"].map(is_inactive_injury).sum())
    if "Injury" in pool.columns
    else 0
)
st.subheader("Player pool")
sleeper_refresh = st.button("Refresh Sleeper projections", key="refresh_sleeper")
if sleeper_refresh:
    st.session_state.sleeper_force = True
    st.rerun()
st.caption(sleeper_status_caption(st.session_state.sleeper_summary))
if not is_showdown:
    considered = pool
    if exclude_inactive and "Injury" in pool.columns:
        considered = pool.loc[~pool["Injury"].map(is_inactive_injury)].copy()
    starters = drop_backup_players(considered)
    starter_ids = set(starters["ID"].astype(str)) if "ID" in starters.columns else set()
    backup_names = (
        considered.loc[~considered["ID"].astype(str).isin(starter_ids), "Name"]
        .astype(str)
        .drop_duplicates()
        .tolist()
        if "ID" in considered.columns
        else []
    )
    if backup_names:
        preview = ", ".join(backup_names[:4])
        extra = f" +{len(backup_names) - 4}" if len(backup_names) > 4 else ""
        st.caption(f"Leaving out {len(backup_names)} backups: {preview}{extra}")
    elif "DepthOrder" in pool.columns and pd.to_numeric(pool["DepthOrder"], errors="coerce").notna().any():
        st.caption("No backup players in this pool.")
injury_note = (
    f" · {inactive_in_pool} row(s) marked OUT/IR/Doubtful"
    if inactive_in_pool
    else ""
)
lock_note = ""
if preset is ContestPreset.CLASSIC_CASH:
    lock_note = " Lock a skill player under $3,800 to force that punt in."
elif preset is ContestPreset.SHOWDOWN_CASH:
    lock_note = " Lock a FLEX player under $3,500 to force that punt in."
elif is_showdown:
    lock_note = " Lock a FLEX player under $1,200 to force that punt in."
st.caption(
    f"{len(pool)} rows loaded · {roster_n}-player roster · salary cap ${DK_SALARY_CAP:,}"
    f"{injury_note}.{lock_note} Check Exclude to drop a player."
)
if "Sleeper" not in pool.columns:
    pool = pool.copy()
    pool["Sleeper"] = pd.Series(pd.NA, index=pool.index, dtype="Float64")
editable_cols = [
    "Name",
    "Position",
    "TeamAbbrev",
    "Salary",
    "AvgPointsPerGame",
    "Sleeper",
    "Projection",
    "Delta",
]
has_injury_status = (
    "Injury" in pool.columns and pool["Injury"].astype(str).str.strip().ne("").any()
)
display_extra = [
    c
    for c in ["ID", "Name + ID", "Roster Position", "RosterSlot", "Game Info", "Ownership", "Injury"]
    if c in pool.columns and (c != "Injury" or has_injury_status)
]
if "DepthOrder" in pool.columns and pd.to_numeric(pool["DepthOrder"], errors="coerce").notna().any():
    display_extra.append("DepthOrder")
if "DepthPosition" in pool.columns and pool["DepthPosition"].astype(str).str.strip().ne("").any():
    display_extra.append("DepthPosition")
pool = apply_projection_delta(pool, vegas_lines, weather_by_home)
editor_frame = pool[editable_cols + display_extra].copy()
editor_frame.insert(0, "Exclude", False)
editor_frame.insert(0, "Lock", False)
edited = st.data_editor(
    editor_frame,
    width="stretch",
    hide_index=True,
    height=320,
    disabled=[
        c for c in editor_frame.columns if c not in {"Projection", "Lock", "Exclude"}
    ],
    column_config={
        "Lock": st.column_config.CheckboxColumn(
            "Lock",
            help="Force this player into the lineup, including a cash punt under the salary floor.",
        ),
        "Exclude": st.column_config.CheckboxColumn(
            "Exclude",
            help="Leave this player out. On Showdown, both the captain and flex rows are dropped.",
        ),
        "AvgPointsPerGame": st.column_config.NumberColumn(
            "DraftKings",
            format="%.2f",
            help="Average points per game from the DraftKings salary file.",
        ),
        "Sleeper": st.column_config.NumberColumn(
            "Sleeper",
            format="%.2f",
            help="Original Sleeper weekly PPR projection. This column does not include Delta.",
        ),
        "Projection": st.column_config.NumberColumn(
            "Projection",
            format="%.2f",
            help=(
                "Sleeper, or the DraftKings average when Sleeper is blank, plus Delta. "
                "This is the number the optimizer uses. Edit to override."
            ),
        ),
        "Delta": st.column_config.NumberColumn(
            "Delta",
            format="%+.2f",
            help=(
                "Points already added to or removed from Projection. "
                "Vegas implied team total moves RB, WR, and TE by +5% at 26 or above, "
                "or -5% at 18 or below. Outdoor wind can cut the number further. "
                "Sleeper stays the original weekly value."
            ),
        ),
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "DepthOrder": st.column_config.NumberColumn(
            "Depth",
            format="%d",
            help="Sleeper depth chart order. 1 is the top player at that spot.",
        ),
        "DepthPosition": st.column_config.TextColumn(
            "Depth spot",
            help="Sleeper depth chart spot, such as QB, RB, LWR, RWR, or SWR.",
        ),
    },
    key=(
        f"player_editor_{contest_format.value}_"
        f"{(st.session_state.sleeper_summary or {}).get('season')}-"
        f"{(st.session_state.sleeper_summary or {}).get('week')}-"
        f"{(st.session_state.sleeper_summary or {}).get('matched')}-"
        f"{int(st.session_state.sleeper_version)}-projection-plus-delta"
    ),
)

working_pool = pool.copy().reset_index(drop=True)
edited = edited.reset_index(drop=True)
working_pool["Projection"] = pd.to_numeric(edited["Projection"], errors="coerce").fillna(
    working_pool["AvgPointsPerGame"]
).to_numpy()
excluded_keys: set[Tuple[str, str]] = set()
if "Exclude" in edited.columns and "Name" in edited.columns:
    excluded_mask = edited["Exclude"].fillna(False).astype(bool)
    if excluded_mask.any() and "TeamAbbrev" in edited.columns:
        excluded_keys = {
            (str(name).strip(), str(team).strip().upper())
            for name, team in zip(
                edited.loc[excluded_mask, "Name"],
                edited.loc[excluded_mask, "TeamAbbrev"],
            )
        }
        person = list(
            zip(
                working_pool["Name"].astype(str).str.strip(),
                working_pool["TeamAbbrev"].astype(str).str.strip().str.upper(),
            )
        )
        working_pool = working_pool.loc[
            [pair not in excluded_keys for pair in person]
        ].copy()

if generate:
    progress = st.progress(0, text="Building optimizer…")
    try:
        if working_pool.empty:
            raise RuntimeError("Every player is excluded.")
        solve_lines = frame_to_lines(edited_vegas)
        solve_teams = set(working_pool["TeamAbbrev"].astype(str).str.strip().str.upper())
        solve_low_total = is_showdown and low_scoring_showdown(solve_lines, solve_teams)
        solve_weather = fetch_slate_weather(
            working_pool,
            force_refresh=False,
        )
        st.session_state.weather_by_home = solve_weather
        # Projection in the player pool already includes Delta, so do not apply it again.
        solve_pool = working_pool
        optimizer = get_nfl_optimizer(contest_format)
        optimizer.set_exclude_inactive(exclude_inactive)
        optimizer.load_from_dataframe(solve_pool)
        locked_ids = []
        if "Lock" in edited.columns and "ID" in edited.columns:
            locked_mask = edited["Lock"].fillna(False).astype(bool)
            if excluded_keys and "Name" in edited.columns and "TeamAbbrev" in edited.columns:
                keep_lock = [
                    (str(name).strip(), str(team).strip().upper()) not in excluded_keys
                    for name, team in zip(edited["Name"], edited["TeamAbbrev"])
                ]
                locked_mask = locked_mask & pd.Series(keep_lock, index=edited.index)
            locked_ids = edited.loc[locked_mask, "ID"].astype(str).tolist()
        optimizer.set_locked_players(locked_ids)
        apply_preset(
            optimizer,
            preset,
            lineup_count=int(lineup_count),
            jitter_seed=int(jitter_seed),
        )
        _apply_strategy_overrides(optimizer, strategy_overrides)
        if confirmed_qb_keys is not None and hasattr(optimizer, "set_confirmed_quarterbacks"):
            optimizer.set_confirmed_quarterbacks(confirmed_qb_keys)
        if solve_low_total and preset is not ContestPreset.SHOWDOWN_CASH:
            optimizer.allow_kicker_captain = True
            optimizer.kicker_max_exposure = KICKER_CAPTAIN_MAX_EXPOSURE
        optimizer.set_exposure_limits(
            min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
            max_exposure_pct=float(max_exposure_pct) if use_max_exposure else None,
        )
        progress.progress(30, text=f"Generating {lineup_count} lineups…")
        if game_script_buckets is not None:
            if not game_script_buckets:
                raise RuntimeError(
                    "Upload a Classic salary file before assigning lineups to games."
                )
            assigned = sum(bucket.lineups for bucket in game_script_buckets)
            if assigned != int(lineup_count):
                raise RuntimeError(
                    f"Game-script lineup counts add up to {assigned}. "
                    f"Set them to {int(lineup_count)}."
                )
            generated = optimize_game_scripts(optimizer, game_script_buckets)
        elif enable_tiered_salary:
            generated = optimize_tiered_salaries(
                optimizer,
                total_n=int(lineup_count),
                leave_money_ratio=float(leave_money_pct) / 100.0,
                standard_cap=DK_SALARY_CAP,
                discount_cap=int(discount_cap),
                discount_floor=int(discount_floor),
            )
        else:
            generated = optimizer.generate_lineups()
        progress.progress(100, text="Done")
        rule_check = None
        if preset is ContestPreset.CLASSIC_LARGE_GPP and game_script_buckets is None:
            rule_check = {
                "overlap_cap": optimizer.last_overlap_cap,
                "game_count": optimizer.last_game_count,
                "stack_exposure": optimizer.last_stack_exposure,
                "salary_min": int(discount_floor) if enable_tiered_salary else None,
                "salary_max": DK_SALARY_CAP,
                "flex_positions": strategy_overrides.get("flex_positions"),
            }
        st.session_state.lineups = generated
        st.session_state.generated_run = {
            "preset": preset.value,
            "preset_label": preset_cfg.label,
            "rows": live_setting_rows,
            "rule_check": rule_check,
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
    if any(lineup.attrs.get("game_script") for lineup in lineups):
        st.markdown("#### Lineups per game")
        st.dataframe(
            game_script_summary(lineups),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Player exposure")
        st.dataframe(
            player_exposure_table(lineups),
            width="stretch",
            hide_index=True,
            column_config={
                "Lineups": st.column_config.NumberColumn("Lineups", format="%d"),
                "Exposure %": st.column_config.NumberColumn(
                    "Exposure %", format="%.1f"
                ),
            },
        )

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
    rule_meta = (used or {}).get("rule_check") if used else None
    if rule_meta:
        games = rule_meta.get("game_count")
        overlap_cap = rule_meta.get("overlap_cap")
        stack_exposure = rule_meta.get("stack_exposure")
        notes = []
        if games:
            notes.append(f"{int(games)}-game slate")
        if overlap_cap is not None:
            notes.append(f"overlap cap {int(overlap_cap)}")
        if stack_exposure is not None:
            notes.append(f"team stack exposure {int(round(float(stack_exposure) * 100))}%")
        st.markdown("#### Large-field rule check")
        if notes:
            st.caption(". ".join(notes) + ".")
        rule_table = classic_gpp_rule_check(
            lineups,
            overlap_cap=overlap_cap,
            salary_min=rule_meta.get("salary_min"),
            salary_max=int(rule_meta.get("salary_max") or DK_SALARY_CAP),
            flex_positions=rule_meta.get("flex_positions"),
        )
        st.dataframe(
            rule_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Salary remaining": st.column_config.NumberColumn(
                    "Salary remaining",
                    format="$%d",
                ),
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
    if "ID" in lineup.columns and "ID" in pool.columns and "Delta" in pool.columns:
        delta_by_id = dict(zip(pool["ID"].astype(str), pool["Delta"]))
        lineup_view["Delta"] = lineup["ID"].astype(str).map(delta_by_id)
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
            "Delta": st.column_config.NumberColumn(
                "Delta",
                format="%+.2f",
                help="Points added or removed from the player-pool projection by Vegas and wind.",
            ),
        },
    )
