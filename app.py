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

from src.contest_parser import (
    contest_id_from_filename,
    fetch_dk_contest_metadata,
    open_contest_standings,
    parse_dk_contest_csv,
)
from src.database import (
    get_chalk_and_cutline_summary,
    get_recent_slates,
    save_contest,
    save_salary_pool_if_new,
    search_player_performance,
)
from src.data_loader import (
    KICKER_CAPTAIN_MAX_EXPOSURE,
    lines_to_frame,
    low_scoring_showdown,
    projection_adjustment_labels,
    sync_vegas_editor,
    visible_odds_warnings,
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
    exposure_report_csv,
    player_exposure_table,
    quarterback_label,
    showdown_lineup_grid,
    tiered_lineup_counts,
    unique_quarterbacks,
)
from src.dk_scoring import SCORING_VERSION
from src.parser import DKCSVValidationError, is_inactive_injury, parse_dk_csv
from src.player_pool import (
    apply_pool_edits,
    carry_choices,
    clear_projection_overrides,
    collapse_for_editor,
    filter_editor_frame,
    game_choices,
    generation_pool,
    locked_player_ids,
    position_choices,
    person_key,
    projections_for_solve,
    set_captain,
    set_included,
    stamp_projection_baseline,
)
from src.sleeper import (
    SCORING_COLUMNS,
    drop_stale_scoring_session,
    session_contract_current,
    overlay_sleeper_projections,
    scoring_approximation_lines,
    scoring_fallback_lines,
    scoring_metadata_current,
    sleeper_status_caption,
    unmatched_players_warning,
)
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
    if drop_stale_scoring_session(st.session_state):
        st.session_state.sleeper_version = int(st.session_state.get("sleeper_version") or 0) + 1


def _player_editor_key(
    contest_format: ContestFormat,
    pool_signature: str,
    sleeper_version: int,
    automatic_key: str,
    name_query: str,
    selected_positions: list[str] | None,
    selected_games: list[str] | None,
    generation: int,
) -> str:
    """One identity for the player table.

    Filters and odds are part of it. Reset bumps ``generation`` so a retained
    projection edit is not replayed. Lock and include already live on the pool.
    """
    return (
        f"player_editor_{contest_format.value}_"
        f"{pool_signature}_"
        f"{int(sleeper_version)}_"
        f"{automatic_key}_"
        f"{name_query}_"
        f"{','.join(selected_positions or [])}_"
        f"{','.join(selected_games or [])}_"
        f"{int(generation)}"
    )


def _automatic_projection_key(frame: pd.DataFrame) -> str:
    """Editor identity for the current automatic projections.

    Odds or wind changes remount the grid. An explicit override lives on the
    pool, so the remount shows that number again instead of the old delta.
    """
    if frame is None or frame.empty or "Projection" not in frame.columns:
        return "none"
    names = frame["Name"] if "Name" in frame.columns else [""] * len(frame)
    teams = frame["TeamAbbrev"] if "TeamAbbrev" in frame.columns else [""] * len(frame)
    values = pd.to_numeric(frame["Projection"], errors="coerce")
    parts = []
    for name, team, value in zip(names, teams, values):
        text = "na" if pd.isna(value) else f"{float(value):.2f}"
        parts.append(f"{name}|{team}|{text}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


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
    at_most_one_qb: bool = False,
    team_split: Tuple[int, int] | None = None,
    team_split_team: str | None = None,
) -> List[Tuple[str, str, str]]:
    """UI-only settings that are not part of a contest preset."""
    rows: List[Tuple[str, str, str]] = []
    if int(jitter_seed) != 42:
        rows.append(("Jitter seed", "42", str(int(jitter_seed))))
    if use_min_exposure:
        rows.append(("Min exposure", "Off", f"{int(min_exposure_pct)}%"))
    if at_most_one_qb:
        rows.append(("Quarterbacks", "No limit", "At most 1"))
    if team_split is not None:
        label = f"{int(team_split[0])}-{int(team_split[1])}"
        if team_split[0] != team_split[1]:
            label = f"{label}, {team_split_team or 'either team'}"
        rows.append(("Team split", "Any", label))
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
        at_most_one_qb = False
        team_split: Tuple[int, int] | None = None
        team_split_team: str | None = None
        if preset is ContestPreset.SHOWDOWN_GPP:
            at_most_one_qb = st.toggle(
                "At most one quarterback",
                value=False,
                key=_preset_key(preset, "at_most_one_qb"),
                help=(
                    "A lineup includes one quarterback at most, so both teams' "
                    "quarterbacks cannot be paired. The captain counts as that quarterback."
                ),
            )
            split_labels = ("Any", "3-3", "4-2", "5-1")
            split_choice = st.segmented_control(
                "Players from each team",
                options=split_labels,
                default="Any",
                required=True,
                width="stretch",
                wrap=True,
                key=_preset_key(preset, "team_split"),
                help=(
                    "Counts include the captain. 4-2 is four players from one team "
                    "and two from the other. DraftKings allows at most 5 from one team."
                ),
            )
            split_label = str(split_choice or "Any")
            if split_label != "Any":
                high_s, low_s = split_label.split("-")
                team_split = (int(high_s), int(low_s))
            if team_split is not None and team_split[0] != team_split[1]:
                slate_teams = _showdown_team_abbrevs()
                heavy_choice = st.selectbox(
                    "More players from",
                    options=["Either team", *slate_teams],
                    key=_preset_key(
                        preset,
                        "team_split_side_" + ("-".join(slate_teams) or "none"),
                    ),
                    help=(
                        "The first number comes from this team. Either team lets "
                        "the optimizer choose the side."
                    ),
                )
                if heavy_choice and heavy_choice != "Either team":
                    team_split_team = str(heavy_choice)
                if not slate_teams:
                    st.caption("Upload a salary file to choose which team gets the larger side.")

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
            max_quarterbacks=1 if at_most_one_qb else None,
            team_split=team_split,
            team_split_team=team_split_team,
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
        if at_most_one_qb:
            changed.append("Quarterbacks")
        if team_split is not None:
            changed.append("Team split")

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


def _showdown_team_abbrevs() -> List[str]:
    """Team abbreviations on the loaded salary file, once a Showdown pool exists."""
    pool = st.session_state.get("player_pool")
    if pool is None or "TeamAbbrev" not in pool.columns:
        return []
    teams = {
        str(team).strip().upper()
        for team in pool["TeamAbbrev"]
        if str(team).strip() and str(team).strip().lower() != "nan"
    }
    return sorted(teams)


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
            max_quarterbacks=overrides.get("max_quarterbacks"),
            team_split=overrides.get("team_split"),
            team_split_team=overrides.get("team_split_team"),
        )

    optimizer.set_max_repeating_players(overrides.get("max_repeating_players"))
    optimizer.max_cumulative_ownership = overrides.get("max_cumulative_ownership")


def _archive_salaries(pool: pd.DataFrame | None, slate_type: str) -> str:
    """Save this salary file once. A second generate of the same file does not write again."""
    signature = str(st.session_state.get("pool_signature") or "").strip()
    if pool is None or pool.empty or not signature:
        return ""
    try:
        saved = save_salary_pool_if_new(
            pool,
            slate_type,
            signature,
            slate_name=str(st.session_state.get("pool_source_name") or "DK salaries"),
        )
    except Exception as exc:  # noqa: BLE001 - lineup generation still stands
        return f" Salaries were not saved: {exc}"
    if saved:
        return " Saved player salaries."
    return " Salaries were already saved for this file."


def render_archive_view() -> None:
    """Import DraftKings contest exports and browse the local SQLite archive."""
    st.header("Historical archive")
    st.caption(
        "Upload a Game Center file named like contest-standings-197055070.csv or .zip. "
        "Contest name, date, entry fee, and contest type come from DraftKings. "
        "Player salaries are saved when you generate lineups."
    )

    with st.form("contest_archive_import"):
        standings_file = st.file_uploader(
            "Contest standings CSV",
            type=["csv", "zip"],
            help="DraftKings CSV, or a zip that contains that CSV.",
        )
        submitted = st.form_submit_button("Import and archive contest")

    if submitted:
        if standings_file is None:
            st.error("Upload a contest standings CSV or zip.")
        else:
            filename = str(getattr(standings_file, "name", "") or "")
            try:
                standings_text, id_name = open_contest_standings(standings_file, filename)
                contest_id = contest_id_from_filename(id_name)
                metadata = fetch_dk_contest_metadata(contest_id)
                note = ""
                if not contest_id:
                    note = "The file name has no contest id, so DraftKings metadata was not loaded."
                elif not str(metadata.get("slate_name") or "").strip():
                    note = "DraftKings did not return contest details. The archive used default metadata."
                parsed = parse_dk_contest_csv(
                    standings_text,
                    contest_id,
                    metadata,
                )
                save_contest(parsed["slate"], parsed["players"], parsed["standings"])
            except Exception as exc:  # noqa: BLE001 - show the import error in the page
                st.error(str(exc))
            else:
                owned = sorted(
                    parsed["players"],
                    key=lambda player: float(player.get("ownership_pct") or -1),
                    reverse=True,
                )[:5]
                st.session_state["last_import"] = {
                    "contest_id": parsed["slate"]["contest_id"],
                    "slate_name": parsed["slate"]["slate_name"],
                    "slate_type": parsed["slate"]["slate_type"],
                    "contest_date": parsed["slate"]["contest_date"],
                    "entry_fee": parsed["slate"]["entry_fee"],
                    "total_entries": parsed["slate"]["total_entries"],
                    "cash_cutline": parsed["slate"]["cash_cutline"],
                    "note": note,
                    "top_owned": [
                        {
                            "Player": player["player_name"],
                            "Ownership %": None
                            if player.get("ownership_pct") is None
                            else round(float(player["ownership_pct"]) * 100, 2),
                            "Actual FPTS": player.get("actual_fpts"),
                        }
                        for player in owned
                    ],
                }

    last_import = st.session_state.get("last_import")
    if last_import:
        slate_name = str(last_import.get("slate_name") or "").strip() or last_import["contest_id"]
        st.success(
            f"Imported {last_import.get('slate_type', 'Classic')} contest {slate_name}."
        )
        if last_import.get("note"):
            st.warning(str(last_import["note"]))
        fee = last_import.get("entry_fee")
        metric_date, metric_fee, metric_cash = st.columns(3)
        metric_date.metric("Contest date", str(last_import.get("contest_date") or "—"))
        metric_fee.metric("Entry fee", "—" if fee is None else f"${float(fee):g}")
        cash_line = last_import.get("cash_cutline")
        metric_cash.metric(
            "Cash line",
            "—" if cash_line is None else f"{float(cash_line):.2f}",
        )
        st.caption("Top chalk plays")
        st.dataframe(
            pd.DataFrame(last_import["top_owned"]),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Recent slates")
    recent = get_recent_slates()
    st.dataframe(recent, width="stretch", hide_index=True)

    st.subheader("Player archive")
    name_filter = st.text_input("Player name", key="archive_player_name")
    slate_labels = ["All slates"]
    slate_ids = [""]
    if not recent.empty:
        for _, slate in recent.iterrows():
            slate_ids.append(str(slate["Contest ID"]))
            slate_labels.append(
                f"{slate['Date']} {slate['Slate Name']} ({slate['Contest ID']})"
            )
    slate_choice = st.selectbox(
        "Slate",
        options=list(range(len(slate_labels))),
        format_func=lambda index: slate_labels[index],
        key="archive_slate",
    )
    players = search_player_performance(
        player_name=name_filter,
        contest_id=slate_ids[int(slate_choice)],
    )
    st.dataframe(players, width="stretch", hide_index=True)

    st.subheader("Summary")
    st.caption(
        "Cash line is the lowest score that finished in the money. Chalk average "
        "is the mean actual points of players owned at 15% or more."
    )
    st.dataframe(get_chalk_and_cutline_summary(), width="stretch", hide_index=True)


_init_state()

view = st.segmented_control(
    "View",
    ["Player pool", "Optimizer", "Historical archive"],
    default="Optimizer",
    label_visibility="collapsed",
    key="app_view",
)
if view == "Historical archive":
    render_archive_view()
    st.stop()

# None until a Showdown pool is loaded. An empty list excludes every quarterback.
confirmed_qb_keys: List[Tuple[str, str]] | None = None

# ---------------------------------------------------------------------------
# Sidebar — format, preset, and editable overrides
# ---------------------------------------------------------------------------
vegas_editor_needs_rerun = False
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
                    "Starts with the Sleeper depth-chart QB1 on each team, using that "
                    "player's Sleeper projection. Any quarterback you leave out is excluded. "
                    "Quarterbacks under 12.0 Sleeper points are removed automatically."
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
            fallback_names = []
            if "StarterFallback" in loaded_pool.columns:
                fallback_names = (
                    loaded_pool.loc[
                        loaded_pool["StarterFallback"].fillna(False).astype(bool),
                        "Name",
                    ]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
            under_floor = [name for name in under_floor if name not in set(fallback_names)]
            if under_floor:
                st.caption(
                    "Under 12.0 points, removed: "
                    + ", ".join(dict.fromkeys(under_floor))
                    + "."
                )
            if fallback_names:
                st.caption(
                    "Confirmed starter with no weekly projection, DraftKings average kept: "
                    + ", ".join(fallback_names)
                    + ". This is not a Sleeper projection and is not removed by the 12-point line."
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
    signature = json.dumps(fetched_lines, sort_keys=True, default=str)
    if (
        force_vegas_refresh
        or st.session_state.vegas_frame is None
        or signature != st.session_state.vegas_source_json
    ):
        st.session_state.vegas_source_json = signature
        st.session_state.vegas_frame = lines_to_frame(fetched_lines)
        # A new fetch, including a successful empty feed, has to remount the
        # grid. Otherwise the previous cells stay on screen.
        st.session_state.vegas_editor_version = int(st.session_state.vegas_editor_version) + 1
    st.caption(format_cache_status())
    vegas_baseline = st.session_state.vegas_frame
    if vegas_baseline is not None:
        vegas_baseline = vegas_baseline.copy()
    edited_vegas = st.data_editor(
        st.session_state.vegas_frame,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        height=240,
        disabled=["ITT"],
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
                format="%.2f",
                width="small",
                disabled=True,
                help=(
                    "Derived from the shared game total and this team's signed spread. "
                    "Editing one total or one spread updates both teams."
                ),
            ),
        },
    )
    reconciled = sync_vegas_editor(
        vegas_baseline,
        edited_vegas,
        int(st.session_state.vegas_editor_version),
    )
    if reconciled["changed"]:
        st.session_state.vegas_frame = reconciled["frame"]
        st.session_state.vegas_editor_version = reconciled["editor_version"]
        vegas_editor_needs_rerun = True
    vegas_warnings = visible_odds_warnings(
        reconciled["lines"],
        st.session_state.player_pool,
        coverage_warnings(),
        reconciled["warnings"],
    )
    for warning in vegas_warnings:
        st.warning(warning)

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

if vegas_editor_needs_rerun:
    st.rerun()
vegas_lines = reconciled["lines"]
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
    at_most_one_qb=strategy_overrides.get("max_quarterbacks") == 1,
    team_split=strategy_overrides.get("team_split"),
    team_split_team=strategy_overrides.get("team_split_team"),
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
# Main — Player pool, or the optimizer action bar
# ---------------------------------------------------------------------------
is_showdown = contest_format is ContestFormat.SHOWDOWN
for warning in vegas_warnings:
    st.warning(warning)
if drop_stale_scoring_session(st.session_state):
    st.session_state.sleeper_version = int(st.session_state.get("sleeper_version") or 0) + 1
lineups = st.session_state.lineups
_generated_for_export = st.session_state.generated_run or {}
_requested_lineups = _generated_for_export.get("requested")
csv_text = (
    lineups_to_dk_upload_csv(lineups, contest_format=contest_format) if lineups else ""
)
exposure_csv = (
    exposure_report_csv(
        lineups,
        requested=None if _requested_lineups is None else int(_requested_lineups),
    )
    if lineups
    else ""
)
download_name = (
    "dk_nfl_showdown_upload.csv" if is_showdown else "dk_nfl_classic_upload.csv"
)
exposure_name = (
    "showdown_player_exposure.csv" if is_showdown else "classic_player_exposure.csv"
)
pool: pd.DataFrame | None = st.session_state.player_pool

if view == "Player pool":
    st.title("Player pool")
    upload_help = (
        "Official DraftKings NFL Showdown / Captain Mode DKSalaries.csv "
        "(CPT + FLEX rows per player)."
        if is_showdown
        else "Official DraftKings NFL Classic salary export (DKSalaries.csv)."
    )
    st.caption(
        "Load the salary file here. Include, captain, ownership, lock, and projection "
        "are saved with the pool and used when you generate lineups."
    )
    uploaded = st.file_uploader(
        "DraftKings player pool CSV",
        type=["csv"],
        help=upload_help,
        key=f"uploader_{contest_format.value}",
    )
    if uploaded is not None:
        try:
            parsed_pool = parse_dk_csv(uploaded, contest_format=contest_format)
            force_sleeper = bool(st.session_state.pop("sleeper_force", False))
            parsed_pool, sleeper_summary = overlay_sleeper_projections(
                parsed_pool,
                force_refresh=force_sleeper,
            )
            parsed_pool = stamp_projection_baseline(parsed_pool)
            if force_sleeper:
                st.session_state.sleeper_version = int(st.session_state.sleeper_version) + 1
            signature = hashlib.sha1(uploaded.getvalue()).hexdigest()
            pool_changed = st.session_state.get("pool_signature") != signature
            if not pool_changed:
                parsed_pool = carry_choices(st.session_state.player_pool, parsed_pool)
            if pool_changed or not sleeper_summary.get("applied"):
                st.session_state.lineups = []
                st.session_state.generated_run = None
            st.session_state.player_pool = parsed_pool
            st.session_state.sleeper_summary = sleeper_summary
            st.session_state.pool_signature = signature
            st.session_state.pool_source_name = str(getattr(uploaded, "name", "") or "DK salaries")
            st.session_state.error = None
            pool = parsed_pool
            # Sidebar reads the pool before this upload runs, so refresh once.
            if pool_changed:
                st.rerun()
        except DKCSVValidationError as exc:
            st.session_state.error = str(exc)
            st.session_state.player_pool = None
            pool = None

    if st.session_state.error:
        st.error(st.session_state.error)

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
    sleeper_refresh = st.button("Refresh Sleeper projections", key="refresh_sleeper")
    if sleeper_refresh:
        st.session_state.sleeper_force = True
        st.rerun()
    if not session_contract_current(pool, st.session_state.sleeper_summary):
        drop_stale_scoring_session(st.session_state)
        st.session_state.sleeper_version = int(st.session_state.get("sleeper_version") or 0) + 1
        st.error(
            "Stored projections are missing current DraftKings scoring metadata and were not used."
        )
        st.stop()
    st.caption(sleeper_status_caption(st.session_state.sleeper_summary, pool))
    unmatched_warning = unmatched_players_warning(st.session_state.sleeper_summary)
    if unmatched_warning:
        st.warning(unmatched_warning)
    fallback_line = scoring_fallback_lines(pool)
    if fallback_line:
        st.warning(fallback_line)
    approximation_line = scoring_approximation_lines(pool)
    if approximation_line:
        st.caption(approximation_line)
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
    teams = pool["TeamAbbrev"] if "TeamAbbrev" in pool.columns else [""] * len(pool)
    positions = pool["Position"] if "Position" in pool.columns else [None] * len(pool)
    player_count = len({
        key
        for key in (
            person_key(name, team, position)
            for name, team, position in zip(pool["Name"], teams, positions)
        )
        if key[0]
    })
    st.caption(
        f"{player_count} players · {len(pool)} salary rows · {roster_n}-player roster · "
        f"salary cap ${DK_SALARY_CAP:,}{injury_note}."
    )
    if is_showdown:
        st.caption(
            "One row per player, priced at the flex salary. "
            "Uncheck Include to drop both the captain and flex rows. "
            "Uncheck Captain to keep the player at flex only. "
            "Set Captain % to the share of lineups where this player is the captain."
        )
    else:
        st.caption("Uncheck Include to leave a player out of lineups.")
    if "Sleeper" not in pool.columns:
        pool = pool.copy()
        pool["Sleeper"] = pd.Series(pd.NA, index=pool.index, dtype="Float64")
    pool = stamp_projection_baseline(pool)
    automatic_pool = pool.copy()
    if "ProjectionOverride" in automatic_pool.columns:
        automatic_pool["ProjectionOverride"] = pd.Series(
            pd.NA, index=automatic_pool.index, dtype="Float64"
        )
    automatic = projections_for_solve(automatic_pool, vegas_lines, weather_by_home)
    shown = projections_for_solve(pool, vegas_lines, weather_by_home)
    computed = collapse_for_editor(automatic, showdown=is_showdown)
    editor_frame = collapse_for_editor(shown, showdown=is_showdown)
    pool_signature = str(st.session_state.get("pool_signature") or "pool")
    name_key = f"pool_name_{pool_signature}"
    position_key = f"pool_position_{pool_signature}"
    game_key = f"pool_game_{pool_signature}"
    if st.session_state.pop("pool_clear_filters", False):
        for filter_state_key in (name_key, position_key, game_key):
            st.session_state.pop(filter_state_key, None)
    name_query = st.text_input(
        "Player name",
        placeholder="Search by name",
        key=name_key,
    )
    available_positions = position_choices(editor_frame)
    selected_positions = st.pills(
        "Position",
        available_positions,
        selection_mode="multi",
        key=position_key,
        help="Leave every position unselected to show all of them. Flex is RB, WR, and TE.",
        width="stretch",
    ) if available_positions else []
    available_games = game_choices(editor_frame)
    selected_games = st.pills(
        "Game",
        available_games,
        selection_mode="multi",
        key=game_key,
        help="Leave every game unselected to show the whole slate.",
        width="stretch",
    ) if available_games else []
    visible_players = filter_editor_frame(
        editor_frame,
        name_query=name_query,
        positions=list(selected_positions or []),
        games=list(selected_games or []),
    )
    st.caption(f"Showing {len(visible_players)} of {len(editor_frame)} players.")
    with st.container(horizontal=True):
        include_shown = st.button("Include shown", key="pool_include_shown")
        remove_shown = st.button("Remove shown", key="pool_remove_shown")
        include_captain_shown = False
        remove_captain_shown = False
        if is_showdown:
            include_captain_shown = st.button(
                "Include captain shown",
                key="pool_include_captain_shown",
            )
            remove_captain_shown = st.button(
                "Remove captain shown",
                key="pool_remove_captain_shown",
            )
        clear_filters = st.button("Clear filters", key="pool_clear_filters_button")
        reset_shown = st.button(
            "Reset to automatic",
            key="pool_reset_projections",
            help=(
                "Drop pinned projections for the players on screen. "
                "A number you type stays, even when it matches the automatic projection, "
                "until you change that cell or reset it."
            ),
        )
    if include_shown or remove_shown:
        st.session_state.player_pool = set_included(pool, visible_players, include_shown)
        st.rerun()
    if include_captain_shown or remove_captain_shown:
        st.session_state.player_pool = set_captain(
            pool, visible_players, include_captain_shown
        )
        st.rerun()
    if clear_filters:
        st.session_state.pool_clear_filters = True
        st.rerun()
    if reset_shown:
        st.session_state.player_pool = clear_projection_overrides(pool, visible_players)
        # The editor widget will not accept a rewritten edited_rows payload.
        # A new generation drops the replayed projection cells. Include, lock,
        # and captain edits are already stored on the pool rows.
        st.session_state["player_editor_generation"] = (
            int(st.session_state.get("player_editor_generation", 0)) + 1
        )
        st.rerun()
    if visible_players.empty:
        st.info("No players match these filters.")
        st.stop()
    editable = {"Include", "Lock", "Projection", "Ownership %"}
    if is_showdown:
        editable.update({"Captain", "Captain %"})
    edited_players = st.data_editor(
        visible_players,
        width="stretch",
        hide_index=True,
        height=480,
        disabled=[column for column in editor_frame.columns if column not in editable],
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Uncheck to leave this player out of lineups. On Showdown, both rows are removed.",
                default=True,
            ),
            "Captain": st.column_config.CheckboxColumn(
                "Captain",
                help="Uncheck to keep this player at flex only. Captain position rules still apply.",
                default=True,
            ),
            "Captain %": st.column_config.NumberColumn(
                "Captain %",
                format="%.0f",
                min_value=0.0,
                max_value=100.0,
                help=(
                    "Share of the lineups you generate where this player is the captain. "
                    "Leave blank to let the optimizer choose. "
                    "The percentages together need to add up to 100% or less, "
                    "and the Captain box still has to be checked."
                ),
            ),
            "Lock": st.column_config.CheckboxColumn(
                "Lock",
                help="Force this player into the lineup, including a cash punt under the salary floor.",
            ),
            "Ownership %": st.column_config.NumberColumn(
                "Ownership %",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
                help="Leave blank when you do not want an ownership number for this player.",
            ),
            "AvgPointsPerGame": st.column_config.NumberColumn(
                "DraftKings",
                format="%.2f",
                help="Average points per game from the DraftKings salary file.",
            ),
            "Sleeper": st.column_config.NumberColumn(
                "Sleeper",
                format="%.2f",
                help=(
                    "Weekly DraftKings total from Sleeper component stats, before Delta. "
                    "A PPR fallback is the Sleeper PPR number and is named in Scoring. "
                    "This column does not include the captain multiplier."
                ),
            ),
            "PPR": st.column_config.NumberColumn(
                "PPR",
                format="%.2f",
                help=(
                    "Sleeper pts_ppr for this week. Compare it with Sleeper, which is "
                    "the DraftKings total. A PPR fallback uses this number."
                ),
            ),
            "Scoring": st.column_config.TextColumn(
                "Scoring",
                help="DraftKings when component stats were complete. PPR fallback when they were not.",
            ),
            "ScoringNote": st.column_config.TextColumn(
                "Scoring note",
                help="Why a row used PPR, or which DraftKings input was approximated.",
                width="large",
            ),
            "Projection": st.column_config.NumberColumn(
                "Projection",
                format="%.2f",
                help=(
                    "Weekly DraftKings total, or the salary-file average when Sleeper "
                    "is blank, plus Delta. Milestone bonuses inside the weekly total are "
                    "expected values. Type a number to pin it. The pin stays when odds "
                    "or wind change. Reset to automatic to use the current odds and wind again."
                ),
            ),
            "Delta": st.column_config.NumberColumn(
                "Delta",
                format="%+.2f",
                help=(
                    "Points already added to or removed from Projection. "
                    "Vegas implied team total moves RB, WR, and TE by +5% at 26 or above, "
                    "or -5% at 18 or below. Outdoor wind can cut the number further. "
                    "Sleeper stays the weekly DraftKings total, or the flagged PPR fallback."
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
        key=_player_editor_key(
            contest_format,
            pool_signature,
            int(st.session_state.sleeper_version),
            f"{_automatic_projection_key(computed)}_{SCORING_VERSION}",
            name_query,
            list(selected_positions or []),
            list(selected_games or []),
            int(st.session_state.get("player_editor_generation", 0)),
        ),
    )
    st.session_state.player_pool = apply_pool_edits(
        pool,
        edited_players,
        visible_players,
        showdown=is_showdown,
    )
    st.stop()

title = (
    "DraftKings NFL Showdown Optimizer"
    if is_showdown
    else "DraftKings NFL Classic Lineup Optimizer"
)
st.title(title)
st.caption("Pick a contest preset and export multi-entry lineups from the player pool.")
export_headers = ",".join(upload_columns_for(contest_format))
st.caption(f"Export template: `{export_headers}`")

generate_col, download_col = st.columns([1.2, 1.6], gap="medium")
with generate_col:
    generate = st.button(
        "Generate Lineups",
        type="primary",
        width="stretch",
        disabled=pool is None,
    )
with download_col:
    st.download_button(
        "Download DK Upload CSV",
        data=csv_text,
        file_name=download_name,
        mime="text/csv",
        type="secondary",
        width="stretch",
        disabled=not bool(lineups),
    )
    st.download_button(
        "Download exposure report",
        data=exposure_csv,
        file_name=exposure_name,
        mime="text/csv",
        type="secondary",
        width="stretch",
        disabled=not bool(lineups),
        help=(
            "One row per real player. Showdown captain and flex count together. "
            "This is not the DraftKings upload."
        ),
    )

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

if pool is None:
    st.info("Open Player pool and upload a DraftKings salary file before generating lineups.")
    st.stop()

optimizer_unmatched = unmatched_players_warning(st.session_state.sleeper_summary)
if optimizer_unmatched:
    st.warning(optimizer_unmatched)
working_pool = generation_pool(pool, showdown=is_showdown).reset_index(drop=True)
locked_ids = locked_player_ids(working_pool, showdown=is_showdown)

if generate:
    sleeper_summary = st.session_state.sleeper_summary
    contract_ok = session_contract_current(st.session_state.player_pool, sleeper_summary)
    if (
        not contract_ok
        or not (sleeper_summary or {}).get("applied")
        or any(column not in working_pool.columns for column in SCORING_COLUMNS)
    ):
        if not contract_ok:
            drop_stale_scoring_session(st.session_state)
            st.session_state.sleeper_version = int(st.session_state.get("sleeper_version") or 0) + 1
            st.error(
                "Stored projections are missing current DraftKings scoring metadata and were not used."
            )
        else:
            st.session_state.lineups = []
            st.session_state.generated_run = None
            st.error(sleeper_status_caption(sleeper_summary, st.session_state.player_pool))
        st.stop()
    progress = st.progress(0, text="Building optimizer…")
    try:
        if working_pool.empty:
            raise RuntimeError("Every player is removed from the pool.")
        solve_lines = vegas_lines
        solve_teams = set(working_pool["TeamAbbrev"].astype(str).str.strip().str.upper())
        solve_low_total = is_showdown and low_scoring_showdown(solve_lines, solve_teams)
        solve_weather = fetch_slate_weather(
            working_pool,
            force_refresh=False,
        )
        st.session_state.weather_by_home = solve_weather
        # Current odds and wind, from the clean baseline. Overrides stay as entered.
        solve_pool = projections_for_solve(working_pool, solve_lines, solve_weather)
        optimizer = get_nfl_optimizer(contest_format)
        optimizer.set_exclude_inactive(exclude_inactive)
        optimizer.load_from_dataframe(solve_pool)
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
        limit_message = None
        if getattr(optimizer, "last_shortfall", None) is not None:
            limit_message = optimizer.last_shortfall.message
        st.session_state.lineups = generated
        st.session_state.generated_run = {
            "preset": preset.value,
            "preset_label": preset_cfg.label,
            "rows": live_setting_rows,
            "rule_check": rule_check,
            "limit_message": limit_message,
            "requested": int(lineup_count),
        }
        override_note = (
            f" (overrides: {', '.join(all_changed)})" if all_changed else ""
        )
        salary_note = _archive_salaries(
            st.session_state.player_pool,
            "Showdown" if is_showdown else "Classic",
        )
        st.session_state.flash_success = (
            f"Generated {len(generated)} lineup(s) with preset "
            f"{PRESET_LABELS[preset]}{override_note}.{salary_note}"
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
    limit_message = (used or {}).get("limit_message") if used else None
    if limit_message:
        st.warning(limit_message)
    if any(lineup.attrs.get("game_script") for lineup in lineups):
        st.markdown("#### Lineups per game")
        st.dataframe(
            game_script_summary(lineups),
            width="stretch",
            hide_index=True,
        )
    requested_count = (used or {}).get("requested") if used else None
    exposure = player_exposure_table(
        lineups,
        requested=None if requested_count is None else int(requested_count),
    )
    st.markdown("#### Player exposure")
    st.caption(
        "Each person is counted once per lineup. Showdown captain and flex are the same player. "
        "Lineups is the count. Exported % is the share of the lineups that were built. "
        "Requested % is the share of the number you asked for. "
        "The cap is the floor of that requested share, so 40% of 6 is 2. "
        "The DraftKings upload file is unchanged and does not include this table; "
        "use Download exposure report for the same numbers."
    )
    exposure_config = {
        "Lineups": st.column_config.NumberColumn("Lineups", format="%d"),
        "Exported %": st.column_config.NumberColumn(
            "Exported %",
            format="%.1f",
            help="Share of the lineups that were exported.",
        ),
        "Requested %": st.column_config.NumberColumn(
            "Requested %",
            format="%.1f",
            help="Share of the number of lineups you asked for.",
        ),
    }
    if "CPT count" in exposure.columns:
        exposure_config["CPT count"] = st.column_config.NumberColumn("CPT count", format="%d")
        exposure_config["FLEX count"] = st.column_config.NumberColumn("FLEX count", format="%d")
    st.dataframe(
        exposure,
        width="stretch",
        hide_index=True,
        column_config=exposure_config,
    )

    options = list(range(len(lineups)))

    def _spent(lu: pd.DataFrame) -> int:
        return int(lu.attrs.get("salary", int(lu["Salary"].sum())))

    def _remaining(lu: pd.DataFrame) -> int:
        if "salary_remaining" in lu.attrs:
            return int(lu.attrs["salary_remaining"])
        return DK_SALARY_CAP - _spent(lu)

    salary_columns = {
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "Salary Remaining ($)": st.column_config.NumberColumn(
            "Salary Remaining ($)",
            format="$%d",
            help="Dollars left under the $50,000 DraftKings cap.",
        ),
        "Projection": st.column_config.NumberColumn("Projection", format="%.2f"),
    }
    if is_showdown:
        st.dataframe(
            showdown_lineup_grid(lineups),
            width="stretch",
            hide_index=True,
            column_config=salary_columns,
        )
    else:
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
            column_config=salary_columns,
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

    if not is_showdown:
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
