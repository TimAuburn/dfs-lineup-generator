"""Streamlit UI for the DraftKings NFL Classic / Showdown Lineup Optimizer."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
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
    file_content_hash,
    get_chalk_and_cutline_summary,
    get_recent_slates,
    import_standings,
    save_salary_pool_if_new,
    search_player_performance,
)
from src.data_loader import (
    apply_low_total_kicker_cap,
    kicker_captain_exposure_cap,
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
from src.generation_progress import format_lineup_progress, lineup_progress_percent
from src.lineup_capacity import (
    GENERATE_BUDGET_SECONDS,
    SolvePlan,
    format_capacity,
    format_warning,
    measure,
    pool_fingerprint,
)
from src.upload_check import (
    csv_for_download,
    failure_table,
    resolve_upload_check,
    upload_check_key,
    validate_lineups,
    warning_table,
)
from src.lineup_rules import (
    active_game_shortfall,
    construction_check_frame,
    construction_problem_messages,
    normalize_spreads,
)
from src.showdown_rules import (
    resolve_salary_bounds,
    slate_salary_cap,
    team_split_cause,
)
from src.lineup_rules import ConstructionSettings as _ConstructionSettings
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
from src.parser import (
    DKCSVValidationError,
    assess_salary_upload,
    is_inactive_injury,
    parse_dk_csv,
)
from src.kickoff_forecast import forecasts_for_matchups, weather_label
from src.lineup_view import dataframe_height, lineup_detail_frame, sleeper_proj_caption
from src.slate_games import (
    disabled_for_slate,
    forecast_source_caption,
    game_key,
    games_table_frame,
    generation_blockers,
    matchups_from_pool,
    odds_fetched_at,
    odds_source_caption,
    order_game_labels_by_total,
    pool_without_games,
    spreads_for_pool,
    with_disabled,
)
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
    set_locked,
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
from src.cash_rules import pass_catcher_rule_label, qb_rb_rule_label
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
        "disabled_games_by_slate": {},
        "salary_detect_key": None,
        "salary_detection": None,
        "capacity_cache": {},
        "capacity_notice": None,
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


def _salary_upload_key(name: str, content: bytes) -> str:
    """One uploaded file. A new name or new bytes is a new auto-set."""
    digest = hashlib.sha256(content).hexdigest()
    return f"{name}\n{digest}"


def _detection_state(found) -> dict:
    fmt = found.contest_format
    return {
        "format": None if fmt is None else fmt.value,
        "caption": found.caption,
        "warning": found.warning,
    }


def _format_needs_autodetect(fmt: ContestFormat) -> bool:
    """True when the current format or preset does not match ``fmt``."""
    fitting = {preset.value for preset in presets_for_format(fmt)}
    return (
        st.session_state.get("contest_format") != fmt.value
        or st.session_state.get("contest_preset") not in fitting
    )


def _consume_detected_format() -> None:
    """Apply a stashed salary format before the format widgets exist.

    The upload runs after those widgets, and Streamlit rejects a write to
    ``contest_format`` once the radio is on the page. A later rerun of the
    same file does not call this: the pending value is already gone, so a
    manual choice is left alone.
    """
    pending = st.session_state.pop("salary_format_pending", None)
    if not pending:
        return
    try:
        fmt = ContestFormat(pending)
    except ValueError:
        return
    presets = presets_for_format(fmt)
    fitting = {preset.value for preset in presets}
    if st.session_state.get("contest_format") == fmt.value:
        if st.session_state.get("contest_preset") not in fitting:
            st.session_state.contest_preset = presets[0].value
        return
    st.session_state.contest_format = fmt.value
    st.session_state.contest_preset = presets[0].value
    st.session_state.player_pool = None
    st.session_state.lineups = []
    st.session_state.generated_run = None
    st.session_state.error = None


def _record_salary_upload(name: str, content: bytes) -> None:
    """Detect a new salary file once. Reruns of that file do not auto-set.

    The key is stored for a rejection too. A later rerun of that same file
    returns here and does not try the upload again. Only an accepted file may
    change the format. The pool and lineups stay put until parsing replaces them.
    """
    detect_key = _salary_upload_key(name, content)
    if st.session_state.get("salary_detect_key") == detect_key:
        return
    decision = assess_salary_upload(content, filename=name)
    st.session_state.salary_detect_key = detect_key
    if not decision.accepted:
        # The notice stays with the file. It is not the optimizer's error, and
        # it must not follow the user onto the lineup page.
        st.session_state.salary_detection = {
            "format": None,
            "caption": None,
            "warning": decision.message if decision.level == "warning" else None,
            "error": decision.message if decision.level == "error" else None,
        }
        st.session_state.error = None
        return
    found = decision.detection
    st.session_state.salary_detection = _detection_state(found)
    if found.contest_format is not None and _format_needs_autodetect(found.contest_format):
        st.session_state.salary_format_pending = found.contest_format.value
        st.rerun()


def _render_salary_detection() -> None:
    found = st.session_state.get("salary_detection") or {}
    warning = found.get("warning")
    if warning:
        st.warning(str(warning))
        return
    error = found.get("error")
    if error:
        st.error(str(error))
        return
    caption = found.get("caption")
    if caption:
        st.caption(str(caption))


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


def _render_upload_check(check, *, have_lineups: bool) -> None:
    """Show the pure upload check. The page does not decide the rules itself."""
    if not have_lineups:
        return
    if check is None:
        st.error("Upload check is out of date for this slate and was not reused.")
        return
    if check.status_message:
        st.warning(check.status_message)
    if check.blocked:
        st.error("Download blocked until every lineup passes.")
        table = failure_table(check)
        if not table.empty:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Reasons": st.column_config.TextColumn("Reasons", width="large"),
                },
            )
    if check.warnings:
        st.warning(
            "Questionable players do not block the download."
            if not check.blocked
            else (
                "Questionable players are a warning only. "
                "The download is blocked for the other reasons."
            )
        )
        warned = warning_table(check)
        if not warned.empty:
            st.dataframe(warned, hide_index=True, width="stretch")


def _prepare_dk_upload(lineups, contest_format: ContestFormat, is_showdown: bool):
    """Recompute the upload check from the slate, lineups, and game toggles.

    A stored result is kept only when it was produced for this same key, and
    only by replacing it in this run. Kickoff uses the clock inside the check.
    """
    pool = st.session_state.get("player_pool")
    salary_pool = pool if isinstance(pool, pd.DataFrame) else None
    disabled = [] if is_showdown else _disabled_game_keys()
    key = upload_check_key(
        lineups,
        salary_pool,
        contest_format=contest_format,
        disabled_games=disabled,
        pool_signature=str(st.session_state.get("pool_signature") or ""),
    )

    def compute():
        produced = ""
        export_error = None
        if lineups:
            try:
                produced = lineups_to_dk_upload_csv(lineups, contest_format=contest_format)
            except ValueError as exc:
                export_error = str(exc)
        clock = st.session_state.get("upload_check_now")
        check = validate_lineups(
            lineups,
            salary_pool,
            contest_format=contest_format,
            disabled_games=disabled,
            now=clock if hasattr(clock, "tzinfo") else None,
            csv_text=None if export_error else produced,
            export_error=export_error,
        )
        return check, produced, export_error

    result = resolve_upload_check(st.session_state, key=key, compute=compute)
    if not isinstance(result, tuple) or len(result) != 3:
        return None, ""
    check, produced, export_error = result
    payload = "" if export_error else produced
    return check, csv_for_download(check, payload)


def _disabled_game_keys() -> List[str]:
    """Classic games turned off for the loaded slate. Showdown never filters."""
    if ContestFormat(st.session_state.get("contest_format")) is ContestFormat.SHOWDOWN:
        return []
    pool = st.session_state.get("player_pool")
    if not isinstance(pool, pd.DataFrame):
        return []
    signature = str(st.session_state.get("pool_signature") or "")
    games = [game.key for game in matchups_from_pool(pool)]
    return disabled_for_slate(
        st.session_state.get("disabled_games_by_slate"),
        signature,
        games,
    )


def _toggle_is_off(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return not bool(value)


def _load_kickoff_forecasts(matchups, slate_key: str):
    """Hourly kickoff forecast. Pytest runs stay off the network."""
    kwargs = {"slate_key": slate_key}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        def _refuse(_url: str):
            raise OSError("kickoff forecast is not called from tests")

        kwargs["http_get"] = _refuse
        kwargs["cache_path"] = Path("/tmp/dfs-kickoff-pytest-cache.json")
    try:
        return forecasts_for_matchups(matchups, **kwargs)
    except Exception:
        return []


def _render_slate_games(pool: pd.DataFrame, lines: dict, *, showdown: bool) -> None:
    """Games, odds, kickoff weather, and Classic on/off toggles."""
    matchups = matchups_from_pool(pool)
    if not matchups:
        return
    signature = str(st.session_state.get("pool_signature") or "slate")
    store = st.session_state.get("disabled_games_by_slate") or {}
    disabled = (
        []
        if showdown
        else disabled_for_slate(store, signature, [game.key for game in matchups])
    )
    forecasts = _load_kickoff_forecasts(matchups, signature)
    labels = {item.game: weather_label(item) for item in forecasts}
    frame = games_table_frame(
        matchups,
        lines,
        labels,
        disabled,
        showdown=showdown,
    )
    fetched = [item.fetched_at for item in forecasts if item.fetched_at is not None]
    st.markdown("#### Games")
    st.caption(odds_source_caption(odds_fetched_at()))
    st.caption(forecast_source_caption(max(fetched) if fetched else None))
    st.caption(
        "Stadium coordinates and roof type come from the local stadium table. "
        "Dome and closed-roof games show Dome and skip the forecast. "
        "A retractable roof shows the hourly forecast and is marked retractable. "
        "A missing total, spread, or forecast stays missing and does not block lineups."
    )
    if showdown:
        st.caption("Showdown is one game, so there is no on/off toggle.")
        st.dataframe(frame, hide_index=True, width="stretch")
        return
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        height=min(56 + 36 * max(len(frame), 1), 520),
        disabled=[column for column in frame.columns if column != "On"],
        column_config={
            "On": st.column_config.CheckboxColumn(
                "On",
                default=True,
                help="Off removes every player in this game, including the defense, from generation.",
            ),
            "Matchup": st.column_config.TextColumn("Matchup"),
            "Kickoff": st.column_config.TextColumn("Kickoff"),
            "Total": st.column_config.TextColumn(
                "Total",
                help="Game over/under from The Odds API. Missing stays missing.",
            ),
            "Spread": st.column_config.TextColumn(
                "Spread",
                help="Favorite and the signed line. Missing stays missing.",
            ),
            "Weather": st.column_config.TextColumn("Weather", width="large"),
        },
        # Edits are stored by row position. The order token drops that state
        # when the rows move, so a checkbox stays on the matchup it was set for.
        key=(
            f"slate_games_{signature}_"
            + hashlib.sha1("\n".join(frame["Matchup"].astype(str)).encode()).hexdigest()[:16]
        ),
    )
    turned_off: List[str] = []
    if "On" in edited.columns and "Matchup" in edited.columns:
        for _, row in edited.iterrows():
            if not _toggle_is_off(row.get("On")):
                continue
            key = game_key(row.get("Matchup"))
            if key:
                turned_off.append(key)
    st.session_state.disabled_games_by_slate = with_disabled(store, signature, turned_off)


def _game_script_buckets(preset: ContestPreset, lineup_count: int) -> List[GameScriptBucket]:
    """Editable per-game lineup counts and salary caps for a loaded Classic pool."""
    pool = st.session_state.get("player_pool")
    games = games_from_pool(pool) if isinstance(pool, pd.DataFrame) else []
    disabled = set(_disabled_game_keys())
    if disabled:
        games = [game for game in games if game not in disabled]
    if not games:
        st.caption("Upload a salary file to assign lineups to games.")
        return []
    signature = st.session_state.get("pool_signature") or "none"
    if disabled:
        signature = (
            f"{signature}_"
            + hashlib.sha1(",".join(sorted(disabled)).encode()).hexdigest()[:8]
        )
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


def _construction_controls(
    preset: ContestPreset,
    preset_cfg: PresetConfig,
) -> Tuple[Dict[str, Any], List[str]]:
    """On/off controls for the Large-Field lineup rules. Other presets default them off."""
    st.markdown("**Lineup construction**")
    st.caption(
        "These rules choose who can share a lineup. They do not change a projection. "
        "A defense counts toward the players from the quarterback's game. "
        "It is never a pass catcher and never a bring-back."
    )
    primary_stack = st.checkbox(
        "Primary stack",
        value=bool(preset_cfg.primary_stack),
        key=_preset_key(preset, "primary_stack"),
        help=(
            "The lineup has a quarterback, at least N of his team's WR/TE, and at least M "
            "RB/WR/TE from the opposing team. The players from that game, including a DST "
            "from either team, stay inside the min and max."
        ),
    )
    pass_catchers = st.number_input(
        "Pass catchers with the QB",
        min_value=0,
        max_value=4,
        value=int(preset_cfg.primary_pass_catchers),
        step=1,
        disabled=not primary_stack,
        key=_preset_key(preset, "primary_pass_catchers"),
        help="At least this many WR/TE from the quarterback's team. A defense does not count.",
    )
    bring_backs = st.number_input(
        "Bring-backs from the opposing team",
        min_value=0,
        max_value=4,
        value=int(preset_cfg.primary_bring_backs),
        step=1,
        disabled=not primary_stack,
        key=_preset_key(preset, "primary_bring_backs"),
        help="At least this many RB, WR, or TE from the other team in the quarterback's game.",
    )
    game_min = st.number_input(
        "Players from the QB's game (min)",
        min_value=1,
        max_value=9,
        value=int(preset_cfg.primary_game_min),
        step=1,
        disabled=not primary_stack,
        key=_preset_key(preset, "primary_game_min"),
        help="Counts every player from either team in that game, including the defense.",
    )
    game_max = st.number_input(
        "Players from the QB's game (max)",
        min_value=1,
        max_value=9,
        value=int(preset_cfg.primary_game_max),
        step=1,
        disabled=not primary_stack,
        key=_preset_key(preset, "primary_game_max"),
        help="The defense counts toward this maximum. It is still not a pass catcher or a bring-back.",
    )
    bring_back_wr1 = st.checkbox(
        "Bring-back must be the opposing WR1 (highest salary)",
        value=bool(preset_cfg.primary_bring_back_wr1),
        disabled=not primary_stack,
        key=_preset_key(preset, "primary_bring_back_wr1"),
        help=(
            "WR1 is the highest-salary wide receiver on the opposing team. "
            "A salary tie counts, so every WR at that salary qualifies. "
            "This uses DraftKings salary, not the projection."
        ),
    )
    mini_stack = st.checkbox(
        "Secondary mini-stack",
        value=bool(preset_cfg.mini_stack),
        key=_preset_key(preset, "mini_stack"),
        help=(
            "A two-player same-game stack is at least two offensive players "
            "(QB, RB, WR, or TE) from one game other than the quarterback's game. "
            "Teammates count, and so do players on opposite sides of that game. "
            "Each game counts once. A defense does not count toward the two players."
        ),
    )
    mini_count = st.number_input(
        "Extra two-player same-game stacks",
        min_value=0,
        max_value=2,
        value=int(preset_cfg.mini_stack_count),
        step=1,
        disabled=not mini_stack,
        key=_preset_key(preset, "mini_stack_count"),
        help="How many games besides the quarterback's game must supply two offensive players.",
    )
    rb_dst = st.checkbox(
        "RB + own DST for a big favorite",
        value=bool(preset_cfg.rb_dst_favorite),
        key=_preset_key(preset, "rb_dst_favorite"),
        help=(
            "The lineup's defense must come with a running back from the same team, "
            "and that team must be favored by at least the margin below on the Vegas spread. "
            "Exactly the margin counts. A missing spread does not, and is not filled in."
        ),
    )
    favorite_margin = st.number_input(
        "Favorite margin (points)",
        min_value=0.0,
        max_value=30.0,
        value=float(preset_cfg.favorite_margin),
        step=0.5,
        disabled=not rb_dst,
        key=_preset_key(preset, "favorite_margin"),
        help="The team's own spread must be this many points or more in its favor. Exactly this number counts.",
    )
    ban_dogs = st.checkbox(
        "Restrict underdog RBs",
        value=bool(preset_cfg.ban_underdog_rbs),
        key=_preset_key(preset, "ban_underdog_rbs"),
        help=(
            "Leave running backs off teams that are underdogs by at least the margin below. "
            "This only removes them from the lineup. It does not change a projection. "
            "Exactly the margin is excluded. A missing spread is not. A locked running back stays."
        ),
    )
    underdog_margin = st.number_input(
        "Underdog margin (points)",
        min_value=0.0,
        max_value=30.0,
        value=float(preset_cfg.underdog_margin),
        step=0.5,
        disabled=not ban_dogs,
        key=_preset_key(preset, "underdog_margin"),
        help="A team dog by exactly this many points loses its running backs. One point less does not.",
    )
    settings = _ConstructionSettings(
        primary_stack=bool(primary_stack),
        pass_catchers=int(pass_catchers),
        bring_backs=int(bring_backs),
        game_min=int(game_min),
        game_max=int(game_max),
        bring_back_wr1=bool(bring_back_wr1),
        mini_stack=bool(mini_stack),
        mini_count=int(mini_count),
        rb_dst_favorite=bool(rb_dst),
        favorite_margin=float(favorite_margin),
        ban_underdog_rbs=bool(ban_dogs),
        underdog_margin=float(underdog_margin),
    )
    for message in construction_problem_messages(settings):
        st.warning(message)
    overrides = {
        "primary_stack": settings.primary_stack,
        "primary_pass_catchers": settings.pass_catchers,
        "primary_bring_backs": settings.bring_backs,
        "primary_game_min": settings.game_min,
        "primary_game_max": settings.game_max,
        "primary_bring_back_wr1": settings.bring_back_wr1,
        "mini_stack": settings.mini_stack,
        "mini_stack_count": settings.mini_count,
        "rb_dst_favorite": settings.rb_dst_favorite,
        "favorite_margin": settings.favorite_margin,
        "ban_underdog_rbs": settings.ban_underdog_rbs,
        "underdog_margin": settings.underdog_margin,
    }
    changed: List[str] = []
    if settings.primary_stack != bool(preset_cfg.primary_stack):
        changed.append("Primary stack")
    elif settings.primary_stack and (
        settings.pass_catchers != int(preset_cfg.primary_pass_catchers)
        or settings.bring_backs != int(preset_cfg.primary_bring_backs)
        or settings.game_min != int(preset_cfg.primary_game_min)
        or settings.game_max != int(preset_cfg.primary_game_max)
        or settings.bring_back_wr1 != bool(preset_cfg.primary_bring_back_wr1)
    ):
        changed.append("Primary stack")
    if settings.mini_stack != bool(preset_cfg.mini_stack):
        changed.append("Secondary mini-stack")
    elif settings.mini_stack and settings.mini_count != int(preset_cfg.mini_stack_count):
        changed.append("Secondary mini-stack")
    if settings.rb_dst_favorite != bool(preset_cfg.rb_dst_favorite):
        changed.append("Favorite RB + DST")
    elif settings.rb_dst_favorite and float(settings.favorite_margin) != float(preset_cfg.favorite_margin):
        changed.append("Favorite RB + DST")
    if settings.ban_underdog_rbs != bool(preset_cfg.ban_underdog_rbs):
        changed.append("Underdog RB restriction")
    elif settings.ban_underdog_rbs and float(settings.underdog_margin) != float(preset_cfg.underdog_margin):
        changed.append("Underdog RB restriction")
    return overrides, changed


def _showdown_rule_controls(
    preset: ContestPreset,
    preset_cfg: PresetConfig,
    *,
    window_on: bool,
    window_min: int,
    window_max: int,
    allow_kicker_captain: bool,
    team_split: Tuple[int, int] | None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Toggles for the three Showdown lineup rules. Values stay on the preset."""
    st.markdown("**Showdown lineup rules**")
    st.caption(
        "These rules choose who can share a lineup and how much salary it spends. "
        "They do not change a projection. K and DST can still be FLEX. "
        "With the team minimum off, DraftKings still allows at most five from one team."
    )
    ban_captain = st.checkbox(
        "No kicker or DST captain",
        value=bool(preset_cfg.ban_k_dst_captain),
        key=_preset_key(preset, "ban_k_dst_captain"),
        help=(
            "K and DST stay eligible at FLEX and cannot be CPT. "
            "A lock on the CPT row, or a captain share above 0, blocks generation "
            "and names the player. The lock is not dropped."
        ),
    )
    min_team = st.checkbox(
        "Minimum players per team",
        value=bool(preset_cfg.min_players_per_team_enabled),
        key=_preset_key(preset, "min_players_per_team_enabled"),
        help=(
            "CPT and FLEX each count as one player. "
            "Off leaves DraftKings' own limit of five players from one team."
        ),
    )
    min_team_count = st.number_input(
        "Minimum from each team",
        min_value=1,
        max_value=3,
        value=int(preset_cfg.min_players_per_team),
        step=1,
        disabled=not min_team,
        key=_preset_key(preset, "min_players_per_team"),
        help="At least this many from each team. 2 is two, three, or four from one side.",
    )
    salary_floor = st.checkbox(
        "Salary floor",
        value=bool(preset_cfg.salary_floor),
        key=_preset_key(preset, "salary_floor"),
        help=(
            "The number is the most salary a lineup may leave unused. "
            "It has to spend at least the slate cap minus that amount. "
            "Captain salary counts at 1.5x."
        ),
    )
    max_unused = st.number_input(
        "Maximum unused salary",
        min_value=0,
        max_value=20_000,
        value=int(preset_cfg.max_unused_salary),
        step=100,
        disabled=not salary_floor,
        key=_preset_key(preset, "max_unused_salary"),
        help="Unused salary equal to this amount passes. One dollar more does not.",
    )
    cap = slate_salary_cap(st.session_state.get("player_pool"))
    bounds = resolve_salary_bounds(
        slate_cap=cap,
        window_min=int(window_min) if window_on else None,
        window_max=int(window_max) if window_on else None,
        salary_floor=bool(salary_floor),
        max_unused=int(max_unused),
    )
    if bounds.conflict:
        st.error(bounds.conflict)
    elif bounds.note:
        st.caption(bounds.note)
    if ban_captain and allow_kicker_captain:
        st.caption(
            "No kicker or DST captain is on, so Allow kicker captain does not put a kicker at CPT."
        )
    split_cause = team_split_cause(bool(min_team), int(min_team_count), team_split)
    if split_cause:
        st.error(split_cause)
    overrides = {
        "ban_k_dst_captain": bool(ban_captain),
        "min_players_per_team_enabled": bool(min_team),
        "min_players_per_team": int(min_team_count),
        "salary_floor": bool(salary_floor),
        "max_unused_salary": int(max_unused),
    }
    changed: List[str] = []
    if bool(ban_captain) != bool(preset_cfg.ban_k_dst_captain):
        changed.append("No kicker or DST captain")
    if bool(min_team) != bool(preset_cfg.min_players_per_team_enabled):
        changed.append("Minimum players per team")
    elif min_team and int(min_team_count) != int(preset_cfg.min_players_per_team):
        changed.append("Minimum players per team")
    if bool(salary_floor) != bool(preset_cfg.salary_floor):
        changed.append("Salary floor")
    elif salary_floor and int(max_unused) != int(preset_cfg.max_unused_salary):
        changed.append("Salary floor")
    return overrides, changed


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
        if preset_cfg.pass_catcher_team_cap:
            strict_pass_catchers = st.checkbox(
                "Strict one pass catcher per team",
                value=bool(preset_cfg.strict_pass_catcher_cap),
                key=_preset_key(preset, "strict_pass_catcher_cap"),
                help=(
                    "On: at most one WR or TE from each team, even when that "
                    "team's quarterback is rostered. Off: more pass catchers "
                    "from a team are allowed when its quarterback is in the lineup."
                ),
            )
            overrides["strict_pass_catcher_cap"] = bool(strict_pass_catchers)
            st.caption(pass_catcher_rule_label(bool(strict_pass_catchers)))
            if bool(strict_pass_catchers) != bool(preset_cfg.strict_pass_catcher_cap):
                changed.append("Pass catchers")
        if preset_cfg.ban_same_team_qb_rb:
            st.caption(qb_rb_rule_label())
        construction_overrides, construction_changed = _construction_controls(preset, preset_cfg)
        overrides.update(construction_overrides)
        changed.extend(construction_changed)
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

        rule_overrides, rule_changed = _showdown_rule_controls(
            preset,
            preset_cfg,
            window_on=bool(use_salary_window),
            window_min=int(salary_lo),
            window_max=int(salary_hi),
            allow_kicker_captain=bool(allow_kicker_captain),
            team_split=team_split,
        )
        overrides.update(rule_overrides)
        changed.extend(rule_changed)
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


def _construction_settings_from_overrides(overrides: Dict[str, Any]) -> _ConstructionSettings | None:
    """Classic construction knobs currently on screen. Showdown has none."""
    if "primary_stack" not in overrides:
        return None
    return _ConstructionSettings(
        primary_stack=bool(overrides["primary_stack"]),
        pass_catchers=int(overrides["primary_pass_catchers"]),
        bring_backs=int(overrides["primary_bring_backs"]),
        game_min=int(overrides["primary_game_min"]),
        game_max=int(overrides["primary_game_max"]),
        bring_back_wr1=bool(overrides["primary_bring_back_wr1"]),
        mini_stack=bool(overrides["mini_stack"]),
        mini_count=int(overrides["mini_stack_count"]),
        rb_dst_favorite=bool(overrides["rb_dst_favorite"]),
        favorite_margin=float(overrides["favorite_margin"]),
        ban_underdog_rbs=bool(overrides["ban_underdog_rbs"]),
        underdog_margin=float(overrides["underdog_margin"]),
    )


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
    if "strict_pass_catcher_cap" in overrides and hasattr(optimizer, "strict_pass_catcher_cap"):
        optimizer.strict_pass_catcher_cap = bool(overrides["strict_pass_catcher_cap"])
    if "primary_stack" in overrides and hasattr(optimizer, "primary_stack"):
        optimizer.primary_stack = bool(overrides["primary_stack"])
        optimizer.primary_pass_catchers = int(overrides["primary_pass_catchers"])
        optimizer.primary_bring_backs = int(overrides["primary_bring_backs"])
        optimizer.primary_game_min = int(overrides["primary_game_min"])
        optimizer.primary_game_max = int(overrides["primary_game_max"])
        optimizer.primary_bring_back_wr1 = bool(overrides["primary_bring_back_wr1"])
        optimizer.mini_stack = bool(overrides["mini_stack"])
        optimizer.mini_stack_count = int(overrides["mini_stack_count"])
        optimizer.rb_dst_favorite = bool(overrides["rb_dst_favorite"])
        optimizer.favorite_margin = float(overrides["favorite_margin"])
        optimizer.ban_underdog_rbs = bool(overrides["ban_underdog_rbs"])
        optimizer.underdog_margin = float(overrides["underdog_margin"])

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
            ban_k_dst_captain=bool(overrides.get("ban_k_dst_captain", False)),
            min_players_per_team_enabled=bool(overrides.get("min_players_per_team_enabled", False)),
            min_players_per_team=int(overrides.get("min_players_per_team", 2)),
            salary_floor=bool(overrides.get("salary_floor", False)),
            max_unused_salary=int(overrides.get("max_unused_salary", 1_500)),
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
        "A file name without a contest id is stored under its own key. "
        "Importing the same file again is skipped. "
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
                raw_bytes = standings_file.getvalue()
                standings_text, id_name = open_contest_standings(raw_bytes, filename)
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
                if not contest_id and not str(parsed["slate"].get("slate_name") or "").strip():
                    parsed["slate"]["slate_name"] = filename
                result = import_standings(
                    parsed["slate"],
                    parsed["players"],
                    parsed["standings"],
                    file_content_hash(raw_bytes),
                    source_contest_id=contest_id,
                    imported_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as exc:  # noqa: BLE001 - show the import error in the page
                st.error(str(exc))
            else:
                warnings = []
                for item in (note, result.message):
                    if item and item not in warnings:
                        warnings.append(item)
                owned = sorted(
                    parsed["players"],
                    key=lambda player: float(player.get("ownership_pct") or -1),
                    reverse=True,
                )[:5]
                st.session_state["last_import"] = {
                    "status": result.status,
                    "message": result.message,
                    "contest_id": result.contest_id,
                    "slate_name": parsed["slate"]["slate_name"],
                    "slate_type": parsed["slate"]["slate_type"],
                    "contest_date": parsed["slate"]["contest_date"],
                    "entry_fee": parsed["slate"]["entry_fee"],
                    "total_entries": parsed["slate"]["total_entries"],
                    "cash_cutline": parsed["slate"]["cash_cutline"],
                    "note": note,
                    "warnings": warnings,
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
        if last_import.get("status") == "duplicate":
            st.warning(str(last_import.get("message") or ""))
        else:
            slate_name = str(last_import.get("slate_name") or "").strip() or last_import["contest_id"]
            st.success(
                f"Imported {last_import.get('slate_type', 'Classic')} contest {slate_name}."
            )
            for warning in last_import.get("warnings") or []:
                st.warning(str(warning))
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
_consume_detected_format()

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
    st.caption(
        "Under streamlit run, a root-level secret is copied into the environment "
        "before this script starts, so Streamlit secrets win over the process "
        "environment and a local `.env` file. The line above names which source "
        "supplied the odds key and does not show the key. Weather uses the "
        "National Weather Service and does not need a key."
    )

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
if low_scoring_game and not strategy_overrides.get("ban_k_dst_captain"):
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
    exposure_cap = kicker_captain_exposure_cap(
        low_scoring=True,
        ban_k_dst_captain=bool(strategy_overrides.get("ban_k_dst_captain")),
        cash_preset=preset is ContestPreset.SHOWDOWN_CASH,
    )
    extra_setting_rows.append(
        (
            "Kicker exposure",
            "No cap",
            "No cap"
            if exposure_cap is None
            else f"{int(round(exposure_cap * 100))}% (game total ≤ 38.5)",
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
    # Stable across formats. A format-specific key would drop the file on the
    # rerun that applies auto-detect, and the two keys would collide.
    uploaded = st.file_uploader(
        "DraftKings player pool CSV",
        type=["csv"],
        help=upload_help,
        key="salary_uploader",
    )
    if uploaded is None:
        st.session_state.salary_detect_key = None
        st.session_state.salary_detection = None
    else:
        _record_salary_upload(
            str(getattr(uploaded, "name", "") or ""),
            uploaded.getvalue(),
        )
    _render_salary_detection()
    detected_format = (st.session_state.get("salary_detection") or {}).get("format")
    if uploaded is not None and detected_format:
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
            label = str(getattr(uploaded, "name", "") or "").strip() or "salary file"
            st.session_state.error = f"{label}: {exc}"

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
    _render_slate_games(pool, vegas_lines, showdown=is_showdown)
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
    available_games = order_game_labels_by_total(
        game_choices(editor_frame),
        matchups_from_pool(pool),
        vegas_lines,
    )
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
        lock_shown = st.button("Lock shown", key="pool_lock_shown")
        unlock_shown = st.button("Unlock shown", key="pool_unlock_shown")
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
    if lock_shown or unlock_shown:
        st.session_state.player_pool = set_locked(pool, visible_players, lock_shown)
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
# Filled after this run's generate attempt. A failure clears the stored
# lineups first, so the same render cannot offer the previous CSV.
export_slot = download_col.empty()
upload_slot = st.empty()
_exports_painted = False


def _render_capacity_notice(notice, *, preset_value, rows, frame) -> None:
    """Show the unique-lineup sentence only for the pool and rules that produced it."""
    if not notice:
        return
    if notice.get("preset") != preset_value:
        return
    if notice.get("rows") != rows:
        return
    if notice.get("pool") != pool_fingerprint(frame):
        return
    warning = notice.get("warning")
    message = notice.get("message")
    if warning:
        st.warning(warning)
    if message:
        st.info(message)


def _paint_exports() -> None:
    """Download buttons and the upload check for the lineups stored right now."""
    global _exports_painted
    if _exports_painted:
        return
    _exports_painted = True
    current = list(st.session_state.lineups or [])
    upload_check, upload_csv = _prepare_dk_upload(current, contest_format, is_showdown)
    generated_for_export = st.session_state.generated_run or {}
    requested = generated_for_export.get("requested")
    min_players = generated_for_export.get("min_players")
    exposure_csv = (
        exposure_report_csv(
            current,
            requested=None if requested is None else int(requested),
            min_players=min_players,
        )
        if current
        else ""
    )
    with export_slot.container():
        st.download_button(
            "Download DK Upload CSV",
            data=upload_csv,
            file_name=download_name,
            mime="text/csv",
            type="secondary",
            width="stretch",
            disabled=(
                not bool(current)
                or upload_check is None
                or upload_check.blocked
            ),
            key="download_dk_upload",
        )
        st.download_button(
            "Download exposure report",
            data=exposure_csv,
            file_name=exposure_name,
            mime="text/csv",
            type="secondary",
            width="stretch",
            disabled=not bool(current),
            help=(
                "One row per real player. Showdown captain and flex count together. "
                "This is not the DraftKings upload."
            ),
            key="download_exposure_report",
        )
    with upload_slot.container():
        _render_upload_check(upload_check, have_lineups=bool(current))

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
    if actual_cfg.pass_catcher_team_cap:
        st.caption(pass_catcher_rule_label(bool(actual_cfg.strict_pass_catcher_cap)))
    if actual_cfg.ban_same_team_qb_rb:
        st.caption(qb_rb_rule_label())
    st.caption(settings_note)
    if low_scoring_game:
        if strategy_overrides.get("ban_k_dst_captain"):
            st.caption(
                "Game total is 38.5 or lower. No kicker or DST captain is on, "
                "so a kicker still cannot be captain."
            )
        else:
            st.caption(
                "Game total is 38.5 or lower, so a kicker can be captain with a 10% exposure cap."
            )
    _render_settings_comparison(live_setting_rows)

if pool is None:
    _paint_exports()
    st.info("Open Player pool and upload a DraftKings salary file before generating lineups.")
    st.stop()

optimizer_unmatched = unmatched_players_warning(st.session_state.sleeper_summary)
if optimizer_unmatched:
    st.warning(optimizer_unmatched)
disabled_games = [] if is_showdown else _disabled_game_keys()
working_pool = generation_pool(pool, showdown=is_showdown).reset_index(drop=True)
if disabled_games:
    working_pool = pool_without_games(working_pool, disabled_games).reset_index(drop=True)
locked_ids = locked_player_ids(working_pool, showdown=is_showdown)

if generate:
    blockers = generation_blockers(
        pool,
        disabled_games,
        showdown=is_showdown,
        min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
        settings=_construction_settings_from_overrides(strategy_overrides),
    )
    if blockers:
        st.session_state.lineups = []
        st.session_state.generated_run = None
        st.error(" ".join(blockers))
        _paint_exports()
        st.stop()
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
            st.session_state.lineups = []
            st.session_state.generated_run = None
            st.error(
                "Stored projections are missing current DraftKings scoring metadata and were not used."
            )
        else:
            st.session_state.lineups = []
            st.session_state.generated_run = None
            st.error(sleeper_status_caption(sleeper_summary, st.session_state.player_pool))
        _paint_exports()
        st.stop()
    progress_slot = st.empty()
    requested_lineups = int(lineup_count)
    generation_started = time.perf_counter()
    progress_bar = progress_slot.progress(
        lineup_progress_percent(0, requested_lineups),
        text=format_lineup_progress(0, requested_lineups, 0.0),
    )

    def _on_lineup_progress(built: int) -> None:
        progress_bar.progress(
            lineup_progress_percent(built, requested_lineups),
            text=format_lineup_progress(
                built,
                requested_lineups,
                time.perf_counter() - generation_started,
            ),
        )

    optimizer = None
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
        optimizer.on_lineup_progress = _on_lineup_progress
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
        if disabled_games and not is_showdown and hasattr(optimizer, "construction_settings"):
            rule_blocks = active_game_shortfall(
                len(games_from_pool(solve_pool)),
                optimizer.construction_settings(),
            )
            if rule_blocks:
                raise RuntimeError(" ".join(rule_blocks))
        if hasattr(optimizer, "set_vegas_spreads"):
            if disabled_games:
                optimizer.set_vegas_spreads(spreads_for_pool(solve_lines, solve_pool))
            else:
                optimizer.set_vegas_spreads(normalize_spreads(solve_lines))
        if confirmed_qb_keys is not None and hasattr(optimizer, "set_confirmed_quarterbacks"):
            optimizer.set_confirmed_quarterbacks(confirmed_qb_keys)
        apply_low_total_kicker_cap(
            optimizer,
            low_scoring=solve_low_total,
            cash_preset=preset is ContestPreset.SHOWDOWN_CASH,
        )
        optimizer.set_exposure_limits(
            min_exposure_pct=float(min_exposure_pct) if use_min_exposure else None,
            max_exposure_pct=float(max_exposure_pct) if use_max_exposure else None,
        )
        if game_script_buckets is not None:
            capacity_plan = SolvePlan(
                kind="game_script",
                buckets=tuple(
                    (str(bucket.game), int(bucket.lineups), int(bucket.salary_cap))
                    for bucket in game_script_buckets
                ),
            )
        elif enable_tiered_salary:
            capacity_plan = SolvePlan(
                kind="tiered",
                leave_money_ratio=float(leave_money_pct) / 100.0,
                discount_cap=int(discount_cap),
                discount_floor=int(discount_floor),
                standard_cap=DK_SALARY_CAP,
            )
        else:
            capacity_plan = SolvePlan()
        progress_bar.progress(0, text="Counting unique lineups…")
        try:
            capacity_report = measure(
                optimizer,
                requested=requested_lineups,
                budget_seconds=GENERATE_BUDGET_SECONDS,
                cache=st.session_state.capacity_cache,
                plan=capacity_plan,
            )
        except Exception:
            # A counting failure must not change the lineups Generate exports.
            st.session_state.capacity_notice = None
        else:
            capacity_message = format_capacity(capacity_report, requested_lineups)
            capacity_warning = format_warning(capacity_report, requested_lineups)
            st.session_state.capacity_notice = {
                "preset": preset.value,
                "rows": live_setting_rows,
                "pool": pool_fingerprint(solve_pool),
                "message": capacity_message,
                "warning": capacity_warning,
            }
            if capacity_warning:
                st.warning(capacity_warning)
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
        limit_parts = []
        if getattr(optimizer, "last_shortfall", None) is not None:
            limit_parts.append(optimizer.last_shortfall.message)
        min_report = getattr(optimizer, "last_min_report", None)
        if min_report is not None and min_report.message:
            limit_parts.append(min_report.message)
        construction_report = getattr(optimizer, "last_construction_report", None)
        if construction_report is not None and construction_report.message:
            limit_parts.append(construction_report.message)
        construction_applied = (
            None if construction_report is None else construction_report.applied
        )
        limit_message = " ".join(limit_parts) if limit_parts else None
        min_players = list(min_report.rows) if min_report is not None else None
        st.session_state.lineups = generated
        st.session_state.generated_run = {
            "preset": preset.value,
            "preset_label": preset_cfg.label,
            "rows": live_setting_rows,
            "rule_check": rule_check,
            "limit_message": limit_message,
            "min_players": min_players,
            "requested": int(lineup_count),
            "construction_applied": construction_applied,
            "construction_spreads": (
                dict(getattr(optimizer, "vegas_spreads", {}) or {})
            ),
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
        if optimizer is not None:
            optimizer.on_lineup_progress = None
        progress_slot.empty()

_render_capacity_notice(
    st.session_state.get("capacity_notice"),
    preset_value=preset.value,
    rows=live_setting_rows,
    frame=projections_for_solve(
        working_pool,
        vegas_lines,
        st.session_state.weather_by_home or {},
    ),
)

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
    min_players = (used or {}).get("min_players") if used else None
    exposure = player_exposure_table(
        lineups,
        requested=None if requested_count is None else int(requested_count),
        min_players=min_players,
    )
    st.markdown("#### Player exposure")
    exposure_caption = (
        "Each person is counted once per lineup. Showdown captain and flex are the same player. "
        "Lineups is the count. Exported % is the share of the lineups that were built. "
        "Requested % is the share of the number you asked for. "
        "The cap is the floor of that requested share, so 40% of 6 is 2. "
        "The DraftKings upload file is unchanged and does not include this table; "
        "use Download exposure report for the same numbers."
    )
    if "Min needed" in exposure.columns:
        exposure_caption += (
            " Min needed is the ceiling of the minimum share of the lineups you asked for, "
            "so 50% of 5 is 3. Min met compares Lineups with that target. "
            "A player who was never rostered stays on this table."
        )
    st.caption(exposure_caption)
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
    if "Min needed" in exposure.columns:
        exposure_config["Min needed"] = st.column_config.NumberColumn(
            "Min needed",
            format="%d",
            help="Appearances required across the lineups you asked for.",
        )
        exposure_config["Min met"] = st.column_config.TextColumn(
            "Min met",
            help="Yes when Lineups is at least Min needed. The counts are in parentheses.",
        )
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
        applied_rules = (used or {}).get("construction_applied") if used else None
        if applied_rules is not None and applied_rules.any_enabled:
            st.markdown("#### Lineup construction")
            st.caption(
                "A defense counts toward the quarterback's game total and is never a pass catcher "
                "or a bring-back. A two-player same-game stack is two offensive players "
                "(QB, RB, WR, or TE) from another game. WR1 is the highest-salary WR on the opposing team."
            )
            st.dataframe(
                construction_check_frame(
                    lineups,
                    applied_rules,
                    pool=pool,
                    spreads=(used or {}).get("construction_spreads") or {},
                ),
                width="stretch",
                hide_index=True,
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
    lineup_view = lineup_detail_frame(lineup, pool if isinstance(pool, pd.DataFrame) else None)
    sleeper_note = sleeper_proj_caption(st.session_state.get("sleeper_summary"))
    if sleeper_note:
        st.caption(sleeper_note)
    st.dataframe(
        lineup_view,
        width="stretch",
        hide_index=True,
        height=dataframe_height(len(lineup_view)),
        column_config={
            "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "Projection": st.column_config.NumberColumn("Projection", format="%.2f"),
            "Sleeper Proj": st.column_config.NumberColumn(
                "Sleeper Proj",
                format="%.2f",
                help=(
                    "Sleeper's weekly projection for this player. "
                    "Complete stat lines are scored on the DraftKings table. "
                    "An incomplete line uses Sleeper full PPR (pts_ppr). "
                    "Standard scoring is not used, a captain is shown at 1x, "
                    "and a blank cell has no matched projection. "
                    "This column is not used to build the lineup."
                ),
            ),
            "Delta": st.column_config.NumberColumn(
                "Delta",
                format="%+.2f",
                help="Points added or removed from the player-pool projection by Vegas and wind.",
            ),
        },
    )

_paint_exports()
