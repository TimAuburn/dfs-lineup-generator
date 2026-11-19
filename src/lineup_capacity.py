"""How many unique lineups the current pool and rules can produce.

``measure`` asks the same optimizer Generate uses. It does not change
projections, the objective, or the lineup that Generate exports. Optimizer
fields touched while counting are copied off and put back.

What a label proves
-------------------
``up to N``
    The maximum is N. Either there is no cross-lineup packing rule and the
    search listed every distinct feasible lineup, or a packed set of N was
    built and the singles search proved there is no (N+1)th lineup.

``at least L``
    A feasible set of L lineups was built with every active rule, including
    overlap and exposure. A greedy stop is not a maximum. A timeout uses
    this label for the lineups finished before the clock ran out.

``at most U``
    U distinct lineups satisfy the single-lineup rules, and the next one was
    proved infeasible. Overlap, exposure, captain share, kicker share, and
    stack-exposure caps were turned off for this pass only. The packed
    maximum cannot be larger than U. It can be smaller.

``at least L and at most U``
    Both of the above, and L < U, so the maximum is still unknown.

``150+`` (or another cap)
    The search built ``cap`` lineups and stopped because the cap was reached,
    without proving that another lineup is impossible. That proves at least
    ``cap``. It does not prove that ``cap`` is the maximum.

A requested count M is too high only when a proved ceiling is below M.
``at least L`` with L < M is not that ceiling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import pandas as pd
from pydfs_lineup_optimizer.exceptions import GenerateLineupException
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver

from .optimizer import (
    DK_SALARY_CAP,
    GAME_SCRIPT_FLEX,
    STANDARD_TIER_FLOOR,
    GameScriptBucket,
    begin_capacity_budget,
    capacity_timed_out,
    clear_capacity_timeout,
    even_lineup_counts,
    optimize_game_scripts,
    optimize_tiered_salaries,
    overlap_cap_for_slate,
)
from .slate_week import slate_games_from_pool

DEFAULT_CAP = 150
# Short on purpose. A full-slate enumeration takes seconds; Generate should not.
GENERATE_BUDGET_SECONDS = 0.4

_FINGERPRINT_COLUMNS = (
    "ID",
    "Name",
    "Position",
    "Roster Position",
    "Salary",
    "TeamAbbrev",
    "Game Info",
    "Projection",
    "AvgPointsPerGame",
    "ProjectionBaseline",
    "ProjectionOverride",
    "Delta",
    "Include",
    "Lock",
    "Captain",
    "CaptainExposure",
    "Ownership",
    "DepthOrder",
    "DepthPosition",
    "Injury",
    "Sleeper",
)

_SETTING_ATTRS = (
    "contest_format",
    "exclude_inactive",
    "min_exposure",
    "max_exposure",
    "max_repeating_players",
    "exclude_vs_dst",
    "max_cumulative_ownership",
    "jitter_pct",
    "position_jitter_sds",
    "jitter_seed",
    "min_salary_cap",
    "max_salary_cap",
    "enable_qb_stack",
    "stack_pass_catchers",
    "enable_bring_back",
    "bring_back_position",
    "require_wr_in_stack",
    "slate_aware_gpp",
    "flex_positions",
    "min_skill_salary",
    "pass_catcher_team_cap",
    "strict_pass_catcher_cap",
    "ban_same_team_qb_rb",
    "primary_stack",
    "primary_pass_catchers",
    "primary_bring_backs",
    "primary_game_min",
    "primary_game_max",
    "primary_bring_back_wr1",
    "mini_stack",
    "mini_stack_count",
    "rb_dst_favorite",
    "favorite_margin",
    "ban_underdog_rbs",
    "underdog_margin",
    "vegas_spreads",
    "min_teams",
    "game_script_correlation",
    "cpt_qb_pass_catchers",
    "max_k_dst_combined",
    "max_kickers",
    "allow_kicker_captain",
    "kicker_max_exposure",
    "captain_positions",
    "captain_min_salary",
    "flex_min_salary",
    "max_dst",
    "ban_dst",
    "max_below_salary_players",
    "below_salary_threshold",
    "max_rbs_per_team",
    "confirmed_quarterbacks",
    "require_starting_core",
    "ban_k_dst_captain",
    "min_players_per_team_enabled",
    "min_players_per_team",
    "salary_floor",
    "max_unused_salary",
    "max_quarterbacks",
    "team_split",
    "team_split_team",
)


@dataclass(frozen=True)
class SolvePlan:
    """How Generate will ask for lineups. ``single`` is one ``generate_lineups`` call."""

    kind: str = "single"
    leave_money_ratio: float = 0.4
    discount_cap: int = 49_500
    discount_floor: int = 48_000
    standard_cap: int = DK_SALARY_CAP
    # (game label ``AWAY@HOME``, requested count, salary cap) per game-script bucket.
    buckets: tuple = ()


@dataclass(frozen=True)
class CapacityReport:
    """Bounds that were actually proved. Missing sides stay ``None``."""

    exact: Optional[int] = None
    lower: Optional[int] = None
    upper: Optional[int] = None
    capped: bool = False
    cap: int = DEFAULT_CAP
    timed_out: bool = False
    trivial: bool = False
    lower_exhausted: bool = False
    upper_exhausted: bool = False
    elapsed_seconds: float = 0.0
    from_cache: bool = False
    passes: tuple = ()


def pool_fingerprint(frame: Optional[pd.DataFrame]) -> str:
    """Stable hash of the solve pool columns that can change feasibility."""
    if frame is None or frame.empty:
        return "empty"
    columns = [column for column in _FINGERPRINT_COLUMNS if column in frame.columns]
    if not columns:
        payload = f"rows:{len(frame)}"
    else:
        payload = frame.loc[:, columns].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def format_capacity(report: CapacityReport, requested: int) -> str:
    """One sentence for the Generate screen. The number is only what was proved."""
    requested = int(requested)
    if report.exact is not None:
        number = int(report.exact)
        return (
            f"Your pool and rules allow up to {number} unique {_word(number)} "
            f"(requested {requested})."
        )
    if report.capped:
        return (
            f"Your pool and rules allow {int(report.cap)}+ unique lineups "
            f"(requested {requested})."
        )
    ceiling_proved = report.upper_exhausted and report.upper is not None
    if report.timed_out and (report.lower is None or int(report.lower) <= 0) and not ceiling_proved:
        return (
            "The unique-lineup count hit the time limit before a lineup was proved "
            f"(requested {requested})."
        )
    parts = []
    if report.lower is not None and int(report.lower) > 0:
        parts.append(("at least", int(report.lower)))
    if ceiling_proved and report.exact is None:
        parts.append(("at most", int(report.upper)))
    if not parts:
        return (
            "The unique-lineup count hit the time limit before a lineup was proved "
            f"(requested {requested})."
        )
    sentence = (
        f"Your pool and rules allow {_bound_phrase(parts)} (requested {requested})."
    )
    if report.timed_out:
        sentence += (
            " The count stopped at the time limit, so a bound that is still open "
            "is not a maximum."
        )
    return sentence


def format_warning(report: CapacityReport, requested: int) -> Optional[str]:
    """Warn only when M is above a proved ceiling.

    ``at least L`` does not say the pool cannot reach M. ``cap+`` does not
    say the pool stops at the cap.
    """
    requested = int(requested)
    if report.exact is not None:
        ceiling = int(report.exact)
    elif report.upper_exhausted and report.upper is not None:
        ceiling = int(report.upper)
    else:
        return None
    if requested <= ceiling:
        return None
    return (
        f"You requested {requested} lineups, but this pool and these rules "
        f"can produce at most {ceiling} unique {_word(ceiling)}."
    )


def measure(
    optimizer,
    *,
    requested: int = 1,
    cap: int = DEFAULT_CAP,
    budget_seconds: Optional[float] = GENERATE_BUDGET_SECONDS,
    cache: Optional[dict] = None,
    plan: Optional[SolvePlan] = None,
) -> CapacityReport:
    """Count unique lineups the configured optimizer can reach.

    ``requested`` is M, used by the formatters. It does not change the count.
    ``cap`` stops the search; hitting it without a failed next lineup is
    labeled ``cap+``. ``budget_seconds`` ``None`` waits for the solver.
    The optimizer's fields are restored before this returns, including when
    the count raises.
    """
    plan = plan or SolvePlan()
    cap = max(1, min(DEFAULT_CAP, int(cap)))
    key = _cache_key(optimizer, plan, cap, budget_seconds)
    if cache is not None and key in cache:
        return replace(cache[key], from_cache=True)
    started = time.perf_counter()
    saved_limit = PuLPSolver.LP_SOLVER.timeLimit
    saved, exposures = _snapshot(optimizer)
    callback = optimizer.on_lineup_progress
    if budget_seconds is None:
        deadline = None
    else:
        deadline = started + max(0.0, float(budget_seconds))
    begin_capacity_budget(deadline)

    def restore() -> None:
        _restore(optimizer, saved, exposures)
        optimizer.on_lineup_progress = None

    try:
        optimizer.on_lineup_progress = None
        report = _measure_body(optimizer, plan, cap, restore)
    finally:
        _restore(optimizer, saved, exposures)
        optimizer.on_lineup_progress = callback
        begin_capacity_budget(None)
        PuLPSolver.LP_SOLVER.timeLimit = saved_limit
    report = replace(report, elapsed_seconds=time.perf_counter() - started, cap=cap)
    if cache is not None:
        cache[key] = report
    return report


def cross_lineup_rules_active(optimizer, plan: Optional[SolvePlan] = None) -> bool:
    """True when the maximum set is a packing problem, not a singles count.

    Overlap at least ``roster - 1`` only forbids an identical lineup, which
    the singles search already does. Exposure of 100% and a minimum of 0 do
    not shrink the set. Tiered salary and game-script portfolios do.
    """
    plan = plan or SolvePlan()
    if plan.kind in {"tiered", "game_script"}:
        return True
    roster = len(getattr(optimizer, "roster_slots", ()) or ())
    repeating = getattr(optimizer, "max_repeating_players", None)
    slate_aware = bool(getattr(optimizer, "slate_aware_gpp", False))
    if repeating is not None and roster:
        tightened = overlap_cap_for_slate(
            int(repeating),
            _game_count(getattr(optimizer, "_player_df", None)),
            slate_aware,
        )
        if tightened is not None and int(tightened) < roster - 1:
            return True
    exposure = getattr(optimizer, "max_exposure", None)
    if exposure is not None and float(exposure) < 1.0 - 1e-9:
        return True
    minimum = getattr(optimizer, "min_exposure", None)
    if minimum is not None and float(minimum) > 1e-12:
        return True
    if slate_aware:
        return True
    kicker = getattr(optimizer, "kicker_max_exposure", None)
    if kicker is not None and float(kicker) < 1.0 - 1e-9:
        return True
    if _captain_share_active(optimizer):
        return True
    return False


def _measure_body(optimizer, plan: SolvePlan, cap: int, restore) -> CapacityReport:
    trivial = not cross_lineup_rules_active(optimizer, plan)
    lower_frames, lower_timed = _run(optimizer, plan, cap, singles=False)
    lower_count = len(lower_frames)
    # Reaching the cap is a finished cap, even if the clock expired on that last solve.
    if lower_count >= cap:
        lower_timed = False
    lower_exhausted = (not lower_timed) and lower_count < cap
    if trivial:
        if lower_timed:
            return _report(
                lower=lower_count,
                timed_out=True,
                trivial=True,
                passes=("rules",),
            )
        if lower_count >= cap:
            return _report(
                lower=lower_count,
                capped=True,
                trivial=True,
                passes=("rules",),
            )
        return _report(
            exact=lower_count,
            lower=lower_count,
            upper=lower_count,
            trivial=True,
            lower_exhausted=True,
            upper_exhausted=True,
            passes=("rules",),
        )
    if lower_count >= cap and not lower_timed:
        return _report(
            lower=lower_count,
            capped=True,
            trivial=False,
            passes=("rules",),
        )
    restore()
    clear_capacity_timeout()
    upper_frames, upper_timed = _run(optimizer, plan, cap, singles=True)
    upper_count = len(upper_frames)
    if upper_count >= cap:
        upper_timed = False
    upper_exhausted = (not upper_timed) and upper_count < cap
    if upper_exhausted and upper_count < lower_count:
        # A singles search that missed a lineup we already built is not an upper bound.
        upper_exhausted = False
        upper_count_report = None
    else:
        upper_count_report = upper_count if upper_exhausted else None
    exact = None
    timed_out = lower_timed or upper_timed
    if upper_exhausted and upper_count == lower_count and not (lower_timed and upper_count != lower_count):
        if upper_count == lower_count:
            exact = lower_count
            timed_out = False
    if exact is not None:
        return _report(
            exact=exact,
            lower=lower_count,
            upper=upper_count,
            trivial=False,
            lower_exhausted=lower_exhausted,
            upper_exhausted=True,
            passes=("rules", "singles"),
        )
    return _report(
        lower=lower_count,
        upper=upper_count_report,
        timed_out=timed_out,
        trivial=False,
        lower_exhausted=lower_exhausted,
        upper_exhausted=bool(upper_count_report is not None),
        passes=("rules", "singles"),
    )


def _report(**kwargs) -> CapacityReport:
    return CapacityReport(**kwargs)


def _run(optimizer, plan: SolvePlan, cap: int, *, singles: bool):
    """Return ``(frames, timed_out)`` for one pass."""
    clear_capacity_timeout()
    optimizer._capacity_probe = True
    optimizer._capacity_singles = bool(singles)
    if singles:
        _clear_packing(optimizer)
    try:
        if plan.kind == "game_script" and singles:
            frames = _game_singles(optimizer, plan, cap)
        elif plan.kind == "tiered" and singles:
            frames = _window_singles(optimizer, plan, cap)
        elif plan.kind == "game_script":
            frames = _game_portfolio(optimizer, plan, cap)
        elif plan.kind == "tiered":
            frames = optimize_tiered_salaries(
                optimizer,
                total_n=int(cap),
                leave_money_ratio=float(plan.leave_money_ratio),
                standard_cap=int(plan.standard_cap),
                discount_cap=int(plan.discount_cap),
                discount_floor=int(plan.discount_floor),
            )
        else:
            frames = _generate(optimizer, cap)
    except (GenerateLineupException, RuntimeError):
        frames = []
    frames = list(frames or [])
    if len(frames) > cap:
        frames = frames[:cap]
    return frames, capacity_timed_out()


def _generate(optimizer, count: int, **kwargs):
    count = int(count)
    if count < 1:
        return []
    optimizer._capacity_probe = True
    try:
        return list(optimizer.generate_lineups(min(count, DEFAULT_CAP), **kwargs))
    except (GenerateLineupException, RuntimeError):
        return []


def _clear_packing(optimizer) -> None:
    """Drop cross-lineup caps. Single-lineup rules stay."""
    optimizer.max_exposure = None
    optimizer.min_exposure = None
    if hasattr(optimizer, "set_max_repeating_players"):
        optimizer.set_max_repeating_players(None)
    else:
        optimizer.max_repeating_players = None
    if hasattr(optimizer, "slate_aware_gpp"):
        optimizer.slate_aware_gpp = False
    if hasattr(optimizer, "kicker_max_exposure"):
        optimizer.kicker_max_exposure = None
    optimizer._capacity_singles = True
    optimizer._capacity_probe = True


def _window_singles(optimizer, plan: SolvePlan, cap: int):
    """Distinct lineups that fit either tier's salary window."""
    windows = (
        (STANDARD_TIER_FLOOR, int(plan.standard_cap)),
        (int(plan.discount_floor), int(plan.discount_cap)),
    )
    return _union_searches(
        optimizer,
        cap,
        (
            {"min_salary_cap": low, "max_salary_cap": high}
            for low, high in windows
        ),
    )


def _game_portfolio(optimizer, plan: SolvePlan, cap: int):
    buckets = _scaled_buckets(plan, cap)
    if not buckets:
        return []
    return list(optimize_game_scripts(optimizer, buckets))


def _game_singles(optimizer, plan: SolvePlan, cap: int):
    """Per-game lineups with that game's rules, ignoring cross-game overlap."""
    _apply_game_script_singles(optimizer)
    searches = []
    for game, _count, salary in plan.buckets:
        away, home = str(game).split("@")
        searches.append(
            {
                "max_salary_cap": int(salary),
                "min_salary_cap": None,
                "quarterback_teams": (away.strip().upper(), home.strip().upper()),
                "game_script": f"{away.strip().upper()}@{home.strip().upper()}",
            }
        )
    return _union_searches(optimizer, cap, searches)


def _apply_game_script_singles(optimizer) -> None:
    """The per-lineup game-script rules. Overlap across games stays off."""
    optimizer.set_qb_stack(True, pass_catchers=1, bring_back=True, bring_back_position="SKILL")
    optimizer.primary_stack = False
    optimizer.mini_stack = False
    optimizer.rb_dst_favorite = False
    optimizer.ban_underdog_rbs = False
    optimizer.primary_bring_back_wr1 = False
    optimizer.require_wr_in_stack = False
    optimizer.slate_aware_gpp = False
    optimizer.flex_positions = GAME_SCRIPT_FLEX
    optimizer.exclude_vs_dst = True
    optimizer.set_max_repeating_players(None)
    optimizer._capacity_singles = True
    optimizer._capacity_probe = True


def _union_searches(optimizer, cap: int, searches: Sequence[dict]):
    """Lineups from each search, counted once. Stops at ``cap`` or the deadline."""
    seen = set()
    frames = []
    for options in searches:
        if len(frames) >= cap or capacity_timed_out():
            break
        generate_kwargs = {}
        if "min_salary_cap" in options:
            optimizer.set_min_salary_cap(options["min_salary_cap"])
        if "max_salary_cap" in options:
            optimizer.set_max_salary_cap(options["max_salary_cap"])
        if "quarterback_teams" in options:
            generate_kwargs["quarterback_teams"] = options["quarterback_teams"]
        if "game_script" in options:
            generate_kwargs["game_script"] = options["game_script"]
        got = _generate(optimizer, cap, **generate_kwargs)
        if capacity_timed_out() and len(got) >= cap:
            # The cap was filled; the clock does not erase those lineups.
            clear_capacity_timeout()
        for frame in got:
            identity = _lineup_identity(frame)
            if identity in seen:
                continue
            seen.add(identity)
            frames.append(frame)
            if len(frames) >= cap:
                break
    return frames


def _scaled_buckets(plan: SolvePlan, cap: int):
    if not plan.buckets:
        return []
    counts = even_lineup_counts(len(plan.buckets), int(cap))
    buckets = []
    for (game, _count, salary), count in zip(plan.buckets, counts):
        if int(count) <= 0:
            continue
        buckets.append(GameScriptBucket(str(game), int(count), int(salary)))
    return buckets


def _lineup_identity(frame: pd.DataFrame):
    if frame is None or frame.empty or "ID" not in frame.columns:
        return ()
    slots = (
        frame["Slot"].astype(str).tolist()
        if "Slot" in frame.columns
        else [""] * len(frame)
    )
    return tuple(sorted(zip(slots, frame["ID"].astype(str).tolist())))


def _captain_share_active(optimizer) -> bool:
    frame = getattr(optimizer, "_player_df", None)
    if frame is None or frame.empty or "CaptainExposure" not in frame.columns:
        return False
    values = pd.to_numeric(frame["CaptainExposure"], errors="coerce")
    return bool(values.notna().any())


def _game_count(frame) -> int:
    if frame is None or getattr(frame, "empty", True):
        return 0
    return len(slate_games_from_pool(frame))


def _cache_key(optimizer, plan: SolvePlan, cap: int, budget_seconds: Optional[float]) -> str:
    payload = {
        "pool": pool_fingerprint(getattr(optimizer, "_player_df", None)),
        "locked": sorted(str(player_id) for player_id in (optimizer.locked_player_ids or ())),
        "settings": _settings(optimizer),
        "plan": {
            "kind": plan.kind,
            "leave_money_ratio": plan.leave_money_ratio,
            "discount_cap": plan.discount_cap,
            "discount_floor": plan.discount_floor,
            "standard_cap": plan.standard_cap,
            "buckets": list(plan.buckets),
        },
        "cap": int(cap),
        "budget": None if budget_seconds is None else round(float(budget_seconds), 3),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _settings(optimizer) -> dict:
    found = {}
    for name in _SETTING_ATTRS:
        if not hasattr(optimizer, name):
            continue
        found[name] = _jsonable(getattr(optimizer, name))
    return found


def _jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=lambda key: str(key))}
    return str(value)


def _snapshot(optimizer):
    saved = {key: _clone(value) for key, value in optimizer.__dict__.items()}
    exposures = [
        (player, getattr(player, "min_exposure", None), getattr(player, "max_exposure", None))
        for player in list(getattr(optimizer, "_players", None) or [])
    ]
    return saved, exposures


def _restore(optimizer, saved, exposures) -> None:
    for key in list(optimizer.__dict__):
        if key not in saved:
            del optimizer.__dict__[key]
    for key, value in saved.items():
        optimizer.__dict__[key] = _clone(value)
    for player, minimum, maximum in exposures:
        player.min_exposure = minimum
        player.max_exposure = maximum


def _clone(value):
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _word(number: int) -> str:
    return "lineup" if int(number) == 1 else "lineups"


def _bound_phrase(parts) -> str:
    if len(parts) == 1:
        label, number = parts[0]
        return f"{label} {number} unique {_word(number)}"
    body = " and ".join(f"{label} {number}" for label, number in parts)
    return f"{body} unique lineups"
