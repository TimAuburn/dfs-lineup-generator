"""Contest preset definitions and solver configuration for Classic / Showdown."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer import BaseNFLOptimizer


class ContestFormat(str, Enum):
    CLASSIC = "classic"
    SHOWDOWN = "showdown"


class ContestPreset(str, Enum):
    CLASSIC_CASH = "classic_cash"
    CLASSIC_SMALL_GPP = "classic_small_gpp"
    CLASSIC_LARGE_GPP = "classic_large_gpp"
    SHOWDOWN_GPP = "showdown_gpp"
    SHOWDOWN_CASH = "showdown_cash"


CLASSIC_PRESETS = (
    ContestPreset.CLASSIC_CASH,
    ContestPreset.CLASSIC_SMALL_GPP,
    ContestPreset.CLASSIC_LARGE_GPP,
)

SHOWDOWN_PRESETS = (
    ContestPreset.SHOWDOWN_CASH,
    ContestPreset.SHOWDOWN_GPP,
)

PRESET_LABELS = {
    ContestPreset.CLASSIC_CASH: "Cash (50/50s & Double-Ups)",
    ContestPreset.CLASSIC_SMALL_GPP: "Small-Field / Single-Entry GPP",
    ContestPreset.CLASSIC_LARGE_GPP: "Large-Field Multi-Entry GPP (20–150 Max)",
    ContestPreset.SHOWDOWN_GPP: "Showdown GPP",
    ContestPreset.SHOWDOWN_CASH: "Showdown Cash",
}

# Position-specific relative SDs for large-field Classic GPP (within ~12–18%).
LARGE_FIELD_POSITION_SDS = {
    "QB": 0.12,
    "RB": 0.15,
    "WR": 0.18,
    "TE": 0.16,
    "DST": 0.14,
    "K": 0.14,
}


@dataclass(frozen=True)
class PresetConfig:
    """Declarative solver rules for a contest preset."""

    format: ContestFormat
    label: str
    lineup_count_range: Tuple[int, int]
    default_lineup_count: int
    # Jitter: either fixed 0.0, a (min, max) uniform relative band, or position SDs.
    jitter_pct: Optional[Tuple[float, float]] = None
    position_jitter_sds: Optional[Dict[str, float]] = None
    enable_qb_stack: bool = False
    stack_pass_catchers: int = 1
    enable_bring_back: bool = False
    bring_back_position: str = "WR"
    # Double stacks are QB + WR + WR/TE. Single stacks stay QB + WR/TE.
    require_wr_in_stack: bool = False
    # FLEX eligibility. None keeps RB, WR, and TE. ("RB", "WR") blocks a second tight end.
    flex_positions: Optional[Tuple[str, ...]] = None
    # Classic cash drops WR/RB/TE under this salary unless the user locks them.
    min_skill_salary: Optional[int] = None
    # Fewer than 10 games tightens the repeating-player cap; 4 games lowers stack exposure.
    slate_aware_gpp: bool = False
    exclude_vs_dst: bool = False
    max_repeating_players: Optional[int] = None
    # Cumulative ownership cap as a fraction (1.25 == 125%). None = do not apply.
    max_cumulative_ownership: Optional[float] = None
    # Fraction of the batch one player may occupy. None leaves exposure uncapped.
    max_player_exposure: Optional[float] = None
    tiered_salary: bool = False
    tiered_leave_ratio: float = 0.40
    tiered_discount_floor: int = 48_000
    tiered_discount_cap: int = 49_500
    # Showdown-specific
    min_salary_cap: Optional[int] = None
    max_salary_cap: Optional[int] = None
    min_teams: Optional[int] = None
    game_script_correlation: bool = False
    cpt_qb_pass_catchers: int = 2
    max_k_dst_combined: Optional[int] = None
    # Showdown roster cap. 1 blocks both teams' kickers in one lineup. None = no cap.
    max_kickers: Optional[int] = None
    # Showdown cash: at most one defense, same idea as the kicker cap.
    max_dst: Optional[int] = None
    allow_kicker_captain: bool = False
    # Showdown cash captain pool. None keeps the default CPT positions.
    captain_positions: Optional[Tuple[str, ...]] = None
    captain_min_salary: Optional[int] = None
    flex_min_salary: Optional[int] = None
    description: str = ""


PRESET_CONFIGS: Dict[ContestPreset, PresetConfig] = {
    ContestPreset.CLASSIC_CASH: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_CASH],
        lineup_count_range=(1, 1),
        default_lineup_count=1,
        jitter_pct=(0.0, 0.0),
        enable_qb_stack=False,
        flex_positions=("RB", "WR"),
        min_skill_salary=3_800,
        exclude_vs_dst=False,
        max_repeating_players=None,
        description=(
            "Single deterministic cash lineup. No stacks. FLEX is WR or RB. "
            "WR, RB, and TE under $3,800 are excluded unless locked."
        ),
    ),
    ContestPreset.CLASSIC_SMALL_GPP: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_SMALL_GPP],
        lineup_count_range=(1, 5),
        default_lineup_count=3,
        jitter_pct=(0.05, 0.08),
        enable_qb_stack=True,
        stack_pass_catchers=1,
        enable_bring_back=False,
        exclude_vs_dst=True,
        max_repeating_players=6,
        description="Small-field GPP: light jitter, QB + 1 WR/TE, exclude vs DST, max 6 repeats.",
    ),
    ContestPreset.CLASSIC_LARGE_GPP: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_LARGE_GPP],
        lineup_count_range=(20, 150),
        default_lineup_count=20,
        jitter_pct=(0.12, 0.18),
        position_jitter_sds=LARGE_FIELD_POSITION_SDS,
        enable_qb_stack=True,
        stack_pass_catchers=2,
        enable_bring_back=True,
        bring_back_position="SKILL",
        require_wr_in_stack=True,
        flex_positions=("RB", "WR"),
        slate_aware_gpp=True,
        exclude_vs_dst=True,
        max_repeating_players=5,
        max_cumulative_ownership=1.25,
        max_player_exposure=0.50,
        tiered_salary=True,
        tiered_leave_ratio=0.40,
        tiered_discount_floor=49_200,
        tiered_discount_cap=49_700,
        description=(
            "Large-field multi-entry: QB + WR + WR/TE, skill bring-back, "
            "FLEX is WR or RB, slate-aware overlap, 50% player exposure, "
            "salary from $49,200 to $50,000."
        ),
    ),
    ContestPreset.SHOWDOWN_GPP: PresetConfig(
        format=ContestFormat.SHOWDOWN,
        label=PRESET_LABELS[ContestPreset.SHOWDOWN_GPP],
        lineup_count_range=(1, 150),
        default_lineup_count=10,
        jitter_pct=(0.08, 0.14),
        min_salary_cap=48_000,
        max_salary_cap=49_800,
        min_teams=2,
        game_script_correlation=True,
        cpt_qb_pass_catchers=2,
        max_k_dst_combined=2,
        max_kickers=1,
        max_repeating_players=3,
        allow_kicker_captain=False,
        flex_min_salary=1_200,
        description=(
            "Showdown GPP: salary window, both teams, CPT QB needs 2 WR/TE "
            "plus an opposing skill player, no K or DST captains, at most one kicker, "
            "no FLEX under $1,200, drop players at 1.0 points or less, max 3 repeats."
        ),
    ),
    ContestPreset.SHOWDOWN_CASH: PresetConfig(
        format=ContestFormat.SHOWDOWN,
        label=PRESET_LABELS[ContestPreset.SHOWDOWN_CASH],
        lineup_count_range=(1, 5),
        default_lineup_count=1,
        jitter_pct=(0.0, 0.0),
        min_salary_cap=49_500,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_k_dst_combined=2,
        max_kickers=1,
        max_dst=1,
        max_repeating_players=None,
        allow_kicker_captain=False,
        captain_positions=("QB", "RB", "WR"),
        captain_min_salary=8_000,
        flex_min_salary=2_000,
        description=(
            "Showdown Cash: spend $49,500–$50,000, captain a QB/RB/WR at $8,000+, "
            "no FLEX under $2,000, at most one kicker and one DST."
        ),
    ),
}


def presets_for_format(contest_format: ContestFormat) -> Tuple[ContestPreset, ...]:
    if contest_format is ContestFormat.SHOWDOWN:
        return SHOWDOWN_PRESETS
    return CLASSIC_PRESETS


def get_preset_config(preset: ContestPreset) -> PresetConfig:
    return PRESET_CONFIGS[preset]


# Identity fields stay on the preset. Lineup count is applied separately.
_LOCKED_OVERRIDE_FIELDS = frozenset(
    {"format", "label", "description", "lineup_count_range", "default_lineup_count"}
)


def config_with_overrides(
    config: PresetConfig,
    *,
    lineup_count: int,
    overrides: Dict[str, Any],
) -> PresetConfig:
    """
    Copy ``config`` with solver fields replaced by UI overrides.

    Unknown keys are ignored. ``lineup_count`` becomes the count shown as the
    actual lineup count (the allowed range stays the preset's).
    """
    allowed = {f.name for f in fields(PresetConfig)} - _LOCKED_OVERRIDE_FIELDS
    updates = {key: overrides[key] for key in allowed if key in overrides}
    updates["default_lineup_count"] = int(lineup_count)
    return replace(config, **updates)


def _fmt_pct_band(band: Optional[Tuple[float, float]]) -> str:
    if band is None:
        return "None"
    lo, hi = float(band[0]), float(band[1])
    if lo == 0.0 and hi == 0.0:
        return "None (deterministic)"
    if lo == hi:
        return f"{lo * 100:.0f}%"
    return f"{lo * 100:.0f}–{hi * 100:.0f}%"


FLEX_POSITION_LABELS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("WR or RB", ("RB", "WR")),
    ("WR", ("WR",)),
    ("RB", ("RB",)),
    ("Any", ("RB", "TE", "WR")),
)


def flex_positions_for_label(label: str) -> Tuple[str, ...]:
    """Map a flex radio label to the positions allowed in the FLEX slot."""
    for name, positions in FLEX_POSITION_LABELS:
        if name == label:
            return positions
    raise ValueError("flex label must be WR or RB, WR, RB, or Any")


def flex_position_label(positions: Optional[Tuple[str, ...]]) -> str:
    """Label for a flex-position tuple. Missing values mean the full DraftKings flex."""
    if not positions:
        return "Any"
    wanted = frozenset(str(pos).upper() for pos in positions)
    for name, allowed in FLEX_POSITION_LABELS:
        if frozenset(allowed) == wanted:
            return name
    return ", ".join(sorted(wanted))


def format_preset_summary(config: PresetConfig) -> List[Tuple[str, str]]:
    """
    Build readable (setting, value) rows for UI display from a PresetConfig.

    Single source of truth: derive labels from the config fields / description,
    do not hardcode per-preset copy beyond PresetConfig itself.
    """
    lo, hi = config.lineup_count_range
    rows: List[Tuple[str, str]] = [
        (
            "Lineup count",
            f"{config.default_lineup_count} (allowed {lo}–{hi})",
        ),
    ]

    if config.position_jitter_sds:
        band = _fmt_pct_band(config.jitter_pct)
        rows.append(
            (
                "Projection jitter",
                f"Position SDs (~{band} relative)",
            )
        )
    else:
        rows.append(("Projection jitter", _fmt_pct_band(config.jitter_pct)))

    if config.format is ContestFormat.CLASSIC:
        if config.enable_qb_stack:
            if config.require_wr_in_stack and config.stack_pass_catchers >= 2:
                stack = f"QB + WR + {config.stack_pass_catchers - 1} WR/TE"
            else:
                stack = f"QB + {config.stack_pass_catchers} WR/TE"
            if config.enable_bring_back:
                bring = "skill" if config.bring_back_position == "SKILL" else config.bring_back_position
                stack += f" + {bring} bring-back"
            rows.append(("Stacks", stack))
        else:
            rows.append(("Stacks", "None"))
        if config.flex_positions:
            rows.append(("Flex", flex_position_label(config.flex_positions)))
        if config.min_skill_salary is not None:
            rows.append(("Skill salary floor", f"${config.min_skill_salary:,}"))
        rows.append(("Exclude vs DST", "Yes" if config.exclude_vs_dst else "No"))
        rows.append(
            (
                "Max player exposure",
                (
                    f"{config.max_player_exposure * 100:.0f}%"
                    if config.max_player_exposure is not None
                    else "Off"
                ),
            )
        )
    else:
        min_sal = config.min_salary_cap
        max_sal = config.max_salary_cap
        if min_sal is not None and max_sal is not None:
            rows.append(("Salary window", f"${min_sal:,}–${max_sal:,}"))
        elif min_sal is not None:
            rows.append(("Salary window", f"min ${min_sal:,}"))
        elif max_sal is not None:
            rows.append(("Salary window", f"max ${max_sal:,}"))
        else:
            rows.append(("Salary window", "Full cap"))

        rows.append(
            (
                "Min teams",
                str(config.min_teams) if config.min_teams is not None else "No minimum",
            )
        )
        rows.append(
            (
                "Game-script correlation",
                "Yes" if config.game_script_correlation else "No",
            )
        )
        if config.game_script_correlation:
            rows.append(
                ("CPT QB pass-catchers", str(config.cpt_qb_pass_catchers))
            )
        rows.append(
            (
                "Kicker captain",
                "Yes" if config.allow_kicker_captain else "No",
            )
        )
        if config.captain_positions:
            captain = ", ".join(config.captain_positions)
            if config.captain_min_salary is not None:
                captain = f"{captain} at ${config.captain_min_salary:,}+"
            rows.append(("Captain", captain))
        if config.flex_min_salary is not None:
            rows.append(("Flex floor", f"${config.flex_min_salary:,}"))
        rows.append(
            (
                "Max K+DST combined",
                (
                    str(config.max_k_dst_combined)
                    if config.max_k_dst_combined is not None
                    else "No limit"
                ),
            )
        )
        rows.append(
            (
                "Max kickers",
                str(config.max_kickers) if config.max_kickers is not None else "No limit",
            )
        )
        if config.max_dst is not None:
            rows.append(("Max DST", str(config.max_dst)))

    rows.append(
        (
            "Max repeating players",
            (
                str(config.max_repeating_players)
                if config.max_repeating_players is not None
                else "No limit"
            ),
        )
    )

    if config.max_cumulative_ownership is not None:
        rows.append(
            (
                "Ownership cap",
                f"{config.max_cumulative_ownership * 100:.0f}% cumulative",
            )
        )
    else:
        rows.append(("Ownership cap", "None"))

    rows.append(("Tiered salary", _tiered_salary_label(config)))

    if config.description:
        rows.append(("Strategy notes", config.description))

    return rows


def _tiered_salary_label(config: PresetConfig) -> str:
    if not config.tiered_salary:
        return "Off"
    total = int(config.default_lineup_count)
    standard_n = int(round((1.0 - float(config.tiered_leave_ratio)) * total))
    standard_n = max(0, min(total, standard_n))
    discount_n = total - standard_n
    return (
        f"{standard_n} at $49,800–$50,000; "
        f"{discount_n} at ${int(config.tiered_discount_floor):,}–${int(config.tiered_discount_cap):,}"
    )


def format_settings_comparison(
    preset: PresetConfig,
    actual: PresetConfig,
    extra_rows: Optional[List[Tuple[str, str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Align preset defaults with the settings that will actually be solved.

    Each row is ``(setting, preset default, actual)``. Strategy notes stay on
    the preset description, not in this comparison. ``extra_rows`` appends
    UI-only settings (exposure, non-default jitter seed) in the same shape.
    """
    actual_map = dict(format_preset_summary(actual))
    rows = [
        (setting, default, actual_map.get(setting, "—"))
        for setting, default in format_preset_summary(preset)
        if setting != "Strategy notes"
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return rows


def apply_preset(
    optimizer: "BaseNFLOptimizer",
    preset: ContestPreset,
    *,
    lineup_count: Optional[int] = None,
    jitter_seed: Optional[int] = 42,
) -> PresetConfig:
    """
    Configure ``optimizer`` from a contest preset.

    Jitter is applied later (at generate time) using ``optimizer.jitter_*`` fields
    so editable projections in the UI remain the baseline.
    """
    config = get_preset_config(preset)
    if config.format.value != optimizer.contest_format:
        raise ValueError(
            f"Preset {preset.value} is for {config.format.value}, "
            f"but optimizer is {optimizer.contest_format}"
        )

    lo, hi = config.lineup_count_range
    count = config.default_lineup_count if lineup_count is None else int(lineup_count)
    count = max(lo, min(hi, count))
    optimizer.set_lineup_count(count)

    optimizer.jitter_pct = config.jitter_pct
    optimizer.position_jitter_sds = (
        dict(config.position_jitter_sds) if config.position_jitter_sds else None
    )
    optimizer.jitter_seed = jitter_seed

    optimizer.set_max_repeating_players(config.max_repeating_players)
    optimizer.exclude_vs_dst = config.exclude_vs_dst
    optimizer.max_cumulative_ownership = config.max_cumulative_ownership

    if hasattr(optimizer, "set_qb_stack"):
        optimizer.set_qb_stack(
            enabled=config.enable_qb_stack,
            pass_catchers=config.stack_pass_catchers,
            bring_back=config.enable_bring_back,
            bring_back_position=config.bring_back_position,
        )
    if hasattr(optimizer, "require_wr_in_stack"):
        optimizer.require_wr_in_stack = bool(config.require_wr_in_stack)
        optimizer.slate_aware_gpp = bool(config.slate_aware_gpp)
    if hasattr(optimizer, "flex_positions"):
        optimizer.flex_positions = config.flex_positions
    if hasattr(optimizer, "min_skill_salary"):
        optimizer.min_skill_salary = config.min_skill_salary

    if hasattr(optimizer, "configure_showdown_rules"):
        optimizer.configure_showdown_rules(
            min_salary_cap=config.min_salary_cap,
            max_salary_cap=config.max_salary_cap,
            min_teams=config.min_teams,
            game_script_correlation=config.game_script_correlation,
            cpt_qb_pass_catchers=config.cpt_qb_pass_catchers,
            max_k_dst_combined=config.max_k_dst_combined,
            max_kickers=config.max_kickers,
            allow_kicker_captain=config.allow_kicker_captain,
            captain_positions=config.captain_positions,
            captain_min_salary=config.captain_min_salary,
            flex_min_salary=config.flex_min_salary,
            max_dst=config.max_dst,
        )

    return config
