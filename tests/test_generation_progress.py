"""Progress text is pure. The optimizer hook reports counts and keeps lineups."""

from __future__ import annotations

from pathlib import Path

from src.generation_progress import format_lineup_progress, lineup_progress_percent
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.parser import parse_dk_csv
from src.presets import ContestPreset, apply_preset

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"


def test_progress_text_counts_lineups_and_elapsed_time():
    assert format_lineup_progress(0, 150, 0) == "Lineups built 0/150 · 0s"
    assert format_lineup_progress(1, 150, 9.8) == "Lineups built 1/150 · 9s"
    assert format_lineup_progress(150, 150, 90) == "Lineups built 150/150 · 1m 30s"
    assert format_lineup_progress(12, 20, 3661) == "Lineups built 12/20 · 1h 01m 01s"
    assert format_lineup_progress(3, 10, -4) == "Lineups built 3/10 · 0s"
    assert format_lineup_progress(-2, -5, 4) == "Lineups built 0/0 · 4s"


def test_progress_percent_tracks_the_ratio_and_stays_in_range():
    assert lineup_progress_percent(0, 150) == 0
    assert lineup_progress_percent(1, 150) == 1
    assert lineup_progress_percent(75, 150) == 50
    assert lineup_progress_percent(150, 150) == 100
    assert lineup_progress_percent(151, 150) == 100
    assert lineup_progress_percent(-3, 10) == 0
    assert lineup_progress_percent(1, 0) == 0


def test_progress_hook_is_optional_and_reports_the_count_it_is_given():
    optimizer = NFLClassicOptimizer()
    seen: list[int] = []
    optimizer._note_lineup_progress(3)
    assert seen == []
    optimizer.on_lineup_progress = seen.append
    optimizer._note_lineup_progress(1)
    optimizer._note_lineup_progress(4)
    optimizer.on_lineup_progress = None
    optimizer._note_lineup_progress(9)
    assert seen == [1, 4]


def _ids(lineups) -> list[list[str]]:
    return [list(map(str, lineup["ID"].tolist())) for lineup in lineups]


def test_progress_hook_does_not_change_seed_42_classic_or_showdown_lineups():
    classic = parse_dk_csv(MOCK_CLASSIC)
    plain = NFLClassicOptimizer()
    plain.load_from_dataframe(classic)
    apply_preset(plain, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    plain_ids = _ids(plain.generate_lineups())

    hooked = NFLClassicOptimizer()
    hooked.load_from_dataframe(classic)
    apply_preset(hooked, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    classic_counts: list[int] = []
    hooked.on_lineup_progress = classic_counts.append
    hooked_ids = _ids(hooked.generate_lineups())
    assert hooked_ids == plain_ids
    assert classic_counts == list(range(1, len(hooked_ids) + 1))

    showdown_pool = parse_dk_csv(MOCK_SHOWDOWN)
    show_plain = NFLShowdownOptimizer()
    show_plain.load_from_dataframe(showdown_pool)
    apply_preset(show_plain, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    show_plain_ids = _ids(show_plain.generate_lineups())

    show_hooked = NFLShowdownOptimizer()
    show_hooked.load_from_dataframe(showdown_pool)
    apply_preset(show_hooked, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    show_counts: list[int] = []
    show_hooked.on_lineup_progress = show_counts.append
    show_ids = _ids(show_hooked.generate_lineups())
    assert show_ids == show_plain_ids
    assert show_counts == list(range(1, len(show_ids) + 1))
