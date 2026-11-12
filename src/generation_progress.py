"""Text and percent for the lineup progress bar.

The optimizer calls back with how many lineups are built. These functions
only format that count. They do not read projections or change a lineup.
"""

from __future__ import annotations


def format_lineup_progress(built: int, total: int, elapsed_seconds: float) -> str:
    """``Lineups built X/N`` plus the elapsed clock.

    Elapsed time is whole seconds, floored, and never negative. Hours appear
    only after 60 minutes.
    """
    done = max(0, int(built))
    requested = max(0, int(total))
    whole = int(max(0.0, float(elapsed_seconds)))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        clock = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes:
        clock = f"{minutes}m {seconds:02d}s"
    else:
        clock = f"{seconds}s"
    return f"Lineups built {done}/{requested} · {clock}"


def lineup_progress_percent(built: int, total: int) -> int:
    """Integer percent for ``st.progress``, from 0 through 100.

    The first lineup moves the bar off zero. A count past ``total`` stays at
    100. A non-positive total stays at 0.
    """
    requested = int(total)
    if requested <= 0:
        return 0
    done = int(built)
    if done <= 0:
        return 0
    if done >= requested:
        return 100
    return min(100, max(1, round(done * 100 / requested)))
