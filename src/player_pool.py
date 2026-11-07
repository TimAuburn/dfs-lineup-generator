"""Collapse a DraftKings pool to one row per player and apply pool-page edits."""

from __future__ import annotations

import pandas as pd

from .data_loader import projection_deltas

CHOICE_COLUMNS = (
    "Include",
    "Captain",
    "Lock",
    "ProjectionOverride",
    "Ownership",
    "CaptainExposure",
)
# Optional DraftKings scoring columns. Present only after that adapter runs.
# The editor keeps them so a later whitelist cannot drop the pool contract.
OPTIONAL_SCORING_COLUMNS = ("PPR", "Scoring", "ScoringNote")
_POSITION_ORDER = ("QB", "RB", "WR", "TE", "FLEX", "DST", "K")
_FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})


def person_key(name: object, team: object) -> tuple[str, str]:
    return (str(name or "").strip().casefold(), str(team or "").strip().upper())


def collapse_for_editor(pool: pd.DataFrame, *, showdown: bool) -> pd.DataFrame:
    """One row per player. Showdown salary is the flex price."""
    if pool is None or pool.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, group in pool.groupby(_group_key(pool), sort=False):
        row = group.loc[_representative_index(group, showdown)]
        rows.append(_editor_row(row, group, showdown=showdown))
    frame = pd.DataFrame(rows)
    frame.attrs.update(dict(pool.attrs))
    return frame


def stamp_projection_baseline(pool: pd.DataFrame) -> pd.DataFrame:
    """Record the clean projection base without replacing one already stored.

    The base is the Sleeper weekly number when that cell is filled, otherwise
    the DraftKings average. A projection that already includes Vegas or wind
    is not a base. An explicit override is stored separately and is not a base.
    """
    if pool is None:
        return pool
    updated = pool.copy()
    if updated.empty:
        return updated
    if "ProjectionBaseline" in updated.columns:
        existing = pd.to_numeric(updated["ProjectionBaseline"], errors="coerce")
    else:
        existing = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    derived = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    for column in ("Sleeper", "AvgPointsPerGame", "Projection"):
        if column not in updated.columns:
            continue
        derived = derived.fillna(pd.to_numeric(updated[column], errors="coerce"))
    updated["ProjectionBaseline"] = existing.fillna(derived)
    return updated


def projections_for_solve(
    pool: pd.DataFrame,
    lines: dict | None,
    weather_by_home: dict | None = None,
) -> pd.DataFrame:
    """Projection the optimizer should receive for these odds and this wind.

    Automatic rows are the clean baseline plus the current Vegas and wind
    delta. ``ProjectionOverride`` replaces that number exactly. The baseline
    is never the adjusted result, so a second call does not compound, and a
    removed total or a calmer wind returns the baseline.
    """
    updated = stamp_projection_baseline(pool)
    if updated is None or updated.empty or "Projection" not in updated.columns:
        return updated
    baseline = pd.to_numeric(updated["ProjectionBaseline"], errors="coerce")
    working = updated.copy()
    working["Projection"] = baseline.to_numpy()
    deltas = projection_deltas(working, lines or {}, weather_by_home or {})
    delta_values = pd.to_numeric(
        pd.Series(deltas.to_numpy(), index=updated.index),
        errors="coerce",
    ).fillna(0.0)
    automatic = (baseline + delta_values).round(2)
    if "ProjectionOverride" in updated.columns:
        override = pd.to_numeric(updated["ProjectionOverride"], errors="coerce")
    else:
        override = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    updated["Delta"] = delta_values.to_numpy()
    updated["Projection"] = automatic.where(override.isna(), override).to_numpy()
    return updated


def apply_pool_edits(
    pool: pd.DataFrame,
    edited: pd.DataFrame,
    computed: pd.DataFrame,
    *,
    showdown: bool,
) -> pd.DataFrame:
    """Write include, captain, lock, ownership, captain share, and projection onto every salary row."""
    updated = pool.copy()
    if updated.empty or edited is None or edited.empty:
        return updated
    for column, default in (("Include", True), ("Captain", True), ("Lock", False)):
        if column not in updated.columns:
            updated[column] = default
    if "ProjectionOverride" not in updated.columns:
        updated["ProjectionOverride"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Ownership" not in updated.columns:
        updated["Ownership"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if showdown and "CaptainExposure" not in updated.columns:
        updated["CaptainExposure"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Projection" not in updated.columns:
        updated["Projection"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")

    computed_projection = _projection_by_key(computed)
    for _, edit in edited.iterrows():
        key = person_key(edit.get("Name"), edit.get("TeamAbbrev"))
        mask = pd.Series(
            [person_key(name, team) == key for name, team in _key_list(updated)],
            index=updated.index,
        )
        if not bool(mask.any()):
            continue
        include = _as_bool(edit.get("Include"), default=True)
        captain = _as_bool(edit.get("Captain"), default=True) if showdown else True
        lock = _as_bool(edit.get("Lock"), default=False)
        updated.loc[mask, "Include"] = include
        updated.loc[mask, "Captain"] = captain
        updated.loc[mask, "Lock"] = lock
        ownership = _percent_to_fraction(edit.get("Ownership %"))
        updated.loc[mask, "Ownership"] = pd.NA if pd.isna(ownership) else ownership
        if showdown:
            captain_rate = _percent_to_fraction(edit.get("Captain %"))
            updated.loc[mask, "CaptainExposure"] = (
                pd.NA if pd.isna(captain_rate) else captain_rate
            )
        projection = pd.to_numeric(edit.get("Projection"), errors="coerce")
        auto = computed_projection.get(key)
        if pd.isna(projection):
            continue
        updated.loc[mask, "Projection"] = float(projection)
        if auto is not None and abs(float(projection) - auto) < 0.011:
            updated.loc[mask, "ProjectionOverride"] = pd.NA
        else:
            updated.loc[mask, "ProjectionOverride"] = float(projection)
    return updated


def generation_pool(pool: pd.DataFrame, *, showdown: bool) -> pd.DataFrame:
    """Drop players who are not included, and captain rows that are turned off."""
    if pool is None or pool.empty:
        return pool
    frame = pool.copy()
    if "Include" in frame.columns:
        excluded = {
            person_key(name, team)
            for name, team, include in zip(
                frame["Name"], frame["TeamAbbrev"], frame["Include"]
            )
            if not _as_bool(include, default=True)
        }
        if excluded:
            keep = pd.Series(
                [
                    person_key(name, team) not in excluded
                    for name, team in _key_list(frame)
                ],
                index=frame.index,
            )
            frame = frame.loc[keep].copy()
    if showdown and "Captain" in frame.columns and "RosterSlot" in frame.columns and not frame.empty:
        blocked = {
            person_key(name, team)
            for name, team, captain in zip(
                frame["Name"], frame["TeamAbbrev"], frame["Captain"]
            )
            if not _as_bool(captain, default=True)
        }
        if blocked:
            slot = frame["RosterSlot"].astype(str).str.strip().str.upper()
            keep = pd.Series(
                [
                    not (person_key(name, team) in blocked and roster == "CPT")
                    for name, team, roster in zip(frame["Name"], frame["TeamAbbrev"], slot)
                ],
                index=frame.index,
            )
            frame = frame.loc[keep].copy()
    return frame


def locked_player_ids(pool: pd.DataFrame, *, showdown: bool) -> list[str]:
    """Ids to force in. Showdown locks the flex row so the player is not forced to captain."""
    if pool is None or pool.empty or "Lock" not in pool.columns or "ID" not in pool.columns:
        return []
    locked: list[str] = []
    seen: set[tuple[str, str]] = set()
    for _, group in pool.groupby(_group_key(pool), sort=False):
        row = group.loc[_representative_index(group, showdown)]
        if not _as_bool(row.get("Include"), default=True):
            continue
        if not _as_bool(row.get("Lock"), default=False):
            continue
        key = person_key(row.get("Name"), row.get("TeamAbbrev"))
        if key in seen:
            continue
        seen.add(key)
        locked.append(str(row["ID"]))
    return locked


def game_label(raw: object) -> str:
    """``AWAY@HOME`` from a DraftKings game info cell."""
    if raw is None:
        return ""
    try:
        if pd.isna(raw):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(raw).strip()
    if not text or text in {"In Progress", "Final", "nan"}:
        return ""
    try:
        teams, *_rest = text.rsplit(" ", 3)
        away, home = teams.split("@")
    except ValueError:
        return ""
    away = away.strip().upper()
    home = home.strip().upper()
    if not away or not home:
        return ""
    return f"{away}@{home}"


def position_choices(frame: pd.DataFrame) -> list[str]:
    """Football positions present on the editor table, in lineup order."""
    if frame is None or frame.empty or "Position" not in frame.columns:
        return []
    found: set[str] = set()
    for value in frame["Position"]:
        found.update(_position_tokens(value))
    if found & _FLEX_POSITIONS:
        found.add("FLEX")
    ordered = [position for position in _POSITION_ORDER if position in found]
    ordered.extend(sorted(found - set(_POSITION_ORDER)))
    return ordered


def game_choices(frame: pd.DataFrame) -> list[str]:
    """Matchups on the editor table, in first-seen order."""
    if frame is None or frame.empty or "Game Info" not in frame.columns:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for raw in frame["Game Info"]:
        label = game_label(raw)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def filter_editor_frame(
    frame: pd.DataFrame,
    *,
    name_query: str = "",
    positions: list[str] | None = None,
    games: list[str] | None = None,
) -> pd.DataFrame:
    """Keep editor rows that match the name, position, and game filters.

    An empty filter shows every row.
    """
    if frame is None or frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    query = str(name_query or "").strip().casefold()
    if query and "Name" in frame.columns:
        mask &= frame["Name"].astype(str).str.casefold().str.contains(query, regex=False)
    chosen_positions = {str(position).strip().upper() for position in (positions or []) if str(position).strip()}
    if "FLEX" in chosen_positions:
        chosen_positions.discard("FLEX")
        chosen_positions.update(_FLEX_POSITIONS)
    if chosen_positions and "Position" in frame.columns:
        mask &= frame["Position"].map(lambda value: bool(_position_tokens(value) & chosen_positions))
    chosen_games = {str(game).strip() for game in (games or []) if str(game).strip()}
    if chosen_games and "Game Info" in frame.columns:
        mask &= frame["Game Info"].map(game_label).isin(chosen_games)
    return frame.loc[mask].copy()


def set_included(pool: pd.DataFrame, shown: pd.DataFrame, included: bool) -> pd.DataFrame:
    """Set Include on every salary row for the players in ``shown``."""
    return _set_player_flag(pool, shown, "Include", included, default=True)


def set_captain(pool: pd.DataFrame, shown: pd.DataFrame, captain: bool) -> pd.DataFrame:
    """Set Captain on every salary row for the players in ``shown``."""
    return _set_player_flag(pool, shown, "Captain", captain, default=True)


def _set_player_flag(
    pool: pd.DataFrame,
    shown: pd.DataFrame,
    column: str,
    value: bool,
    *,
    default: bool,
) -> pd.DataFrame:
    updated = pool.copy()
    if updated.empty or shown is None or shown.empty:
        return updated
    if column not in updated.columns:
        updated[column] = default
    keys = {
        person_key(name, team)
        for name, team in zip(shown["Name"], shown["TeamAbbrev"])
    }
    mask = pd.Series(
        [person_key(name, team) in keys for name, team in _key_list(updated)],
        index=updated.index,
    )
    updated.loc[mask, column] = bool(value)
    return updated


def carry_choices(previous: pd.DataFrame | None, parsed: pd.DataFrame) -> pd.DataFrame:
    """Keep include, captain, lock, ownership, captain share, and projection overrides for the same players."""
    if previous is None or previous.empty or parsed is None or parsed.empty:
        return parsed
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in previous.iterrows():
        lookup.setdefault(person_key(row.get("Name"), row.get("TeamAbbrev")), row)
    updated = parsed.copy()
    for column in CHOICE_COLUMNS:
        if column not in previous.columns:
            continue
        if column not in updated.columns:
            updated[column] = pd.NA
        values = []
        for name, team in _key_list(updated):
            prior = lookup.get(person_key(name, team))
            values.append(pd.NA if prior is None else prior.get(column))
        updated[column] = values
    if "ProjectionOverride" in updated.columns:
        override = pd.to_numeric(updated["ProjectionOverride"], errors="coerce")
        if "Projection" in updated.columns:
            base = pd.to_numeric(updated["Projection"], errors="coerce")
            updated["Projection"] = override.where(override.notna(), base)
    return updated


def _editor_row(row: pd.Series, group: pd.DataFrame, *, showdown: bool) -> dict:
    ownership = _fraction_to_percent(row.get("Ownership"))
    projection = row.get("ProjectionOverride", pd.NA)
    if pd.isna(projection):
        projection = row.get("Projection")
    record = {
        "Include": _as_bool(row.get("Include"), default=True),
        "Lock": _as_bool(row.get("Lock"), default=False),
        "Name": row.get("Name"),
        "Position": row.get("Position"),
        "TeamAbbrev": row.get("TeamAbbrev"),
        "Salary": row.get("Salary"),
        "Ownership %": ownership,
        "Projection": projection,
    }
    if showdown:
        record = {
            "Include": record["Include"],
            "Captain": _as_bool(row.get("Captain"), default=True),
            "Captain %": _fraction_to_percent(row.get("CaptainExposure")),
            "Lock": record["Lock"],
            **{key: value for key, value in record.items() if key not in {"Include", "Lock"}},
        }
    for column in (
        "Sleeper",
        "Delta",
        "AvgPointsPerGame",
        "Injury",
        "DepthOrder",
        "DepthPosition",
        "Game Info",
        *OPTIONAL_SCORING_COLUMNS,
    ):
        if column in group.columns:
            record[column] = row.get(column)
    return record


def _representative_index(group: pd.DataFrame, showdown: bool):
    if showdown and "RosterSlot" in group.columns:
        flex = group.index[group["RosterSlot"].astype(str).str.strip().str.upper() != "CPT"]
        if len(flex):
            return flex[0]
    return group.index[0]


def _group_key(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [f"{name}\0{team}" for name, team in _key_list(frame)],
        index=frame.index,
    )


def _key_list(frame: pd.DataFrame) -> list[tuple[str, str]]:
    names = frame["Name"] if "Name" in frame.columns else [""] * len(frame)
    teams = frame["TeamAbbrev"] if "TeamAbbrev" in frame.columns else [""] * len(frame)
    return [person_key(name, team) for name, team in zip(names, teams)]


def _projection_by_key(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    found: dict[tuple[str, str], float] = {}
    if frame is None or frame.empty or "Projection" not in frame.columns:
        return found
    for _, row in frame.iterrows():
        value = pd.to_numeric(row.get("Projection"), errors="coerce")
        if pd.isna(value):
            continue
        found[person_key(row.get("Name"), row.get("TeamAbbrev"))] = float(value)
    return found


def _fraction_to_percent(value: object):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    number = float(number)
    if number > 1:
        return number
    return number * 100.0


def _percent_to_fraction(value: object):
    if value is None or (isinstance(value, str) and not value.strip()):
        return pd.NA
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return pd.NA
    return float(number) / 100.0


def _position_tokens(value: object) -> set[str]:
    tokens = set()
    for part in str(value or "").split("/"):
        token = part.strip().upper()
        if token and token not in {"FLEX", "CPT", "UTIL", "NAN"}:
            tokens.add(token)
    return tokens


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)
