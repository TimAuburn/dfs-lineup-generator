"""One player-name normalizer for every join.

DraftKings, Sleeper, and the editor do not spell names the same way. A hyphen,
a period, an apostrophe, a generational suffix, or an accent used to make an
injury or a projection miss, and it used to split locks and exposure. Every
join calls ``normalize_player_name`` and keys the result with the team.

The key is ``(normalized name, team)``. A defense is ``("dst", team)`` because
the nickname is not shared across feeds. Two players with the same normalized
name stay separate when their teams differ.

Team abbreviations use the map already in ``slate_week.dk_team`` (WSH/WAS,
JAC/JAX, LA/LAR, and the older STL, SD, and OAK codes). No other aliases are
added.

``person_key`` is computed from the name, team, and position columns. It is
not written to the Sleeper cache, the editor, or the archive. Pins, locks,
and projection overrides live on the salary rows and are re-keyed on each
read, so this function does not need a pin migration. A saved Sleeper index
stores names already passed through an older normalizer. Matching runs those
stored names through this function again. The new form is idempotent, and the
previous hyphenated form (``amon-ra``) becomes the spaced form. The slate
matching contract (``MATCH_CACHE_VERSION``, slate key, fingerprint) is unchanged.
"""

from __future__ import annotations

import re
import unicodedata

from .slate_week import dk_team

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_STATUS_TAG_RE = re.compile(
    r"\((?:q|o|d|ir|ir-r|irr|pup|sus|na|nwt|p|out|doubtful|questionable|probable|suspended)\)",
    re.IGNORECASE,
)
_DASHES = str.maketrans({
    "-": " ",
    "‐": " ",
    "‑": " ",
    "–": " ",
    "—": " ",
    "−": " ",
})
_DROPPED_MARKS = str.maketrans("", "", ".'’‘`ʼ′´")
_DST_POSITIONS = frozenset({"DST", "DEF"})
_SLOT_POSITIONS = frozenset({"CPT", "FLEX", "UTIL"})


def normalize_player_name(name: object) -> str:
    """Casefold a player name so punctuation and suffix variants join.

    Hyphens become spaces. Periods and apostrophes are removed. Accents are
    stripped. Whitespace collapses. A trailing Jr, Sr, II, III, IV, or V is
    removed. A parenthetical injury tag such as ``(Q)`` is not part of the name.
    """
    if name is None:
        return ""
    try:
        if name != name:  # NaN
            return ""
    except TypeError:
        pass
    text = str(name).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return ""
    text = _STATUS_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().translate(_DASHES).translate(_DROPPED_MARKS)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    parts = [part for part in text.split() if part]
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _position_token(position: object) -> str:
    return str(position or "").split("/")[0].strip().upper()


def person_key(name: object, team: object, position: object = None) -> tuple[str, str]:
    """Identity for one player: normalized name and team, or team for a defense."""
    club = dk_team(team)
    if _position_token(position) in _DST_POSITIONS:
        return ("dst", club)
    return (normalize_player_name(name), club)


def player_person_key(player: object) -> tuple[str, str]:
    """``person_key`` for a pydfs player, using the football position when it is known."""
    original = getattr(player, "original_positions", None) or ()
    positions = tuple(original) if original else tuple(getattr(player, "positions", None) or ())
    position = positions[0] if positions else None
    if _position_token(position) in _SLOT_POSITIONS:
        position = None
    return person_key(
        getattr(player, "full_name", ""),
        getattr(player, "team", ""),
        position,
    )
