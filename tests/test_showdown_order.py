"""Showdown FLEX columns follow salary, then name, then player ID.

The solver row order is left alone. Display rows and the DraftKings upload
file both read the same ordered roster. Classic slots stay where they are.
"""

from __future__ import annotations

import pandas as pd

from src.exporter import lineups_to_dk_upload_csv
from src.lineup_view import lineup_detail_frame
from src.optimizer import showdown_lineup_grid
from src.showdown_order import order_showdown_roster


def _player(slot, name, player_id, salary, team="BUF") -> dict:
    return {
        "Slot": slot,
        "ID": str(player_id),
        "Name": name,
        "Name + ID": f"{name} ({player_id})",
        "Position": "WR",
        "Team": team,
        "Salary": salary,
        "Projection": salary / 1000,
    }


def _frame(rows: list[dict], **attrs) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs["salary"] = int(sum(row["Salary"] for row in rows))
    frame.attrs["projection"] = float(sum(row["Projection"] for row in rows))
    frame.attrs.update(attrs)
    return frame


def _flex_names(lineup: pd.DataFrame) -> list[str]:
    ordered = order_showdown_roster(lineup)
    slots = ordered["Slot"].astype(str).str.strip().str.upper()
    return ordered.loc[slots.eq("FLEX"), "Name"].tolist()


def test_flex_is_salary_descending_and_captain_stays_first():
    lineup = _frame(
        [
            _player("FLEX", "Cheap", "5", 3000),
            _player("FLEX", "Mid", "4", 7000),
            _player("CPT", "Captain", "1", 2000),
            _player("FLEX", "Top", "2", 11000),
            _player("FLEX", "Next", "3", 9000),
            _player("FLEX", "Low", "6", 4000),
        ]
    )
    before = lineup.copy()
    ordered = order_showdown_roster(lineup)

    assert list(ordered["Name"]) == ["Captain", "Top", "Next", "Mid", "Low", "Cheap"]
    assert list(ordered["Slot"]) == ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    assert list(ordered["ID"])[0] == "1"
    assert int(ordered.attrs["salary"]) == int(lineup.attrs["salary"])
    assert float(ordered.attrs["projection"]) == float(lineup.attrs["projection"])
    pd.testing.assert_frame_equal(lineup, before)


def test_equal_salaries_use_name_then_id():
    lineup = _frame(
        [
            _player("FLEX", "zack", "30", 8000),
            _player("CPT", "Captain", "1", 15000),
            _player("FLEX", "bob", "20", 8000),
            _player("FLEX", "Amy", "10", 8000),
            _player("FLEX", "alex", "300", 5000),
            _player("FLEX", "ALEX", "100", 5000),
            _player("FLEX", "Alex", "200", 5000),
        ]
    )

    assert _flex_names(lineup) == ["Amy", "bob", "zack", "ALEX", "Alex", "alex"]
    assert list(order_showdown_roster(lineup)["ID"]) == [
        "1",
        "10",
        "20",
        "30",
        "100",
        "200",
        "300",
    ]


def test_shared_player_column_follows_the_other_salaries():
    """The same flex player lands in one column whenever the salaries match."""
    shared = _player("FLEX", "Shared", "s", 7000)
    companions = [
        _player("FLEX", "High", "h", 9000),
        _player("FLEX", "Mid", "m", 8000),
        _player("FLEX", "Low", "l", 5000),
        _player("FLEX", "Lower", "w", 4000),
    ]
    captains = [
        _player("CPT", "Captain A", "c1", 15000, team="KC"),
        _player("CPT", "Captain B", "c2", 11000, team="BUF"),
    ]
    permutations = [
        [captains[0], shared, companions[0], companions[3], companions[1], companions[2]],
        [companions[3], companions[2], captains[1], shared, companions[1], companions[0]],
        [companions[1], shared, companions[0], companions[2], captains[0], companions[3]],
    ]
    grid = showdown_lineup_grid([_frame(rows) for rows in permutations])
    flex_columns = [f"FLEX {number}" for number in range(1, 6)]
    shown = [list(grid.loc[index, flex_columns]) for index in range(3)]

    assert shown[0] == shown[1] == shown[2]
    assert shown[0] == [
        "High (BUF)",
        "Mid (BUF)",
        "Shared (BUF)",
        "Low (BUF)",
        "Lower (BUF)",
    ]
    assert list(grid["Captain"]) == ["Captain A (KC)", "Captain B (BUF)", "Captain A (KC)"]
    assert list(grid["Salary"]) == [48000, 44000, 48000]

    cheaper_mid = [
        captains[0],
        shared,
        companions[0],
        _player("FLEX", "Mid", "m", 6000),
        companions[2],
        companions[3],
    ]
    shifted = showdown_lineup_grid([_frame(cheaper_mid)])
    assert list(shifted.loc[0, flex_columns]) == [
        "High (BUF)",
        "Shared (BUF)",
        "Mid (BUF)",
        "Low (BUF)",
        "Lower (BUF)",
    ]


def test_detail_and_upload_use_the_display_order():
    lineup = _frame(
        [
            _player("FLEX", "Ty Johnson", "92008", 4800),
            _player("FLEX", "Patrick Mahomes", "91002", 11200, team="KC"),
            _player("CPT", "Josh Allen", "91003", 17400),
            _player("FLEX", "Marquise Brown", "93010", 6400, team="KC"),
            _player("FLEX", "Clyde Edwards-Helaire", "92006", 5200, team="KC"),
            _player("FLEX", "Harrison Butker", "95002", 5000, team="KC"),
        ]
    )
    ordered = order_showdown_roster(lineup)
    grid = showdown_lineup_grid([lineup])
    detail = lineup_detail_frame(lineup, None)
    exported = lineups_to_dk_upload_csv([lineup], contest_format="showdown")
    header, body = exported.splitlines()

    flex_ids = [str(player_id) for player_id in ordered.loc[ordered["Slot"].eq("FLEX"), "ID"]]
    assert header == "CPT,FLEX,FLEX,FLEX,FLEX,FLEX"
    assert body.split(",") == ["91003", *flex_ids]
    assert body.split(",") == ["91003", "91002", "93010", "92006", "95002", "92008"]
    assert list(detail["Name"]) == list(ordered["Name"])
    assert list(detail["Slot"]) == ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    assert [grid.loc[0, f"FLEX {number}"] for number in range(1, 6)] == [
        "Patrick Mahomes (KC)",
        "Marquise Brown (KC)",
        "Clyde Edwards-Helaire (KC)",
        "Harrison Butker (KC)",
        "Ty Johnson (BUF)",
    ]
    assert grid.loc[0, "Salary"] == int(lineup.attrs["salary"])
    assert grid.loc[0, "Projection"] == float(lineup.attrs["projection"])
    # Numeric IDs only. The header and the ID text are unchanged apart from order.
    assert all(cell.isdigit() for cell in body.split(","))


def test_classic_roster_order_and_upload_bytes_stay_put():
    rows = [
        _player("QB", "Cheap QB", "1", 3000),
        _player("RB", "Pricey RB", "2", 9000),
        _player("RB", "Other RB", "3", 2000),
        _player("WR", "Wide", "4", 8000),
        _player("WR", "Second", "5", 2500),
        _player("WR", "Third", "6", 7000),
        _player("TE", "Tight", "7", 4000),
        _player("FLEX", "Top Flex", "8", 9500),
        _player("DST", "Defense", "9", 1500),
    ]
    lineup = _frame(rows)
    assert order_showdown_roster(lineup) is lineup

    exported = lineups_to_dk_upload_csv([lineup], contest_format="classic")
    assert exported == (
        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n"
        "Cheap QB (1),Pricey RB (2),Other RB (3),Wide (4),Second (5),"
        "Third (6),Tight (7),Top Flex (8),Defense (9)\n"
    )
