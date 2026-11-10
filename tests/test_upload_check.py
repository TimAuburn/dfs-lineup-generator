"""Upload checks that run before a DraftKings CSV can be downloaded.

The lineups below are the seed-42 cash results, or that roster with one
thing changed. A passing file is the exporter's own bytes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.exporter import lineups_to_dk_upload_csv, upload_columns_for
from src.names import person_key as name_person_key
from src.optimizer import (
    NFLClassicOptimizer,
    NFLShowdownOptimizer,
)
from src.parser import parse_dk_csv
from src.portfolio import person_key
from src.presets import ContestFormat, ContestPreset, apply_preset
from src.slate_games import EASTERN
from src.upload_check import (
    csv_for_download,
    failure_table,
    fresh_upload_check,
    resolve_upload_check,
    salary_cap_for,
    upload_check_key,
    validate_lineups,
    warning_table,
)

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
GOLDEN_CLASSIC = Path(__file__).parent / "golden" / "classic_cash_seed42.csv"
GOLDEN_SHOWDOWN = Path(__file__).parent / "golden" / "showdown_cash_seed42.csv"

# main at 5c27f96, seed 42, one cash lineup each.
CASH_IDS = ["1001", "2001", "2003", "3012", "3010", "3013", "4004", "2002", "5004"]
SHOWDOWN_IDS = ["91003", "92006", "95002", "93010", "91002", "92008"]
CLASSIC_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
# Before every kickoff on the mock slate.
BEFORE = datetime(2025, 9, 10, 12, 0, tzinfo=EASTERN)


def _by_id(pool: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["ID"]): row for _, row in pool.iterrows()}


def _lineup(pool: pd.DataFrame, slots: list[str], ids: list[str]) -> pd.DataFrame:
    indexed = _by_id(pool)
    rows = []
    for slot, player_id in zip(slots, ids):
        source = indexed.get(str(player_id))
        if source is None:
            rows.append(
                {
                    "Slot": slot,
                    "ID": str(player_id),
                    "Name": "",
                    "Position": "",
                    "Team": "",
                    "Salary": 0,
                }
            )
            continue
        rows.append(
            {
                "Slot": slot,
                "ID": str(player_id),
                "Name": source["Name"],
                "Name + ID": source["Name + ID"],
                "Position": source["Position"],
                "Team": source["TeamAbbrev"],
                "Salary": int(source["Salary"]),
                "Projection": float(source["Projection"]),
            }
        )
    return pd.DataFrame(rows)


def _classic(pool: pd.DataFrame, ids: list[str] | None = None) -> pd.DataFrame:
    return _lineup(pool, CLASSIC_SLOTS, ids or CASH_IDS)


def _showdown(pool: pd.DataFrame, ids: list[str] | None = None) -> pd.DataFrame:
    return _lineup(pool, SHOWDOWN_SLOTS, ids or SHOWDOWN_IDS)


def _check(lineup, pool, **kwargs):
    fmt = kwargs.pop("contest_format", ContestFormat.CLASSIC)
    return validate_lineups(
        [lineup] if not isinstance(lineup, list) else lineup,
        pool,
        contest_format=fmt,
        now=kwargs.pop("now", BEFORE),
        disabled_games=kwargs.pop("disabled_games", ()),
        csv_text=kwargs.pop("csv_text", None),
        export_error=kwargs.pop("export_error", None),
    )


def _blocks(check) -> list[str]:
    return [reason for finding in check.findings for reason in finding.blocks]


def test_salary_at_the_cap_passes_and_one_dollar_over_blocks():
    pool = parse_dk_csv(MOCK_CLASSIC)
    lineup = _classic(pool)
    indexed = _by_id(pool)
    spent = sum(int(indexed[player_id]["Salary"]) for player_id in CASH_IDS)
    assert spent == salary_cap_for(ContestFormat.CLASSIC, pool)
    assert spent == NFLClassicOptimizer.salary_cap
    passing = _check(lineup, pool)
    assert passing.blocked is False
    assert passing.salary_cap == NFLClassicOptimizer.salary_cap
    assert not any("salary cap" in reason for reason in _blocks(passing))

    over = pool.copy()
    over.loc[over["ID"] == "5004", "Salary"] = 2601
    blocked = _check(lineup, over)
    assert blocked.blocked is True
    assert _blocks(blocked) == ["Salary is $50,001, over the $50,000 salary cap."]


def test_slate_salary_cap_overrides_the_format_cap():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.attrs["salary_cap"] = 40_000
    check = _check(_classic(pool), pool)
    assert check.salary_cap == 40_000
    assert check.blocked is True
    assert "over the $40,000 salary cap" in _blocks(check)[0]


def test_ineligible_position_blocks_and_showdown_captain_id_and_price_are_separate():
    classic = parse_dk_csv(MOCK_CLASSIC)
    # Sutton is a WR. The second RB slot does not accept him.
    swapped = CASH_IDS.copy()
    swapped[2] = "3014"
    blocked = _check(_classic(classic, swapped), classic)
    assert blocked.blocked is True
    assert _blocks(blocked) == ["Courtland Sutton (DEN) is WR and is not eligible for RB."]

    showdown = parse_dk_csv(MOCK_SHOWDOWN)
    flex_as_captain = SHOWDOWN_IDS.copy()
    flex_as_captain[0] = "91004"
    wrong_id = _check(
        _showdown(showdown, flex_as_captain),
        showdown,
        contest_format=ContestFormat.SHOWDOWN,
    )
    assert wrong_id.blocked is True
    assert _blocks(wrong_id) == [
        "CPT uses DraftKings ID 91004, which is the FLEX row for Josh Allen (BUF), not the CPT row."
    ]

    priced = showdown.copy()
    priced.loc[priced["ID"] == "91003", "Salary"] = 11_600
    wrong_price = _check(_showdown(priced), priced, contest_format=ContestFormat.SHOWDOWN)
    assert _blocks(wrong_price) == [
        "CPT salary for Josh Allen (BUF) is $11,600, not 1.5x the $11,600 flex salary."
    ]

    short = _showdown(showdown).iloc[:-1].reset_index(drop=True)
    missing = _check(short, showdown, contest_format=ContestFormat.SHOWDOWN)
    assert any("Showdown needs CPT, FLEX, FLEX, FLEX, FLEX, FLEX" in reason for reason in _blocks(missing))


def test_same_person_at_captain_and_flex_is_a_duplicate():
    showdown = parse_dk_csv(MOCK_SHOWDOWN)
    allen_cpt = showdown.loc[showdown["ID"] == "91003"].iloc[0]
    allen_flex = showdown.loc[showdown["ID"] == "91004"].iloc[0]
    assert person_key(allen_cpt["Name"], allen_cpt["TeamAbbrev"], allen_cpt["Position"]) == name_person_key(
        allen_flex["Name"], allen_flex["TeamAbbrev"], allen_flex["Position"]
    )
    ids = ["91003", "91004", "92008", "92006", "93012", "95004"]
    check = _check(_showdown(showdown, ids), showdown, contest_format=ContestFormat.SHOWDOWN)
    assert check.blocked is True
    assert _blocks(check) == ["Josh Allen (BUF) appears twice (CPT and FLEX)."]

    classic = parse_dk_csv(MOCK_CLASSIC)
    clone = classic.loc[classic["ID"] == "2001"].copy()
    clone["ID"] = "200199"
    clone["Name + ID"] = "Christian McCaffrey (200199)"
    clone["Salary"] = 5000
    both = pd.concat([classic, clone], ignore_index=True)
    ids = CASH_IDS.copy()
    ids[-2] = "200199"
    duplicated = _check(_classic(both, ids), both)
    assert _blocks(duplicated) == ["Christian McCaffrey (SF) appears twice (RB and FLEX)."]


def test_stale_player_id_is_not_in_the_salary_file():
    pool = parse_dk_csv(MOCK_CLASSIC)
    ids = CASH_IDS.copy()
    ids[-1] = "999999"
    check = _check(_classic(pool, ids), pool)
    assert check.blocked is True
    assert _blocks(check) == ["Player ID 999999 is not in the salary file."]


@pytest.mark.parametrize(
    ("raw", "phrase"),
    [("OUT", "is Out"), ("O", "is Out"), ("IR", "is IR"), ("D", "is Doubtful"), ("PUP", "is PUP")],
)
def test_out_ir_and_doubtful_block(raw, phrase):
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["ID"] == "1001", "Injury"] = raw
    check = _check(_classic(pool), pool)
    assert check.blocked is True
    assert _blocks(check) == [f"Patrick Mahomes (KC) {phrase}."]
    assert check.warnings == ()


def test_questionable_warns_and_does_not_block():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["ID"] == "1001", "Injury"] = "Q"
    pool.loc[pool["ID"] == "2001", "Injury"] = "QUESTIONABLE"
    check = _check(_classic(pool), pool)
    assert check.blocked is False
    assert check.status_field == "Injury"
    assert check.status_message == ""
    assert check.findings[0].blocks == ()
    assert check.findings[0].warnings == (
        "Patrick Mahomes (KC) is Questionable.",
        "Christian McCaffrey (SF) is Questionable.",
    )
    warned = warning_table(check)
    assert list(warned["Lineup"]) == [1]
    assert "Questionable" in warned.loc[0, "Warnings"]
    assert failure_table(check).empty
    csv_text = lineups_to_dk_upload_csv([_classic(pool)])
    assert csv_for_download(check, csv_text) == csv_text


def test_probable_and_blank_status_do_not_warn():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["ID"] == "1001", "Injury"] = "P"
    check = _check(_classic(pool), pool)
    assert check.blocked is False
    assert check.warnings == ()


def test_missing_status_field_is_reported_and_does_not_pass_silently():
    pool = parse_dk_csv(MOCK_CLASSIC).drop(columns=["Injury"])
    check = _check(_classic(pool), pool)
    assert check.status_field is None
    assert check.blocked is False
    assert "no injury status field" in check.status_message
    assert "Out, IR, and Doubtful were not checked" in check.status_message


def test_toggled_off_game_blocks_only_players_in_that_game():
    pool = parse_dk_csv(MOCK_CLASSIC)
    check = _check(_classic(pool), pool, disabled_games=["KC @ BUF"])
    reasons = _blocks(check)
    assert check.blocked is True
    assert "Patrick Mahomes (KC) is in KC@BUF, which is off." in reasons
    assert "Khalil Shakir (BUF) is in KC@BUF, which is off." in reasons
    assert not any("Christian McCaffrey" in reason for reason in reasons)
    assert not any("kickoff has passed" in reason for reason in reasons)

    showdown = parse_dk_csv(MOCK_SHOWDOWN)
    showdown_check = _check(
        _showdown(showdown),
        showdown,
        contest_format=ContestFormat.SHOWDOWN,
        disabled_games=["KC@BUF"],
    )
    assert showdown_check.blocked is True
    assert all("which is off" in reason for reason in _blocks(showdown_check))


def test_kickoff_uses_the_injected_clock_and_eastern_zone():
    pool = parse_dk_csv(MOCK_CLASSIC)
    lineup = _classic(pool)
    noon = _check(lineup, pool, now=datetime(2025, 9, 10, 12, 59, tzinfo=EASTERN))
    assert noon.blocked is False

    at_one = _check(lineup, pool, now=datetime(2025, 9, 10, 13, 0, tzinfo=EASTERN))
    reasons = _blocks(at_one)
    assert "Derrick Henry (BAL) is in BAL@CIN, and kickoff has passed (09/10/2025 01:00PM ET)." in reasons
    assert "Chris Godwin (TB) is in TB@ATL, and kickoff has passed (09/10/2025 01:00PM ET)." in reasons
    assert not any("Patrick Mahomes" in reason for reason in reasons)
    assert not any("Christian McCaffrey" in reason for reason in reasons)

    naive = _check(lineup, pool, now=datetime(2025, 9, 10, 13, 30))
    assert any("Derrick Henry" in reason and "kickoff has passed" in reason for reason in _blocks(naive))
    assert not any("Patrick Mahomes" in reason for reason in _blocks(naive))

    started = pool.copy()
    started.loc[started["ID"] == "3012", "Game Info"] = "TB@ATL In Progress"
    underway = _check(_classic(started), started)
    assert "Chris Godwin (TB) is in TB@ATL, which is already underway." in _blocks(underway)
    assert not any("Godwin" in reason and "kickoff has passed" in reason for reason in _blocks(underway))


def test_omitted_now_uses_the_module_clock(monkeypatch):
    fixed = datetime(2025, 9, 10, 12, 0, tzinfo=EASTERN)

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed.replace(tzinfo=None)
            return fixed.astimezone(tz)

    monkeypatch.setattr("src.upload_check.datetime", _Frozen)
    pool = parse_dk_csv(MOCK_CLASSIC)
    check = validate_lineups([_classic(pool)], pool, contest_format=ContestFormat.CLASSIC)
    assert check.blocked is False


def test_csv_header_must_match_the_template_for_the_format():
    pool = parse_dk_csv(MOCK_CLASSIC)
    lineup = _classic(pool)
    wrong = _check(lineup, pool, csv_text="QB,RB\n")
    assert wrong.blocked is True
    assert wrong.findings[0].blocks == ()
    assert wrong.export_blocks == (
        "CSV header is QB,RB, expected QB,RB,RB,WR,WR,WR,TE,FLEX,DST.",
    )
    table = failure_table(wrong)
    assert list(table["Lineup"]) == ["Export"]
    assert csv_for_download(wrong, "QB,RB\n") == ""

    showdown = parse_dk_csv(MOCK_SHOWDOWN)
    showdown_lineup = _showdown(showdown)
    reordered = _check(
        showdown_lineup,
        showdown,
        contest_format=ContestFormat.SHOWDOWN,
        csv_text="FLEX,CPT,FLEX,FLEX,FLEX,FLEX\n1,2,3,4,5,6\n",
    )
    assert reordered.export_blocks == (
        "CSV header is FLEX,CPT,FLEX,FLEX,FLEX,FLEX, expected CPT,FLEX,FLEX,FLEX,FLEX,FLEX.",
    )
    assert upload_columns_for(ContestFormat.SHOWDOWN) == SHOWDOWN_SLOTS


def test_clean_classic_and_showdown_pass_and_match_the_seed_42_export():
    classic = parse_dk_csv(MOCK_CLASSIC)
    cash = NFLClassicOptimizer()
    cash.load_from_dataframe(classic)
    apply_preset(cash, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    classic_lineups = cash.generate_lineups()
    assert [str(player_id) for player_id in classic_lineups[0]["ID"]] == CASH_IDS
    classic_check = _check(classic_lineups, classic)
    assert classic_check.blocked is False
    assert classic_check.warnings == ()
    assert classic_check.status_field == "Injury"
    classic_csv = lineups_to_dk_upload_csv(classic_lineups, contest_format=ContestFormat.CLASSIC)
    assert classic_csv == GOLDEN_CLASSIC.read_text(encoding="utf-8")
    assert csv_for_download(classic_check, classic_csv) == classic_csv
    assert classic_csv.splitlines()[0] == "QB,RB,RB,WR,WR,WR,TE,FLEX,DST"

    showdown_pool = parse_dk_csv(MOCK_SHOWDOWN)
    showdown = NFLShowdownOptimizer()
    showdown.load_from_dataframe(showdown_pool)
    apply_preset(showdown, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    showdown_lineups = showdown.generate_lineups()
    assert [str(player_id) for player_id in showdown_lineups[0]["ID"]] == SHOWDOWN_IDS
    showdown_check = _check(showdown_lineups, showdown_pool, contest_format=ContestFormat.SHOWDOWN)
    assert showdown_check.blocked is False
    assert showdown_check.warnings == ()
    assert showdown_check.salary_cap == NFLShowdownOptimizer.salary_cap
    showdown_csv = lineups_to_dk_upload_csv(showdown_lineups, contest_format=ContestFormat.SHOWDOWN)
    assert showdown_csv == GOLDEN_SHOWDOWN.read_text(encoding="utf-8")
    assert csv_for_download(showdown_check, showdown_csv) == showdown_csv
    assert showdown_csv.splitlines()[0] == "CPT,FLEX,FLEX,FLEX,FLEX,FLEX"


def test_only_the_failing_lineup_is_listed():
    pool = parse_dk_csv(MOCK_CLASSIC)
    clean = _classic(pool)
    stale_ids = CASH_IDS.copy()
    stale_ids[0] = "999999"
    stale = _classic(pool, stale_ids)
    check = _check([clean, stale], pool)
    assert check.blocked is True
    assert check.findings[0].blocks == ()
    assert failure_table(check).to_dict("records") == [
        {"Lineup": 2, "Reasons": "Player ID 999999 is not in the salary file."}
    ]


def test_stored_check_is_recomputed_and_never_reused_for_other_inputs():
    pool = parse_dk_csv(MOCK_CLASSIC)
    lineup = _classic(pool)
    first = upload_check_key(
        [lineup],
        pool,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=[],
        pool_signature="slate-a",
    )
    assert first == upload_check_key(
        [lineup],
        pool,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=[],
        pool_signature="slate-a",
    )
    toggled = upload_check_key(
        [lineup],
        pool,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=["KC@BUF"],
        pool_signature="slate-a",
    )
    other_slate = upload_check_key(
        [lineup],
        pool,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=[],
        pool_signature="slate-b",
    )
    injured = pool.copy()
    injured.loc[injured["ID"] == "1001", "Injury"] = "OUT"
    other_status = upload_check_key(
        [lineup],
        injured,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=[],
        pool_signature="slate-a",
    )
    other_lineup = upload_check_key(
        [_classic(pool, ["1002", *CASH_IDS[1:]])],
        pool,
        contest_format=ContestFormat.CLASSIC,
        disabled_games=[],
        pool_signature="slate-a",
    )
    assert len({first, toggled, other_slate, other_status, other_lineup}) == 5

    previous = validate_lineups([lineup], pool, contest_format="classic", now=BEFORE)
    state = {"upload_check": previous, "upload_check_key": "stale"}
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return validate_lineups(
            [lineup],
            pool,
            contest_format="classic",
            now=BEFORE,
            disabled_games=["KC@BUF"],
        )

    current = resolve_upload_check(state, key=toggled, compute=compute)
    assert current is not previous
    assert current.blocked is True
    assert calls["n"] == 1
    again = resolve_upload_check(state, key=toggled, compute=compute)
    assert again is not current
    assert calls["n"] == 2
    assert fresh_upload_check(previous, "stale", toggled) is None
    assert fresh_upload_check(again, toggled, toggled) is again


def _contract(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from src.dk_scoring import SCORING_VERSION
    from src.slate_week import slate_games_from_pool, slate_key
    from src.sleeper import MATCH_CACHE_VERSION

    stamped = pool.copy()
    stamped["PPR"] = pd.Series(pd.NA, index=stamped.index, dtype="Float64")
    stamped["Scoring"] = ""
    stamped["ScoringNote"] = ""
    key = slate_key(slate_games_from_pool(stamped))
    stamped.attrs["scoring_version"] = SCORING_VERSION
    stamped.attrs["match_schema"] = MATCH_CACHE_VERSION
    stamped.attrs["slate_key"] = key
    stamped.attrs["fingerprint"] = "upload-check-ui"
    summary = {
        "ok": False,
        "applied": False,
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
        "fingerprint": "upload-check-ui",
        "slate_key": key,
        "matched": 0,
        "rows": len(stamped),
        "ppr_fallback_rows": 0,
    }
    return stamped, summary


def _optimizer_app(pool: pd.DataFrame, lineup: pd.DataFrame):
    app_test = pytest.importorskip("streamlit.testing.v1")
    stamped, summary = _contract(pool)
    app = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=90,
    )
    app.session_state["app_view"] = "Optimizer"
    app.session_state["contest_format"] = "classic"
    app.session_state["player_pool"] = stamped
    app.session_state["sleeper_summary"] = summary
    app.session_state["lineups"] = [lineup]
    app.session_state["pool_signature"] = "upload-check-ui"
    app.session_state["upload_check_now"] = BEFORE
    app.run()
    return app


def test_streamlit_blocks_the_download_and_lists_the_reason():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["ID"] == "1001", "Injury"] = "OUT"
    app = _optimizer_app(pool, _classic(pool))
    assert not app.exception
    button = next(item for item in app.download_button if item.label == "Download DK Upload CSV")
    assert button.disabled
    frames = [frame.value for frame in app.dataframe]
    reasons = next(frame for frame in frames if "Reasons" in frame.columns)
    assert list(reasons["Reasons"]) == ["Patrick Mahomes (KC) is Out."]
    exposure = next(item for item in app.download_button if item.label == "Download exposure report")
    assert exposure.disabled is False


def test_streamlit_questionable_warning_leaves_the_download_available():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["ID"] == "1001", "Injury"] = "Q"
    lineup = _classic(pool)
    app = _optimizer_app(pool, lineup)
    assert not app.exception
    button = next(item for item in app.download_button if item.label == "Download DK Upload CSV")
    assert button.disabled is False
    frames = [frame.value for frame in app.dataframe]
    warnings = next(frame for frame in frames if "Warnings" in getattr(frame, "columns", []))
    assert "Questionable" in " ".join(warnings["Warnings"].astype(str))
    assert not any("Reasons" in getattr(frame, "columns", []) for frame in frames)
    check, produced, export_error = app.session_state["upload_check"]
    assert export_error is None
    assert check.blocked is False
    assert produced == lineups_to_dk_upload_csv([lineup])
