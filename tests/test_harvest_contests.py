"""Tuesday harvest of completed contest standings. No live HTTP."""

from __future__ import annotations

import importlib.util
import io
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.request import Request

import pandas as pd
import pytest

from src.database import connect, get_recent_slates, save_contest, search_player_performance

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "harvest_contests",
    ROOT / "scripts" / "harvest_contests.py",
)
harvest = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(harvest)

SIDE_BY_SIDE = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,,,Player,Roster Position,%Drafted,FPTS
1,100,Alpha,0,180.0,CPT Josh Allen FLEX Nico Collins,,,,Josh Allen,QB,18.4%,30.2
2,200,Bravo,0,150.0,CPT Nico Collins FLEX Josh Allen,,,,Nico Collins,WR,10%,22.0
"""

STANDINGS_ONLY = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup
1,100,Alpha,0,180.0,QB Josh Allen RB Joe Mixon
2,200,Bravo,0,150.0,QB Josh Allen WR Nico Collins
"""

COMPLETED = {
    "slate_name": "NFL $15K Quarter Jukebox",
    "entry_fee": 0.25,
    "contest_date": "2026-10-27",
    "paid_positions": 2,
    "slate_type": "Showdown",
    "contest_state": "Completed",
}


def _seed(db_path: Path, contest_id: str, slate_name: str) -> None:
    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO monitored_contests (
                    contest_id, slate_name, entry_fee, prize_pool, draft_group_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (contest_id, slate_name, 0.25, 1500, "100"),
            )
    finally:
        conn.close()


def test_completed_unfinished_and_imported_are_logged(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "1", "NFL $15K Quarter Jukebox")
    _seed(db_path, "2", "Thursday main")
    _seed(db_path, "3", "Already saved")
    save_contest(
        {
            "contest_id": "3",
            "slate_name": "Already saved",
            "slate_type": "Classic",
            "contest_date": "2026-10-01",
            "total_entries": 1,
            "entry_fee": 1,
            "cash_cutline": 10,
            "first_place_score": 20,
        },
        [
            {
                "player_name": "Josh Allen",
                "position": "QB",
                "team": None,
                "salary": None,
                "ownership_pct": 0.5,
                "actual_fpts": 10,
            }
        ],
        [],
        db_path,
    )
    downloaded: list[str] = []

    def metadata(contest_id: str):
        if contest_id == "2":
            return {**COMPLETED, "slate_name": "Thursday main", "contest_state": "Live"}
        return dict(COMPLETED)

    def download(contest_id: str) -> bytes:
        downloaded.append(contest_id)
        return SIDE_BY_SIDE.encode()

    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=metadata,
        downloader=download,
    )
    assert code == 0
    assert downloaded == ["1"]
    recent = get_recent_slates(db_path=db_path)
    jukebox = recent.loc[recent["Contest ID"] == "1"].iloc[0]
    assert float(jukebox["Cash Cutline"]) == 150.0
    players = search_player_performance(contest_id="1", db_path=db_path)
    allen = players.loc[players["Player"] == "Josh Allen"].iloc[0]
    assert float(allen["Ownership %"]) == 18.4
    assert float(allen["Actual FPTS"]) == 30.2

    log = log_path.read_text(encoding="utf-8")
    assert "1 NFL $15K Quarter Jukebox saved entries=2 players=2" in log
    assert "2 Thursday main skipped unfinished" in log
    assert "3 Already saved skipped already imported" in log
    assert "Saved 1. Skipped unfinished 1. Skipped already imported 1. Failed 0." in log


def test_login_page_stops_the_run_and_is_logged(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "9", "NFL $6M Midseason Main Event")

    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=lambda _contest_id: dict(COMPLETED),
        downloader=lambda _contest_id: b"<!doctype html><html><title>Sign Up</title>",
    )
    assert code == 1
    log = log_path.read_text(encoding="utf-8")
    assert "9 NFL $6M Midseason Main Event failed DraftKings login expired." in log
    assert "DraftKings login expired." in log
    assert "Failed 1." in log


def test_standings_without_drafted_column_counts_lineup_ownership(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "5", "NFL $25K Pylon")
    metadata = {
        **COMPLETED,
        "slate_name": "NFL $25K Pylon",
        "slate_type": "Classic",
        "paid_positions": 1,
    }
    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=lambda _contest_id: metadata,
        downloader=lambda _contest_id: STANDINGS_ONLY.encode(),
    )
    assert code == 0
    players = search_player_performance(contest_id="5", db_path=db_path).set_index("Player")
    assert float(players.loc["Josh Allen", "Ownership %"]) == 100.0
    assert float(players.loc["Joe Mixon", "Ownership %"]) == 50.0
    assert float(players.loc["Nico Collins", "Ownership %"]) == 50.0
    assert pd.isna(players.loc["Josh Allen", "Actual FPTS"])
    recent = get_recent_slates(db_path=db_path)
    assert float(recent.loc[recent["Contest ID"] == "5", "Cash Cutline"].iloc[0]) == 180.0


def test_missing_cookie_exits_with_a_log_line(tmp_path):
    log_path = tmp_path / "data" / "harvest.log"
    code = harvest.run_harvest(tmp_path / "dfs.db", log_path, "")
    assert code == 1
    assert "DK_COOKIE is missing." in log_path.read_text(encoding="utf-8")


# Larger than the 8,192-byte request-header limit S3 enforces, and larger
# than the 9 KB bar for this fix. The value is fake and must never be logged.
_LARGE_COOKIE = "session=" + ("a" * 10_000)
_PAYLOAD = b"Rank,EntryId,EntryName\n1,100,Alpha\n"
_CONTEST_ID = "99"


class _ScriptedBody(io.BytesIO):
    def __init__(self, code: int, headers: Message, body: bytes):
        super().__init__(body)
        self.code = code
        self.msg = "OK" if 200 <= code < 300 else "Found"
        self._headers = headers

    def info(self):
        return self._headers


def _cookie_items(request: Request) -> list[tuple[str, str]]:
    found = []
    for header_map in (request.headers, request.unredirected_hdrs):
        for name, value in header_map.items():
            if name.lower() == "cookie":
                found.append((name, value))
    return found


def _assert_cookie_absent(request: Request, cookie: str) -> None:
    assert _cookie_items(request) == []
    for header_map in (request.headers, request.unredirected_hdrs):
        for value in header_map.values():
            assert cookie not in value


def _redirect(location: str):
    def factory():
        headers = Message()
        headers["Location"] = location
        return _ScriptedBody(302, headers, b"")

    return factory


def _payload(body: bytes = _PAYLOAD):
    def factory():
        return _ScriptedBody(200, Message(), body)

    return factory


def _follow(monkeypatch, cookie: str, routes: dict):
    seen: list[Request] = []

    def open_request(self, req):
        seen.append(req)
        factory = routes.get(req.full_url)
        if factory is None:
            raise AssertionError(f"unexpected redirect target {req.full_url}")
        return factory()

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", open_request)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", open_request)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    body = harvest.download_standings(_CONTEST_ID, cookie)
    return body, seen


def _export_url() -> str:
    return harvest.EXPORT_URL.format(contest_id=_CONTEST_ID)


def test_s3_redirect_strips_a_large_cookie_and_returns_the_file(monkeypatch, capsys):
    assert len(_LARGE_COOKIE.encode("ascii")) > 9 * 1024
    export_url = _export_url()
    s3_url = (
        "https://s3.amazonaws.com/dk-exports/99.csv"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc"
    )
    body, seen = _follow(
        monkeypatch,
        _LARGE_COOKIE,
        {export_url: _redirect(s3_url), s3_url: _payload()},
    )
    assert body == _PAYLOAD
    assert [req.full_url for req in seen] == [export_url, s3_url]
    assert _cookie_items(seen[0]) == [("Cookie", _LARGE_COOKIE)]
    _assert_cookie_absent(seen[1], _LARGE_COOKIE)
    captured = capsys.readouterr()
    assert _LARGE_COOKIE not in captured.out
    assert _LARGE_COOKIE not in captured.err


@pytest.mark.parametrize(
    "location",
    [
        "https://api.draftkings.com/standings.csv",
        "https://draftkings.com/standings.csv",
        "https://WWW.DraftKings.COM/standings.csv",
        "https://stats.draftkings.com:443/standings.csv",
    ],
)
def test_draftkings_https_redirect_keeps_the_cookie(monkeypatch, location):
    export_url = _export_url()
    body, seen = _follow(
        monkeypatch,
        _LARGE_COOKIE,
        {export_url: _redirect(location), location: _payload()},
    )
    assert body == _PAYLOAD
    assert _cookie_items(seen[0]) == [("Cookie", _LARGE_COOKIE)]
    assert _cookie_items(seen[1]) == [("Cookie", _LARGE_COOKIE)]


def test_http_downgrade_strips_the_cookie(monkeypatch):
    export_url = _export_url()
    downgraded = "http://www.draftkings.com/standings.csv"
    body, seen = _follow(
        monkeypatch,
        _LARGE_COOKIE,
        {export_url: _redirect("HTTP://www.draftkings.com/standings.csv"), downgraded: _payload()},
    )
    assert body == _PAYLOAD
    assert seen[1].full_url == downgraded
    assert _cookie_items(seen[0]) == [("Cookie", _LARGE_COOKIE)]
    _assert_cookie_absent(seen[1], _LARGE_COOKIE)


@pytest.mark.parametrize(
    "location",
    [
        "https://evil-draftkings.com/standings.csv",
        "https://notdraftkings.com/standings.csv",
        "https://draftkings.com.attacker.test/standings.csv",
    ],
)
def test_lookalike_host_strips_the_cookie(monkeypatch, location):
    export_url = _export_url()
    body, seen = _follow(
        monkeypatch,
        _LARGE_COOKIE,
        {export_url: _redirect(location), location: _payload()},
    )
    assert body == _PAYLOAD
    assert seen[1].full_url == location
    _assert_cookie_absent(seen[1], _LARGE_COOKIE)


def test_same_domain_hop_keeps_cookie_until_s3(monkeypatch):
    export_url = _export_url()
    api_url = "https://api.draftkings.com/contest/exportfullstandingscsv/99"
    s3_url = "https://s3.amazonaws.com/dk-exports/99.csv"
    body, seen = _follow(
        monkeypatch,
        _LARGE_COOKIE,
        {
            export_url: _redirect(api_url),
            api_url: _redirect(s3_url),
            s3_url: _payload(),
        },
    )
    assert body == _PAYLOAD
    assert [req.full_url for req in seen] == [export_url, api_url, s3_url]
    assert _cookie_items(seen[0]) == [("Cookie", _LARGE_COOKIE)]
    assert _cookie_items(seen[1]) == [("Cookie", _LARGE_COOKIE)]
    _assert_cookie_absent(seen[2], _LARGE_COOKIE)


@pytest.mark.parametrize(
    "target",
    [
        "https://",
        "https:///standings.csv",
        "ftp://files.draftkings.com/standings.csv",
    ],
)
def test_redirect_request_strips_a_missing_host_or_non_https_scheme(target):
    handler = harvest._DraftKingsRedirectHandler()
    source = Request(
        _export_url(),
        headers={"Cookie": _LARGE_COOKIE, "Accept": "text/csv"},
    )
    redirected = handler.redirect_request(source, None, 302, "Found", {}, target)
    assert redirected is not None
    _assert_cookie_absent(redirected, _LARGE_COOKIE)
    assert redirected.headers["Accept"] == "text/csv"


def test_external_redirect_strips_mixed_case_cookie_from_both_maps(monkeypatch):
    """Plant both header maps. urllib would only forward ``headers``."""
    planted = Request("https://s3.amazonaws.com/bucket/standings.csv")
    planted.headers["Cookie"] = "session=headers"
    planted.headers["COOKIE"] = "session=HEADERS"
    planted.unredirected_hdrs["cookie"] = "session=unredirected"
    planted.unredirected_hdrs["CoOkIe"] = "session=UNREDIRECTED"

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return planted

    monkeypatch.setattr(urllib.request.HTTPRedirectHandler, "redirect_request", redirect_request)
    handler = harvest._DraftKingsRedirectHandler()
    source = Request(_export_url(), headers={"Cookie": "session=original"})
    redirected = handler.redirect_request(
        source,
        None,
        302,
        "Found",
        {},
        planted.full_url,
    )
    assert redirected is planted
    assert _cookie_items(redirected) == []


def test_draftkings_redirect_keeps_mixed_case_cookie_on_both_maps(monkeypatch):
    planted = Request("https://api.draftkings.com/standings.csv")
    planted.headers["COOKIE"] = "session=headers"
    planted.unredirected_hdrs["CoOkIe"] = "session=unredirected"

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return planted

    monkeypatch.setattr(urllib.request.HTTPRedirectHandler, "redirect_request", redirect_request)
    handler = harvest._DraftKingsRedirectHandler()
    source = Request(_export_url(), headers={"Cookie": "session=original"})
    redirected = handler.redirect_request(
        source,
        None,
        302,
        "Found",
        {},
        planted.full_url,
    )
    assert _cookie_items(redirected) == [
        ("COOKIE", "session=headers"),
        ("CoOkIe", "session=unredirected"),
    ]
