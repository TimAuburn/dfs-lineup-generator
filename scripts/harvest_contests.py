"""Download completed contest results for IDs already stored by the lobby sync.

Run after scripts/sync_contests.py, with the project virtualenv:

    venv\\Scripts\\python.exe scripts\\harvest_contests.py

Requires DK_COOKIE in .env. Writes data/harvest.log next to the database.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contest_parser import fetch_dk_contest_metadata, parse_contest_export
from src.database import connect, save_contest
from src.env_config import get_env_key, load_env

DB_PATH = ROOT / "data" / "dfs_tracker.db"
LOG_PATH = ROOT / "data" / "harvest.log"
EXPORT_URL = "https://www.draftkings.com/contest/exportfullstandingscsv/{contest_id}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

MetadataLoader = Callable[[str], Mapping[str, Any]]
Downloader = Callable[[str], bytes]


def run_harvest(
    db_path: Path,
    log_path: Path,
    cookie: str,
    *,
    metadata_loader: Optional[MetadataLoader] = None,
    downloader: Optional[Downloader] = None,
) -> int:
    """Import finished contests. Returns 0, or 1 when the login or cookie fails."""
    load_metadata = metadata_loader or fetch_dk_contest_metadata
    fetch_file = downloader or (lambda contest_id: download_standings(contest_id, cookie))
    saved = skipped_unfinished = skipped_imported = failed = 0
    _log(log_path, f"{datetime.now():%Y-%m-%d %H:%M:%S} harvest started")

    if not str(cookie or "").strip():
        _log(log_path, "DK_COOKIE is missing.")
        _log(log_path, _summary(saved, skipped_unfinished, skipped_imported, failed))
        return 1

    for contest_id, slate_name in _monitored_contests(db_path):
        label = slate_name or contest_id
        if _already_imported(db_path, contest_id):
            skipped_imported += 1
            _log(log_path, f"{contest_id} {label} skipped already imported")
            continue
        metadata = dict(load_metadata(contest_id))
        if not metadata.get("slate_name"):
            metadata["slate_name"] = slate_name
        state = str(metadata.get("contest_state") or "").strip().casefold()
        if state != "completed":
            skipped_unfinished += 1
            _log(log_path, f"{contest_id} {label} skipped unfinished")
            continue
        try:
            payload = fetch_file(contest_id)
            if _is_login_page(payload):
                failed += 1
                _log(log_path, f"{contest_id} {label} failed DraftKings login expired.")
                _log(log_path, "DraftKings login expired.")
                _log(log_path, _summary(saved, skipped_unfinished, skipped_imported, failed))
                return 1
            parsed = parse_contest_export(payload, contest_id, metadata)
            save_contest(parsed["slate"], parsed["players"], parsed["standings"], db_path)
        except (OSError, ValueError, HTTPError, URLError, TimeoutError) as exc:
            failed += 1
            _log(log_path, f"{contest_id} {label} failed {exc}")
            continue
        saved += 1
        _log(
            log_path,
            f"{contest_id} {parsed['slate'].get('slate_name') or label} "
            f"saved entries={parsed['slate'].get('total_entries')} "
            f"players={len(parsed['players'])}",
        )

    _log(log_path, _summary(saved, skipped_unfinished, skipped_imported, failed))
    return 0


def download_standings(contest_id: str, cookie: str) -> bytes:
    """Download one contest export. The cookie is sent and never logged.

    DraftKings redirects the export to a presigned S3 URL. urllib would
    forward Cookie on that hop, which leaks the session and is rejected
    once the header block exceeds 8 KB.
    """
    request = Request(
        EXPORT_URL.format(contest_id=str(contest_id).strip()),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,application/zip,*/*",
            "Cookie": _cookie_value(cookie),
        },
    )
    opener = build_opener(_DraftKingsRedirectHandler)
    with opener.open(request, timeout=60) as response:
        return response.read()


def _is_draftkings_https(url: str) -> bool:
    """True when the resolved URL is HTTPS on draftkings.com."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return (parsed.scheme or "").lower() == "https" and (
        host == "draftkings.com" or host.endswith(".draftkings.com")
    )


def _strip_cookie(request: Request) -> None:
    """Remove Cookie from both header maps, whatever the key's case is."""
    for header_map in (request.headers, request.unredirected_hdrs):
        for name in [name for name in header_map if name.lower() == "cookie"]:
            del header_map[name]


class _DraftKingsRedirectHandler(HTTPRedirectHandler):
    """Keep Cookie on DraftKings HTTPS redirects and drop it on every other hop.

    ``HTTPRedirectHandler.redirect_request`` copies ``req.headers`` onto the
    next ``Request`` and does not copy ``unredirected_hdrs``. Each hop repeats
    that copy, so a Cookie left in ``headers`` follows the whole chain.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not _is_draftkings_https(redirected.full_url):
            _strip_cookie(redirected)
        return redirected


def _monitored_contests(db_path: Path) -> list[tuple[str, str]]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT contest_id, slate_name
            FROM monitored_contests
            ORDER BY contest_id
            """
        ).fetchall()
    finally:
        conn.close()
    return [(str(row["contest_id"]), str(row["slate_name"] or "")) for row in rows]


def _already_imported(db_path: Path, contest_id: str) -> bool:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM player_performance WHERE contest_id = ? LIMIT 1",
            (contest_id,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _is_login_page(payload: bytes | str) -> bool:
    raw = payload.encode("utf-8", errors="ignore") if isinstance(payload, str) else payload
    head = raw[:800].lstrip().lower()
    return head.startswith((b"<!doctype", b"<html")) or b"<html" in head[:300]


def _cookie_value(cookie: str) -> str:
    text = str(cookie or "").strip()
    if text.lower().startswith("cookie:"):
        return text.split(":", 1)[1].strip()
    return text


def _summary(saved: int, unfinished: int, imported: int, failed: int) -> str:
    return (
        f"Saved {saved}. Skipped unfinished {unfinished}. "
        f"Skipped already imported {imported}. Failed {failed}."
    )


def _log(log_path: Path, line: str) -> None:
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    load_env(ROOT / ".env")
    cookie = get_env_key("DK_COOKIE") or ""
    return run_harvest(DB_PATH, LOG_PATH, cookie)


if __name__ == "__main__":
    raise SystemExit(main())
