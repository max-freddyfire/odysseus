"""Pin the calendar-extraction retry guarantee in the background email poller.

`_auto_summarize_pass_single` in `routes/email_pollers.py` runs three
independent per-email steps: summary, AI-reply, and calendar event
extraction. The summary step records the email in `email_summaries`
only inside `if summary:` (the success branch), and the reply step
records in `email_ai_replies` only inside `if reply:`. So when their
LLM call raises, the `except` only logs and the email is reprocessed
on the next poll.

The calendar step was the inconsistent outlier. Its
`INSERT ... INTO email_calendar_extractions` (and the matching
`_cal_existing.add(message_id)`) sat AFTER the try/except, at the
`if need_cal:` body indent, not inside the success path. So a transient
LLM failure (timeout, model down) still marked the email as processed.
Once `message_id` was in `email_calendar_extractions` / `_cal_existing`,
`need_cal` was False forever and the meeting was never extracted on any
later run.

The fix moves the record into the success branch, matching summary/reply.

The regression test below drives one calendar-only pass with the
calendar LLM call forced to raise, and asserts `message_id` is NOT in
`email_calendar_extractions` afterwards, so the email is retried.
Pre-fix the assertion fails because the email was recorded despite the
failure; post-fix the record only happens on success.
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path


# Point every data-dir-using dependency at a per-process tmp dir BEFORE
# any `from routes...` import runs, so module-import-time DB setup does
# not try to open `./data/app.db` on a bare machine. (Same guard as
# tests/test_email_polly_imap_leak.py.)
_TMP_DATA = Path(tempfile.mkdtemp(prefix="odysseus-cal-skip-"))
os.environ.setdefault("DATA_DIR", str(_TMP_DATA))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP_DATA / 'app.db'}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_RAW_EMAIL = (
    b"Message-ID: <cal-skip-test@local>\r\n"
    b"From: Sam <sam@example.com>\r\n"
    b"Subject: Lunch Tuesday 12:00\r\n"
    b"Date: Mon, 01 Jun 2026 09:00:00 +0000\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Can we do lunch Tuesday at noon?\r\n"
)
_MESSAGE_ID = "<cal-skip-test@local>"


class _FakeConn:
    """Minimal IMAP stand-in: one INBOX message, no Sent folder."""

    def select(self, folder, readonly=True):
        # Only INBOX exists; reject the Sent-folder probes so
        # folders_to_scan stays at ["INBOX"].
        if isinstance(folder, str) and folder.strip('"').upper() == "INBOX":
            return ("OK", [b"1"])
        return ("NO", [b""])

    def uid(self, command, *args):
        if command == "SEARCH":
            return ("OK", [b"1"])
        if command == "FETCH":
            return ("OK", [(b"1 (RFC822 {len})", _RAW_EMAIL)])
        return ("NO", [b""])

    def logout(self):
        pass


def _setup(monkeypatch, *, llm_raises):
    """Wire one calendar-only pass against a temp SCHEDULED_DB.

    Returns the temp DB path so the caller can inspect
    email_calendar_extractions after the pass.
    """
    import routes.email_helpers as email_helpers
    import routes.email_pollers as email_pollers
    import src.endpoint_resolver as endpoint_resolver
    import core.database as core_db

    db_path = str(_TMP_DATA / f"scheduled-{'fail' if llm_raises else 'ok'}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Both modules hold their own binding to SCHEDULED_DB; patch both,
    # then build the schema in the temp DB.
    monkeypatch.setattr(email_helpers, "SCHEDULED_DB", db_path)
    monkeypatch.setattr(email_pollers, "SCHEDULED_DB", db_path)
    email_helpers._init_scheduled_db()

    # Calendar-only pass: avoids the summary/reply HTTP paths entirely.
    monkeypatch.setattr(
        email_pollers, "_load_settings",
        lambda: {"email_auto_calendar": True},
    )
    monkeypatch.setattr(
        email_pollers, "_owner_for_email_account", lambda account_id: "",
    )
    monkeypatch.setattr(
        email_pollers, "_imap_connect",
        lambda account_id=None, owner="": _FakeConn(),
    )
    # Endpoint resolution is imported inside the pass as a local name.
    monkeypatch.setattr(
        endpoint_resolver, "resolve_endpoint",
        lambda kind, owner="": ("http://endpoint.invalid", "test-model", {}),
    )
    # get_upcoming_events is imported inside the calendar block.
    monkeypatch.setattr(core_db, "get_upcoming_events", lambda *a, **k: [])

    async def _fake_llm(*args, **kwargs):
        if llm_raises:
            raise RuntimeError("simulated calendar LLM failure")
        # On success: no actionable ops, so nothing hits the calendar tool.
        return "[]"

    monkeypatch.setattr(email_pollers, "llm_call_async", _fake_llm)

    return email_pollers, db_path


def _recorded(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT message_id FROM email_calendar_extractions",
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


async def test_calendar_extraction_not_recorded_when_llm_fails(monkeypatch):
    """A failing calendar LLM call must NOT mark the email processed, so
    the meeting can be extracted on a later poll. Pre-fix the email was
    recorded in email_calendar_extractions despite the failure."""
    email_pollers, db_path = _setup(monkeypatch, llm_raises=True)

    await email_pollers._auto_summarize_pass_single(
        account_id="acct-1", progress_cb=None,
    )

    assert _MESSAGE_ID not in _recorded(db_path), (
        "On a failed calendar extraction the email must not be recorded "
        "in email_calendar_extractions; otherwise need_cal is False on "
        "every later poll and the meeting is never extracted. The summary "
        "and reply steps only record on success — the calendar step must "
        "match. Pre-fix the INSERT ran after the try/except unconditionally."
    )


async def test_calendar_extraction_recorded_when_llm_succeeds(monkeypatch):
    """Control: a successful calendar pass DOES record the email, so we
    don't re-LLM it. This proves the failure-path test discriminates
    success from failure rather than always passing."""
    email_pollers, db_path = _setup(monkeypatch, llm_raises=False)

    await email_pollers._auto_summarize_pass_single(
        account_id="acct-1", progress_cb=None,
    )

    assert _MESSAGE_ID in _recorded(db_path), (
        "On a successful calendar extraction the email should be recorded "
        "so the poller does not re-run the LLM on it next pass."
    )
