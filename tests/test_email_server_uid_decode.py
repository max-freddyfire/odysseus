"""Regression: some IMAP servers return message UIDs as str, not bytes.

`_list_emails` and `_search_emails` built each result row with a bare
`uid.decode()`. On a server that hands back str UIDs, that call raised
AttributeError. The per-message `except Exception: continue` swallowed it,
so matching emails were silently dropped from the listing/search.

The sibling `_read_email` already guarded with
`uid.decode() if isinstance(uid, bytes) else str(uid)`; this aligns the two
list/search builders with it. Tested with a fake IMAP connection so no live
server or docker is needed.
"""
import os
import tempfile
from pathlib import Path

import pytest

_tmp_data = Path(tempfile.mkdtemp(prefix="odysseus-email-uid-test-"))
os.environ.setdefault("DATA_DIR", str(_tmp_data))

pytest.importorskip("mcp")

import mcp_servers.email_server as es


_HEADER = (
    b"Subject: Hello\r\n"
    b"From: Alice <alice@example.com>\r\n"
    b"Date: Mon, 1 Jan 2024 00:00:00 +0000\r\n"
    b"Message-ID: <abc@example.com>\r\n"
    b"\r\n"
)


class _FakeConn:
    """Fake IMAP connection. `search_blob` is what `uid('SEARCH', ...)`
    returns as data[0]; bytes mimics a normal server, str mimics the
    servers that trigger the bug."""

    def __init__(self, search_blob):
        self._blob = search_blob
        self.logged_out = False

    def select(self, mailbox, readonly=False):
        return ("OK", [b"1"])

    def uid(self, cmd, *args):
        if cmd == "SEARCH":
            return ("OK", [self._blob])
        if cmd == "FETCH":
            return ("OK", [(b"1 (RFC822.HEADER {n}", _HEADER)])
        return ("OK", [None])

    def logout(self):
        self.logged_out = True


def _patch(monkeypatch, search_blob):
    monkeypatch.setattr(es, "_imap_connect", lambda account=None: _FakeConn(search_blob))
    monkeypatch.setattr(es, "_get_cached_summaries", lambda: {})


def test_list_emails_bytes_uid(monkeypatch):
    _patch(monkeypatch, b"1 2 3")
    results = es._list_emails()
    assert [r["uid"] for r in results] == ["3", "2", "1"]


def test_list_emails_str_uid_does_not_drop(monkeypatch):
    # The bug case: a server that returns str UIDs must not crash and must
    # still yield every message, not silently drop them.
    _patch(monkeypatch, "1 2 3")
    results = es._list_emails()
    assert [r["uid"] for r in results] == ["3", "2", "1"]


def test_search_emails_bytes_uid(monkeypatch):
    _patch(monkeypatch, b"1 2 3")
    results = es._search_emails("hello", folders=["INBOX"])
    assert [r["uid"] for r in results] == ["3", "2", "1"]


def test_search_emails_str_uid_does_not_drop(monkeypatch):
    _patch(monkeypatch, "1 2 3")
    results = es._search_emails("hello", folders=["INBOX"])
    assert [r["uid"] for r in results] == ["3", "2", "1"]
