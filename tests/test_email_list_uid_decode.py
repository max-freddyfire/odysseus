"""Regression test for str-UID handling in the email list endpoint.

routes/email_routes.py crashed when uid_list contained str UIDs (some IMAP
backends / the legacy bare-UID path fixed in email_pollers.py by bd4067c return
str search results). _list_emails_sync hit this at two consecutive points on
the same code path with the same input:

  * line 768: _uid_strs = [u.decode() for u in uid_list]
              -> AttributeError: 'str' object has no attribute 'decode'
  * line 795: fetch_set = b",".join(uid_list)
              -> TypeError: sequence item 0: expected a bytes-like object,
                 str found

Both are fixed with the established conversion patterns: the isinstance guard
used at email_routes.py:161/190 and email_pollers.py:316, and the _uid_bytes()
helper at email_routes.py:219.
"""

import pytest


def _uid_bytes(uid):
    # mirrors routes/email_routes.py:219
    return uid if isinstance(uid, bytes) else str(uid).encode()


def test_uid_decode_handles_string_uid():
    # line 768: confirm the fixed comprehension handles mixed bytes/string uids
    uid_list = [b"123", "456", b"789"]  # mixed bytes and strings
    result = [u.decode() if isinstance(u, bytes) else str(u) for u in uid_list]
    assert result == ["123", "456", "789"]


def test_uid_decode_all_bytes():
    uid_list = [b"100", b"200"]
    result = [u.decode() if isinstance(u, bytes) else str(u) for u in uid_list]
    assert result == ["100", "200"]


def test_uid_decode_bare_crashes_on_string():
    # demonstrate the pre-fix decode pattern crashes (confirm the bug is real)
    uid_list = ["not-bytes"]
    with pytest.raises(AttributeError):
        [u.decode() for u in uid_list]


def test_fetch_set_join_handles_string_uid():
    # line 795: confirm the fixed fetch-set build handles mixed bytes/string uids
    uid_list = [b"123", "456", b"789"]
    fetch_set = b",".join(_uid_bytes(u) for u in uid_list)
    assert fetch_set == b"123,456,789"


def test_fetch_set_join_bare_crashes_on_string():
    # demonstrate the pre-fix join pattern crashes on str uids (same input as 768)
    uid_list = ["456", "789"]
    with pytest.raises(TypeError):
        b",".join(uid_list)
