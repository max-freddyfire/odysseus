"""Issue #2179 — a reopened (non-active) document must be readable, not just editable.

`is_active` flags the single document currently open in a session (see
core.database.Document). Importing or opening another document flips every other
owned doc to `is_active=False` (routes/email_routes.py). So at most one owned
document is active at a time; every other reopenable doc is inactive.

The `manage_documents` read action filtered `is_active == True`, while
`edit_document` did not. Reading a reopened (inactive) document therefore failed
with "not found", yet editing the same document succeeded — exactly the reported
"read error but edits anyway" behaviour. Read must match edit: any owned
document is readable by explicit id, regardless of the single-active flag.
"""

import asyncio
import json
import sys
import types
from types import SimpleNamespace


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return (self.name, "eq", value)

    def desc(self):
        return (self.name, "desc")

    def ilike(self, value):
        return (self.name, "ilike", value)


class _Document:
    id = _Column("id")
    owner = _Column("owner")
    is_active = _Column("is_active")
    title = _Column("title")
    language = _Column("language")
    updated_at = _Column("updated_at")


class _Query:
    def __init__(self, first_doc=None):
        self.filters = []
        self.first_doc = first_doc

    def filter(self, *clauses):
        self.filters.extend(clauses)
        return self

    def order_by(self, *args):
        return self

    def first(self):
        return self.first_doc


class _Db:
    def __init__(self, query):
        self.query_obj = query

    def query(self, *args):
        return self.query_obj

    def close(self):
        pass


def _install_database_stub(monkeypatch, module_name, query):
    db = _Db(query)
    db_mod = types.ModuleType(module_name)
    db_mod.SessionLocal = lambda: db
    db_mod.Document = _Document
    db_mod.DocumentVersion = object
    db_mod.Session = object
    monkeypatch.setitem(sys.modules, module_name, db_mod)
    return db


def test_read_returns_inactive_owned_document(monkeypatch):
    from src import tool_implementations as tools

    inactive_doc = SimpleNamespace(
        id="doc-1",
        title="Reopened doc",
        language="markdown",
        current_content="hello world",
    )
    query = _Query(first_doc=inactive_doc)
    _install_database_stub(monkeypatch, "core.database", query)

    result = asyncio.run(
        tools.do_manage_documents(
            json.dumps({"action": "read", "document_id": "doc-1"}), owner="alice"
        )
    )

    # Read must not require the single-active flag; it should return the doc.
    assert result["exit_code"] == 0
    assert result["document"]["id"] == "doc-1"
    assert ("is_active", "eq", True) not in query.filters
    # Owner scoping must still hold.
    assert ("id", "eq", "doc-1") in query.filters
    assert ("owner", "eq", "alice") in query.filters
