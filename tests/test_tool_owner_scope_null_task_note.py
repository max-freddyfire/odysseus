"""Regression: null-owner records must not be writable by other users.

The HTTP route layer (routes/task_routes.py, routes/note_routes.py) uses a
strict ownership gate: a record whose `owner` is None is treated as belonging
to nobody, so an authenticated caller is blocked. The tool-implementation
layer used a lenient `record.owner and ...` short-circuit that let ANY
authenticated caller mutate a null-owner row. These tests pin the strict
behavior for do_manage_tasks and do_manage_notes.
"""

import asyncio
import sys
import types

from src import tool_implementations as tools


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return (self.name, "eq", value)

    def startswith(self, value):
        return (self.name, "startswith", value)


class _ScheduledTask:
    id = _Column("id")
    owner = _Column("owner")
    created_at = _Column("created_at")


class _Note:
    id = _Column("id")
    owner = _Column("owner")
    archived = _Column("archived")
    due_date = _Column("due_date")


class _Query:
    """Minimal query stub: filter/order_by are no-ops, first() returns the
    preset record so the ownership gate (not the lookup) is what is tested."""

    def __init__(self, first_obj=None):
        self.first_obj = first_obj

    def filter(self, *clauses):
        return self

    def order_by(self, *args):
        return self

    def limit(self, *args):
        return self

    def all(self):
        return []

    def first(self):
        return self.first_obj


class _Db:
    def __init__(self, query):
        self.query_obj = query
        self.committed = False
        self.deleted = []

    def query(self, *args):
        return self.query_obj

    def add(self, obj):
        pass

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True

    def close(self):
        pass


def _install_task_db(monkeypatch, query):
    db = _Db(query)
    db_mod = types.ModuleType("core.database")
    db_mod.SessionLocal = lambda: db
    db_mod.ScheduledTask = _ScheduledTask
    monkeypatch.setitem(sys.modules, "core.database", db_mod)
    return db


def _install_note_db(monkeypatch, query):
    db = _Db(query)
    db_mod = types.ModuleType("core.database")
    db_mod.SessionLocal = lambda: db
    db_mod.Note = _Note
    monkeypatch.setitem(sys.modules, "core.database", db_mod)
    # do_manage_notes imports flag_modified from sqlalchemy at function entry;
    # the env has no sqlalchemy, so stub the attribute module.
    attrs_mod = types.ModuleType("sqlalchemy.orm.attributes")
    attrs_mod.flag_modified = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm.attributes", attrs_mod)
    return db


class _FakeTask:
    def __init__(self, owner):
        self.id = "t1"
        self.owner = owner
        self.name = "legacy task"
        self.status = "active"
        self.trigger_type = "schedule"


class _FakeNote:
    def __init__(self, owner):
        self.id = "n1note"
        self.owner = owner
        self.title = "legacy note"
        self.items = None


# --- tasks -----------------------------------------------------------------


def test_edit_null_owner_task_denies_other_user(monkeypatch):
    db = _install_task_db(monkeypatch, _Query(first_obj=_FakeTask(owner=None)))

    result = asyncio.run(
        tools.do_manage_tasks(
            '{"action":"edit","task_id":"t1","name":"hijacked"}', owner="alice"
        )
    )

    assert result.get("error") == "Access denied"
    assert result.get("exit_code") == 1
    assert db.committed is False


def test_delete_null_owner_task_denies_other_user(monkeypatch):
    db = _install_task_db(monkeypatch, _Query(first_obj=_FakeTask(owner=None)))

    result = asyncio.run(
        tools.do_manage_tasks('{"action":"delete","task_id":"t1"}', owner="alice")
    )

    assert result.get("error") == "Access denied"
    assert result.get("exit_code") == 1
    assert db.deleted == []


def test_pause_null_owner_task_denies_other_user(monkeypatch):
    db = _install_task_db(monkeypatch, _Query(first_obj=_FakeTask(owner=None)))

    result = asyncio.run(
        tools.do_manage_tasks('{"action":"pause","task_id":"t1"}', owner="alice")
    )

    assert result.get("error") == "Access denied"
    assert result.get("exit_code") == 1
    assert db.committed is False


def test_edit_owned_task_still_allowed_for_owner(monkeypatch):
    """Sanity: the strict gate must not lock owners out of their own tasks."""
    db = _install_task_db(monkeypatch, _Query(first_obj=_FakeTask(owner="alice")))

    result = asyncio.run(
        tools.do_manage_tasks(
            '{"action":"edit","task_id":"t1","name":"renamed"}', owner="alice"
        )
    )

    assert result.get("exit_code") == 0
    assert db.committed is True


# --- notes -----------------------------------------------------------------


def test_update_null_owner_note_denies_other_user(monkeypatch):
    db = _install_note_db(monkeypatch, _Query(first_obj=_FakeNote(owner=None)))

    result = asyncio.run(
        tools.do_manage_notes(
            '{"action":"update","id":"n1note","title":"hijacked"}', owner="alice"
        )
    )

    assert result.get("error") == "Note not found"
    assert result.get("exit_code") == 1
    assert db.committed is False


def test_delete_null_owner_note_denies_other_user(monkeypatch):
    db = _install_note_db(monkeypatch, _Query(first_obj=_FakeNote(owner=None)))

    result = asyncio.run(
        tools.do_manage_notes('{"action":"delete","id":"n1note"}', owner="alice")
    )

    assert result.get("error") == "Note not found"
    assert result.get("exit_code") == 1
    assert db.deleted == []


def test_toggle_item_null_owner_note_denies_other_user(monkeypatch):
    db = _install_note_db(monkeypatch, _Query(first_obj=_FakeNote(owner=None)))

    result = asyncio.run(
        tools.do_manage_notes(
            '{"action":"toggle_item","id":"n1note","index":0}', owner="alice"
        )
    )

    assert result.get("error") == "Note not found"
    assert result.get("exit_code") == 1
    assert db.committed is False


def test_update_owned_note_still_allowed_for_owner(monkeypatch):
    """Sanity: the strict gate must not lock owners out of their own notes."""
    db = _install_note_db(monkeypatch, _Query(first_obj=_FakeNote(owner="alice")))

    result = asyncio.run(
        tools.do_manage_notes(
            '{"action":"update","id":"n1note","title":"renamed"}', owner="alice"
        )
    )

    assert result.get("exit_code") == 0
    assert db.committed is True
