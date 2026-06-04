import asyncio
import sys
import types
from types import SimpleNamespace

from src import bg_monitor


def _install_session(monkeypatch, sess):
    """Wire src.ai_interaction.get_session_manager and core.models.ChatMessage
    so _run_followup's lazy imports resolve to fakes returning `sess`."""
    sm = SimpleNamespace(
        get_session=lambda sid: sess,
        add_message=lambda *a, **k: None,
        save_sessions=lambda: None,
    )

    ai_interaction = types.ModuleType("src.ai_interaction")
    ai_interaction.get_session_manager = lambda: sm
    monkeypatch.setitem(sys.modules, "src.ai_interaction", ai_interaction)

    core_models = types.ModuleType("core.models")
    core_models.ChatMessage = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    monkeypatch.setitem(sys.modules, "core.models", core_models)

    return sm


def _make_session():
    return SimpleNamespace(
        id="s1",
        model="model",
        get_context_messages=lambda: [],
    )


def test_run_followup_defers_when_is_active_raises(monkeypatch):
    """If agent_runs.is_active raises, the guard must not be bypassed: defer
    (return False) instead of injecting into a possibly-live session."""
    sess = _make_session()
    _install_session(monkeypatch, sess)

    def boom(_sid):
        raise RuntimeError("agent_runs unavailable")

    agent_runs = types.ModuleType("src.agent_runs")
    agent_runs.is_active = boom
    monkeypatch.setitem(sys.modules, "src.agent_runs", agent_runs)

    drained = {"called": False}

    async def fake_drain(*a, **k):
        drained["called"] = True
        return ("", [])

    monkeypatch.setattr(bg_monitor, "_drain_agent", fake_drain)

    result = asyncio.run(bg_monitor._run_followup({"id": "j1", "session_id": "s1"}))

    assert result is False
    assert drained["called"] is False  # must not inject when guard check failed


def test_run_followup_defers_when_session_active(monkeypatch):
    """Existing behavior: a busy (live-turn) session defers."""
    sess = _make_session()
    _install_session(monkeypatch, sess)

    agent_runs = types.ModuleType("src.agent_runs")
    agent_runs.is_active = lambda _sid: True
    monkeypatch.setitem(sys.modules, "src.agent_runs", agent_runs)

    drained = {"called": False}

    async def fake_drain(*a, **k):
        drained["called"] = True
        return ("", [])

    monkeypatch.setattr(bg_monitor, "_drain_agent", fake_drain)

    result = asyncio.run(bg_monitor._run_followup({"id": "j1", "session_id": "s1"}))

    assert result is False
    assert drained["called"] is False


def test_run_followup_proceeds_when_session_idle(monkeypatch):
    """Happy path: an idle session injects the result and reports completion."""
    sess = _make_session()
    _install_session(monkeypatch, sess)

    agent_runs = types.ModuleType("src.agent_runs")
    agent_runs.is_active = lambda _sid: False
    monkeypatch.setitem(sys.modules, "src.agent_runs", agent_runs)

    monkeypatch.setattr(bg_monitor.bg_jobs, "result_text", lambda rec: "job output")

    drained = {"called": False}

    async def fake_drain(*a, **k):
        drained["called"] = True
        return ("done", [])

    monkeypatch.setattr(bg_monitor, "_drain_agent", fake_drain)

    result = asyncio.run(bg_monitor._run_followup({"id": "j1", "session_id": "s1"}))

    assert result is True
    assert drained["called"] is True
