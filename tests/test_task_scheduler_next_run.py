import asyncio
from datetime import timedelta

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

import src.task_scheduler as ts
from src.task_scheduler import TaskScheduler, _utcnow


def _setup_db(tmp_path, monkeypatch):
    import core.database as cd

    base = declarative_base()

    class ScheduledTask(base):
        __tablename__ = "scheduled_tasks"

        id = Column(String, primary_key=True)
        owner = Column(String)
        name = Column(String)
        task_type = Column(String, default="llm")
        status = Column(String, default="active")
        trigger_type = Column(String, default="schedule")
        schedule = Column(String)
        scheduled_time = Column(String)
        scheduled_day = Column(String)
        scheduled_date = Column(String)
        cron_expression = Column(String)
        notifications_enabled = Column(String)
        last_run = Column(DateTime)
        next_run = Column(DateTime)
        run_count = Column(String)

    class TaskRun(base):
        __tablename__ = "task_runs"

        id = Column(String, primary_key=True)
        task_id = Column(String)
        started_at = Column(DateTime)
        finished_at = Column(DateTime)
        status = Column(String)
        result = Column(Text)
        error = Column(Text)
        model = Column(String)

    engine = create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")
    base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(cd, "SessionLocal", session_local)
    monkeypatch.setattr(cd, "ScheduledTask", ScheduledTask)
    monkeypatch.setattr(cd, "TaskRun", TaskRun)
    return session_local, ScheduledTask, TaskRun


def _make_scheduler():
    scheduler = TaskScheduler.__new__(TaskScheduler)
    scheduler._executing = set()
    scheduler._executing_lock = asyncio.Lock()
    scheduler._task_handles = {}
    scheduler._task_defer_counts = {}
    scheduler._last_run_model = None
    return scheduler


def test_next_run_advances_when_compute_raises_on_error_path(tmp_path, monkeypatch):
    """When the task executor fails AND compute_next_run raises in the error
    path, next_run must be pushed into the future (5-min stall) so the broken
    task doesn't busy-loop the scheduler with a stale past date."""
    session_local, ScheduledTask, TaskRun = _setup_db(tmp_path, monkeypatch)

    past = _utcnow() - timedelta(hours=1)
    db = session_local()
    db.add(ScheduledTask(
        id="t1", owner="alice", name="Broken Task",
        task_type="llm", status="active", trigger_type="schedule",
        schedule="daily", next_run=past,
    ))
    db.add(TaskRun(id="r1", task_id="t1", status="queued"))
    db.commit()
    db.close()

    # Force the executor to fail so we enter the error path.
    async def boom_exec(self, task, db):
        raise RuntimeError("executor blew up")

    monkeypatch.setattr(TaskScheduler, "_execute_llm_task", boom_exec)

    # Avoid DB-coupled timezone lookup; the bug is about compute_next_run.
    monkeypatch.setattr(ts, "_resolve_task_timezone", lambda db, task: None)

    # The recompute itself raises — this is the path the fix guards.
    def boom_compute(*args, **kwargs):
        raise ValueError("bad cron")

    monkeypatch.setattr(ts, "compute_next_run", boom_compute)

    before = _utcnow()
    asyncio.run(_make_scheduler()._execute_task_locked("t1", "r1"))
    after = _utcnow()

    db = session_local()
    try:
        task = db.query(ScheduledTask).filter(ScheduledTask.id == "t1").first()
        assert task.next_run is not None
        # Advanced past the original stale date and past "now".
        assert task.next_run > after, "next_run must not remain in the past"
        # Roughly a 5-minute deferral from when the error path ran.
        lo = before + timedelta(minutes=5) - timedelta(seconds=5)
        hi = after + timedelta(minutes=5) + timedelta(seconds=5)
        assert lo <= task.next_run <= hi
    finally:
        db.close()
