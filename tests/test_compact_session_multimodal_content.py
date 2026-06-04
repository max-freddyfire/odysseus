"""Regression: manual compaction must send plain text, not list repr.

The /session/{id}/compact endpoint builds the summarizer prompt from the
older history. A message's ``content`` is a plain string for normal turns,
a multimodal list of content blocks for image/vision turns, and ``None`` for
assistant turns that persisted only native tool_calls.

``compact_session`` built ``convo_text`` with ``(m.content or '')[:2000]``.
For a multimodal list, ``(list or '')`` is truthy so the or-guard never
fires, and ``list[:2000]`` is a list slice that the f-string renders as a
Python repr (``[{'type': 'text', ...}, {'type': 'image_url', ...}]``). The
LLM summarizer then receives that repr instead of the real text, so the
compaction summary is garbled for any session that included an image turn.

``_content_to_text`` (already defined in session_routes.py and used by the
txt/html/md export paths) coerces all three shapes to plain text. These tests
drive the real route handler and assert no list repr leaks into the
summarizer prompt.

The handler is called directly (not via TestClient) so the test runs without
httpx; auth, endpoint resolution, and the LLM call are stubbed so only the
prompt-construction path under test runs.
"""
import asyncio

import pytest

import routes.session_routes as sr
import src.endpoint_resolver as endpoint_resolver
import src.llm_core as llm_core
from core.models import ChatMessage, Session


def _compact_handler(manager):
    """Return the compact_session coroutine registered on the router."""
    router = sr.setup_session_routes(manager, {})
    for route in router.routes:
        if getattr(route, "name", None) == "compact_session":
            return route.endpoint
    raise AssertionError("compact_session route not found")


@pytest.fixture
def captured_convo(monkeypatch):
    """Run the compact handler with auth/endpoint/LLM stubbed; capture the
    convo_text the summarizer would receive."""
    captured = {}

    multimodal = [
        {"type": "text", "text": "describe this picture"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "in one sentence"},
    ]
    # 6+ messages so older[] is non-empty after the recent-tail keep.
    history = [
        ChatMessage(role="user", content="hello there"),
        ChatMessage(role="assistant", content="hi, how can I help?"),
        ChatMessage(role="user", content=multimodal),
        ChatMessage(role="assistant", content=None),  # native tool_calls only
        ChatMessage(role="user", content="and another thing"),
        ChatMessage(role="assistant", content="sure"),
        ChatMessage(role="user", content="more recent"),
        ChatMessage(role="assistant", content="ok"),
    ]

    class FakeManager:
        sessions = {}

        def get_session(self, session_id):
            return Session(
                id=session_id, name="chat",
                endpoint_url="http://localhost:11434", model="gpt-4o",
                owner="alice", history=list(history),
            )

        def replace_messages(self, session_id, messages):
            return True

    # Bypass ownership/auth: the bug is in prompt construction, not gating.
    monkeypatch.setattr(sr, "_verify_session_owner", lambda *a, **k: None)
    monkeypatch.setattr(sr, "get_current_user", lambda request: "alice")
    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint",
                        lambda *a, **k: ("http://localhost:11434",
                                         "gpt-4o", {}))

    async def fake_llm_call_async(url, model, messages, **kwargs):
        # messages == [system prompt, {"role": "user", "content": convo_text}]
        captured["convo_text"] = messages[-1]["content"]
        return "a summary"

    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm_call_async)

    handler = _compact_handler(FakeManager())
    result = asyncio.run(handler(request=object(), session_id="sess-1"))
    assert result["ok"] is True
    return captured["convo_text"]


def test_multimodal_turn_is_not_python_repr(captured_convo):
    # The summarizer must get the real text, never a Python list repr.
    assert "[{'type'" not in captured_convo
    assert "image_url" not in captured_convo
    # The actual text blocks survive.
    assert "describe this picture" in captured_convo
    assert "in one sentence" in captured_convo


def test_none_content_does_not_render_as_literal_none(captured_convo):
    # The assistant turn with content=None must not show as "None".
    assert "ASSISTANT: None" not in captured_convo


def test_plain_string_turns_pass_through(captured_convo):
    assert "USER: hello there" in captured_convo
    assert "ASSISTANT: hi, how can I help?" in captured_convo
