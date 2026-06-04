"""Tests for ResearchHandler.synthesize_query with multimodal/None content.

ChatMessage.content carries three shapes: a plain string (text turn), a list of
content blocks (vision/image turn, e.g. [{"type":"text","text":"..."},
{"type":"image_url",...}]), or None (assistant turns that persisted only native
tool_calls). synthesize_query passed content straight to ``.strip()`` and
``[:500]`` assuming str, so a session with an image turn crashed in the fallback
path and rendered Python list repr into the synthesis prompt.
"""
import pytest

from core.models import ChatMessage, Session
from src.research_handler import ResearchHandler


def _session(history):
    return Session(
        id="s1", name="t", endpoint_url="http://local.test", model="m",
        history=[ChatMessage(role, content) for role, content in history],
    )


@pytest.fixture
def handler():
    return ResearchHandler()


async def _raise(*args, **kwargs):
    raise RuntimeError("synthesis unavailable")


@pytest.mark.asyncio
async def test_multimodal_history_does_not_crash_in_fallback(handler, monkeypatch):
    # Latest message is a bare affirmation, so _fallback() scans history for the
    # original ask. An earlier image turn carries list content; the old code did
    # (list or "").strip() -> AttributeError. It must instead skip the image
    # block's empty text and return the substantive text ask.
    monkeypatch.setattr("src.llm_core.llm_call_async", _raise)
    sess = _session([
        ("user", [
            {"type": "text", "text": "What breed is this dog?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]),
        ("assistant", "Want me to research this breed in depth?"),
    ])
    result = await handler.synthesize_query(sess, "yes", "http://local.test", "m")
    assert result == "What breed is this dog?"


@pytest.mark.asyncio
async def test_multimodal_content_flattened_in_convo_prompt(handler, monkeypatch):
    # The convo builder must pass plain text, not Python list repr, to the LLM.
    captured = {}

    async def _capture(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return "synthesized query about dog breeds"

    monkeypatch.setattr("src.llm_core.llm_call_async", _capture)
    sess = _session([
        ("user", [
            {"type": "text", "text": "What breed is this dog?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]),
        ("assistant", "Any constraints on the answer?"),
    ])
    await handler.synthesize_query(
        sess, "focus on temperament and size", "http://local.test", "m",
    )
    prompt = captured["messages"][0]["content"]
    assert "What breed is this dog?" in prompt
    assert "'type': 'image_url'" not in prompt
    assert "[{" not in prompt


@pytest.mark.asyncio
async def test_none_content_in_history_handled(handler, monkeypatch):
    # A tool-call assistant turn persists content=None. It must not crash and
    # must not be treated as the substantive user ask.
    monkeypatch.setattr("src.llm_core.llm_call_async", _raise)
    sess = _session([
        ("user", "Compare national healthcare systems."),
        ("assistant", None),
        ("assistant", "Want me to go ahead?"),
    ])
    result = await handler.synthesize_query(sess, "yes", "http://local.test", "m")
    assert result == "Compare national healthcare systems."
