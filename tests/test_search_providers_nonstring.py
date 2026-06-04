"""Regression: searxng_search_api must tolerate a non-string query.

`searxng_search_api` did `query.lower()` directly to detect news queries, before
the surrounding try block. A None / non-string query (e.g. from a caller that
didn't coerce) therefore raised an UNCAUGHT AttributeError that aborted the
search. The sibling helpers in services/search/query.py already guard with
`if not isinstance(query, str)`; this pins the same contract on providers.

The crash sits on the `q_lc = query.lower()` line, before any HTTP call, so the
test stays offline: httpx.get is mocked and _get_search_instance is patched.
"""

from services.search import providers


class _JSONResponse:
    """A valid SearXNG JSON API response."""

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {"title": "Result", "url": "https://example.com", "content": "Snippet"}
            ]
        }


def _patch_offline(monkeypatch, fake_get):
    monkeypatch.setattr(providers, "_get_search_instance", lambda: "http://searx.test")
    monkeypatch.setattr(providers, "_get_search_settings", lambda: {"search_safesearch": "off"})
    monkeypatch.setattr(providers.httpx, "get", fake_get)


def test_searxng_api_none_query_does_not_raise(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen["params"] = kwargs["params"]
        return _JSONResponse()

    _patch_offline(monkeypatch, fake_get)

    # Without the guard this raises AttributeError on query.lower() before the
    # try block, so it would never reach the JSON path at all.
    results = providers.searxng_search_api(None, count=1)

    # Coerced to "", the JSON API path completes and returns the parsed result.
    assert results == [
        {"title": "Result", "url": "https://example.com", "snippet": "Snippet"}
    ]
    assert seen["params"]["format"] == "json"
    assert seen["params"]["q"] == ""


def test_searxng_api_nonstring_query_does_not_raise(monkeypatch):
    def fake_get(url, **kwargs):
        return _JSONResponse()

    _patch_offline(monkeypatch, fake_get)

    # An int (or any non-string truthy value) hit the same query.lower() crash.
    results = providers.searxng_search_api(123, count=1)

    assert results == [
        {"title": "Result", "url": "https://example.com", "snippet": "Snippet"}
    ]


def test_searxng_api_valid_query_still_works(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen["params"] = kwargs["params"]
        return _JSONResponse()

    _patch_offline(monkeypatch, fake_get)

    results = providers.searxng_search_api("odysseus", count=1)

    assert results
    assert seen["params"]["q"] == "odysseus"
