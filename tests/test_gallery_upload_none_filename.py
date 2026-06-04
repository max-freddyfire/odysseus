"""gallery_upload must not 500 when the uploaded part has no filename.

``UploadFile.filename`` is typed ``Optional[str]`` and is ``None`` when a
multipart part carries no ``filename`` (the handler must not assume a string).
The gallery upload handler did ``"." in file.filename``, which raises
``TypeError: argument of type 'NoneType' is not iterable`` and surfaces as a
500. Every sibling upload handler (calendar/email/memory) already guards this;
this one did not.

Calls the async route handler DIRECTLY with a minimal fake request (same style
as test_caldav_writeback_route.py) so we exercise the filename logic without
the auth middleware or a live multipart parse.
"""

import asyncio
import io
from types import SimpleNamespace

import pytest
from starlette.datastructures import FormData, UploadFile

import routes.gallery_routes as g


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None


class _FakeDB:
    """Captures the GalleryImage that would be persisted."""

    def __init__(self, sink):
        self._sink = sink

    def query(self, *a, **k):
        return _FakeQuery()

    def add(self, obj):
        self._sink.append(obj)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def upload_handler(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "get_current_user", lambda req: "tester")

    added = []
    monkeypatch.setattr(g, "SessionLocal", lambda: _FakeDB(added))

    # Keep written bytes out of the repo's data/ dir.
    monkeypatch.chdir(tmp_path)

    router = g.setup_gallery_routes()
    for r in router.routes:
        if getattr(r, "path", "") == "/api/gallery/upload" and "POST" in getattr(r, "methods", set()):
            return r.endpoint, added
    raise RuntimeError("POST /api/gallery/upload not found")


def _req(upload_file):
    form = FormData([("file", upload_file)])

    async def _form():
        return form

    return SimpleNamespace(form=_form, state=SimpleNamespace(current_user="tester"))


def test_upload_with_none_filename_does_not_500(upload_handler):
    handler, added = upload_handler
    uf = UploadFile(filename=None, file=io.BytesIO(b"\x89PNG\r\n\x1a\nfake-image-bytes"))

    res = asyncio.run(handler(_req(uf)))

    # Must succeed with a fallback name, not raise TypeError -> 500.
    assert res.get("ok") is True
    assert added, "image row should have been persisted"
    # original_name falls back to a sensible default when filename is missing.
    assert added[0].prompt == "upload"
