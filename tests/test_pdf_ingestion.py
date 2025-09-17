import asyncio
import json
import pytest
import types

import ai_math_tutor.extract_content_from_pdf as mod

def make_dummy_pdf(num_pages: int):
    """Return a context-manager that mimics `pymupdf.open(...)` with num_page pages."""
    class DummyDoc:
        def __enter__(self):
            # pages can be any objects; our mocked `async_call_llm` ignores them
            self._pages = [object() for _ in range(num_pages)]
            return self._pages  # iterable over pages

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self._pages)

    return DummyDoc()


def test_store_pages_in_json(tmp_path):
    data = [{"page_num": 1, "page_content": "hello"}]
    out = tmp_path / "out.json"
    path = mod.store_pages_in_json(data, str(out))
    assert path == str(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == data


@pytest.mark.asyncio
async def test_extract_content_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        await mod.extract_content(str(tmp_path / "missing.pdf"))

@pytest.mark.asyncio
async def test_extract_content_single_page_success(monkeypatch, tmp_path):
    # 1 page pdf
    monkeypatch.setattr(mod, "_client", lambda: object())
    monkeypatch.setattr(mod, "pymupdf", types.SimpleNamespace(
        open=lambda path: make_dummy_pdf(1),
    ))

    async def fake_async_call_llm(page, i, client):
        await asyncio.sleep(0)  # yield control to simulate async work
        return {"page_content": f"content for page {i}", "page_num": i}

    monkeypatch.setattr(mod, "async_call_llm", fake_async_call_llm)

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.touch()  # create dummy file to avoid filenotfounderror

    results = await mod.extract_content(str(pdf_path))
    assert results == [{"page_content": "content for page 1", "page_num": 1}]

@pytest.mark.asyncio
async def test_extract_content_multiple_pages_and_error(monkeypatch, tmp_path):
    # 3 pages where page 2 raises an error and should be replaced by placeholder
    monkeypatch.setattr(mod, "_client", lambda: object())
    monkeypatch.setattr(mod, "pymupdf", types.SimpleNamespace(
        open=lambda path: make_dummy_pdf(3),
    ))

    async def fake_async_call_llm(page, i, client):
        if i == 2:
            raise RuntimeError("LLM server failed")
        return {"page_content": f"ok-{i}", "page_num": i}

    monkeypatch.setattr(mod, "async_call_llm", fake_async_call_llm)

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.touch()  # create dummy file to avoid filenotfounderror

    results = await mod.extract_content(str(pdf_path))
    assert len(results) == 3
    # Success pages preserved
    assert results[0] == {"page_content": "ok-1", "page_num": 1}
    assert results[2] == {"page_content": "ok-3", "page_num": 3}
    # Failed page replaced with placeholder and error info present
    assert results[1]["page_num"] == 2
    assert results[1]["page_content"] == mod.MISSING_PAGE_PLACEHOLDER
    assert "error" in results[1] and "Error processing page 2" in results[1]["error"]

@pytest.mark.asyncio
async def test_extract_content_preserves_page_order(monkeypatch, tmp_path):
    # Ensure results map page_num 1..N correctly even if tasks finish out of order
    monkeypatch.setattr(mod, "_client", lambda: object())
    monkeypatch.setattr(mod, "pymupdf", types.SimpleNamespace(
        open=lambda path: make_dummy_pdf(4),
    ))

    async def fake_async_call_llm(page, i, client):
        # Make even pages slower to finish
        if i % 2 == 0:
            await asyncio.sleep(0.02)
        else:
            await asyncio.sleep(0.001)
        return {"page_content": f"p{i}", "page_num": i}

    monkeypatch.setattr(mod, "async_call_llm", fake_async_call_llm)

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.touch()  # create dummy file to avoid filenotfounderror

    results = await mod.extract_content(str(pdf_path))
    # The function returns results in the order tasks were gathered (page order)
    assert [r["page_num"] for r in results] == [1, 2, 3, 4]
    assert [r["page_content"] for r in results] == ["p1", "p2", "p3", "p4"]