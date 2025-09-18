import json
import pytest
import sqlite3

from langchain.docstore.document import Document



from ai_math_tutor.chunk_and_embed import store_index, BookIndex, IndexEntry, extract_index




def make_fakes(to_return, holder, to_raise=None):
    """
    Build minimal fakes for:
      - init_chat_model() → FakeModel
      - ChatPromptTemplate.from_messages(...) → FakePrompt
    that support `.with_structured_output`, `.with_retry`, piping (`prompt | model`),
    and `.invoke({...})`.
    `holder` captures the last kwargs passed to `invoke` so the test can assert on index_text.
    """

    class FakeChain:
        def __init__(self):
            self.to_return = to_return
            self.to_raise = to_raise

        def invoke(self, kwargs):
            holder["invoke_kwargs"] = kwargs
            if self.to_raise:
                raise self.to_raise
            return self.to_return

    class FakeModel:
        def with_structured_output(self, schema):
            # mimic returning a runnable-like object
            return self

        def with_retry(self, retry_if_exception_type=None, wait_exponential_jitter=None, stop_after_attempt=None):
            # return a runnable-ish chain
            return FakeChain()

    class FakePrompt:
        # emulate `prompt | runnable` by returning the right-hand side
        def __or__(self, other):
            return other

    return FakeModel(), FakePrompt()


def docs(n, start_page=1, content_fn=None):
    if content_fn is None:
        content_fn = lambda i: f"Page {i} content"
    return [Document(page_content=content_fn(i), metadata={"page_num": i}) for i in range(start_page, start_page + n)]


def test_extract_index_success(monkeypatch):
    # Prepare a valid structured result
    expected = BookIndex(
        index_found=True,
        entries=[IndexEntry(term="Limits", pages=["10"]), IndexEntry(term="Series", pages=["12-13"])],
    )
    holder = {}
    fake_model, fake_prompt = make_fakes(expected, holder)

    # Patch: init_chat_model returns our fake model; ChatPromptTemplate.from_messages returns fake prompt
    import ai_math_tutor.chunk_and_embed as mod
    monkeypatch.setattr(mod, "init_chat_model", lambda *a, **k: fake_model)
    monkeypatch.setattr(mod.ChatPromptTemplate, "from_messages", lambda *a, **k: fake_prompt)

    # Feed 100 pages so lookback logic takes the last 10% (or up to MAX_LOOKBACK)
    ds = docs(100)

    result = extract_index(ds)
    assert result == expected

    # Assert the prompt input contained only the tail pages per lookback logic
    index_text = holder["invoke_kwargs"]["index_text"]
    # Should include high page numbers; quick sanity checks:
    assert "--- Page 100 ---" in index_text
    assert "--- Page 1 ---" not in index_text


def test_extract_index_handles_invoke_error(monkeypatch):
    holder = {}
    # Simulate the chain raising at invoke (e.g., server failure)
    fake_model, fake_prompt = make_fakes(to_return=None, holder=holder, to_raise=RuntimeError("boom"))

    import ai_math_tutor.chunk_and_embed as mod
    monkeypatch.setattr(mod, "init_chat_model", lambda *a, **k: fake_model)
    monkeypatch.setattr(mod.ChatPromptTemplate, "from_messages", lambda *a, **k: fake_prompt)

    ds = docs(5)
    result = extract_index(ds)
    assert result.index_found is False
    assert result.entries is None


def test_extract_index_lookback_window(monkeypatch):
    # With 30 pages, start_page = max(30 - MAX_LOOKBACK, int(30*0.9)) = max(30-50, 27) = 27
    holder = {}
    expected = BookIndex(index_found=True, entries=[])
    fake_model, fake_prompt = make_fakes(expected, holder)

    import ai_math_tutor.chunk_and_embed as mod
    monkeypatch.setattr(mod, "init_chat_model", lambda *a, **k: fake_model)
    monkeypatch.setattr(mod.ChatPromptTemplate, "from_messages", lambda *a, **k: fake_prompt)

    ds = docs(30)  # pages 1..30
    _ = extract_index(ds)

    index_text = holder["invoke_kwargs"]["index_text"]
    # Should include only pages 27..30
    assert "--- Page 26 ---" not in index_text
    assert all(f"--- Page {p} ---" in index_text for p in (27, 28, 29, 30))

def read_all(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

def test_store_index_inserts_and_populates_fts(tmp_path):
    db_path = tmp_path / "content.db"
    book_id = "book-1"
    idx = BookIndex(
        index_found=True,
        entries=[
            IndexEntry(term="Algebra", pages=["1", "2-3"]),
            IndexEntry(term="Limits", pages=["10"]),
        ],
    )

    store_index(idx, book_id, content_db_path=str(db_path))

    with sqlite3.connect(db_path) as conn:
        rows = read_all(conn, "SELECT book_id, term, pages_json FROM book_index WHERE book_id = ?", (book_id,))
        assert len(rows) == 2
        # terms stored lowercased
        terms = {r[1] for r in rows}
        assert terms == {"algebra", "limits"}
        # pages_json is a JSON string of the list
        pages_map = {r[1]: json.loads(r[2]) for r in rows}
        assert pages_map["algebra"] == ["1", "2-3"]
        assert pages_map["limits"] == ["10"]

        # FTS table populated from main table for this book
        fts_rows = read_all(conn, "SELECT term, pages_json, book_id FROM book_index_fts WHERE book_id = ?", (book_id,))
        assert len(fts_rows) == 2
        assert {r[0] for r in fts_rows} == {"algebra", "limits"}

def test_store_index_upsert_and_fts_refresh(tmp_path):
    db_path = tmp_path / "content.db"
    book_id = "book-1"

    # First insert
    store_index(
        BookIndex(index_found=True, entries=[IndexEntry(term="Limits", pages=["10"])]),
        book_id,
        content_db_path=str(db_path),
    )
    # Second call replaces and refreshes FTS (deletes old then inserts new)
    store_index(
        BookIndex(index_found=True, entries=[
            IndexEntry(term="Limits", pages=["11"]),  # updated pages
            IndexEntry(term="Derivatives", pages=["12-14"]),
        ]),
        book_id,
        content_db_path=str(db_path),
    )

    with sqlite3.connect(db_path) as conn:
        rows = read_all(conn, "SELECT term, pages_json FROM book_index WHERE book_id = ?", (book_id,))
        assert {r[0] for r in rows} == {"limits", "derivatives"}
        pages = {r[0]: json.loads(r[1]) for r in rows}
        assert pages["limits"] == ["11"]  # upsert took effect

        fts_rows = read_all(conn, "SELECT term FROM book_index_fts WHERE book_id = ?", (book_id,))
        # No duplicates; reflects the latest set exactly
        assert {r[0] for r in fts_rows} == {"limits", "derivatives"}

def test_store_index_handles_empty_entries(tmp_path):
    db_path = tmp_path / "content.db"
    book_id = "book-empty"
    idx = BookIndex(index_found=True, entries=[])

    store_index(idx, book_id, content_db_path=str(db_path))

    with sqlite3.connect(db_path) as conn:
        rows = read_all(conn, "SELECT * FROM book_index WHERE book_id = ?", (book_id,))
        assert rows == []
        fts_rows = read_all(conn, "SELECT * FROM book_index_fts WHERE book_id = ?", (book_id,))
        assert fts_rows == []

def test_store_index_bubbles_sql_errors(monkeypatch, tmp_path):
    # Force sqlite3.connect to raise to verify exceptions bubble up
    def boom(_):
        raise sqlite3.OperationalError("cannot open database file")

    monkeypatch.setattr(sqlite3, "connect", boom)
    with pytest.raises(sqlite3.OperationalError):
        store_index(BookIndex(index_found=True, entries=[]), "book-x", content_db_path=str(tmp_path / "x.db"))



def test_extract_index_happy_path(monkeypatch, tmp_path):
    # test 
    pass