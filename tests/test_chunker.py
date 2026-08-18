"""Unit tests for ingestion.chunker – page-aware text chunking (Milestone 3)."""

from __future__ import annotations

import hashlib

import pytest

from ingestion.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunk,
    _make_chunk_id,
    _normalize_text,
    _split_page,
    chunk_pages,
)
from ingestion.pdf_loader import PageContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pages(*texts: str) -> list[PageContent]:
    """Build a list of PageContent objects (1-based page numbers)."""
    return [PageContent(page_number=i + 1, text=t) for i, t in enumerate(texts)]


DOC_ID = "doc-test-001"


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _normalize_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self) -> None:
        assert _normalize_text("foo    bar") == "foo bar"

    def test_collapses_tabs(self) -> None:
        assert _normalize_text("foo\t\tbar") == "foo bar"

    def test_collapses_mixed_horizontal_whitespace(self) -> None:
        assert _normalize_text("a \t b") == "a b"

    def test_preserves_single_newline(self) -> None:
        result = _normalize_text("line one\nline two")
        assert result == "line one\nline two"

    def test_collapses_multiple_blank_lines(self) -> None:
        result = _normalize_text("para one\n\n\n\npara two")
        assert result == "para one\n\npara two"

    def test_empty_string(self) -> None:
        assert _normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        assert _normalize_text("   \t\n  ") == ""

    def test_crlf_normalised(self) -> None:
        assert _normalize_text("a\r\nb") == "a\nb"

    def test_meaningful_text_preserved(self) -> None:
        text = "Section 1: Introduction\nThis tender requires compliance."
        assert _normalize_text(text) == text


# ---------------------------------------------------------------------------
# _split_page
# ---------------------------------------------------------------------------

class TestSplitPage:
    def test_short_text_single_chunk(self) -> None:
        chunks = _split_page("hello world", chunk_size=50, overlap=10)
        assert chunks == ["hello world"]

    def test_exact_size_single_chunk(self) -> None:
        text = "a" * 50
        chunks = _split_page(text, chunk_size=50, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self) -> None:
        text = "x" * 200
        chunks = _split_page(text, chunk_size=100, overlap=0)
        assert len(chunks) == 2

    def test_overlap_content_shared(self) -> None:
        # With size=10 and overlap=5: step=5
        # "0123456789abcdefghij" (20 chars) → starts [0,5,10,15] → 4 windows
        text = "0123456789abcdefghij"  # 20 chars
        chunks = _split_page(text, chunk_size=10, overlap=5)
        assert len(chunks) == 4
        # Every adjacent pair shares the trailing/leading 5 characters
        for i in range(len(chunks) - 1):
            assert chunks[i][-5:] == chunks[i + 1][:5]

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            _split_page("text", chunk_size=0, overlap=0)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            _split_page("text", chunk_size=10, overlap=-1)

    def test_overlap_gte_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            _split_page("text", chunk_size=10, overlap=10)


# ---------------------------------------------------------------------------
# _make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_deterministic(self) -> None:
        id1 = _make_chunk_id("doc-a", 1, 0)
        id2 = _make_chunk_id("doc-a", 1, 0)
        assert id1 == id2

    def test_different_documents_differ(self) -> None:
        assert _make_chunk_id("doc-a", 1, 0) != _make_chunk_id("doc-b", 1, 0)

    def test_different_pages_differ(self) -> None:
        assert _make_chunk_id("doc-a", 1, 0) != _make_chunk_id("doc-a", 2, 0)

    def test_different_indices_differ(self) -> None:
        assert _make_chunk_id("doc-a", 1, 0) != _make_chunk_id("doc-a", 1, 1)

    def test_returns_16_hex_chars(self) -> None:
        cid = _make_chunk_id("doc-a", 1, 0)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_sha256_derivation(self) -> None:
        raw = "doc-a:p1:c0"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert _make_chunk_id("doc-a", 1, 0) == expected


# ---------------------------------------------------------------------------
# chunk_pages – main public API
# ---------------------------------------------------------------------------

class TestChunkPages:
    # -----------------------------------------------------------------------
    # Basic structural tests
    # -----------------------------------------------------------------------

    def test_short_text_produces_one_chunk(self) -> None:
        pages = _pages("Short text.")
        chunks = chunk_pages(pages, DOC_ID, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self) -> None:
        long_text = "word " * 300  # ~1500 chars
        pages = _pages(long_text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=200, overlap=40)
        assert len(chunks) > 1

    def test_returns_chunk_instances(self) -> None:
        pages = _pages("Hello TenderLens.")
        chunks = chunk_pages(pages, DOC_ID)
        assert all(isinstance(c, Chunk) for c in chunks)

    # -----------------------------------------------------------------------
    # Page metadata preservation
    # -----------------------------------------------------------------------

    def test_page_metadata_preserved_single(self) -> None:
        pages = _pages("Content on page 1.")
        chunks = chunk_pages(pages, DOC_ID)
        assert all(c.page == 1 for c in chunks)

    def test_page_metadata_preserved_multi(self) -> None:
        pages = _pages("Page one content.", "Page two content.", "Page three content.")
        chunks = chunk_pages(pages, DOC_ID, chunk_size=500, overlap=50)
        page_nums = [c.page for c in chunks]
        assert page_nums == [1, 2, 3]

    def test_page_number_matches_source(self) -> None:
        """Chunks from page 7 must carry page=7."""
        pages = [PageContent(page_number=7, text="Tender clause 7 content here.")]
        chunks = chunk_pages(pages, DOC_ID)
        assert chunks[0].page == 7

    # -----------------------------------------------------------------------
    # Document ID preservation
    # -----------------------------------------------------------------------

    def test_document_id_preserved(self) -> None:
        pages = _pages("Some document text.")
        chunks = chunk_pages(pages, DOC_ID)
        assert all(c.document_id == DOC_ID for c in chunks)

    # -----------------------------------------------------------------------
    # Sequential chunk indices
    # -----------------------------------------------------------------------

    def test_chunk_indices_sequential_from_zero(self) -> None:
        long_text = "w " * 400  # produces multiple chunks
        pages = _pages(long_text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=100, overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_indices_sequential_across_pages(self) -> None:
        pages = _pages("a " * 200, "b " * 200, "c " * 200)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=100, overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    # -----------------------------------------------------------------------
    # Overlap correctness
    # -----------------------------------------------------------------------

    def test_overlap_is_preserved_in_content(self) -> None:
        # Use a predictable text so overlap is verifiable
        text = "abcdefghij" * 10  # 100 chars, known content
        pages = _pages(text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=30, overlap=10)
        # The last 10 chars of chunk[0] should appear at the start of chunk[1]
        if len(chunks) >= 2:
            tail_of_first = chunks[0].text[-10:]
            head_of_second = chunks[1].text[:10]
            assert tail_of_first == head_of_second

    # -----------------------------------------------------------------------
    # Whitespace normalisation
    # -----------------------------------------------------------------------

    def test_whitespace_normalised_in_chunk_text(self) -> None:
        pages = _pages("  tender   document   clause  ")
        chunks = chunk_pages(pages, DOC_ID)
        assert chunks[0].text == "tender document clause"

    def test_tab_normalised_in_chunk_text(self) -> None:
        pages = _pages("item\tone\ttwo")
        chunks = chunk_pages(pages, DOC_ID)
        assert chunks[0].text == "item one two"

    # -----------------------------------------------------------------------
    # Empty / whitespace-only pages
    # -----------------------------------------------------------------------

    def test_empty_page_produces_no_chunks(self) -> None:
        pages = _pages("")
        chunks = chunk_pages(pages, DOC_ID)
        assert chunks == []

    def test_whitespace_only_page_produces_no_chunks(self) -> None:
        pages = _pages("   \t\n   ")
        chunks = chunk_pages(pages, DOC_ID)
        assert chunks == []

    def test_empty_pages_list_produces_no_chunks(self) -> None:
        chunks = chunk_pages([], DOC_ID)
        assert chunks == []

    def test_mixed_empty_and_nonempty_pages(self) -> None:
        pages = _pages("", "real content here", "   ")
        chunks = chunk_pages(pages, DOC_ID, chunk_size=500, overlap=50)
        # Only the middle page contributes
        assert len(chunks) == 1
        assert chunks[0].page == 2

    # -----------------------------------------------------------------------
    # Page boundaries not crossed
    # -----------------------------------------------------------------------

    def test_no_cross_page_chunks(self) -> None:
        page1 = "alpha " * 200
        page2 = "beta " * 200
        pages = _pages(page1, page2)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=100, overlap=20)
        page1_chunks = [c for c in chunks if c.page == 1]
        page2_chunks = [c for c in chunks if c.page == 2]
        # Every page-1 chunk should only contain "alpha"
        for c in page1_chunks:
            assert "beta" not in c.text
        # Every page-2 chunk should only contain "beta"
        for c in page2_chunks:
            assert "alpha" not in c.text

    # -----------------------------------------------------------------------
    # Chunk ID determinism
    # -----------------------------------------------------------------------

    def test_chunk_ids_deterministic(self) -> None:
        pages = _pages("Hello deterministic chunking!")
        chunks1 = chunk_pages(pages, DOC_ID)
        chunks2 = chunk_pages(pages, DOC_ID)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_ids_unique_within_document(self) -> None:
        long_text = "z " * 500
        pages = _pages(long_text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=100, overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    # -----------------------------------------------------------------------
    # Configurable parameters
    # -----------------------------------------------------------------------

    def test_custom_chunk_size_respected(self) -> None:
        text = "a" * 1000
        pages = _pages(text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=200, overlap=0)
        # 1000 / 200 = 5 chunks
        assert len(chunks) == 5

    def test_zero_overlap_no_repetition(self) -> None:
        text = "0123456789" * 10  # 100 chars
        pages = _pages(text)
        chunks = chunk_pages(pages, DOC_ID, chunk_size=50, overlap=0)
        combined = "".join(c.text for c in chunks)
        assert combined == text

    def test_default_parameters_exist(self) -> None:
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_OVERLAP >= 0
        assert DEFAULT_OVERLAP < DEFAULT_CHUNK_SIZE
