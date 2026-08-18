"""Tests for Milestone 5: retrieval.retrieval_service.RetrievalService.

Covers:
- Relevant query returns expected chunk first (semantic ranking).
- top_k is respected.
- document_id filtering works.
- Invalid / empty query is handled (InvalidQueryError).
- No-result / empty-store behaviour (StoreNotFoundError propagated).
- RetrievalService correctly delegates to VectorStore / EmbeddingService.
- Existing Milestone-4 tests continue to pass (verified by running the full
  test suite; no imports from test_retrieval are duplicated here).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.chunker import Chunk, _make_chunk_id
from retrieval.retrieval_service import InvalidQueryError, RetrievalService
from retrieval.vector_store import SearchResult, StoreNotFoundError, VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str,
    document_id: str = "doc-001",
    page: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    """Build a synthetic :class:`Chunk` for testing."""
    return Chunk(
        chunk_id=_make_chunk_id(document_id, page, chunk_index),
        document_id=document_id,
        page=page,
        chunk_index=chunk_index,
        text=text,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    """Fresh :class:`VectorStore` in a temp directory."""
    return VectorStore(store_dir=tmp_path)


@pytest.fixture()
def service(store: VectorStore) -> RetrievalService:
    """A :class:`RetrievalService` backed by a fresh temporary store."""
    return RetrievalService(vector_store=store)


@pytest.fixture()
def populated_store(tmp_path: Path) -> VectorStore:
    """VectorStore pre-loaded with three semantically distinct chunks."""
    chunks = [
        _make_chunk(
            "The procurement authority requires all bidders to submit financial statements.",
            document_id="doc-A",
            page=1,
            chunk_index=0,
        ),
        _make_chunk(
            "Technical specifications for the construction of a new bridge over the river.",
            document_id="doc-A",
            page=2,
            chunk_index=1,
        ),
        _make_chunk(
            "Environmental impact assessment must be completed before project commencement.",
            document_id="doc-B",
            page=1,
            chunk_index=0,
        ),
    ]
    vs = VectorStore(store_dir=tmp_path)
    vs.add_chunks(chunks)
    return vs


@pytest.fixture()
def populated_service(populated_store: VectorStore) -> RetrievalService:
    """RetrievalService backed by the pre-populated store."""
    return RetrievalService(vector_store=populated_store)


# ===========================================================================
# Semantic relevance
# ===========================================================================


class TestSemanticRanking:
    def test_relevant_query_returns_expected_chunk_first(
        self, populated_service: RetrievalService
    ) -> None:
        """A financial-domain query should rank the financial chunk at rank 1."""
        results = populated_service.search(
            "financial statements required from bidders", top_k=3
        )
        assert len(results) >= 1
        assert results[0].rank == 1
        top_text = results[0].text.lower()
        assert "financial" in top_text or "bidder" in top_text

    def test_bridge_query_returns_bridge_chunk_first(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search(
            "bridge construction technical specifications", top_k=3
        )
        assert len(results) >= 1
        assert "bridge" in results[0].text.lower() or "construction" in results[0].text.lower()

    def test_scores_are_in_descending_order(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("procurement scope", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_from_one(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("environmental assessment", top_k=3)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))


# ===========================================================================
# top_k respected
# ===========================================================================


class TestTopK:
    def test_top_k_limits_results(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("construction bridge", top_k=1)
        assert len(results) == 1

    def test_top_k_two_returns_at_most_two(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("procurement financial", top_k=2)
        assert len(results) <= 2

    def test_top_k_exceeds_total_returns_all(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("something", top_k=100)
        assert len(results) == 3  # only 3 chunks in the store

    def test_top_k_default_is_five(self, tmp_path: Path) -> None:
        """Default top_k=5 with fewer than 5 chunks returns all available."""
        chunks = [_make_chunk(f"chunk text {i}", chunk_index=i) for i in range(3)]
        vs = VectorStore(store_dir=tmp_path)
        vs.add_chunks(chunks)
        svc = RetrievalService(vector_store=vs)
        results = svc.search("chunk text")
        # Should return all 3 (< 5)
        assert len(results) == 3


# ===========================================================================
# document_id filtering
# ===========================================================================


class TestDocumentIdFilter:
    def test_filter_returns_only_matching_document(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("procurement", top_k=5, document_id="doc-A")
        assert len(results) >= 1
        assert all(r.document_id == "doc-A" for r in results)

    def test_filter_excludes_other_documents(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search(
            "environmental impact", top_k=5, document_id="doc-B"
        )
        assert all(r.document_id == "doc-B" for r in results)

    def test_filter_nonexistent_document_returns_empty(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search(
            "anything", top_k=5, document_id="no-such-doc"
        )
        assert results == []

    def test_no_filter_returns_chunks_from_all_documents(
        self, populated_service: RetrievalService
    ) -> None:
        results = populated_service.search("assessment procurement bridge", top_k=5)
        doc_ids = {r.document_id for r in results}
        assert "doc-A" in doc_ids
        assert "doc-B" in doc_ids


# ===========================================================================
# Invalid / empty query handling
# ===========================================================================


class TestInvalidQuery:
    def test_empty_string_raises_invalid_query_error(
        self, service: RetrievalService
    ) -> None:
        with pytest.raises(InvalidQueryError):
            service.search("")

    def test_whitespace_only_query_raises(self, service: RetrievalService) -> None:
        with pytest.raises(InvalidQueryError):
            service.search("   ")

    def test_tab_only_query_raises(self, service: RetrievalService) -> None:
        with pytest.raises(InvalidQueryError):
            service.search("\t\t")

    def test_newline_only_query_raises(self, service: RetrievalService) -> None:
        with pytest.raises(InvalidQueryError):
            service.search("\n\n")

    def test_invalid_query_error_is_value_error_subclass(
        self, service: RetrievalService
    ) -> None:
        """InvalidQueryError must be a ValueError so callers can catch broadly."""
        with pytest.raises(ValueError):
            service.search("")

    def test_invalid_top_k_propagated(
        self, populated_service: RetrievalService
    ) -> None:
        """VectorStore.ValueError for bad top_k should propagate unchanged."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            populated_service.search("financial", top_k=0)


# ===========================================================================
# Empty store behaviour
# ===========================================================================


class TestEmptyStore:
    def test_search_on_empty_store_raises_store_not_found(
        self, service: RetrievalService
    ) -> None:
        """StoreNotFoundError propagated from VectorStore for empty store."""
        with pytest.raises(StoreNotFoundError):
            service.search("anything")

    def test_valid_query_on_empty_store_still_raises(
        self, service: RetrievalService
    ) -> None:
        with pytest.raises(StoreNotFoundError):
            service.search("financial statements", top_k=3)


# ===========================================================================
# Delegation to VectorStore / EmbeddingService
# ===========================================================================


class TestDelegation:
    """Verify RetrievalService delegates to VectorStore without duplicating logic."""

    def test_search_calls_vector_store_search(self, tmp_path: Path) -> None:
        """RetrievalService.search must call VectorStore.search exactly once."""
        mock_store = MagicMock(spec=VectorStore)
        mock_result = SearchResult(
            rank=1,
            score=0.95,
            chunk_id="abc123",
            document_id="doc-A",
            page=1,
            chunk_index=0,
            text="financial statements",
        )
        mock_store.search.return_value = [mock_result]

        svc = RetrievalService(vector_store=mock_store)
        results = svc.search("financial statements", top_k=3, document_id="doc-A")

        mock_store.search.assert_called_once_with(
            query="financial statements",
            top_k=3,
            document_id="doc-A",
        )
        assert results == [mock_result]

    def test_search_strips_query_before_delegating(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace is stripped before passing to the store."""
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search.return_value = []

        svc = RetrievalService(vector_store=mock_store)
        svc.search("  financial statements  ", top_k=2)

        mock_store.search.assert_called_once_with(
            query="financial statements",
            top_k=2,
            document_id=None,
        )

    def test_invalid_query_does_not_call_vector_store(self) -> None:
        """VectorStore.search must NOT be called when the query is invalid."""
        mock_store = MagicMock(spec=VectorStore)
        svc = RetrievalService(vector_store=mock_store)

        with pytest.raises(InvalidQueryError):
            svc.search("")

        mock_store.search.assert_not_called()

    def test_store_not_found_error_propagated_unchanged(self) -> None:
        """StoreNotFoundError from VectorStore must bubble up unmodified."""
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search.side_effect = StoreNotFoundError("empty store")

        svc = RetrievalService(vector_store=mock_store)
        with pytest.raises(StoreNotFoundError, match="empty store"):
            svc.search("valid query")

    def test_returns_search_results_verbatim(self) -> None:
        """RetrievalService returns whatever VectorStore.search returns."""
        expected = [
            SearchResult(
                rank=1,
                score=0.88,
                chunk_id="aaaa",
                document_id="doc-X",
                page=2,
                chunk_index=3,
                text="some text",
            ),
            SearchResult(
                rank=2,
                score=0.72,
                chunk_id="bbbb",
                document_id="doc-X",
                page=3,
                chunk_index=4,
                text="other text",
            ),
        ]
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search.return_value = expected

        svc = RetrievalService(vector_store=mock_store)
        results = svc.search("query", top_k=2)
        assert results == expected


# ===========================================================================
# Metadata preservation
# ===========================================================================


class TestMetadataPreservation:
    def test_chunk_id_preserved(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert results[0].chunk_id != ""

    def test_document_id_preserved(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert results[0].document_id in {"doc-A", "doc-B"}

    def test_page_number_preserved(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert results[0].page >= 1

    def test_chunk_index_preserved(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert results[0].chunk_index >= 0

    def test_text_preserved(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert results[0].text != ""

    def test_score_is_float(self, populated_service: RetrievalService) -> None:
        results = populated_service.search("procurement financial")
        assert isinstance(results[0].score, float)
