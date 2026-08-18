"""Tests for Milestone 4: retrieval.embedding_service and retrieval.vector_store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingestion.chunker import Chunk, _make_chunk_id
from retrieval.embedding_service import EMBEDDING_DIM, EmbeddingService, embedding_service
from retrieval.vector_store import (
    DimensionMismatchError,
    StoreNotFoundError,
    VectorStore,
)


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


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    """Fresh :class:`VectorStore` pointing at a temp directory."""
    return VectorStore(store_dir=tmp_path)


@pytest.fixture()
def three_chunks() -> list[Chunk]:
    """Three synthetic chunks with distinct topics."""
    return [
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


# ===========================================================================
# EmbeddingService unit tests
# ===========================================================================


class TestEmbeddingServiceShape:
    def test_embed_returns_correct_shape(self) -> None:
        texts = ["tender clause one", "tender clause two", "tender clause three"]
        vecs = embedding_service.embed(texts)
        assert vecs.shape == (3, EMBEDDING_DIM)

    def test_embed_single_text(self) -> None:
        vecs = embedding_service.embed(["single text"])
        assert vecs.shape == (1, EMBEDDING_DIM)

    def test_embed_returns_float32(self) -> None:
        vecs = embedding_service.embed(["hello"])
        assert vecs.dtype == np.float32


class TestEmbeddingServiceNormalization:
    def test_embed_normalized(self) -> None:
        """Every row must have unit L2-norm (within float tolerance)."""
        texts = ["alpha", "beta", "gamma", "delta"]
        vecs = embedding_service.embed(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)

    def test_embed_single_normalized(self) -> None:
        vecs = embedding_service.embed(["single"])
        norm = float(np.linalg.norm(vecs[0]))
        assert abs(norm - 1.0) < 1e-5


class TestEmbeddingServiceErrors:
    def test_embed_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            embedding_service.embed([])


class TestEmbeddingServiceSingleton:
    def test_model_loaded_once(self) -> None:
        """The module-level singleton is the same object on repeated imports."""
        # Re-importing should return the exact same singleton object.
        import retrieval.embedding_service as _mod

        assert _mod.embedding_service is embedding_service

    def test_singleton_is_embedding_service_instance(self) -> None:
        assert isinstance(embedding_service, EmbeddingService)


# ===========================================================================
# VectorStore unit tests
# ===========================================================================


class TestVectorStoreAddSearch:
    def test_add_chunks_and_search_returns_results(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("financial requirements for bidders")
        assert len(results) >= 1

    def test_search_top_k_respected(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("bridge construction", top_k=2)
        assert len(results) <= 2

    def test_search_returns_all_when_k_exceeds_total(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("something", top_k=100)
        assert len(results) == 3  # only 3 chunks exist

    def test_search_ranks_start_at_one(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("procurement", top_k=3)
        assert results[0].rank == 1

    def test_search_ranks_are_sequential(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("environmental assessment", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_search_scores_descending(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("bridge technical specs", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestVectorStoreDocumentFilter:
    def test_search_document_id_filter(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        """Filter by document_id should only return chunks from that document."""
        store.add_chunks(three_chunks)
        results = store.search("procurement", top_k=5, document_id="doc-A")
        assert len(results) >= 1
        assert all(r.document_id == "doc-A" for r in results)

    def test_filter_excludes_other_documents(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("environmental impact", top_k=5, document_id="doc-B")
        assert all(r.document_id == "doc-B" for r in results)

    def test_filter_nonexistent_document_returns_empty(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("anything", top_k=5, document_id="no-such-doc")
        assert results == []


class TestVectorStoreErrors:
    def test_search_empty_store_raises(self, store: VectorStore) -> None:
        with pytest.raises(StoreNotFoundError):
            store.search("query")

    def test_search_invalid_top_k_zero_raises(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            store.search("query", top_k=0)

    def test_search_invalid_top_k_negative_raises(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            store.search("query", top_k=-5)

    def test_add_empty_chunks_is_noop(self, store: VectorStore) -> None:
        """add_chunks([]) must not raise and the store remains empty."""
        store.add_chunks([])  # should not raise
        with pytest.raises(StoreNotFoundError):
            store.search("query")

    def test_dimension_mismatch_raises(self, store: VectorStore) -> None:
        """Injecting a wrong-dimension vector into an existing index raises."""
        import faiss as _faiss

        # Build a store with dim=384 (normal).
        chunk = _make_chunk("normal text")
        store.add_chunks([chunk])

        # Manually inject a wrong-dimension index.
        wrong_dim = 128
        store._index = _faiss.IndexFlatIP(wrong_dim)

        bad_chunk = _make_chunk("another text", chunk_index=1)
        with pytest.raises(DimensionMismatchError):
            store.add_chunks([bad_chunk])


class TestVectorStoreMetadata:
    def test_search_result_preserves_chunk_id(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("procurement")
        chunk_ids = {c.chunk_id for c in three_chunks}
        assert results[0].chunk_id in chunk_ids

    def test_search_result_preserves_document_id(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("procurement")
        doc_ids = {"doc-A", "doc-B"}
        assert results[0].document_id in doc_ids

    def test_search_result_has_text(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("procurement")
        assert results[0].text != ""

    def test_search_result_has_page(
        self, store: VectorStore, three_chunks: list[Chunk]
    ) -> None:
        store.add_chunks(three_chunks)
        results = store.search("something")
        assert results[0].page >= 1


class TestVectorStorePersistence:
    def test_persistence_roundtrip(
        self, tmp_path: Path, three_chunks: list[Chunk]
    ) -> None:
        """Save and load: search works correctly after reload."""
        store1 = VectorStore(store_dir=tmp_path)
        store1.add_chunks(three_chunks)

        # Simulate a fresh process by loading from the same directory.
        store2 = VectorStore(store_dir=tmp_path)
        results = store2.search("bridge construction", top_k=3)
        assert len(results) >= 1

    def test_persistence_files_created(
        self, tmp_path: Path, three_chunks: list[Chunk]
    ) -> None:
        store = VectorStore(store_dir=tmp_path)
        store.add_chunks(three_chunks)
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_clear_removes_files(
        self, tmp_path: Path, three_chunks: list[Chunk]
    ) -> None:
        store = VectorStore(store_dir=tmp_path)
        store.add_chunks(three_chunks)
        store.clear()
        assert not (tmp_path / "index.faiss").exists()
        assert not (tmp_path / "metadata.json").exists()

    def test_clear_resets_store(
        self, tmp_path: Path, three_chunks: list[Chunk]
    ) -> None:
        store = VectorStore(store_dir=tmp_path)
        store.add_chunks(three_chunks)
        store.clear()
        with pytest.raises(StoreNotFoundError):
            store.search("query")


# ===========================================================================
# Integration test
# ===========================================================================


class TestRetrievalIntegration:
    def test_retrieval_integration(self, tmp_path: Path) -> None:
        """End-to-end: 3 synthetic chunks, query returns semantically correct chunk at rank 1.

        The query "financial statements for bidders" should retrieve the
        procurement chunk ahead of the bridge/environmental ones.
        """
        chunks = [
            _make_chunk(
                "The procurement authority requires all bidders to submit financial statements.",
                document_id="int-doc",
                page=1,
                chunk_index=0,
            ),
            _make_chunk(
                "Technical specifications for the construction of a new bridge over the river.",
                document_id="int-doc",
                page=2,
                chunk_index=1,
            ),
            _make_chunk(
                "Environmental impact assessment must be completed before project commencement.",
                document_id="int-doc",
                page=3,
                chunk_index=2,
            ),
        ]

        store = VectorStore(store_dir=tmp_path)
        store.add_chunks(chunks)

        results = store.search("financial statements required from bidders", top_k=3)

        assert len(results) >= 1
        assert results[0].rank == 1
        # The procurement/financial chunk should be the most similar.
        assert "financial" in results[0].text.lower() or "bidder" in results[0].text.lower()
