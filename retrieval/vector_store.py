"""FAISS-backed vector store for TenderLens (Milestone 4).

Converts :class:`~ingestion.chunker.Chunk` objects into normalised embeddings,
indexes them with a ``faiss.IndexFlatIP`` (inner-product → cosine similarity on
unit-norm vectors), and persists both the FAISS index and chunk metadata to
disk.

Typical usage::

    from retrieval.vector_store import VectorStore

    store = VectorStore()          # default storage: data/vector_store/
    store.add_chunks(chunks)       # embed + index + auto-save
    results = store.search("procurement scope", top_k=5)
    for r in results:
        print(r.rank, r.score, r.text[:80])

Persistence layout::

    data/vector_store/
    ├── index.faiss    – serialised FAISS index
    └── metadata.json  – JSON list of ChunkMeta dicts

Custom exceptions
-----------------
:class:`StoreNotFoundError`
    Raised when :meth:`VectorStore.search` is called on an empty store.
:class:`DimensionMismatchError`
    Raised when newly embedded vectors have a different dimension than the
    existing FAISS index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import faiss
import numpy as np

from ingestion.chunker import Chunk
from retrieval.embedding_service import embedding_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STORE_DIR: Final[Path] = Path("data/vector_store")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class StoreNotFoundError(RuntimeError):
    """Raised when :meth:`VectorStore.search` is called on an empty store."""


class DimensionMismatchError(ValueError):
    """Raised when new vectors have a different dimension than the existing index."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ChunkMeta:
    """Metadata preserved for every indexed chunk.

    Attributes:
        chunk_id:    Deterministic chunk identifier (from :class:`~ingestion.chunker.Chunk`).
        document_id: Parent document identifier.
        page:        1-based source page number.
        chunk_index: Sequential position within the document.
        text:        The original chunk text.
    """

    chunk_id: str
    document_id: str
    page: int
    chunk_index: int
    text: str


@dataclass
class SearchResult:
    """A single result returned by :meth:`VectorStore.search`.

    Attributes:
        rank:        1-based rank (1 = most similar).
        score:       Inner-product similarity score (0–1 for normalised vectors).
        chunk_id:    Source chunk identifier.
        document_id: Source document identifier.
        page:        Source page number.
        chunk_index: Source chunk index.
        text:        Chunk text.
    """

    rank: int
    score: float
    chunk_id: str
    document_id: str
    page: int
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """FAISS-backed store for chunk embeddings with persistent metadata.

    Args:
        store_dir: Directory used for ``index.faiss`` and ``metadata.json``.
            Defaults to :data:`DEFAULT_STORE_DIR` (``data/vector_store/``).
    """

    _INDEX_FILE: Final[str] = "index.faiss"
    _META_FILE: Final[str] = "metadata.json"

    def __init__(self, store_dir: Path | None = None) -> None:
        self._store_dir: Path = store_dir if store_dir is not None else DEFAULT_STORE_DIR
        self._index: faiss.Index | None = None
        self._metadata: list[ChunkMeta] = []

        # Attempt to load an existing persisted store.
        if (self._store_dir / self._INDEX_FILE).exists():
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load existing vector store from '%s': %s – starting fresh.",
                    self._store_dir,
                    exc,
                )
                self._index = None
                self._metadata = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed *chunks*, add them to the FAISS index, and persist.

        Args:
            chunks: Chunks to index.  An empty list is a no-op.

        Raises:
            DimensionMismatchError: If the embedded vectors have a different
                dimension than the existing index.
        """
        if not chunks:
            logger.debug("add_chunks() called with empty list – no-op.")
            return

        texts = [c.text for c in chunks]
        vectors: np.ndarray = embedding_service.embed(texts)  # (N, D) float32 normalised
        dim: int = vectors.shape[1]

        if self._index is None:
            logger.info("Creating new IndexFlatIP with dim=%d.", dim)
            self._index = faiss.IndexFlatIP(dim)
        else:
            existing_dim: int = self._index.d
            if dim != existing_dim:
                raise DimensionMismatchError(
                    f"New vectors have dimension {dim} but the existing index "
                    f"expects dimension {existing_dim}."
                )

        self._index.add(vectors)

        for chunk in chunks:
            self._metadata.append(
                ChunkMeta(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )
            )

        logger.info(
            "Indexed %d chunk(s) → total %d in store.",
            len(chunks),
            self._index.ntotal,
        )
        self.save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search the store for chunks most similar to *query*.

        Args:
            query:       Query string.
            top_k:       Maximum number of results to return (≥ 1).
            document_id: When supplied, only results from this document are
                         returned.  All indexed vectors are searched and results
                         are filtered afterwards.

        Returns:
            A list of :class:`SearchResult` objects, ordered by descending
            similarity score, with ``rank`` starting at 1.  May be shorter
            than *top_k* if fewer matching results exist.

        Raises:
            ValueError:        If *top_k* < 1.
            StoreNotFoundError: If the store is empty (no chunks indexed).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {top_k!r}.")
        if self._index is None or self._index.ntotal == 0:
            raise StoreNotFoundError(
                "The vector store is empty.  Call add_chunks() before searching."
            )

        query_vec: np.ndarray = embedding_service.embed([query])  # (1, D) normalised

        # Fetch all stored vectors so we can apply the document_id filter.
        # This prioritises correctness over optimisation for this milestone.
        n_fetch: int = self._index.ntotal
        scores_matrix, idx_matrix = self._index.search(query_vec, n_fetch)

        scores: np.ndarray = scores_matrix[0]   # shape (n_fetch,)
        indices: np.ndarray = idx_matrix[0]     # shape (n_fetch,)

        results: list[SearchResult] = []
        rank = 1
        for raw_score, meta_idx in zip(scores, indices):
            if meta_idx < 0:
                continue  # FAISS sentinel for "not enough results"
            meta: ChunkMeta = self._metadata[meta_idx]
            if document_id is not None and meta.document_id != document_id:
                continue
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(raw_score),
                    chunk_id=meta.chunk_id,
                    document_id=meta.document_id,
                    page=meta.page,
                    chunk_index=meta.chunk_index,
                    text=meta.text,
                )
            )
            rank += 1
            if len(results) == top_k:
                break

        logger.debug(
            "Search returned %d result(s) for query=%r (top_k=%d, filter=%r).",
            len(results),
            query[:50],
            top_k,
            document_id,
        )
        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata to :attr:`_store_dir`.

        Creates the directory if it does not exist.  Silently skips if the
        index has never been initialised.
        """
        if self._index is None:
            return

        self._store_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._store_dir / self._INDEX_FILE
        faiss.write_index(self._index, str(index_path))

        meta_path = self._store_dir / self._META_FILE
        meta_path.write_text(
            json.dumps([asdict(m) for m in self._metadata], indent=2),
            encoding="utf-8",
        )

        logger.debug(
            "Saved FAISS index (%d vectors) to '%s'.",
            self._index.ntotal,
            self._store_dir,
        )

    def load(self) -> None:
        """Load a previously persisted FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If index or metadata files are missing.
            json.JSONDecodeError: If the metadata file is malformed.
        """
        index_path = self._store_dir / self._INDEX_FILE
        meta_path = self._store_dir / self._META_FILE

        self._index = faiss.read_index(str(index_path))

        raw_meta: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        self._metadata = [ChunkMeta(**m) for m in raw_meta]

        logger.info(
            "Loaded FAISS index (%d vectors) from '%s'.",
            self._index.ntotal,
            self._store_dir,
        )

    def clear(self) -> None:
        """Reset the store and remove persisted files if they exist."""
        self._index = None
        self._metadata = []

        for fname in (self._INDEX_FILE, self._META_FILE):
            fpath = self._store_dir / fname
            if fpath.exists():
                fpath.unlink()
                logger.debug("Removed '%s'.", fpath)

        logger.info("VectorStore cleared.")
