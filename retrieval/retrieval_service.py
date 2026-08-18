"""Semantic retrieval service for TenderLens (Milestone 5).

Provides a thin orchestration layer that:

1. Validates the user query.
2. Delegates embedding to :class:`~retrieval.embedding_service.EmbeddingService`
   (via :class:`~retrieval.vector_store.VectorStore`).
3. Runs FAISS similarity search through :class:`~retrieval.vector_store.VectorStore`.
4. Returns ranked :class:`~retrieval.vector_store.SearchResult` objects.

No embedding or FAISS logic is duplicated here.

Typical usage::

    from retrieval.retrieval_service import RetrievalService
    from retrieval.vector_store import VectorStore

    store = VectorStore()
    service = RetrievalService(vector_store=store)

    results = service.search("procurement financial statements", top_k=5)
    for r in results:
        print(r.rank, r.score, r.text[:80])

Custom exceptions
-----------------
:class:`InvalidQueryError`
    Raised when the query string is empty or contains only whitespace.
"""

from __future__ import annotations

import logging
from typing import Optional

from retrieval.vector_store import SearchResult, StoreNotFoundError, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InvalidQueryError(ValueError):
    """Raised when a search query is empty or whitespace-only."""


# ---------------------------------------------------------------------------
# RetrievalService
# ---------------------------------------------------------------------------


class RetrievalService:
    """Orchestrates semantic search over indexed tender chunks.

    Args:
        vector_store: A :class:`~retrieval.vector_store.VectorStore` instance
            that already contains (or will contain) indexed chunks.  The
            caller is responsible for populating the store via
            :meth:`~retrieval.vector_store.VectorStore.add_chunks` before
            calling :meth:`search`.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store: VectorStore = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most semantically similar chunks for *query*.

        The method embeds *query* (via the underlying
        :class:`~retrieval.vector_store.VectorStore`) and performs an
        inner-product (cosine) search against all indexed chunk vectors.
        Results are already ranked by descending similarity score; ``rank``
        starts at 1.

        Args:
            query:       Natural-language question or keyword string.
            top_k:       Maximum number of results to return (>= 1).
            document_id: Optional document filter.  When provided, only chunks
                         from that document are returned.  Set to ``None``
                         (default) to search across all documents.

        Returns:
            A list of :class:`~retrieval.vector_store.SearchResult` objects
            ordered by descending similarity score.  The list may be shorter
            than *top_k* if fewer matching chunks exist.  Returns an **empty
            list** when no chunks match the document_id filter or the store
            contains no results for the query.

        Raises:
            InvalidQueryError:  If *query* is empty or whitespace-only.
            ValueError:         If *top_k* < 1 (propagated from VectorStore).
            StoreNotFoundError: If the vector store is empty (propagated from
                                VectorStore).
        """
        stripped = query.strip() if isinstance(query, str) else ""
        if not stripped:
            raise InvalidQueryError(
                "search() requires a non-empty query string; got an empty or "
                f"whitespace-only value: {query!r}"
            )

        logger.debug(
            "RetrievalService.search called: query=%r top_k=%d document_id=%r",
            stripped[:80],
            top_k,
            document_id,
        )

        results: list[SearchResult] = self._store.search(
            query=stripped,
            top_k=top_k,
            document_id=document_id,
        )

        logger.debug(
            "RetrievalService.search returned %d result(s).",
            len(results),
        )
        return results
