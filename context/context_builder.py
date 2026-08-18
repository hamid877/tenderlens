"""Context construction for TenderLens (Milestone 6).

Converts a ranked list of :class:`~retrieval.vector_store.SearchResult` objects
into a deterministic, structured evidence context string ready for passing to
an LLM prompt.

No retrieval, embedding, or LLM calls are performed here.  The builder is a
pure transformation layer: it accepts already-retrieved results and formats
them into a clearly delimited evidence block string.

Typical usage::

    from retrieval.vector_store import SearchResult
    from context.context_builder import ContextBuilder

    results: list[SearchResult] = retrieval_service.search("procurement scope")
    builder = ContextBuilder(max_chunks=5)
    ctx = builder.build(results)
    # ctx.context_string is ready to be injected into an LLM prompt

Custom exceptions
-----------------
:class:`EmptyEvidenceError`
    Raised when :meth:`ContextBuilder.build` receives an empty evidence list.
:class:`InvalidMaxChunksError`
    Raised when *max_chunks* is not a positive integer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLOCK_SEPARATOR: str = "\n" + ("=" * 60) + "\n"
_DEFAULT_MAX_CHUNKS: int = 10


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EmptyEvidenceError(ValueError):
    """Raised when :meth:`ContextBuilder.build` receives an empty list."""


class InvalidMaxChunksError(ValueError):
    """Raised when *max_chunks* is not a positive integer (>= 1)."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class BuiltContext:
    """The structured output of :meth:`ContextBuilder.build`.

    Attributes:
        context_string: Deterministic, LLM-ready text with evidence blocks.
        chunks_used:    Number of :class:`~retrieval.vector_store.SearchResult`
                        objects included (after applying *max_chunks*).
        total_results:  Total number of results passed to :meth:`~ContextBuilder.build`
                        before the *max_chunks* cap was applied.
    """

    context_string: str
    chunks_used: int
    total_results: int


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Builds a structured evidence context string from retrieval results.

    Args:
        max_chunks: Maximum number of evidence chunks to include in the built
            context.  Must be >= 1.  Defaults to :data:`_DEFAULT_MAX_CHUNKS`.
            Results beyond this limit (by retrieval rank) are silently dropped.

    Raises:
        InvalidMaxChunksError: If *max_chunks* is less than 1.
    """

    def __init__(self, max_chunks: int = _DEFAULT_MAX_CHUNKS) -> None:
        if not isinstance(max_chunks, int) or max_chunks < 1:
            raise InvalidMaxChunksError(
                f"max_chunks must be a positive integer (>= 1); got {max_chunks!r}."
            )
        self._max_chunks: int = max_chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, results: list[SearchResult]) -> BuiltContext:
        """Build a deterministic evidence context from *results*.

        The input list is expected to be pre-ranked (retrieval order is
        preserved exactly; no re-sorting is performed).  At most
        :attr:`max_chunks` results are included, in their original order.

        Each evidence block has the form::

            ============================================================
            [Evidence 1]
            Source     : <document_id>
            Page       : <page>
            Section    : <section or N/A>
            Chunk ID   : <chunk_id>
            Chunk Index: <chunk_index>
            Rank       : <rank>
            Score      : <score>
            ------------------------------------------------------------
            <text>
            ============================================================

        Args:
            results: Pre-ranked list of :class:`~retrieval.vector_store.SearchResult`
                objects returned by the retrieval layer.  Must be non-empty.

        Returns:
            A :class:`BuiltContext` containing the formatted context string,
            the number of chunks actually used, and the total results count
            before the cap.

        Raises:
            EmptyEvidenceError: If *results* is ``None`` or an empty list.
        """
        if not results:
            raise EmptyEvidenceError(
                "build() requires at least one SearchResult; got an empty list."
            )

        total_results: int = len(results)
        selected: list[SearchResult] = results[: self._max_chunks]

        blocks: list[str] = []
        for position, result in enumerate(selected, start=1):
            block = self._format_block(position, result)
            blocks.append(block)

        context_string: str = _BLOCK_SEPARATOR.join(blocks)

        logger.debug(
            "ContextBuilder.build: %d result(s) in, %d chunk(s) used (max=%d).",
            total_results,
            len(selected),
            self._max_chunks,
        )

        return BuiltContext(
            context_string=context_string,
            chunks_used=len(selected),
            total_results=total_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_block(position: int, result: SearchResult) -> str:
        """Format a single evidence block for *result*.

        Args:
            position: 1-based display position within the built context.
            result:   The :class:`~retrieval.vector_store.SearchResult` to format.

        Returns:
            A multi-line string representing the evidence block.
        """
        # ``section`` is not yet part of SearchResult; handle gracefully.
        section: str = getattr(result, "section", None) or "N/A"

        header = (
            f"[Evidence {position}]\n"
            f"Source     : {result.document_id}\n"
            f"Page       : {result.page}\n"
            f"Section    : {section}\n"
            f"Chunk ID   : {result.chunk_id}\n"
            f"Chunk Index: {result.chunk_index}\n"
            f"Rank       : {result.rank}\n"
            f"Score      : {result.score:.6f}\n"
            f"{'-' * 60}\n"
            f"{result.text}"
        )
        return header
