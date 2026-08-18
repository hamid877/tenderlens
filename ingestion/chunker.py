"""Page-aware text chunking for TenderLens.

This module splits page-level text extracted by :mod:`ingestion.pdf_loader`
into overlapping chunks while preserving page metadata.  Embeddings and
vector-store integration are handled in later milestones.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from ingestion.pdf_loader import PageContent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 500   # characters per chunk
DEFAULT_OVERLAP: int = 100      # overlap characters between consecutive chunks


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk produced from one page of a document.

    Attributes:
        chunk_id:    Deterministic identifier derived from document ID, page
                     number, and chunk index (SHA-256 hex prefix).
        document_id: ID of the parent document.
        page:        1-based page number this chunk was produced from.
        chunk_index: Sequential 0-based position of this chunk within the
                     document (across all pages).
        text:        The chunk text, already normalised.
    """

    chunk_id: str
    document_id: str
    page: int
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalise whitespace in *text*.

    * Collapses runs of any whitespace (spaces, tabs, non-breaking spaces,
      form-feeds, etc.) into a single space character.
    * Strips leading / trailing whitespace from the result.
    * Preserves intentional line breaks as a single newline so paragraph
      structure survives; consecutive blank lines are collapsed to one.
    """
    # Normalise line endings first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace (spaces, tabs, \xA0, …) per line
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    # Collapse multiple consecutive blank lines into one blank line
    normalised_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        normalised_lines.append(line)
        prev_blank = is_blank
    return "\n".join(normalised_lines).strip()


def _make_chunk_id(document_id: str, page: int, chunk_index: int) -> str:
    """Return a short, deterministic identifier for a chunk.

    The ID is the first 16 hex characters of the SHA-256 digest of the
    concatenated ``document_id``, ``page``, and ``chunk_index``.  This
    guarantees uniqueness within a document without using random UUIDs.
    """
    raw = f"{document_id}:p{page}:c{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_page(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split *text* into overlapping windows of at most *chunk_size* chars.

    The algorithm walks the text character-by-character advancing by
    ``chunk_size - overlap`` on each step.  If a page's normalised text fits
    within *chunk_size*, a single-element list is returned.

    Args:
        text:       Pre-normalised page text (non-empty).
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters shared between successive chunks.

    Returns:
        A list of non-empty text strings.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size!r}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative; got {overlap!r}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # guard against whitespace-only windows
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[PageContent],
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Convert page-level text into an ordered list of :class:`Chunk` objects.

    Text is normalised before splitting.  Pages whose normalised text is empty
    (or whitespace-only) produce no chunks.  Page boundaries are never crossed:
    a chunk always belongs to exactly one page.

    Chunk indices are sequential across the whole document (not per-page) so
    that downstream code can order chunks by ``chunk_index`` alone.

    Args:
        pages:       Ordered list of :class:`~ingestion.pdf_loader.PageContent`
                     objects as produced by the PDF loader.
        document_id: Stable identifier for the parent document.
        chunk_size:  Maximum character count per chunk (default
                     :data:`DEFAULT_CHUNK_SIZE`).
        overlap:     Character overlap between adjacent chunks (default
                     :data:`DEFAULT_OVERLAP`).

    Returns:
        A list of :class:`Chunk` objects in page order, with sequential
        ``chunk_index`` values starting from 0.
    """
    result: list[Chunk] = []
    global_index = 0

    for page_content in pages:
        normalised = _normalize_text(page_content.text)
        if not normalised:
            logger.debug(
                "Page %d of document '%s' has no extractable text; skipping.",
                page_content.page_number,
                document_id,
            )
            continue

        text_windows = _split_page(normalised, chunk_size, overlap)
        for window in text_windows:
            chunk = Chunk(
                chunk_id=_make_chunk_id(document_id, page_content.page_number, global_index),
                document_id=document_id,
                page=page_content.page_number,
                chunk_index=global_index,
                text=window,
            )
            result.append(chunk)
            global_index += 1

    logger.debug(
        "Produced %d chunks from %d pages for document '%s'.",
        len(result),
        len(pages),
        document_id,
    )
    return result
