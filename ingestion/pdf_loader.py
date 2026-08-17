"""PDF loading and text extraction using pypdf."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import pypdf

logger = logging.getLogger(__name__)

# Magic bytes for PDF files (%PDF-)
_PDF_MAGIC = b"%PDF-"


@dataclass
class PageContent:
    """Text content extracted from a single PDF page."""

    page_number: int  # 1-based
    text: str


@dataclass
class LoadedPDF:
    """Result of loading and extracting a PDF document."""

    filename: str
    page_count: int
    pages: list[PageContent] = field(default_factory=list)


def is_pdf_bytes(data: bytes) -> bool:
    """Return True if *data* begins with the PDF magic bytes."""
    return data[:5] == _PDF_MAGIC


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> LoadedPDF:
    """Open a PDF from raw bytes and extract text page-by-page.

    Args:
        file_bytes: Raw PDF bytes.
        filename:   Original filename (used only for labelling).

    Returns:
        A :class:`LoadedPDF` instance with one :class:`PageContent` per page.

    Raises:
        ValueError: If *file_bytes* is not a valid PDF.
    """
    if not is_pdf_bytes(file_bytes):
        raise ValueError(f"'{filename}' does not appear to be a PDF file.")

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    page_count = len(reader.pages)

    pages: list[PageContent] = []
    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not extract text from page %d of '%s'; storing empty string.",
                page_number,
                filename,
            )
            text = ""
        pages.append(PageContent(page_number=page_number, text=text))

    logger.debug("Extracted %d pages from '%s'.", page_count, filename)
    return LoadedPDF(filename=filename, page_count=page_count, pages=pages)
