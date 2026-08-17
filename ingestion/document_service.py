"""In-memory document store and ingestion orchestration."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.pdf_loader import LoadedPDF, extract_text_from_pdf

logger = logging.getLogger(__name__)

# Local directory where uploaded files are saved temporarily.
UPLOAD_DIR = Path("data/uploads")


@dataclass
class Document:
    """Represents an ingested document stored in memory."""

    document_id: str
    filename: str
    page_count: int
    status: str  # "processed" | "error"
    pages: list[dict] = field(default_factory=list)


class DocumentService:
    """Orchestrates PDF ingestion and provides an in-memory document store."""

    def __init__(self, upload_dir: Path | None = None) -> None:
        self._upload_dir: Path = upload_dir if upload_dir is not None else UPLOAD_DIR
        self._store: dict[str, Document] = {}
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, filename: str, file_bytes: bytes) -> Document:
        """Validate, extract, and store a PDF document.

        Args:
            filename:   Original filename supplied by the caller.
            file_bytes: Raw bytes of the uploaded file.

        Returns:
            The resulting :class:`Document` with status ``"processed"``.

        Raises:
            ValueError: When the file is not a valid PDF.
        """
        loaded: LoadedPDF = extract_text_from_pdf(file_bytes, filename)

        doc_id = str(uuid.uuid4())
        document = Document(
            document_id=doc_id,
            filename=filename,
            page_count=loaded.page_count,
            status="processed",
            pages=[
                {"page_number": p.page_number, "text": p.text}
                for p in loaded.pages
            ],
        )

        self._store[doc_id] = document

        # Persist the raw file locally (best-effort; never blocks ingestion).
        self._save_to_disk(doc_id, filename, file_bytes)

        logger.info(
            "Ingested document '%s' → id=%s  pages=%d",
            filename,
            doc_id,
            loaded.page_count,
        )
        return document

    def get(self, document_id: str) -> Document | None:
        """Return the :class:`Document` with *document_id*, or ``None``."""
        return self._store.get(document_id)

    @property
    def count(self) -> int:
        """Number of documents currently stored."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_to_disk(
        self, doc_id: str, filename: str, data: bytes
    ) -> None:
        """Write *data* to ``<upload_dir>/<doc_id>_<filename>``."""
        safe_name = Path(filename).name  # strip any path components
        dest = self._upload_dir / f"{doc_id}_{safe_name}"
        try:
            dest.write_bytes(data)
            logger.debug("Saved upload to %s", dest)
        except OSError as exc:
            logger.warning("Could not save upload to disk: %s", exc)


# Module-level singleton – shared across the application lifetime.
document_service = DocumentService()
