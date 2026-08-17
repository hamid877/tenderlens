"""POST /documents – PDF upload and ingestion endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from ingestion.document_service import document_service
from ingestion.pdf_loader import is_pdf_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Maximum accepted upload size: 50 MB
_MAX_BYTES = 50 * 1024 * 1024


class DocumentResponse(BaseModel):
    """JSON body returned after a successful document ingestion."""

    document_id: str
    filename: str
    pages: int
    status: str


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(file: UploadFile) -> DocumentResponse:
    """Accept a PDF upload, extract its text, and return document metadata.

    - Only PDF files are accepted (validated by magic bytes, not just extension).
    - Uploaded files are stored temporarily under ``data/uploads/``.
    - Text is extracted page-by-page using *pypdf*.
    """
    # ---- read -------------------------------------------------------
    file_bytes: bytes = await file.read()

    # ---- size guard -------------------------------------------------
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the 50 MB limit.",
        )

    # ---- PDF validation (magic bytes + optional extension hint) -----
    if not is_pdf_bytes(file_bytes):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Uploaded file is not a valid PDF. "
                "Only PDF documents are accepted."
            ),
        )

    # ---- ingest -----------------------------------------------------
    filename = file.filename or "upload.pdf"
    try:
        doc = document_service.ingest(filename=filename, file_bytes=file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return DocumentResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        pages=doc.page_count,
        status=doc.status,
    )
