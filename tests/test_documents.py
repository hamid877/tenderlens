"""Tests for POST /documents – PDF ingestion endpoint."""

from __future__ import annotations

import io
import struct
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers – build minimal in-memory PDFs without touching the file system
# ---------------------------------------------------------------------------

def _make_minimal_pdf(page_texts: list[str] | None = None) -> bytes:
    """Return the raw bytes of a minimal, structurally valid PDF.

    The PDF is assembled by hand so tests have zero dependency on any external
    PDF-creation library.  Each string in *page_texts* becomes the visible
    content of one page.
    """
    if page_texts is None:
        page_texts = ["TenderLens test page."]

    # We build a single-page (or multi-page) PDF using direct PDF syntax.
    # Object numbering: 1=catalog, 2=pages, then pairs (3,4), (5,6)… for pages.
    objects: dict[int, bytes] = {}
    page_ids: list[int] = []

    obj_id = 3  # start of page objects
    for text in page_texts:
        # Content stream
        stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        stream_obj = (
            f"{obj_id} 0 obj\n"
            f"<< /Length {len(stream_content)} >>\n"
            f"stream\n".encode()
            + stream_content
            + b"\nendstream\nendobj\n"
        )
        objects[obj_id] = stream_obj
        content_id = obj_id
        obj_id += 1

        # Page dictionary
        page_obj = (
            f"{obj_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\n"
            f">>\nendobj\n"
        ).encode()
        objects[obj_id] = page_obj
        page_ids.append(obj_id)
        obj_id += 1

    # Pages dictionary (obj 2)
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = (
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>\n"
        f"endobj\n"
    ).encode()

    # Catalog (obj 1)
    objects[1] = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    # Assemble body
    body = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for oid in sorted(objects):
        offsets[oid] = len(body)
        body += objects[oid]

    # Cross-reference table
    xref_offset = len(body)
    max_id = max(objects)
    xref = f"xref\n0 {max_id + 1}\n0000000000 65535 f \n".encode()
    for oid in range(1, max_id + 1):
        xref += f"{offsets[oid]:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return body + xref + trailer


def _make_pdf_file(
    page_texts: list[str] | None = None,
    filename: str = "tender.pdf",
) -> tuple[str, tuple[str, bytes, str]]:
    """Return a ``(field_name, (filename, bytes, mime))`` tuple for TestClient."""
    return ("file", (filename, _make_minimal_pdf(page_texts), "application/pdf"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUploadSuccess:
    """Successful PDF upload scenarios."""

    def test_returns_201(self) -> None:
        """A valid single-page PDF should yield HTTP 201."""
        response = client.post("/documents", files=[_make_pdf_file()])
        assert response.status_code == 201

    def test_response_shape(self) -> None:
        """Response JSON must contain the four required fields."""
        response = client.post("/documents", files=[_make_pdf_file()])
        data = response.json()
        assert "document_id" in data
        assert "filename" in data
        assert "pages" in data
        assert "status" in data

    def test_status_is_processed(self) -> None:
        """Status field should be 'processed'."""
        response = client.post("/documents", files=[_make_pdf_file()])
        assert response.json()["status"] == "processed"

    def test_filename_preserved(self) -> None:
        """The original filename should be echoed back."""
        response = client.post(
            "/documents",
            files=[_make_pdf_file(filename="my_tender.pdf")],
        )
        assert response.json()["filename"] == "my_tender.pdf"

    def test_single_page_count(self) -> None:
        """A single-page PDF should report pages=1."""
        response = client.post(
            "/documents",
            files=[_make_pdf_file(page_texts=["Page one content."])],
        )
        assert response.json()["pages"] == 1

    def test_multi_page_count(self) -> None:
        """A three-page PDF should report pages=3."""
        response = client.post(
            "/documents",
            files=[
                _make_pdf_file(
                    page_texts=["Page 1.", "Page 2.", "Page 3."],
                    filename="multi.pdf",
                )
            ],
        )
        assert response.json()["pages"] == 3

    def test_document_id_is_uuid(self) -> None:
        """document_id must be a valid UUID string."""
        response = client.post("/documents", files=[_make_pdf_file()])
        doc_id = response.json()["document_id"]
        # Raises ValueError if not a valid UUID
        uuid.UUID(doc_id)

    def test_each_upload_gets_unique_id(self) -> None:
        """Two uploads of the same file must receive different document IDs."""
        r1 = client.post("/documents", files=[_make_pdf_file()])
        r2 = client.post("/documents", files=[_make_pdf_file()])
        assert r1.json()["document_id"] != r2.json()["document_id"]


class TestUploadRejection:
    """Invalid upload scenarios."""

    def test_plain_text_rejected(self) -> None:
        """Uploading a plain-text file should return 422."""
        response = client.post(
            "/documents",
            files=[("file", ("notes.txt", b"This is not a PDF.", "text/plain"))],
        )
        assert response.status_code == 422

    def test_fake_pdf_extension_rejected(self) -> None:
        """A file named .pdf but containing non-PDF bytes should be rejected."""
        response = client.post(
            "/documents",
            files=[("file", ("fake.pdf", b"FAKE CONTENT", "application/pdf"))],
        )
        assert response.status_code == 422

    def test_jpeg_rejected(self) -> None:
        """A JPEG file should be rejected regardless of declared MIME type."""
        jpeg_magic = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client.post(
            "/documents",
            files=[("file", ("image.jpg", jpeg_magic, "image/jpeg"))],
        )
        assert response.status_code == 422

    def test_empty_file_rejected(self) -> None:
        """An empty file should be rejected."""
        response = client.post(
            "/documents",
            files=[("file", ("empty.pdf", b"", "application/pdf"))],
        )
        assert response.status_code == 422

    def test_rejection_body_contains_detail(self) -> None:
        """Rejection response should include a human-readable 'detail' field."""
        response = client.post(
            "/documents",
            files=[("file", ("bad.pdf", b"NOT PDF", "application/pdf"))],
        )
        assert "detail" in response.json()


class TestHealthStillWorks:
    """Ensure the existing health endpoint is unaffected."""

    def test_health_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
